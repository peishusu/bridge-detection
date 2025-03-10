# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2d, chamfer_distance, min_area_polygons
from mmdet.utils import get_device
from mmcv.runner import force_fp32
from mmdet.core import images_to_levels, multi_apply, unmap
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmdet.core.utils import select_single_mlvl
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

from mmrotate.core import (build_assigner, build_sampler,
                           multiclass_nms_rotated, obb2poly, poly2obb)
from ..builder import ROTATED_HEADS, build_loss
from .utils import levels_to_images,get_num_level_anchors_inside,points_center_pts
import numpy as np
import cv2
import torch.nn.functional as F


def ChamferDistance2D(point_set_1,
                      point_set_2,
                      distance_weight=0.05,
                      eps=1e-12):
    """Compute the Chamfer distance between two point sets.

    Args:
        point_set_1 (torch.tensor): point set 1 with shape (N_pointsets,
                                    N_points, 2)
        point_set_2 (torch.tensor): point set 2 with shape (N_pointsets,
                                    N_points, 2)

    Returns:
        dist (torch.tensor): chamfer distance between two point sets
                             with shape (N_pointsets,)
    """
    assert point_set_1.dim() == point_set_2.dim()
    assert point_set_1.shape[-1] == point_set_2.shape[-1]
    assert point_set_1.dim() <= 3
    dist1, dist2, _, _ = chamfer_distance(point_set_1, point_set_2)
    dist1 = torch.sqrt(torch.clamp(dist1, eps))
    dist2 = torch.sqrt(torch.clamp(dist2, eps))
    dist = distance_weight * (dist1.mean(-1) + dist2.mean(-1)) / 2.0

    return dist


##
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out



@ROTATED_HEADS.register_module()
class OrientedQuaRepPointsHead(BaseDenseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 point_feat_channels=256,
                 stacked_convs=3,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 conv_bias='auto',
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_spatial_init=dict(
                     type='SpatialBorderLoss', loss_weight=0.05),
                 loss_spatial_refine=dict(
                     type='SpatialBorderLoss', loss_weight=0.1),
                 conv_cfg=None,
                 norm_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 center_init=True,
                 version='oc',
                 top_ratio=0.4,
                 init_qua_weight=0.2,
                 ori_qua_weight=0.3,
                 poc_qua_weight=0.1,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='reppoints_cls_out',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        super(OrientedQuaRepPointsHead, self).__init__(init_cfg)
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.center_init = center_init

        # we use deform conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))   # 形变卷积
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        ##### deform conv to extract context features
        num_context_points=25
        self.num_context_points=num_context_points
        self.dcn_kernel2 = int(np.sqrt(num_context_points))  #5, 形变卷积
        self.dcn_pad2 = int((self.dcn_kernel2 - 1) / 2) #2
        assert self.dcn_kernel2 * self.dcn_kernel2 == num_context_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel2 % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base2 = np.arange(-self.dcn_pad2,
                             self.dcn_pad2 + 1).astype(np.float64)
        dcn_base_y2 = np.repeat(dcn_base2, self.dcn_kernel2)
        dcn_base_x2 = np.tile(dcn_base2, self.dcn_kernel2)

        self.dcn_base_y2=dcn_base_y2
        self.dcn_base_x2=dcn_base_x2

        dcn_base_offset2 = np.stack([dcn_base_y2, dcn_base_x2], axis=1).reshape(
            (-1))
        self.dcn_base_offset2 = torch.tensor(dcn_base_offset2).view(1, -1, 1, 1)


        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_cls = build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.prior_generator = MlvlPointGenerator(
            self.point_strides, offset=0.)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        if self.train_cfg:
            self.init_assigner = build_assigner(self.train_cfg.init.assigner)
            self.refine_assigner = build_assigner(
                self.train_cfg.refine.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.loss_spatial_init = build_loss(loss_spatial_init)
        self.loss_spatial_refine = build_loss(loss_spatial_refine)
        self.init_qua_weight = init_qua_weight
        self.ori_qua_weight = ori_qua_weight
        self.poc_qua_weight = poc_qua_weight
        self.top_ratio = top_ratio
        self.version = version
        self._init_layers()

        print('——————Oriented Quality RepPoints 5*5Context——————\n')

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.context_convs=nn.ModuleList() #添加

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

            if i==self.stacked_convs:  # context只设计两层
                continue
            ##### 添加
            self.context_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    5,
                    stride=1,
                    padding=2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        pts_out_dim = 2 * self.num_points
        self.reppoints_cls_conv = DeformConv2d(self.feat_channels,
                                               self.point_feat_channels,
                                               self.dcn_kernel, 1,
                                               self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv2d(self.feat_channels,
                                                      self.point_feat_channels,
                                                      self.dcn_kernel, 1,
                                                      self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)


       ######### high level feature
        self.reppoints_cls_conv_h = DeformConv2d(self.feat_channels,
                                               self.point_feat_channels,
                                               self.dcn_kernel, 1,
                                               self.dcn_pad)
        self.reppoints_cls_out_h = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv_h = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out_h = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv_h = DeformConv2d(self.feat_channels,
                                                      self.point_feat_channels,
                                                      self.dcn_kernel, 1,
                                                      self.dcn_pad)
        self.reppoints_pts_refine_out_h = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

        #########context
        context_pts_out_dim = 2 * self.num_context_points
        self.context_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 5,
                                                 1,2)  # 5*5卷积核
        # self.context_pts_init_conv = nn.Conv2d(self.feat_channels,
        #                                          self.point_feat_channels, 3,
        #                                          1, 1)
        self.context_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                context_pts_out_dim, 1, 1, 0)
        self.context_cls_conv=DeformConv2d(self.feat_channels,
                                               self.point_feat_channels,
                                               self.dcn_kernel2, 1,
                                               self.dcn_pad2)
        self.context_cls_out=nn.Conv2d(self.point_feat_channels,
                                       self.cls_out_channels,1,1,0)

        self.reduce_conv=nn.Conv2d(2*self.point_feat_channels,
                                   self.point_feat_channels,1,1,0)



    def forward(self, feats):
        """Forward function."""
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        """Forward feature map of a single FPN level.
        Args:
            x (torch.tensor): single-level feature map sizes.

        Returns:
            cls_out (torch.tensor): classification score prediction
            pts_out_init (torch.tensor): initial point sets prediction
            pts_out_refine (torch.tensor): refined point sets prediction
            base_feat: single-level feature as the basic feature map
        """
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        points_init = 0
        cls_feat = x
        pts_feat = x
        base_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach(
        ) + self.gradient_mul * pts_out_init
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset
        # cls_out = self.reppoints_cls_out(
        #     self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        pts_out_refine = pts_out_refine + pts_out_init.detach()

        ### 对不同层的特征图进行不同操作,因为高层特征图太小了,5*5变形卷积无法有效学习
        ### 该分界阈值设为16
        if x.shape[2]<=16 or x.shape[3]<=16:
            # initialize reppoints
            pts_out_init_h = self.reppoints_pts_init_out_h(
                self.relu(self.reppoints_pts_init_conv_h(pts_feat)))
            pts_out_init_h = pts_out_init_h + points_init
            # refine and classify reppoints
            pts_out_init_grad_mul_h = (1 - self.gradient_mul) * pts_out_init_h.detach(
            ) + self.gradient_mul * pts_out_init_h
            dcn_offset_h = pts_out_init_grad_mul_h - dcn_base_offset
            pts_out_refine_h = self.reppoints_pts_refine_out_h(
                self.relu(self.reppoints_pts_refine_conv_h(pts_feat, dcn_offset_h)))
            pts_out_refine_h = pts_out_refine_h + pts_out_init_h.detach()

            cls_out_h = self.reppoints_cls_out_h(self.relu(self.reppoints_cls_conv_h(cls_feat, dcn_offset)))  # 256->256
            return cls_out_h, pts_out_init_h, pts_out_refine_h, base_feat

        else:
            ##### context features
            context_feat=x
            for ctx_conv in self.context_convs:
                context_feat=ctx_conv(context_feat)  #得到feature
            # init context reppoints
            context_pts_out_init = self.context_pts_init_out(  # 256->50
                self.relu(self.context_pts_init_conv(context_feat)))  # 256->256

            context_dcn_base_offset=self.dcn_base_offset2.type_as(x)
            # 计算偏移量
            context_dcn_offset = context_pts_out_init - context_dcn_base_offset

            context_fea_out = self.context_cls_conv(context_feat, context_dcn_offset)
            cls_fea_out=self.reppoints_cls_conv(cls_feat, dcn_offset)
            final_cls_fea=[]
            final_cls_fea.append(context_fea_out)
            final_cls_fea.append(cls_fea_out)
            final_cls_out=torch.cat(final_cls_fea,dim=1)

            # final_cls_out=self.reduce_conv(final_cls_out)
            final_cls_out=self.relu(self.reduce_conv(final_cls_out))

            # coordattention
            device = get_device()  # 获取当前设备
            in_dim=self.point_feat_channels
            out_dim=self.point_feat_channels
            reduction=32
            coord_att=CoordAtt(in_dim,out_dim,reduction=reduction).to(device)
            final_cls_out=coord_att(final_cls_out)


            cls_out = self.reppoints_cls_out(final_cls_out)

            # 返回类别、初始点、细化点和经过FPN后的feature
            return cls_out, pts_out_init, pts_out_refine, base_feat





    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)

        multi_level_points = self.prior_generator.grid_priors(
            featmap_sizes, device=device, with_stride=True)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'])
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate."""
        pts_list = []
        for i_lvl, _ in enumerate(self.point_strides):
            pts_lvl = []
            for i_img, _ in enumerate(center_list):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def sampling_points(self, polygons, points_num, device):
        """Sample edge points for polygon.

        Args:
            polygons (torch.tensor): polygons with shape (N, 8)
            points_num (int): number of sampling points for each polygon edge.
                              10 by default.

        Returns:
            sampling_points (torch.tensor): sampling points with shape (N,
                             points_num*4, 2)
        """
        polygons_xs, polygons_ys = polygons[:, 0::2], polygons[:, 1::2]
        ratio = torch.linspace(0, 1, points_num).to(device).repeat(
            polygons.shape[0], 1)

        edge_pts_x = []
        edge_pts_y = []
        for i in range(4):
            if i < 3:
                points_x = ratio * polygons_xs[:, i + 1:i + 2] + (
                    1 - ratio) * polygons_xs[:, i:i + 1]
                points_y = ratio * polygons_ys[:, i + 1:i + 2] + (
                    1 - ratio) * polygons_ys[:, i:i + 1]
            else:
                points_x = ratio * polygons_xs[:, 0].unsqueeze(1) + (
                    1 - ratio) * polygons_xs[:, i].unsqueeze(1)
                points_y = ratio * polygons_ys[:, 0].unsqueeze(1) + (
                    1 - ratio) * polygons_ys[:, i].unsqueeze(1)

            edge_pts_x.append(points_x)
            edge_pts_y.append(points_y)

        sampling_points_x = torch.cat(edge_pts_x, dim=1).unsqueeze(dim=2)
        sampling_points_y = torch.cat(edge_pts_y, dim=1).unsqueeze(dim=2)
        sampling_points = torch.cat([sampling_points_x, sampling_points_y],
                                    dim=2)

        return sampling_points

    def get_adaptive_points_feature(self, features,cls_scores,pt_locations, stride):
        """Get the points features from the locations of predicted points.
        # 从预测点的位置得到点的特征?——>根据每层图上每个点预测的9个点位置在对应特征图上采样
        Args:
            features (torch.tensor): base feature with shape (B,C,W,H) FPN后的某一层特征图如(1,256,128,128)
            pt_locations (torch.tensor): locations of points in each point set
                     with shape (B, N_points_set(number of point set),
                     N_points(number of points in each point set) *2)
        Returns:
            tensor: sampling features with (B, C, N_points_set, N_points)
        """

        h = features.shape[2] * stride
        w = features.shape[3] * stride

        tmp_pt1=pt_locations[0][0]
        tmp_pt2 = pt_locations[0][3]
        tmp_pt3 = pt_locations[0][5]

        pt_locations = pt_locations.view(pt_locations.shape[0],
                                         pt_locations.shape[1], -1, 2).clone() #(B,N,9,2)
        pt_locations[..., 0] = pt_locations[..., 0] / (w / 2.) - 1
        pt_locations[..., 1] = pt_locations[..., 1] / (h / 2.) - 1 # 归一化到[-1,1]

        batch_size = features.size(0)
        sampled_features = torch.zeros([
            pt_locations.shape[0],
            features.size(1),
            pt_locations.size(1),
            pt_locations.size(2)
        ]).to(pt_locations.device) #sucn as (1,256,16384,9)

        sampled_cls_scores=torch.zeros([
            pt_locations.shape[0],
            cls_scores.size(1),
            pt_locations.size(1),
            pt_locations.size(2)
        ]).to(pt_locations.device) #(1,1,16384,9)

        for i in range(batch_size):
            # 这里是利用预测的点的位置在单层特征图上采样,上面的reshape为9,2以及归一化到[-1,1]就是为了这里的输入!
            # 以feature为img(1,C,H_in,W_in),pt_location为grid(要映射到-1,1)(1,H_out,W_out,2)
            # 输出为(1,C,H_out,W_out)
            feature = nn.functional.grid_sample(features[i:i + 1], # (256,128,128)
                                                pt_locations[i:i + 1])[0] #(16384,9,2)
            sampled_features[i] = feature # (256,16384,9)

            cls_score=nn.functional.grid_sample(cls_scores[i:i + 1],
                                                pt_locations[i:i + 1])[0]
            sampled_cls_scores[i]=cls_score

        return sampled_features,sampled_cls_scores

    def feature_cosine_similarity(self, points_features):
        """Compute the points features similarity for points-wise correlation.
        计算逐点特征的余弦相似度
        Args:
            points_features (torch.tensor): sampling point feature with
                     shape (N_pointsets, N_points, C)
        Returns:
            max_correlation: max feature similarity in each point set with
                     shape (N_points_set, N_points, C)
        """

        mean_points_feats = torch.mean(points_features, dim=1, keepdim=True)# 如(4,9,256)——>(4,1,256)
        norm_pts_feats = torch.norm(       # 求二范数,即平方和的开方(4,9,1)
            points_features, p=2, dim=2).unsqueeze(dim=2).clamp(min=1e-2)
        norm_mean_pts_feats = torch.norm(  # (4,1,1)
            mean_points_feats, p=2, dim=2).unsqueeze(dim=2).clamp(min=1e-2)

        unity_points_features = points_features / norm_pts_feats
        unity_mean_points_feats = mean_points_feats / norm_mean_pts_feats

        cos_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)
        feats_similarity = 1.0 - cos_similarity(unity_points_features,
                                                unity_mean_points_feats) #(4,1,256),(4,9,256)——>(4,9)

        max_correlation, _ = torch.max(feats_similarity, dim=1)

        return max_correlation

    def pointsets_quality_assessment(self, pts_features, cls_score,
                                     pts_pred_init, pts_pred_refine, label,
                                     bbox_gt, label_weight, bbox_weight,
                                     pos_inds):
        """Assess the quality of each point set from the classification,
        localization, orientation, and point-wise correlation based on
        the assigned point sets samples. 从4个方面来评估每个点集的质量指标，借此选出topk个正样本
        Args:
            pts_features (torch.tensor): points features with shape (N, 9, C) 如(21824,9,256)
            cls_score (torch.tensor): classification scores with
                        shape (N, class_num) 如(21824,15),特征金字塔所有位置的点的类别得分
            pts_pred_init (torch.tensor): initial point sets prediction with
                        shape (N, 9*2) 特征金字塔所有位置对应的init和refine点集,9个点
            pts_pred_refine (torch.tensor): refined point sets prediction with
                        shape (N, 9*2)
            label (torch.tensor): gt label with shape (N)
            bbox_gt(torch.tensor): gt bbox of polygon with shape (N, 8) ?
            label_weight (torch.tensor): label weight with shape (N)
            bbox_weight (torch.tensor): box weight with shape (N)
            pos_inds (torch.tensor): the  inds of  positive point set samples

        Returns:
            qua (torch.tensor) : weighted quality values for positive
                                 point set samples.
        """
        device = cls_score.device
        pos_scores = cls_score[pos_inds]
        pos_pts_pred_init = pts_pred_init[pos_inds]
        pos_pts_pred_refine = pts_pred_refine[pos_inds]
        pos_pts_refine_features = pts_features[pos_inds]
        pos_bbox_gt = bbox_gt[pos_inds]
        pos_label = label[pos_inds] # pos_label为0,其余为1,
        pos_label_weight = label_weight[pos_inds] # 都是1
        pos_bbox_weight = bbox_weight[pos_inds]# 都是1

        cls_max=torch.max(cls_score) # -4.0387
        cls_min=torch.min(cls_score) # -5.3098
        cls_mean=torch.mean(cls_score) # -4.6153
        bbox_gt_tmp=  bbox_gt[bbox_gt>0]

        # quality of point-wise correlation
        qua_poc = self.poc_qua_weight * self.feature_cosine_similarity(
            pos_pts_refine_features)

        qua_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        polygons_pred_init = min_area_polygons(pos_pts_pred_init)     # 得到多边形
        polygons_pred_refine = min_area_polygons(pos_pts_pred_refine) # 得到多边形
        sampling_pts_pred_init = self.sampling_points(
            polygons_pred_init, 10, device=device)  # 在多边形的边界上采样10个点
        sampling_pts_pred_refine = self.sampling_points(
            polygons_pred_refine, 10, device=device)
        sampling_pts_gt = self.sampling_points(pos_bbox_gt, 10, device=device)

        # quality of orientation
        qua_ori_init = self.ori_qua_weight * ChamferDistance2D(
            sampling_pts_gt, sampling_pts_pred_init)
        qua_ori_refine = self.ori_qua_weight * ChamferDistance2D(
            sampling_pts_gt, sampling_pts_pred_refine)

        # quality of localization
        qua_loc_init = self.loss_bbox_refine(
            pos_pts_pred_init,
            pos_bbox_gt,
            pos_bbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')  # 这里计算loc的loss是用预测的9个点和gt box作为函数的输入,而不是9个点转化后得到的OBB
        qua_loc_refine = self.loss_bbox_refine(
            pos_pts_pred_refine,
            pos_bbox_gt,
            pos_bbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        # quality of classification
        qua_cls = qua_cls.sum(-1)

        # weighted inti-stage and refine-stage
        # 权重求和,类别cls+定位loc+方向对齐ori+点相关poc
        # qua = qua_cls + self.init_qua_weight * (
        #     qua_loc_init + qua_ori_init) + (1.0 - self.init_qua_weight) * (
        #         qua_loc_refine + qua_ori_refine) # 去掉qua_poc
        
        qua = qua_cls + self.init_qua_weight * (
            qua_loc_init + qua_ori_init) + (1.0 - self.init_qua_weight) * (
                qua_loc_refine + qua_ori_refine) + qua_poc  # 求和，为什么两项的权重之和为1？

        return qua,

    def dynamic_pointset_samples_selection(self,
                                           quality,
                                           label,
                                           label_weight,
                                           bbox_weight,
                                           pos_inds,
                                           pos_gt_inds,
                                           num_proposals_each_level=None,
                                           num_level=None):
        """The dynamic top k selection of point set samples based on the
        quality assessment values. 基于质量评估动态地选取k个点集
        #  问题：既然已经得到了pos_ids,为什么还要设计这个函数来选择samples？？？
        #  答：因为pos_inds数量可能很多，这一步会选出topk个
        Args:
            quality (torch.tensor): the quality values of positive
                                    point set samples
            label (torch.tensor): gt label with shape (N)
            bbox_gt(torch.tensor): gt bbox of polygon with shape (N, 8)
            label_weight (torch.tensor): label weight with shape (N)
            bbox_weight (torch.tensor): box weight with shape (N)
            pos_inds (torch.tensor): the inds of  positive point set samples
            num_proposals_each_level (list[int]): proposals number of
                                    each level
            num_level (int): the level number
        Returns:
            label: gt label with shape (N)
            label_weight: label weight with shape (N)
            bbox_weight: box weight with shape (N)
            num_pos (int): the number of selected positive point samples
                           with high-qualty
            pos_normalize_term (torch.tensor): the corresponding positive
                             normalize term
        """
        if len(pos_inds) == 0:
            return label, label_weight, bbox_weight, 0, torch.tensor(
                []).type_as(bbox_weight)

        num_gt = pos_gt_inds.max()
        # print('pos ind的数量为:', len(pos_inds))
        # print('quality的长度为:',len(quality))
        # print('GT的数量为:',pos_gt_inds.max())
        num_proposals_each_level_ = num_proposals_each_level.copy()
        num_proposals_each_level_.insert(0, 0)  # (0,16384,4096,1024,256,64)
        inds_level_interval = np.cumsum(num_proposals_each_level_) #(0,16384,20480,...,21824)
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (
                pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)  # 找到pos_inds对应的proposal属于哪一层特征图,
            # 如pos_inds=21733,pos_level_mask=(False,F,F,T,F)
        pos_inds_after_select = []
        ignore_inds_after_select = []

        for gt_ind in range(num_gt):
            pos_inds_select = []
            pos_loss_select = []
            gt_mask = pos_gt_inds == (gt_ind + 1)
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask # 找到每一个gt对应的那个特征层
                value, topk_inds = quality[level_gt_mask].topk(  # 但是传入的quality长度为1,只有一个！这里的topk有啥意义???
                    min(level_gt_mask.sum(), 6), largest=False)
                pos_inds_select.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_select.append(value)
            pos_inds_select = torch.cat(pos_inds_select)
            pos_loss_select = torch.cat(pos_loss_select)

            if len(pos_inds_select) < 2:
                pos_inds_after_select.append(pos_inds_select)
                ignore_inds_after_select.append(pos_inds_select.new_tensor([]))

            else:
                pos_loss_select, sort_inds = pos_loss_select.sort(
                )  # small to large
                pos_inds_select = pos_inds_select[sort_inds]
                # dynamic top k
                topk = math.ceil(pos_loss_select.shape[0] * self.top_ratio)
                pos_inds_select_topk = pos_inds_select[:topk]
                pos_inds_after_select.append(pos_inds_select_topk)
                ignore_inds_after_select.append(
                    pos_inds_select_topk.new_tensor([]))

        pos_inds_after_select = torch.cat(pos_inds_after_select)
        ignore_inds_after_select = torch.cat(ignore_inds_after_select)

        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_select).all(1)
        reassign_ids = pos_inds[reassign_mask]
        label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_select] = 0
        bbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_select)

        pos_level_mask_after_select = []
        for i in range(num_level):
            mask = (pos_inds_after_select >= inds_level_interval[i]) & (
                pos_inds_after_select < inds_level_interval[i + 1])
            pos_level_mask_after_select.append(mask)
        pos_level_mask_after_select = torch.stack(pos_level_mask_after_select,
                                                  0).type_as(label)
        pos_normalize_term = pos_level_mask_after_select * (
            self.point_base_scale *
            torch.as_tensor(self.point_strides).type_as(label)).reshape(-1, 1)
        pos_normalize_term = pos_normalize_term[
            pos_normalize_term > 0].type_as(bbox_weight)
        assert len(pos_normalize_term) == len(pos_inds_after_select)

        return label, label_weight, bbox_weight, num_pos, pos_normalize_term

    def init_loss_single(self, pts_pred_init, bbox_gt_init, bbox_weights_init,
                         stride,sam_weights_init):
        """Single initial stage loss function."""
        normalize_term = self.point_base_scale * stride

        bbox_gt_init = bbox_gt_init.reshape(-1, 8)
        bbox_weights_init = bbox_weights_init.reshape(-1)
        sam_weights_init = sam_weights_init.reshape(-1) # SASM
        # init points loss
        pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
        pos_ind_init = (bbox_weights_init > 0).nonzero(
            as_tuple=False).reshape(-1)

        pts_pred_init_norm = pts_pred_init[pos_ind_init]
        bbox_gt_init_norm = bbox_gt_init[pos_ind_init]
        bbox_weights_pos_init = bbox_weights_init[pos_ind_init]
        sam_weights_pos_init = sam_weights_init[pos_ind_init] # SASM

        # loss_pts_init = self.loss_bbox_init(
        #     pts_pred_init_norm / normalize_term,
        #     bbox_gt_init_norm / normalize_term, bbox_weights_pos_init)
        # SASM
        loss_pts_init = self.loss_bbox_init(
            pts_pred_init_norm / normalize_term,
            bbox_gt_init_norm / normalize_term,
            bbox_weights_pos_init * sam_weights_pos_init)

        loss_border_init = self.loss_spatial_init(
            pts_pred_init_norm.reshape(-1, 2 * self.num_points) /
            normalize_term,
            bbox_gt_init_norm / normalize_term,
            bbox_weights_pos_init* sam_weights_pos_init,
            avg_factor=None)

        return loss_pts_init, loss_border_init

# 采样,为init和refine stage选择合适的样本点
    def _point_target_single(self,
                             flat_proposals,
                             num_level_proposals_list,  # SASM
                             valid_flags,
                             gt_bboxes,
                             gt_bboxes_ignore,
                             gt_labels,
                             overlaps,
                             cls_score_pts_refine,
                             stage='init',
                             unmap_outputs=True):
        """Single point target function for initial and refine stage."""
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 8
        # if not inside_flags.any():
        #     return (None, ) * 9  # SASM是9
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]

        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg.init.pos_weight
        else:
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight


        # convert gt from obb to poly
        gt_bboxes = obb2poly(gt_bboxes, self.version) # 这里将OBB转换为8点表示形式
        # 进行正负样本采样
        # print('当前stage为:',stage)
        # # 得到分配结果
        assign_result = assigner.assign(proposals, gt_bboxes, overlaps,
                                        gt_bboxes_ignore,
                                        None if self.sampling else gt_labels)

        sampling_result = self.sampler.sample(assign_result, proposals,
                                              gt_bboxes)   #这一步就得到正负样本了，在这个方法中直接采样中心点得到

        gt_inds = assign_result.gt_inds
        num_valid_proposals = proposals.shape[0]
        bbox_gt = proposals.new_zeros([num_valid_proposals, 8])
        pos_proposals = torch.zeros_like(proposals)
        proposals_weights = proposals.new_zeros(num_valid_proposals)

        labels = proposals.new_full((num_valid_proposals, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = proposals.new_zeros(
            num_valid_proposals, dtype=torch.float)

        pos_inds = sampling_result.pos_inds

        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            bbox_gt[pos_inds, :] = pos_gt_bboxes
            pos_proposals[pos_inds, :] = proposals[pos_inds, :]
            proposals_weights[pos_inds] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        
        ##### calcute weights
        # cls_weights and qua_weights
        cls_weights=torch.ones_like(label_weights)

        # use la 得到gt的位置和形状
        rbboxes_center, width, height, angles = torch.split(
            poly2obb(bbox_gt, self.version), [2, 1, 1, 1], dim=-1)
        
        gt_xy=rbboxes_center[pos_inds]
        gt_w=width[pos_inds]
        gt_h=height[pos_inds]
        angle=angles[pos_inds]
        # cos_a=math.cos(angles[pos_inds])

        ## 1. vector w、h、d
        v_w_x=torch.mul(gt_w,torch.cos(angle))
        v_w_y=torch.mul(gt_w,torch.sin(angle))
        v_h_x=torch.mul(-gt_h,torch.sin(angle))
        v_h_y=torch.mul(gt_h,torch.cos(angle))

        if stage == 'init':
            points_xy = pos_proposals[:, :2]
            # print('init 阶段正样本数:',len(pos_inds))
            # print('GT数:',len(gt_bboxes))
        else:  # refine
            # 获取proposals也就是9个关键点的平均中心点位置
            points_xy = points_center_pts(pos_proposals,y_first=False)
            pos_9points=pos_proposals[pos_inds]  

            
            ## calculate class score
            # 根据refine预测的9个关键点位置在cls_score上采样
            pos_cls_socre=cls_score_pts_refine[pos_inds] #(N,9,1)
            # print('pos_cls_socre',pos_cls_socre)
            # print('pos_cls_socre[0]',pos_cls_socre[0])
            # print('pos_cls_socre shape',pos_cls_socre.shape)

            # # 计算9个点类别预测的一致性
            # cls_mean=torch.mean(pos_cls_socre,dim=1)
            # cls_std = torch.std(cls_score_pts_refine, dim=1)  # 求9个语义点类别得分的方差
            # # print('mean pos_cls_socre',cls_mean)
            # # print('mean pos_cls_socre[0]',cls_mean[0])

            # # 计算对应权重
            # # pos_cls_socre.to('gpu31')
            # if len(pos_inds)==1:
            #     cls_qua=torch.zeros_like(cls_std)
            # else:
            #     cls_qua=(cls_std-torch.min(cls_std))/(torch.max(cls_std)-torch.min(cls_std)) # 用方差归一化

            # cls_weights=torch.ones_like(cls_std)+cls_std  # cls_weight=1*(1+cls_qua)
            # # cls_weights=label_weights*cls_weights  

            # # pos_std=cls_std[pos_inds]
            # # print('std pos_cls_socre',pos_std)
            # # print('std pos_cls_socre[0]',pos_std[0])
            # # pos_cls_weights=torch.ones_like(pos_std)+(pos_std-torch.min(pos_std))/(torch.max(pos_std)-torch.min(pos_std))
            # # print('pos_cls_weights',pos_cls_weights)


        pos_xy=points_xy[pos_inds]
        # vector d
        v_d_xy=torch.sub(pos_xy,gt_xy)
        neg_y_inds=v_d_xy[:,1]<0
        v_d_xy[neg_y_inds]=torch.mul(-1,v_d_xy[neg_y_inds]) # v_d_y<0则翻转vector

        # delta d(distance)
        dis = torch.zeros_like(width).reshape(-1)
        dis[pos_inds] = torch.sqrt(
            (torch.pow( gt_xy[:, 0] -pos_xy[:, 0], 2)
             +torch.pow( gt_xy[:, 1] -pos_xy[:, 1], 2))
        )
        delta_d=dis[pos_inds].unsqueeze(1)

        ## 2. cos(v_w,v_d),cos(v_h,v_d)
        cos_a_wd=torch.zeros_like(angles).reshape(-1)
        cos_a_hd = torch.zeros_like(angles).reshape(-1)
        # w·d and h·d
        v_d_x=v_d_xy[:,0].unsqueeze(1)
        v_d_y = v_d_xy[:, 1].unsqueeze(1)
        tmp1=torch.mul(v_w_x,v_d_x)
        tmp2=v_w_x*v_d_xy[:,0]

        w_dot_d=torch.mul(v_w_x, v_d_x) + torch.mul(v_w_y, v_d_y)
        h_dot_d=torch.mul(v_h_x, v_d_x) + torch.mul(v_h_y, v_d_y)
        # w·d/|w||d|
        cos_a_wd[pos_inds] = torch.div(w_dot_d , torch.mul(delta_d, gt_w)).squeeze(1)
        # h·d/|h||d|
        cos_a_hd[pos_inds] = torch.div(h_dot_d,torch.mul(delta_d, gt_h)).squeeze(1)
        tmp_cos_wd = cos_a_wd[pos_inds]
        tmp_cos_hd=cos_a_hd[pos_inds]
        ## 3. |d|cos(v_w,v_d),|d|cos(v_h,v_d)——> Qw,Qh
        rela_w=torch.abs(torch.mul(delta_d.squeeze(),cos_a_wd[pos_inds]))
        rela_h=torch.abs(torch.mul(delta_d.squeeze(),cos_a_hd[pos_inds]))

        Qw = torch.zeros_like(width).reshape(-1)
        Qh = torch.zeros_like(height).reshape(-1)
        Qw[pos_inds]=torch.div(torch.mul(rela_w,2),gt_w.squeeze())  # 2*w'/w,宽度相对比例
        Qh[pos_inds]=torch.div(torch.mul(rela_h,2),gt_h.squeeze())  
        Qw[torch.isnan(Qw)] = 0.
        Qh[torch.isnan(Qh)] = 0.
        # bef=label_weights[pos_inds]
        qua_weights=label_weights*(torch.log(Qw+1)+1)*(torch.log(Qh+1)+1) #修改
        # qua_weights = label_weights * (torch.exp(0.5 /(Qw + 1))) * (torch.exp(0.5 /(Qh + 1))) # 缩小范围

        # qua_weights=label_weights*torch.exp(Qw)*torch.exp(Qh)



        if stage=='init':
            cls_weights=torch.ones_like(label_weights)  # init阶段为训练初期,只有单个点的得分且cls_score抖动很大,不适合


        if stage == 'refine':  # 上面的质量加权只对Init阶段进行计算
            qua_weights=torch.ones_like(label_weights)  # refine阶段用中心距离不适合,因为是defom conv
            cls_weights=torch.ones_like(label_weights)  # 实验表明std形式的cls_weights不适合,所以不改动


        # print('-----------stage-------------',stage)
        # print(stage)
        # print('rela_w[pos_inds]',rela_w)
        # print('width[pos_inds]',width[pos_inds])
        # print('rela_h[pos_inds]',rela_h)
        # print('height[pos_inds]',height[pos_inds])
        # print('(torch.log(Qw+1)+1)',(torch.log(Qw+1)+1))
        # print('(torch.log(Qh+1)+1)',(torch.log(Qh+1)+1))
        # print('gt_xy:',gt_xy)
        # print('pos_xy:',pos_xy)
        # print('delta_d:',delta_d)
        # print('angle:',angle)
        # print('tmp_cos_wd:',tmp_cos_wd)
        # print('tmp_cos_hd:',tmp_cos_hd)
        # print('Qw[pos_inds]',Qw[pos_inds])
        # print('Qh[pos_inds]',Qh[pos_inds])
        # print('Qw[4]',Qw[4])
        # print('Qw[5]',Qw[5])
        # print('torch.isnan(Qw)',torch.isnan(Qw))

        # print('\n qua shape:',qua_weights.shape)
        # print('sam_weights shape1:',sam_weights.shape)
        # print('qua_weights[pos_inds]',qua_weights[pos_inds])
        # print('label_weights[pos_inds]',label_weights[pos_inds])
        # print('qua_weights[2]',qua_weights[2])
        # print('label_weights[2]',label_weights[2])
        # print('sam_weights[pos_inds]',sam_weights[pos_inds])
        # print('sam_weights[2]',sam_weights[2])

        # sam_weights=qua_weights  #这一步用于Debug
        # sam_weights = label_weights * (torch.exp(1 / (distances + 1)))
        # sam_weights[sam_weights == float('inf')] = 0.



        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels = unmap(labels, num_total_proposals, inside_flags)
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
            pos_proposals = unmap(pos_proposals, num_total_proposals,
                                  inside_flags)
            proposals_weights = unmap(proposals_weights, num_total_proposals,
                                      inside_flags)
            gt_inds = unmap(gt_inds, num_total_proposals, inside_flags)
            qua_weights = unmap(qua_weights, num_total_proposals, inside_flags) # SASM
            cls_weights = unmap(cls_weights, num_total_proposals, inside_flags) 
        
        return (labels, label_weights, bbox_gt, pos_proposals,
                proposals_weights, pos_inds, neg_inds, gt_inds,
                sampling_result, qua_weights,cls_weights)


    def get_targets(self,
                    proposals_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    cls_score_pts_refine,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    stage='init',
                    label_channels=1,
                    unmap_outputs=True):
        """Compute corresponding GT box and classification targets for
        proposals in initial stage.

        Args:
            proposals_list (list[list]): Multi level points/bboxes of each
                image.
            valid_flag_list (list[list]): Multi level valid flags of each
                image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_bboxes_list (list[Tensor]): Ground truth labels of each box.
            stage (str): `init` or `refine`. Generate target for init stage or
                refine stage
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple (list[Tensor]):

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_gt_list (list[Tensor]): Ground truth bbox of each level.
                - proposal_list (list[Tensor]): Proposals(points/bboxes) of \
                    each level.
                - proposal_weights_list (list[Tensor]): Proposal weights of \
                    each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]
        num_level_proposals_list = [num_level_proposals] * num_imgs #SASM
        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        all_overlaps_rotate_list = [None] * 4
        (all_labels, all_label_weights, all_bbox_gt, all_proposals,
         all_proposal_weights, pos_inds_list, neg_inds_list, all_gt_inds,
         sampling_result,all_qua_weights,all_cls_weights
         ) = multi_apply(
             self._point_target_single,
             proposals_list,
             num_level_proposals_list, # SASM
             valid_flag_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             all_overlaps_rotate_list,
             cls_score_pts_refine,
             stage=stage,
             unmap_outputs=unmap_outputs)  # 其中pos_inds_list为init stage所用的,而refine则是从all_labels里筛选

        if stage == 'init':
            # no valid points
            if any([labels is None for labels in all_labels]):
                return None
            # sampled points of all images
            num_total_pos = sum(
                [max(inds.numel(), 1) for inds in pos_inds_list]) # pos_inds_list由self._point_target_single得到
            num_total_neg = sum(
                [max(inds.numel(), 1) for inds in neg_inds_list])
            labels_list = images_to_levels(all_labels, num_level_proposals)
            label_weights_list = images_to_levels(all_label_weights,
                                                  num_level_proposals)
            bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
            proposals_list = images_to_levels(all_proposals,
                                              num_level_proposals)
            proposal_weights_list = images_to_levels(all_proposal_weights,
                                                     num_level_proposals)
            # Quality
            qua_weights_list = images_to_levels(all_qua_weights,
                                                     num_level_proposals)
            cls_weights_list=images_to_levels(all_cls_weights,num_level_proposals)

            return (labels_list, label_weights_list, bbox_gt_list,
                    proposals_list, proposal_weights_list, num_total_pos, #num_total_pos即为pos_inds_list
                    num_total_neg, None, qua_weights_list,None)

        else: #stage == 'refine'
            pos_inds = []
            pos_gt_index = []
            for i, single_labels in enumerate(all_labels):
                pos_mask = (0 <= single_labels) & (
                    single_labels < self.num_classes)
                pos_inds.append(pos_mask.nonzero(as_tuple=False).view(-1))  #返回非False值的所有索引
                pos_gt_index.append(
                    all_gt_inds[i][pos_mask.nonzero(as_tuple=False).view(-1)])

            # Quality
            qua_weights_list = images_to_levels(all_qua_weights,
                                                     num_level_proposals)
            cls_weights_list=images_to_levels(all_cls_weights,num_level_proposals)
            return (all_labels, all_label_weights, all_bbox_gt, all_proposals,
                    all_proposal_weights, pos_inds, pos_gt_index
                    , qua_weights_list,cls_weights_list)

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             base_features,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Loss function of OrientedRepPoints head."""

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        device = cls_scores[0].device

        center_list, valid_flag_list = self.get_points(
            featmap_sizes, img_metas, device=device)  #得到每层特征图的像素点坐标和有效区域
        # print('img_metas:',img_metas[0]['filename'])

        pts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                       pts_preds_init)

        num_proposals_each_level = [(featmap.size(-1) * featmap.size(-2))
                                    for featmap in cls_scores]

        num_level = len(featmap_sizes)
        assert num_level == len(pts_coordinate_preds_init)

        candidate_list = center_list

        cls_reg_targets_init = self.get_targets(
            candidate_list, # 与refine阶段不同
            valid_flag_list,
            gt_bboxes,
            cls_scores,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='init', # 与refine阶段不同
            label_channels=label_channels)
        (*_, bbox_gt_list_init, candidate_list_init, bbox_weights_list_init,
         num_total_pos_init, num_total_neg_init, _,
         sam_weights_list_init,_) = cls_reg_targets_init  # 这里即为正样本数

        center_list, valid_flag_list = self.get_points(
            featmap_sizes, img_metas, device=device)
        pts_coordinate_preds_refine = self.offset_to_pts(
            center_list, pts_preds_refine)
        
        refine_points_features,refine_cls_scores = multi_apply(self.get_adaptive_points_feature,
                                              base_features,
                                              cls_scores,
                                              pts_coordinate_preds_refine,
                                              self.point_strides)

        features_pts_refine = levels_to_images(refine_points_features)
        features_pts_refine = [
            item.reshape(-1, self.num_points, item.shape[-1])
            for item in features_pts_refine
        ] 

        # get cls scores map
        cls_score_pts_refine=levels_to_images(refine_cls_scores)
        cls_score_pts_refine=[
            item.reshape(-1, self.num_points, item.shape[-1])
            for item in cls_score_pts_refine
        ] #(21824,9,1)

        points_list = []
        for i_img, center in enumerate(center_list):
            points = []
            for i_lvl in range(len(pts_preds_refine)):
                points_preds_init_ = pts_preds_init[i_lvl].detach()
                points_preds_init_ = points_preds_init_.view(
                    points_preds_init_.shape[0], -1,
                    *points_preds_init_.shape[2:])
                points_shift = points_preds_init_.permute(
                    0, 2, 3, 1) * self.point_strides[i_lvl]
                points_center = center[i_lvl][:, :2].repeat(1, self.num_points)
                points.append(
                    points_center +
                    points_shift[i_img].reshape(-1, 2 * self.num_points))
            points_list.append(points)

        cls_reg_targets_refine = self.get_targets(
            points_list, # 与init阶段不同
            valid_flag_list,
            gt_bboxes,
            cls_score_pts_refine,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='refine', # 与init阶段不同
            label_channels=label_channels) # 传入参数stage为'refine',所以得到的为refine阶段的结果

        (labels_list, label_weights_list, bbox_gt_list_refine, _,
         bbox_weights_list_refine, pos_inds_list_refine,
         pos_gt_index_list_refine,
         sam_weights_list_refine,
         cls_weights_list_refine) = cls_reg_targets_refine  # 为什么这里是refine??答:因为传入参数为refine

        cls_scores = levels_to_images(cls_scores)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]

        pts_coordinate_preds_init_img = levels_to_images(
            pts_coordinate_preds_init, flatten=True)
        pts_coordinate_preds_init_img = [
            item.reshape(-1, 2 * self.num_points)
            for item in pts_coordinate_preds_init_img
        ]

        pts_coordinate_preds_refine_img = levels_to_images(
            pts_coordinate_preds_refine, flatten=True)
        pts_coordinate_preds_refine_img = [
            item.reshape(-1, 2 * self.num_points)
            for item in pts_coordinate_preds_refine_img
        ]

        ### 进行质量评估,计算4项Q值并据此选择topk
        with torch.no_grad():
            pos_num=len(pos_inds_list_refine[0])
            if pos_num>1 and pos_num!= gt_bboxes[0].shape[0]:
                tmp=1
                tmp+=1

            quality_assess_list, = multi_apply(
                self.pointsets_quality_assessment, features_pts_refine,
                cls_scores, pts_coordinate_preds_init_img,
                pts_coordinate_preds_refine_img, labels_list,
                bbox_gt_list_refine, label_weights_list,
                bbox_weights_list_refine, pos_inds_list_refine)

            labels_list, label_weights_list, bbox_weights_list_refine, \
                num_pos, pos_normalize_term = multi_apply(
                    self.dynamic_pointset_samples_selection,
                    quality_assess_list,
                    labels_list,
                    label_weights_list,
                    bbox_weights_list_refine, # 使用的是refine阶段的正样本
                    pos_inds_list_refine,
                    pos_gt_index_list_refine,
                    num_proposals_each_level=num_proposals_each_level,
                    num_level=num_level
                )
            num_pos = sum(num_pos)

        # convert all tensor list to a flatten tensor
        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
        pts_preds_refine = torch.cat(pts_coordinate_preds_refine_img, 0).view(
            -1, pts_coordinate_preds_refine_img[0].size(-1))

        labels = torch.cat(labels_list, 0).view(-1)
        labels_weight = torch.cat(label_weights_list, 0).view(-1)
        bbox_gt_refine = torch.cat(bbox_gt_list_refine,
                                   0).view(-1, bbox_gt_list_refine[0].size(-1))
        bbox_weights_refine = torch.cat(bbox_weights_list_refine, 0).view(-1)
        pos_normalize_term = torch.cat(pos_normalize_term, 0).reshape(-1)
        pos_inds_flatten = ((0 <= labels) &
                            (labels < self.num_classes)).nonzero(
                                as_tuple=False).reshape(-1)

        assert len(pos_normalize_term) == len(pos_inds_flatten)

        ### 计算cls loss和refine阶段的损失
        if num_pos:
            # Quality
            sam_weights_refine=sam_weights_list_refine
            sam_weights_refine = [
                item.reshape(-1, self.cls_out_channels) for item in sam_weights_refine
            ]
            # sam_weights_refine = sam_weights_refine.reshape(-1) #这里不是loos_single了,需要展平成一个向量
            sam_weights_refine = torch.cat(sam_weights_refine, 0).view(-1, sam_weights_refine[0].size(-1)).squeeze()

            cls_weights_refine=cls_weights_list_refine
            cls_weights_refine = [
                item.reshape(-1, self.cls_out_channels) for item in cls_weights_refine
            ]
            cls_weights_refine = torch.cat(cls_weights_refine, 0).view(-1, cls_weights_refine[0].size(-1)).squeeze()

            # ##定义不同尺度的不同权重
            # lvl_labels_weight=torch.ones_like(labels_weight)
            # lvl_labels_weight[0:10000]=1.2
            # lvl_labels_weight[10000: 12500] = 1.15
            # lvl_labels_weight[12500: 13125] = 1.1
            # lvl_labels_weight[13125: 13294] = 1.05
            # lvl_labels_weight[13294: 13343]=1
            # # lvl_labels_weight[0:10000]=1
            # # lvl_labels_weight[10000: 12500] = 1.05
            # # lvl_labels_weight[12500: 13125] = 1.1
            # # lvl_labels_weight[13125: 13294] = 1.15
            # # lvl_labels_weight[13294: 13343]=1.2


            losses_cls=self.loss_cls(
                cls_scores,labels,labels_weight*sam_weights_refine,
                avg_factor=num_pos 
            ) #SASM
            # losses_cls=self.loss_cls(
            #     cls_scores,labels,labels_weight*sam_weights_refine*cls_weights_refine,
            #     avg_factor=num_pos  # 这里只传入num_pos?为什么不是全部样本??答:因为用的是FocalLoss
            # ) #SASM
            pos_pts_pred_refine = pts_preds_refine[pos_inds_flatten]
            pos_bbox_gt_refine = bbox_gt_refine[pos_inds_flatten]

            pos_bbox_weights_refine = bbox_weights_refine[pos_inds_flatten]
            
            # SASM
            sam_weights_pos_refine = sam_weights_refine[pos_inds_flatten]
            # lvl_labels_pos_weight=lvl_labels_weight[pos_inds_flatten]

            cls_weights_pos_refine=cls_weights_refine[pos_inds_flatten]
            losses_pts_refine = self.loss_bbox_refine(
                pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1),
                pos_bbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                # pos_bbox_weights_refine*sam_weights_pos_refine*lvl_labels_pos_weight) # SASM把原有权重乘以一个sam_weight
                pos_bbox_weights_refine*sam_weights_pos_refine) # SASM把原有权重乘以一个sam_weight

            # losses_pts_refine = self.loss_bbox_refine(
            #     pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1),
            #     pos_bbox_gt_refine / pos_normalize_term.reshape(-1, 1),
            #     pos_bbox_weights_refine)

            loss_border_refine = self.loss_spatial_refine(
                pos_pts_pred_refine.reshape(-1, 2 * self.num_points) /
                pos_normalize_term.reshape(-1, 1),
                pos_bbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                # pos_bbox_weights_refine*sam_weights_pos_refine*lvl_labels_pos_weight,
                pos_bbox_weights_refine*sam_weights_pos_refine,
                avg_factor=None)

        else:
            losses_cls = cls_scores.sum() * 0
            losses_pts_refine = pts_preds_refine.sum() * 0
            loss_border_refine = pts_preds_refine.sum() * 0

        # 这里调用init_loss_single只计算init阶段的loss,refine阶段则是在上方算出
        losses_pts_init, loss_border_init = multi_apply(
            self.init_loss_single, pts_coordinate_preds_init,
            bbox_gt_list_init, bbox_weights_list_init, self.point_strides,
            sam_weights_list_init) #SASM,只用init因为这里只计算init
        # losses_pts_init, loss_border_init = multi_apply(
        #     self.init_loss_single, pts_coordinate_preds_init,
        #     bbox_gt_list_init, bbox_weights_list_init, self.point_strides)

        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_pts_init': losses_pts_init,
            'loss_pts_refine': losses_pts_refine,
            'loss_spatial_init': loss_border_init,
            'loss_spatial_refine': loss_border_refine
        }
        return loss_dict_all

    @force_fp32(apply_to=('cls_scores', 'pts_preds_init', 'pts_preds_refine'))
    def get_bboxes(self,
                   cls_scores,
                   pts_preds_init,
                   pts_preds_refine,
                   base_feats,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            pts_preds_init (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 18D-tensor, has shape
                (batch_size, num_points * 2, H, W).
            pts_preds_refine (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 18D-tensor, has shape
                (batch_size, num_points * 2, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 4 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(pts_preds_refine)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].device,
            device=cls_scores[0].device)

        result_list = []

        for img_id, _ in enumerate(img_metas):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            point_pred_list = select_single_mlvl(pts_preds_refine, img_id)

            results = self._get_bboxes_single(cls_score_list, point_pred_list,
                                              mlvl_priors, img_meta, cfg,
                                              rescale, with_nms, **kwargs)
            result_list.append(results)

        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           point_pred_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.
        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RepPoints head does not need
                this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 2).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (cx, cy, w, h, a) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """

        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(point_pred_list)
        scale_factor = img_meta['scale_factor']
        # img_name=img_meta['filename']

        mlvl_bboxes = []
        mlvl_scores = []
        for level_idx, (cls_score, points_pred, points) in enumerate(
                zip(cls_score_list, point_pred_list, mlvl_priors)):
            assert cls_score.size()[-2:] == points_pred.size()[-2:]

            # ## 可视化
            # tmp_class_map=cls_score.clone()
            # # tmp_class_map[tmp_class_map<0]=0
            # tmp_class_map=torch.sigmoid((tmp_class_map))
            # #上采样显示
            # tmp_class_map = F.interpolate(tmp_class_map.unsqueeze(0),scale_factor=pow(2,level_idx+3), mode='bilinear', align_corners=True)
            # tmp_class_map=np.array(tmp_class_map.cpu()).squeeze()*255
            
            # tmp_max=np.max(tmp_class_map)
            # tmp_min = np.min(tmp_class_map)
            
            # show_class_map=(tmp_class_map-tmp_min)/(tmp_max-tmp_min)*255
            
            # # masktmp2=np.array(masktmp2)
            # cv2.imwrite('/scratch/luojunwei/test_result/BridgeSubset/oriented_qua_reppoints_r50_fpn_1x_dota_oc_context_coordatt_55conv/cls_vis/'+img_name.split('/')[-1].split('.')[0]
            # +'_'+str(level_idx)+'.png', show_class_map)
            # # tmp_class_map[tmp_class_map < 0] = -100



            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            points_pred = points_pred.permute(1, 2, 0).reshape(
                -1, 2 * self.num_points)
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < scores.shape[0]:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                points_pred = points_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            pts_pred = points_pred.reshape(-1, self.num_points, 2)
            pts_pred_offsety = pts_pred[:, :, 0::2]
            pts_pred_offsetx = pts_pred[:, :, 1::2]
            pts_pred = torch.cat([pts_pred_offsetx, pts_pred_offsety],
                                 dim=2).reshape(-1, 2 * self.num_points)

            pts_pos_center = points[:, :2].repeat(1, self.num_points)
            pts = pts_pred * self.point_strides[level_idx] + pts_pos_center

            polys = min_area_polygons(pts)
            bboxes = poly2obb(polys, self.version)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)

        if rescale:
            mlvl_bboxes[..., :4] /= mlvl_bboxes[..., :4].new_tensor(
                scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        
        # print('quality mlvl_scores[0:4]:',mlvl_scores[0:4])
        # print('quality mlvl_scores shape:',mlvl_scores.shape)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms_rotated(
                mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
                cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            raise NotImplementedError
