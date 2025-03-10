# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
import numpy as np
from mmdet.utils import get_device
from PIL import Image
from .img_split_bridge_tools import *
from mmrotate.core import (build_assigner, build_sampler, rbbox2result,
                           multiclass_nms_rotated, obb2poly, poly2obb)
from .img_split_bridge_tools import *
import torch.nn.functional as F
from torch import nn
from mmcv.cnn import ConvModule

def resize_bboxes_len6(bboxes_out,scale):
    """Resize bounding boxes with scales."""

    for i in range(len(bboxes_out)):
        box_out=bboxes_out[i]
        w_scale = scale
        h_scale = scale
        box_out[:, 0] *= w_scale
        box_out[:, 1] *= h_scale
        box_out[:, 2:4] *= np.sqrt(w_scale * h_scale)

    return bboxes_out

def FullImageCrop(self, imgs, bboxes, labels, patch_shape,
                  gaps,
                  jump_empty_patch=False,
                  mode='train'):
    """
    Args:
        imgs (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        bboxes (list[Tensor]): Each item are the truth boxes for each
            image in [tl_x, tl_y, br_x, br_y] format.
        labels (list[Tensor]): Class indices corresponding to each box
    Returns:
        dict[str, Tensor]: A dictionary of loss components.
    """
    out_imgs = []
    out_bboxes = []
    out_labels = []
    out_metas = []
    device = get_device()
    img_rate_thr = 0.6  # 图片与wins窗口的交并比阈值
    iof_thr = 0.1  # 裁剪后的标签占原标签的比值阈值
    padding_value = [0.0081917211329, -0.004901960784, 0.0055655449953]  # 归一化后的padding值

    if mode == 'train':
        # for i in range(imgs.shape[0]):
        for img, bbox, label in zip(imgs, bboxes, labels):
            p_imgs = []
            p_bboxes = []
            p_labels = []
            p_metas = []
            img = img.cpu()
            # patch
            info = dict()
            info['labels'] = np.array(torch.tensor(label, device='cpu', requires_grad=False))
            info['ann'] = {'bboxes': {}}
            info['width'] = img.shape[1]
            info['height'] = img.shape[2]

            tmp_boxes = torch.tensor(bbox, device='cpu', requires_grad=False)
            info['ann']['bboxes'] = np.array(obb2poly(tmp_boxes, 'oc'))  # 这里将OBB转换为8点表示形式
            bbbox = info['ann']['bboxes']
            sizes = [patch_shape[0]]
            # gaps=[0]
            windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
            window_anns = get_window_obj(info, windows, iof_thr)
            patchs, patch_infos = crop_and_save_img(info, windows, window_anns,
                                                    img,
                                                    no_padding=True,
                                                    # no_padding=False,
                                                    padding_value=padding_value)

            # 对每张大图分解成的子图集合中的每张子图遍历
            for i, patch_info in enumerate(patch_infos):
                if jump_empty_patch:
                    # 如果该patch中不含有效标签,将其跳过不输出,可在训练时使用

                    if patch_info['labels'] == [-1]:
                        # print('Patch does not contain box.\n')
                        continue
                obj = patch_info['ann']
                if min(obj['bboxes'].shape) == 0:  # 张量为空
                    tmp_boxes = poly2obb(torch.tensor(obj['bboxes']), 'oc')  # oc转化可以处理空张量
                else:
                    tmp_boxes = poly2obb(torch.tensor(obj['bboxes']), self.version)  # 转化回5参数
                p_bboxes.append(tmp_boxes.to(device))
                # p_trunc.append(torch.tensor(obj['trunc'],device=device))  # 是否截断,box全部在win内部时为false
                ## 若box超出win范围则trunc为true
                p_labels.append(torch.tensor(patch_info['labels'], device=device))
                p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
                                'y_start': torch.tensor(patch_info['y_start'], device=device),
                                'shape': patch_shape, 'trunc': torch.tensor(obj['trunc'], device=device)})

                patch = patchs[i]
                p_imgs.append(patch.to(device))

            out_imgs.append(p_imgs)
            out_bboxes.append(p_bboxes)
            out_labels.append(p_labels)
            out_metas.append(p_metas)

    elif mode == 'test':
        p_imgs = []
        p_metas = []
        img = imgs.cpu().squeeze(0)
        # patch
        info = dict()
        info['labels'] = np.array(torch.tensor([], device='cpu'))
        info['ann'] = {'bboxes': {}}
        info['width'] = img.shape[1]
        info['height'] = img.shape[2]

        sizes = [patch_shape[0]]
        # gaps=[0]
        windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
        patchs, patch_infos = crop_img_withoutann(info, windows, img,
                                                  no_padding=False,
                                                  padding_value=padding_value)

        # 对每张大图分解成的子图集合中的每张子图遍历
        for i, patch_info in enumerate(patch_infos):
            p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
                            'y_start': torch.tensor(patch_info['y_start'], device=device),
                            'shape': patch_shape, 'img_shape': patch_shape, 'scale_factor': 1})

            patch = patchs[i]
            p_imgs.append(patch.to(device))

        out_imgs.append(p_imgs)
        out_metas.append(p_metas)

        return out_imgs, out_metas

    return out_imgs, out_bboxes, out_labels, out_metas

def list2tensor(img_lists):
    '''
    images: list of list of tensor images
    '''
    inputs = []
    for img in img_lists:
        inputs.append(img.cpu())
    inputs = torch.stack(inputs, dim=0).cpu()
    return inputs

def relocate(idx, local_bboxes, patch_meta):
    # 二阶段的bboxes为array
    # put patches' local bboxes to full img via patch_meta
    meta = patch_meta[idx]
    top = meta['y_start']
    left = meta['x_start']

    local_bboxes_tmp = local_bboxes[0]
    for i in range(len(local_bboxes_tmp)):
        bbox = local_bboxes_tmp[i]
        # print('local_bboxes[i]:',bbox)
        bbox[0] += left
        bbox[1] += top
    return


# 从Global的信息整理成forward格式
def Collect_Global(g_img_infos, img_metas, length_thr):
    g_gt_boxes = []
    g_gt_labels = []

    for idx in range(len(g_img_infos)):
        g_gt_boxes.append(g_img_infos[idx]['gt_box'].squeeze(0))
        g_gt_labels.append(g_img_infos[idx]['labels'].squeeze(0))
        g_img_infos[idx]['img_shape'] = img_metas[0]['img_shape']
        g_img_infos[idx]['pad_shape'] = img_metas[0]['pad_shape']
        g_img_infos[idx]['scale_factor'] = 1.0

    # 各层按阈值进行标签分配(过滤)
    g_gt_boxes, g_gt_labels=filter_small_ann(g_gt_boxes, g_gt_labels, length_thr, g_img_infos)  # 这里进行标签过滤

    return g_gt_boxes, g_gt_labels, g_img_infos


def filter_small_ann(gt_bboxes, gt_labels, length_thr, g_img_infos=None):

    gt_bboxes_global = []
    gt_labels_global = []
    gt_bboxes_global_ignore = []
    gt_labels_global_ignore = []
    
    for gt, (bbox, label) in enumerate(zip(gt_bboxes, gt_labels)):
        # down_ratio = g_img_infos[gt]
        tmp_boxes = gt_bboxes[gt].clone()
        # gt_prepare = tmp_boxes[0].unsqueeze(0)  # 无gt时候补
        # gt_label_prepare = gt_labels[gt][[0]]
        gt_prepare = torch.zeros((0, 5), device=tmp_boxes.device)  # 无符合条件gt时来候补
        gt_label_prepare = torch.tensor([], device=tmp_boxes.device)
        # 根据长度阈值进行筛选
        mask = (tmp_boxes[:, 2] < length_thr) & (tmp_boxes[:, 3] < length_thr)

        tmp_boxes_out_ignore = tmp_boxes[mask]
        keeps_ignore = torch.nonzero(mask).squeeze(1)
        tmp_boxes_out = tmp_boxes[~mask]
        keeps = torch.nonzero(~mask).squeeze(1)

        tmp_labels_out = label[keeps]
        tmp_labels_out_ignore = label[keeps_ignore]

        if len(tmp_boxes_out) < 1:
            gt_bboxes_global.append(gt_prepare)
            gt_labels_global.append(gt_label_prepare)
        else:
            gt_bboxes_global.append(tmp_boxes_out)
            gt_labels_global.append(tmp_labels_out)

        gt_bboxes_global_ignore.append(tmp_boxes_out_ignore)
        gt_labels_global_ignore.append(tmp_labels_out_ignore)
    return gt_bboxes_global, gt_labels_global
    # return gt_bboxes_global, gt_labels_global, gt_bboxes_global_ignore, gt_labels_global_ignore


@ROTATED_DETECTORS.register_module()
class RotatedTwoStageDetectorImgFPN2(RotatedBaseDetector):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 max_ratio_nums=5,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 global_roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedTwoStageDetectorImgFPN2, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.global_backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)
            self.global_neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)
            self.global_rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)
            print('build local_roi_head.')
        if global_roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            global_roi_head.update(train_cfg=rcnn_train_cfg)
            global_roi_head.update(test_cfg=test_cfg.rcnn)
            global_roi_head.pretrained = pretrained     
            self.global_roi_head = build_head(global_roi_head)
            print('build global_roi_head.')
            print('finest_scale global:',self.global_roi_head.bbox_roi_extractor.finest_scale)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.max_ratio_nums=max_ratio_nums  # 多分辨率图像金字塔最高层数

        self.fusion_convs = nn.ModuleList()
        print('-----Fusion convs 卷积核设为3-----')
        for ratio in range(self.max_ratio_nums):
            ch = self.neck.out_channels
            self.fusion_convs.append(ConvModule((ratio+2) * ch, ch,3,padding=1))

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def ConcateFPNFea(self, g_fea, l_fea, g_img_metas):
        '''
        Args:
            l_fea (tuple(tensor)): features of local patches after FPN
            g_fea (tuple(tensor)): features of multi-resolution global imgs after FPN
            g_img_metas (list(dict)): infos of each global img in g_imgs

        Returns:

        '''
        # concate local FPN and global FPN
        same_lvls = []
        gl_lvls = torch.zeros(len(g_fea))
        m_ratio_g_feas = []
        crop_g_feas=[]
        for rat in range(len(g_img_metas)):
            down_ratio = g_img_metas[rat]['down_ratio']
            s_ratio_g_fea = [g[rat, :, :, :].unsqueeze(0) for g in g_fea]
            m_ratio_g_feas.append(s_ratio_g_fea)

            # 得到patch在当前ratio下的特征图中的相对坐标
            rel_x0y0x1y1 = g_img_metas[rat]['rel_x0y0x1y1']
            [left, top, right, down] = rel_x0y0x1y1.squeeze().cpu().numpy()
            # 对当前ratio下的每层特征图进行裁剪,得到与底层patch具有相同空间位置的特征块
            crop_g_fea = [lvl_fea[:,:,
                          int(left*lvl_fea.size()[2]):int(right*lvl_fea.size()[2]),
                          int(top *lvl_fea.size()[3]):int(down *lvl_fea.size()[3])]
                          for lvl_fea in s_ratio_g_fea]
            # 进行特征融合
            crop_g_feas.append(crop_g_fea)

        merge_fea_out=[]


        for lvl in range(len(l_fea)):
            s_lvl_l_fea = l_fea[lvl]
            g_feas = s_lvl_l_fea
            lvls=0  # 总共拼合的层数
            for ratio in range(len(crop_g_feas)):
                g_lvl=lvl-ratio-1 # ratio为0,1,2,3,对应特征比local特征高1,2,3,4层
                if g_lvl>0 and g_lvl<len(crop_g_feas[ratio]):
                    g_fea=crop_g_feas[ratio][g_lvl]
                    g_geas= torch.cat((g_feas, g_fea), dim=1) # 不上/下采样,直接融合
                    lvls+=1

            if lvls>0:  # 如果融合层没有找到匹配的global的特征,则不进行融合
                merge_fea = self.fusion_convs[lvls-1](g_geas)
                merge_fea_out.append(merge_fea)  # 将融合后的特征作为local分支的特征
            else:
                merge_fea_out.append(l_fea[lvl])

        return tuple(merge_fea_out)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feat_global(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.global_backbone(img)
        if self.with_neck:
            x = self.global_neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 5).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      g_img_list=None,
                      g_img_infos=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        """
        losses = dict()
        # local 子图块特征提取
        l_fea = self.extract_feat(img)
        merge_l_fea=l_fea

        if g_img_list is not None: 
            if len(g_img_list)>0:
                if len(g_img_list)>2:
                    g_img_list=g_img_list[0:2]  # 只保留down2 和 down4
                    g_img_infos=g_img_infos[0:2]

                losses_global = dict()

                # 1)读取输入的图像金字塔各层图像,并进行标签过滤
                length_thr = 15  
                g_imgs = torch.stack(g_img_list, dim=0).squeeze(1).permute(0, 3, 2, 1)
                g_gt_boxes, g_gt_labels, g_img_metas = Collect_Global(g_img_infos, img_metas, length_thr)

                g_fea = self.extract_feat_global(g_imgs)

                merge_l_fea = self.ConcateFPNFea(g_fea, l_fea, g_img_metas)

                if self.with_rpn:
                    proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                    self.test_cfg.rpn)
                    g_rpn_losses, g_proposal_list = self.global_rpn_head.forward_train(
                        g_fea,
                        g_img_metas,
                        g_gt_boxes,  # 把多分辨率图像当成一个batch输入,这样get_target_single时会分开分配处理
                        gt_labels=None,
                        gt_bboxes_ignore=gt_bboxes_ignore,
                        proposal_cfg=proposal_cfg,
                        **kwargs)
                    # losses_global.update(g_rpn_losses)
                    losses_global['loss_rpn_cls_global']=g_rpn_losses['loss_rpn_cls']
                    losses_global['loss_rpn_bbox_global'] = g_rpn_losses['loss_rpn_bbox']
                else:
                    g_proposal_list = proposals

                g_roi_losses = self.global_roi_head.forward_train(g_fea, g_img_metas, g_proposal_list,
                                                        g_gt_boxes, g_gt_labels,
                                                        gt_bboxes_ignore, gt_masks,
                                                        **kwargs)
                # losses_global.update(g_roi_losses)
                losses_global['loss_cls_global'] = g_roi_losses['loss_cls']
                losses_global['loss_bbox_global'] = g_roi_losses['loss_bbox']
                losses_global['acc_global'] = g_roi_losses['acc']
                losses.update(losses_global)

            else:
                zero=torch.tensor(0.,device=img.device)
                losses_global = dict()
                g_roi_losses = {'loss_cls': zero, 'loss_bbox': zero, 'acc': zero}
                g_rpn_losses = {'loss_rpn_cls': [zero,zero,zero,zero,zero], 'loss_rpn_bbox': [zero,zero,zero,zero,zero]}
                losses_global['loss_rpn_cls_global'] = g_rpn_losses['loss_rpn_cls']
                losses_global['loss_rpn_bbox_global'] = g_rpn_losses['loss_rpn_bbox']
                losses_global['loss_cls_global'] = g_roi_losses['loss_cls']
                losses_global['loss_bbox_global'] = g_roi_losses['loss_bbox']
                losses_global['acc_global'] = g_roi_losses['acc']
                losses.update(losses_global)



        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                merge_l_fea,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(merge_l_fea, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)
    

    def Test_Concat_Patches_GlobalImg(self, ori_img, ratio, scale, g_fea, patch_shape, gaps, p_bs, proposals, rescale=False):
        """
        对按一定比例scale缩小后的global img进行切块检测,并返回拼接后的完整特征图
        Args:
            ratio: 当前金字塔某一层的global img大小和金字塔最顶层的img大小的比值
            scale: 原始图像的大小和当前金字塔某一层的global img大小的比值
        """
        img = F.interpolate(ori_img, scale_factor=1 / scale, mode='bilinear')
        print('global img shpae:',img.shape)
        patches_bboxes_lists = []
        full_patches = []
        templates = []
        for idx in range(len(g_fea)):
            _, C, H, W = g_fea[idx].size()
            full_patch = torch.zeros(C, int(H * ratio), int(W * ratio), device='cpu', requires_grad=False)
            template = torch.zeros(C, int(H * ratio), int(W * ratio), device='cpu', requires_grad=False)
            full_patches.append(full_patch)
            templates.append(template)

        # Crop full img into patches 'test'
        gt_bboxes = []
        gt_labels = []
        device=get_device()
        p_imgs, p_metas = FullImageCrop(self, img, gt_bboxes, gt_labels, 
                                        patch_shape=patch_shape,gaps=gaps, mode='test')

        for i in range(img.shape[0]):
            j = 0
            patches = list2tensor(p_imgs[i])  # list to tensor
            patches_meta = p_metas[i]
            # patch batchsize
            while j < len(p_imgs[i]):
                if (j + p_bs) >= len(p_imgs[i]):
                    patch = patches[j:]
                    patch_meta = patches_meta[j:]
                else:
                    patch = patches[j:j + p_bs]
                    patch_meta = patches_meta[j:j + p_bs]  # x_start and y_start
              
                with torch.no_grad():
                    # fea_l_neck = self.extract_feat(patch)
                    patch=patch.to(device)
                    patch_fea = self.extract_feat_global(patch)
                    for idx in range(len(g_fea)):
                        s_lvl_g_fea = g_fea[idx]
                        s_lvl_l_fea = patch_fea[idx]
                        _, C, H, W = s_lvl_g_fea.size()
                        C_p, H_p, W_p = s_lvl_l_fea[0].size()

                        x_starts = [d['x_start'] for d in patch_meta]
                        y_starts = [d['y_start'] for d in patch_meta]
                        p_shapes = [d['shape'] for d in patch_meta]

                        for pa in range(s_lvl_l_fea.shape[0]):
                            patch = s_lvl_l_fea[pa]
                            # coordinate after backbone+neck
                            x_start = int(x_starts[pa] / img.shape[2] * int(W * ratio))
                            y_start = int(y_starts[pa] / img.shape[3] * int(H * ratio))
                            
                            full_patches[idx][:, x_start:x_start + W_p, y_start:y_start + H_p] += patch.cpu()
                            templates[idx][:, x_start:x_start + W_p, y_start:y_start + H_p] += 1

                        full_patches[idx] = full_patches[idx] / templates[idx]
                        full_patches[idx][torch.isnan(full_patches[idx])] = 0  # nan置零

                    if proposals is None:
                        proposal_list = self.global_rpn_head.simple_test_rpn(patch_fea, patch_meta)
                    else:
                        proposal_list = proposals
                    # 这里输出每组patch上的预测结果
                    global_bbox_list = self.global_roi_head.simple_test(
                        patch_fea, proposal_list, patch_meta, rescale=rescale)

                    for idx, res_list in enumerate(global_bbox_list):
                        # 1)将每个patch的local boxes按照裁剪时的坐标放置到大图上对应的位置
                        relocate(idx, res_list, patch_meta)
                        # 2)按照缩放倍率放大到原始影像尺寸
                        resize_bboxes_len6(res_list, scale)

                    patches_bboxes_lists.append(global_bbox_list)
                j = j + p_bs

        patches_bboxes_list = merge_results_two_stage(patches_bboxes_lists, iou_thr=0.4)
        # print('scale:',scale)
        print('global_patches_bboxes_list shape:',patches_bboxes_list.shape)
        if patches_bboxes_list.shape[-1]==5:
            patches_bboxes_list = torch.zeros((0, 6), device=device)
        # full_patches_out = [full_patch.cpu() for full_patch in full_patches]  # 转移到cpu
        # full_patches_out =[]
        full_patches_out =full_patches
        return patches_bboxes_list, full_patches_out


    def Test_Concat_Patches_GlobalImg_without_fea(self, ori_img, ratio, scale, g_fea, patch_shape, gaps, p_bs, proposals, rescale=False):
        """
        对按一定比例scale缩小后的global img进行切块检测,并返回拼接后的完整特征图
        Args:
            ratio: 当前金字塔某一层的global img大小和金字塔最顶层的img大小的比值
            scale: 原始图像的大小和当前金字塔某一层的global img大小的比值
        """
        img = F.interpolate(ori_img, scale_factor=1 / scale, mode='bilinear')
        print('global img shpae:',img.shape)
        patches_bboxes_lists = []
        gt_bboxes = []
        gt_labels = []
        device=get_device()
        p_imgs, p_metas = FullImageCrop(self, img, gt_bboxes, gt_labels, 
                                        patch_shape=patch_shape,gaps=gaps, mode='test')

        for i in range(img.shape[0]):
            j = 0
            patches = list2tensor(p_imgs[i])  # list to tensor
            patches_meta = p_metas[i]
            # patch batchsize
            while j < len(p_imgs[i]):
                if (j + p_bs) >= len(p_imgs[i]):
                    patch = patches[j:]
                    patch_meta = patches_meta[j:]
                else:
                    patch = patches[j:j + p_bs]
                    patch_meta = patches_meta[j:j + p_bs]  # x_start and y_start
              
                with torch.no_grad():
                    # fea_l_neck = self.extract_feat(patch)
                    patch=patch.to(device)
                    patch_fea = self.extract_feat_global(patch)
                    for idx in range(len(g_fea)):
                        # l_feas[idx].append(torch.tensor(l_fea[idx]))
                        s_lvl_g_fea = g_fea[idx]
                        s_lvl_l_fea = patch_fea[idx]
                        _, C, H, W = s_lvl_g_fea.size()
                        C_p, H_p, W_p = s_lvl_l_fea[0].size()

                        x_starts = [d['x_start'] for d in patch_meta]
                        y_starts = [d['y_start'] for d in patch_meta]
                        p_shapes = [d['shape'] for d in patch_meta]

                        for pa in range(s_lvl_l_fea.shape[0]):
                            patch = s_lvl_l_fea[pa]
                            # coordinate after backbone+neck
                            x_start = int(x_starts[pa] / img.shape[2] * int(W * ratio))
                            y_start = int(y_starts[pa] / img.shape[3] * int(H * ratio))

                    if proposals is None:
                        proposal_list = self.global_rpn_head.simple_test_rpn(patch_fea, patch_meta)
                    else:
                        proposal_list = proposals
                    # 这里输出每组patch上的预测结果
                    global_bbox_list = self.global_roi_head.simple_test(
                        patch_fea, proposal_list, patch_meta, rescale=rescale)

                    for idx, res_list in enumerate(global_bbox_list):
                        # 1)将每个patch的local boxes按照裁剪时的坐标放置到大图上对应的位置
                        relocate(idx, res_list, patch_meta)
                        # 2)按照缩放倍率放大到原始影像尺寸
                        resize_bboxes_len6(res_list, scale)

                    patches_bboxes_lists.append(global_bbox_list)
                j = j + p_bs

        patches_bboxes_list = merge_results_two_stage(patches_bboxes_lists, iou_thr=0.4)
        # print('scale:',scale)
        print('global_patches_bboxes_list shape:',patches_bboxes_list.shape)
        if patches_bboxes_list.shape[-1]==5:
            patches_bboxes_list = torch.zeros((0, 6), device=device)
        # full_patches_out = [full_patch.cpu() for full_patch in full_patches]  # 转移到cpu
        full_patches_out =[]
        return patches_bboxes_list, full_patches_out


    def Test_Patches_Img(self,img,patch_shape,gaps, p_bs, proposals, rescale=False):
        """
        对输入的img按patch_shape,gaps决定的窗口进行切块检测
        """
        # Crop full img into patches
        gt_bboxes=[]
        gt_labels=[]
        device=get_device()
        local_bboxes_lists=[]
        p_imgs, p_metas = FullImageCrop(self, img, gt_bboxes, gt_labels,
                                        patch_shape=patch_shape,
                                        gaps=gaps, mode='test')

        for i in range(img.shape[0]):
            j = 0
            patches = list2tensor(p_imgs[i])  # list to tensor
            patches_meta = p_metas[i]

            # patch batchsize
            while j < len(p_imgs[i]):
                if (j+p_bs) >= len(p_imgs[i]):
                    patch = patches[j:]
                    patch_meta = patches_meta[j:]
                else:
                    patch = patches[j:j + p_bs]
                    patch_meta = patches_meta[j:j + p_bs]  # x_start and y_start

                with torch.no_grad():
                    # fea_l_neck = self.extract_feat(patch)
                    patch=patch.to(device)
                    x = self.extract_feat(patch)
                    if proposals is None:
                        proposal_list = self.rpn_head.simple_test_rpn(x, patch_meta)
                    else:
                        proposal_list = proposals
                        
                    local_bbox_list = self.roi_head.simple_test(
                        x, proposal_list, patch_meta, rescale=rescale)
                    # 将每个patch的local boxes放置到大图上对应的位置
                    for idx, res_list in enumerate(local_bbox_list):
                        det_bboxes = res_list
                        relocate(idx, det_bboxes, patch_meta)
                    local_bboxes_lists.append(local_bbox_list)
                    # local_bboxes_lists.append([local_bbox_list,local_label_list])

                j = j+p_bs
        # 进行NMS
        # bbox_list = merge_results_two_stage(local_bboxes_lists,iou_thr=0.5)
        bbox_list = merge_results_two_stage(local_bboxes_lists,iou_thr=0.4)
        print('local_patches_bboxes_list shape:',bbox_list.shape)
        if bbox_list.shape[-1]==5:
            bbox_list = torch.zeros((0, 6)).cpu()

        return bbox_list

    def Test_Patches_Merge_Global(self, img, global_fea_list, patch_shape, gaps, p_bs, proposals, rescale=False):
        """
        对输入的img按patch_shape,gaps进行切块后,与global_fea_list中对应位置的特征进行融合,
        使用融合后特征得到patch的检测结果
        """
        # Crop full img into patches
        gt_bboxes=[]
        gt_labels=[]
        device=get_device()
        local_bboxes_lists=[]
        p_imgs, p_metas = FullImageCrop(self, img, gt_bboxes, gt_labels,
                                        patch_shape=patch_shape,
                                        gaps=gaps, mode='test')

        for i in range(img.shape[0]):
            j = 0
            patches = list2tensor(p_imgs[i])  # list to tensor
            patches_meta = p_metas[i]

            # patch batchsize
            while j < len(p_imgs[i]):
                if (j+p_bs) >= len(p_imgs[i]):
                    patch = patches[j:]
                    patch_meta = patches_meta[j:]
                else:
                    patch = patches[j:j + p_bs]
                    patch_meta = patches_meta[j:j + p_bs]  # x_start and y_start

                with torch.no_grad():
                    patch=patch.to(device)
                    local_fea = self.extract_feat(patch)
                    
                    x_starts = [d['x_start'] for d in patch_meta]
                    y_starts = [d['y_start'] for d in patch_meta]
                    p_shape = patch_meta[0]['shape']

                    for lvl in range(len(local_fea)):
                        fusion_fea_lvl = local_fea[lvl]
                        p_num=len(x_starts)

                        for p in range(p_num):
                            fusion = fusion_fea_lvl[p].unsqueeze(0)
                            
                            lvls=0  
                            for ratio, global_fea in enumerate(global_fea_list):
                                g_lvl=lvl-ratio-1 # ratio为0,1,2,3,对应特征比local特征高1,2,3,4层
                                if g_lvl>0 and g_lvl<len(global_fea_list):
                                    g_fea_lvl=global_fea[g_lvl]
                                    C, H, W = g_fea_lvl.size()
                                    # coordinate after backbone+neck
                                    x_start = int(x_starts[p] / img.shape[2] * W)
                                    y_start = int(y_starts[p] / img.shape[3] * H)
                                    # x_start = [int(x_starts[pa] / img.shape[2] * W) for pa in range(p_num)]
                                    # y_start = [int(y_starts[pa] / img.shape[3] * H) for pa in range(p_num)]
                                    p_lenx = int(p_shape[0] / img.shape[2] * W)
                                    p_leny = int(p_shape[1] / img.shape[3] * H)
                                    # 裁剪对应部分
                                    g_fea_cut = g_fea_lvl[:, x_start:x_start + p_lenx,y_start:y_start + p_leny].to(local_fea[lvl].device)  #从cpu转移到gpu上
                                  
                                    fusion = torch.cat((fusion, g_fea_cut),dim=1)
                                    lvls+=1
                            if lvls>0: 
                                fusion_fea = self.fusion_convs[lvls-1](fusion)
                                local_fea[lvl][p] = fusion_fea 

                        if proposals is None:
                            proposal_list = self.rpn_head.simple_test_rpn(local_fea, patch_meta)
                        else:
                            proposal_list = proposals
                    
                        local_bbox_list = self.roi_head.simple_test(
                            local_fea, proposal_list, patch_meta, rescale=rescale)
                   
                        for idx, res_list in enumerate(local_bbox_list):
                            det_bboxes = res_list
                            relocate(idx, det_bboxes, patch_meta)
                        local_bboxes_lists.append(local_bbox_list)
                        # local_bboxes_lists.append([local_bbox_list,local_label_list])

                j = j+p_bs
        # 进行NMS
        bbox_list = merge_results_two_stage(local_bboxes_lists,iou_thr=0.4)

        return bbox_list


    def Test_Single_Global(self,img,img_metas,scale,proposals,rescale=False,mode='bilinear'):
        """
        对输入的img按scale,进行缩放后,直接进行检测得到结果
        """
        # Global feature extraction
        global_img = F.interpolate(img, scale_factor=1 / scale, mode=mode)
        # extract global feature
        x_g = self.extract_feat_global(global_img)
        if proposals is None:
            global_proposal_list = self.global_rpn_head.simple_test_rpn(x_g, img_metas)
        else:
            global_proposal_list = proposals

        global_bbox_list = self.global_roi_head.simple_test(
            x_g, global_proposal_list, img_metas, rescale=rescale)

        # 放大回原图
        global_bbox_list = [resize_bboxes_len6(global_bbox_list[0], scale)]
        return global_bbox_list
    
    # ## 原始切图推理
    # def simple_test(self, img, img_metas, proposals=None, rescale=False):
    #     """Test without augmentation."""
    #     assert self.with_bbox, 'Bbox head must be implemented.'

    #     # gaps = [0]
    #     gaps = [200]
    #     patch_shape = (1024, 1024)
    #     p_bs = 4  # patch batchsize
    #     # 使用新的函数
    #     bbox_list = self.Test_Patches_Img(img, patch_shape, gaps, p_bs, proposals, rescale=False)

    #     final_bbox_list=[bbox_list.numpy()]
    #     return [final_bbox_list]

    # 金字塔推理
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        all_bboxes_lists=[]
        # 缩放原图进行推理
        # # 提取某一缩放尺度对应的global上的box,返回还原到原图尺度的box值
        # global_bbox_list_1 = self.Test_Single_Global(img, img_metas, scale_1, proposals, mode='bilinear')
        # all_bboxes_lists.append(global_bbox_list_1)

        # 每一尺度的图像判断是否超出最小patch size,如果超出则进行裁剪+预测,并返回预测结果+完整的特征图(cpu上)
        # 1)先对最小尺度的global img进行推理
        global_shape_min = img.shape[3]
        
        gloabl_shape_list=[]
        while global_shape_min > 1024:
            global_shape_min = global_shape_min/2
            gloabl_shape_list.append((global_shape_min, global_shape_min))
       
        global_shape_min = (global_shape_min, global_shape_min)
        print('global_shape_min',global_shape_min )
        scale_min = img.shape[3] / global_shape_min[0]
        global_img_min = F.interpolate(img, scale_factor=1 / scale_min, mode='bilinear')

        min_g_feature = self.extract_feat_global(global_img_min)
        if proposals is None:
            proposal_list = self.global_rpn_head.simple_test_rpn(min_g_feature, img_metas)
        else:
            proposal_list = proposals

        min_global_box_list = self.global_roi_head.simple_test(min_g_feature, proposal_list,
                                                        img_metas, rescale=rescale)
        
        for idx, res_list in enumerate(min_global_box_list):
            # 按照缩放倍率放大到原始影像尺寸
            resize_bboxes_len6(res_list, scale_min)

        # 放回到cpu上
        # for fea in min_g_feature:
        #     fea=fea.cpu()
        # 这里不需要添加最顶层,下面循环时会遍历到
        # all_bboxes_lists.append(torch.tensor(min_global_box_list[0][0]))
        # print('min_global_box_list.shape:',all_bboxes_lists[0].shape)

        # 2)再依次向下层得到切块结果以及对应的整特征图
        gaps = [200]
        patch_shape = (1024, 1024)
        p_bs = 4  # patch batchsize
        global_fea_list = []

        for global_shape in gloabl_shape_list:
            # scale: 原始大幅面图像img的大小和当前金字塔某一层的global img大小的比值
            scale = img.shape[3]/global_shape[0]
            # ratio: 当前金字塔某一层的global img大小和金字塔最顶层的global img大小的比值
            ratio = global_shape[0]/global_shape_min[0]
            
            # 控制预测的下采样层数
            scale_int=int(scale)
            
            global_patches_bbox_list, global_full_fea = self.Test_Concat_Patches_GlobalImg_without_fea(img, ratio, scale,
                                                                                           min_g_feature,
                                                                                           patch_shape, gaps, p_bs,
                                                                                           proposals)
            all_bboxes_lists.append(global_patches_bbox_list)
            global_fea_list.append(global_full_fea)

        # (2) 对原始分辨率图像裁剪子图块进行推理
        p_bs=2
        # local_bboxes_list = self.Test_Patches_Merge_Global(img, global_fea_list, patch_shape, gaps, p_bs, proposals, rescale=False)
        # all_bboxes_lists.append(local_bboxes_list)

        local_bboxes_list = self.Test_Patches_Img(img, patch_shape, gaps, p_bs, proposals, rescale=False)
        all_bboxes_lists.append(local_bboxes_list)

        ## 进行NMS
        bbox_list = merge_results_tensor(all_bboxes_lists, iou_thr=0.5).cpu()
        final_bbox_list = [bbox_list.numpy()]
        
        # final_bbox_list = [local_bboxes_list.numpy()]
        return [final_bbox_list]


    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
