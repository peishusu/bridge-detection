# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmrotate.core import (build_assigner, build_sampler,rbbox2result,
                           multiclass_nms_rotated, obb2poly, poly2obb)
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
import numpy as np
from .img_split_bridge_tools import *
# from .single_stage_bridge import *
from mmdet.utils import get_device
import torch.nn.functional as F
from copy import deepcopy


def resize_bboxes(bboxes,scale):
    """Resize bounding boxes with scales."""

    orig_shape = bboxes.shape
    out_boxxes=bboxes.clone().reshape((-1, 5))
    # bboxes = bboxes.reshape((-1, 5))
    w_scale = scale
    h_scale = scale
    out_boxxes[:, 0] *= w_scale
    out_boxxes[:, 1] *= h_scale
    out_boxxes[:, 2:4] *= np.sqrt(w_scale * h_scale)

    return out_boxxes

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

def resize(images, shape, label=False):
    '''
    resize PIL images
    shape: (w, h)
    '''
    resized = list(images)
    for i in range(len(images)):
        if label:
            resized[i] = images[i].resize(shape, Image.NEAREST)
        else:
            resized[i] = images[i].resize(shape, Image.BILINEAR)
    return resized

def list2tensor(img_lists):
    '''
    images: list of list of tensor images
    '''
    inputs = []
    for img in img_lists:
        if img.device!='cpu':
            inputs.append(img.cpu())
    inputs = torch.stack(inputs, dim=0).cpu()
    return inputs

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
    out_imgs=[]
    out_bboxes=[]
    out_labels=[]
    out_metas=[]
    device = imgs.device
    # print('imgs的设备为：',imgs.device)
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
            info['labels'] = np.array(torch.tensor(label, device='cpu',requires_grad=False))
            info['ann'] = {'bboxes': {}}
            info['width'] = img.shape[1]
            info['height'] = img.shape[2]

            tmp_boxes = torch.tensor(bbox, device='cpu', requires_grad=False)
            info['ann']['bboxes'] = np.array(obb2poly(tmp_boxes, self.version))  # 这里将OBB转换为8点表示形式

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
                torch.cuda.empty_cache()

            out_imgs.append(p_imgs)
            out_bboxes.append(p_bboxes)
            out_labels.append(p_labels)
            out_metas.append(p_metas)

    elif mode =='test':
        # for i in range(imgs.shape[0]):
        # for img in zip(imgs):
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
        patchs, patch_infos = crop_img_withoutann(info, windows,img,
                                                  no_padding=False,
                                                  padding_value=padding_value)

        # 对每张大图分解成的子图集合中的每张子图遍历
        for i, patch_info in enumerate(patch_infos):
            p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
                            'y_start': torch.tensor(patch_info['y_start'], device=device),
                            'shape': patch_shape,'img_shape':patch_shape, 'scale_factor':1})
            patch = patchs[i]
            p_imgs.append(patch)
            torch.cuda.empty_cache()

        out_imgs.append(p_imgs)
        out_metas.append(p_metas)

        return out_imgs, out_metas


    return out_imgs, out_bboxes,out_labels, out_metas

def get_single_img(fea_g_necks, i):
    fea_g_neck=[]
    for idx in range(len(fea_g_necks)):
        fea_g_neck.append(fea_g_necks[idx][i])

    return tuple(fea_g_neck)

def relocate(idx, local_bboxes, patch_meta):
    # 二阶段的bboxes为array
    # put patches' local bboxes to full img via patch_meta
    meta=patch_meta[idx]
    top = meta['y_start']
    left = meta['x_start']
    top = float(top.item())  # 转换为标量
    left = float(left.item())  # 转换为标量\float类型
    
    for local_box in local_bboxes:
        if len(local_box)==0:
            continue
        else:
            local_box[:,0]+= left
            local_box[:,1]+= top

    return

def list2array(local_bboxes_list):
    # tmp=[]
    # print('local_bboxes_list:',local_bboxes_list)
    tmp_all=[]
    for idx in range(len(local_bboxes_list)):
        bbox=local_bboxes_list[idx]
        # print('bbox',bbox)
    for idx in range(len(local_bboxes_list)):
        bbox = local_bboxes_list[idx]
        tmp_box=[]
        for j in range(len(bbox)):
            box=bbox[j]
            if len(box[0])==0:
                continue
            tmp_box.append(box)
        if len(tmp_box)==0:
            continue
        # tmp_array=np.stack(tmp_box,axis=1)
        tmp_array=np.concatenate(tmp_box,axis=1)
        tmp_all.append(tmp_array)
    
    if len(tmp_all)==0:
        arrayout=local_bboxes_list[0]
    else:
        arrayout=np.concatenate(tmp_all,axis=1)

    # for i in range(len(local_bboxes_list)):
    #     local_bbox=local_bboxes_list[i]
    return arrayout



@ROTATED_DETECTORS.register_module()
class RotatedTwoStageDetector(RotatedBaseDetector):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedTwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # print('--Use GLobal Branch in normal two-stage!!!--')

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    # def forward_dummy(self, img):
    #     """Used for computing network flops.

    #     See `mmdetection/tools/analysis_tools/get_flops.py`
    #     """
    #     outs = ()
    #     # backbone
    #     x = self.extract_feat(img)
    #     # rpn
    #     if self.with_rpn:
    #         rpn_outs = self.rpn_head(x)
    #         outs = outs + (rpn_outs, )
    #     proposals = torch.randn(1000, 5).to(img.device)
    #     # roi_head
    #     roi_outs = self.roi_head.forward_dummy(x, proposals)
    #     outs = outs + (roi_outs, )
    #     return outs
    
    # 切图推理
    def forward_dummy(self, img):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        outs = ()
        gaps = [200]
        patch_shape = (1024, 1024)
        # patch_shape = (800, 800)
        p_bs = 4  # patch batchsize
        proposals = torch.randn(1000, 6).to(img.device)  # (1000,6) for Oriented RCNN
        # 使用新的函数
        outs = self.Test_Patches_Img_Dummy(img, patch_shape, gaps, p_bs, proposals, outs)

        return outs

    def Test_Patches_Img_Dummy(self,img, patch_shape,gaps, p_bs, proposals, outs):
        """
        对输入的img按patch_shape,gaps决定的窗口进行切块检测
        """
        # Crop full img into patches
        gt_bboxes=[]
        gt_labels=[]
        device=img.device
        # print('当前设备：',device)
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
                    proposals = torch.randn(1000, 6).to(img.device)
                    if self.with_rpn:
                        rpn_outs = self.rpn_head(x)
                        outs = outs + (rpn_outs, )
                    # local的meta设置为1,因为这里未经过缩放
                    local_roi_outs = self.roi_head.forward_dummy(x, proposals)
                    
                    outs = outs + (local_roi_outs)

                j = j+p_bs
                
        return outs

    def fliter_small_ann(self,gt_bboxes,gt_labels,scale):

        # resize后同时小于两个阈值的标签不保留
        ratio_min=4
        # h_min=20

        jump_patch_ext=True

        gt_bboxes_global =[]
        gt_labels_global=[]
        gt_bboxes_global_ignore =[]
        gt_labels_global_ignore=[]
        gt_bboxes_after_resize=[]
        # 剔除resize后过小的标签
        for gt in range(len(gt_bboxes)):
            gt_bbox=gt_bboxes[gt].clone()
            tmp_boxes=resize_bboxes(gt_bbox, 1 / scale)
            gt_bboxes_after_resize.append(tmp_boxes)
            tmp_boxes_out=[]
            keeps=[]
            tmp_boxes_out_ignore=[]
            keeps_ignore=[]
            gt_prepare=tmp_boxes[0].unsqueeze(0) # 无gt时候补
            gt_label_prepare=gt_labels[gt][[0]]
            for idx in range(tmp_boxes.shape[0]):
                tmp_box=tmp_boxes[idx,:]
                w=tmp_box[2]
                h=tmp_box[3]
                ratio=h/w
                if ratio<1:
                    ratio=1/ratio

                if ratio<ratio_min:
                    tmp_boxes_out_ignore.append(tmp_box.unsqueeze(0)) # 这里放入的是缩小后的gtbox
                    # tmp_boxes_out_ignore.append(gt_bboxes[gt][idx].unsqueeze(0))
                    keeps_ignore.append(idx)
                    continue
                else:
                    tmp_boxes_out.append(tmp_box.unsqueeze(0))
                    keeps.append(idx)

            tmp_boxes_out=list2tensor_(tmp_boxes_out,dim=0)
            tmp_labels_out=gt_labels[gt][keeps]
            tmp_boxes_out_ignore=list2tensor_(tmp_boxes_out_ignore,dim=0)
            tmp_labels_out_ignore=gt_labels[gt][keeps_ignore]

            if len(tmp_boxes_out) > 2: # 过滤后还剩有较多标签
                jump_patch_ext=False
            if len(tmp_boxes_out) < 1:
                gt_bboxes_global.append(gt_prepare)
                gt_labels_global.append(gt_label_prepare)
            else:
                gt_bboxes_global.append(tmp_boxes_out)
                gt_labels_global.append(tmp_labels_out)
            
            # print('remain:',len(tmp_boxes_out))

            gt_bboxes_global_ignore.append(tmp_boxes_out_ignore)
            gt_labels_global_ignore.append(tmp_labels_out_ignore)


        return gt_bboxes_global, gt_labels_global, gt_bboxes_global_ignore, gt_labels_global_ignore,jump_patch_ext

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
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
    
    def extract_feat_global(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
       
       
    def Test_Concat_Patches_GlobalImg(self, ori_img, ratio, scale, g_fea, patch_shape, gaps, p_bs, proposals, rescale=False):
        """
        对按一定比例scale缩小后的global img进行切块检测
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

                    if proposals is None:
                        proposal_list = self.rpn_head.simple_test_rpn(patch_fea, patch_meta)
                    else:
                        proposal_list = proposals
                    # 这里输出每组patch上的预测结果
                    global_bbox_list = self.roi_head.simple_test(
                        patch_fea, proposal_list, patch_meta, rescale=rescale)

                    for idx, res_list in enumerate(global_bbox_list):
                        # 1)将每个patch的local boxes按照裁剪时的坐标放置到大图上对应的位置
                        relocate(idx, res_list, patch_meta)
                        # 2)按照缩放倍率放大到原始影像尺寸
                        resize_bboxes_len6(res_list, scale)

                    patches_bboxes_lists.append(global_bbox_list)
                j = j + p_bs
                torch.cuda.empty_cache()

        patches_bboxes_list = merge_results_two_stage(patches_bboxes_lists, iou_thr=0.4)
        for bbox_cls in patches_bboxes_list:
            if bbox_cls.shape[-1]==5:
                bbox_cls = np.zeros((0, 6)) 
        full_patches_out =[]

        return patches_bboxes_list, full_patches_out
    
    def Test_Patches_Img(self,img,patch_shape,gaps, p_bs, proposals, rescale=False):
        """
        对输入的img按patch_shape,gaps决定的窗口进行切块检测
        """
        # Crop full img into patches
        gt_bboxes=[]
        gt_labels=[]
        device=img.device
        # print('当前设备：',device)
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
        # 进行NMS,这里需要改为对多类别的
        bbox_list = merge_results_two_stage(local_bboxes_lists,iou_thr=0.4)
        

        for bbox_cls in bbox_list:
            if bbox_cls.shape[-1]==5:
                # bbox_cls = torch.zeros((0, 6)).cpu()
                bbox_cls = np.zeros((0, 6)) 
            # return [bbox_list.numpy()]

        return bbox_list

    def Test_Patches_Img_Select(self,img,saved_p_metas,patch_shape,gaps, p_bs, proposals, rescale=False):
        """
        对输入的img按patch_shape,gaps决定的窗口进行切块检测
        saved_p_metas:从global推理结果得到的含有多个较高置信度的box对应的patch位置,含有裁剪起始patch
        传入global_box进行精细检测
        """
        # Crop full img into patches
        gt_bboxes=[]
        gt_labels=[]
        device=img.device
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
                    # 这里输出Local的预测结果
                    # outs_local = self.bbox_head(fea_l_neck)
                    # local的meta设置为1,因为这里未经过缩放
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
        bbox_list = merge_results_two_stage(local_bboxes_lists,iou_thr=0.4)
        print('local_patches_bboxes_list shape:',bbox_list.shape)
        if bbox_list.shape[-1]==5:
            bbox_list = torch.zeros((0, 6)).cpu()

        return bbox_list


    # 切图推理
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        # gaps = [0]
        gaps = [200]
        patch_shape = (1024, 1024)
        # patch_shape = (800, 800)
        p_bs = 4  # patch batchsize
        # 使用新的函数
        bbox_list = self.Test_Patches_Img(img, patch_shape, gaps, p_bs, proposals, rescale=False)

        # bbox_list = torch.zeros((0, 6),device='cpu')
        final_bbox_list=bbox_list
        final_bbox_list = [bbox_list.numpy()]
        # print('final_bbox_list:',final_bbox_list)
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
