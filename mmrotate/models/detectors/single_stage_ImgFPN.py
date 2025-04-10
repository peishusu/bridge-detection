# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmrotate.core import rbbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector


import mmcv
from pathlib import Path
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmdet.utils import get_device
import torch
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from .img_split_bridge_tools import *
# from .single_stage_bridge import *
import torch.nn.functional as F
from torch import nn

def resize_bboxes_len6(bboxes_out,scale):
    """Resize bounding boxes with scales."""

    box_out=bboxes_out
    w_scale = scale
    h_scale = scale
    box_out[:, 0] *= w_scale
    box_out[:, 1] *= h_scale
    box_out[:, 2:4] *= np.sqrt(w_scale * h_scale)

    return box_out

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
        inputs.append(img.cpu()) 
    inputs = torch.stack(inputs, dim=0)
    
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


def get_single_img(fea_g_necks, i):
    fea_g_neck=[]
    for idx in range(len(fea_g_necks)):
        fea_g_neck.append(fea_g_necks[idx][i])

    return tuple(fea_g_neck)

def relocate(idx, local_bboxes, patch_meta):
    # put patches' local bboxes to full img via patch_meta
    meta=patch_meta[idx]
    top = meta['y_start']
    left = meta['x_start']

    for i in range(len(local_bboxes)):
        bbox = local_bboxes[i]
        if bbox.size()[0] == 0:
            continue
        bbox[0] += left
        bbox[1] += top

    return

@ROTATED_DETECTORS.register_module()
class RotatedSingleStageDetectorImgFPN(RotatedBaseDetector):
    """Base class for rotated single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedSingleStageDetectorImgFPN, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.global_backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
            self.global_neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.global_bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.version='oc'


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
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def Test_Concat_Patches_GlobalImg_without_fea(self, ori_img, ratio, scale, g_fea, patch_shape, gaps, p_bs, rescale=False):
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
                    
                    global_outs = self.global_bbox_head(patch_fea)
                    global_bbox_list = self.global_bbox_head.get_bboxes(*global_outs,
                                                                        patch_meta,
                                                                        rescale=False)

                    for idx, res_list in enumerate(global_bbox_list):
                        det_bboxes, det_labels = res_list
                        if det_bboxes.shape[0] == 0:
                            det_bboxes = torch.zeros((0, 6), device=device)
                        relocate(idx, det_bboxes, patch_meta)
                        resize_bboxes_len6(det_bboxes, scale)

                    patches_bboxes_lists.append(global_bbox_list)
                j = j + p_bs
        
        # single_stage的merge
        patches_bboxes_list = merge_results([patches_bboxes_lists], iou_thr=0.4)
            
        full_patches_out =[]
        return patches_bboxes_list, full_patches_out



    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetectorImgFPN, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses
    

    # def simple_test(self, img, img_metas, rescale=False):
    #     """Test function without test time augmentation.

    #     Args:
    #         imgs (list[torch.Tensor]): List of multiple images
    #         img_metas (list[dict]): List of image information.
    #         rescale (bool, optional): Whether to rescale the results.
    #             Defaults to False.

    #     Returns:
    #         list[list[np.ndarray]]: BBox results of each image and classes. \
    #             The outer list corresponds to each image. The inner list \
    #             corresponds to each class.
    #     """

    #     x = self.extract_feat(img)
    #     outs = self.bbox_head(x)
    #     bbox_list = self.bbox_head.get_bboxes(
    #         *outs, img_metas, rescale=rescale)

    #     bbox_results = [
    #         rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
    #         for det_bboxes, det_labels in bbox_list
    #     ]
    #     return bbox_results

    #### 金字塔联合推理
    def simple_test(self, img, img_metas, rescale=False):
        # 为了使用裁剪小图策略推理标准模型
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        print('single stage infetence Global-Local!!!!!!')
        # 1)先对最小尺度的global img进行推理
        global_shape_min = img.shape[3]
        
        gloabl_shape_list=[]
        while global_shape_min > 1024:
            global_shape_min = global_shape_min/2
            gloabl_shape_list.append((global_shape_min, global_shape_min))
       
        global_shape_min = (global_shape_min, global_shape_min)
        print('global_shape_min',global_shape_min)
        scale_min = img.shape[3] / global_shape_min[0]
        global_img_min = F.interpolate(img, scale_factor=1 / scale_min, mode='bilinear')
        min_g_feature = self.extract_feat_global(global_img_min)

        # min_g_outs = self.global_bbox_head(min_g_feature)
        # min_g_bbox_list = self.global_bbox_head.get_bboxes(
        #     *min_g_outs, img_metas, rescale=rescale)

        # for idx, res_list in enumerate(min_g_bbox_list):
        #     # 按照缩放倍率放大到原始影像尺寸
        #     resize_bboxes_len6(res_list, scale_min)


        # 2)再依次向下层得到切块结果以及对应的整特征图
        all_bboxes_lists=[]
        gaps = [200]
        device=get_device()
        patch_shape = (1024, 1024)
        p_bs = 2  # patch batchsize
        global_fea_list = []
        print('gloabl_shape_list:', gloabl_shape_list)
        for global_shape in gloabl_shape_list:
            scale = img.shape[3]/global_shape[0]
            ratio = global_shape[0]/global_shape_min[0]

            global_patches_bbox_list, global_full_fea = self.Test_Concat_Patches_GlobalImg_without_fea(img, ratio, scale,
                                                                                                       min_g_feature,
                                                                                                       patch_shape, gaps, p_bs)
            all_bboxes_lists.append(global_patches_bbox_list)
            global_fea_list.append(global_full_fea)


        # Crop full img into patches
        gt_bboxes=[]
        gt_labels=[]
        p_imgs, p_metas = FullImageCrop(self, img, gt_bboxes, gt_labels,
                                        patch_shape=patch_shape,
                                        gaps=gaps, mode='test')

        local_bboxes_lists=[]
        for i in range(img.shape[0]):
            j = 0
            # patches = list2tensor(p_imgs[i])  # list to tensor,此时放在cpu上
            # p_imgs[i]=torch.stack(p_imgs[i], dim=0)
            # patches=p_imgs[i]
            patches = list2tensor(p_imgs[i])
            patches_meta = p_metas[i]

            # patch batchsize
            while j < len(p_imgs[i]):
                if (j+p_bs) >= len(p_imgs[i]):
                    patch = patches[j:]
                    patch = patches[j:]
                    patch_meta = patches_meta[j:]
                else:
                    patch = patches[j:j + p_bs]
                    patch_meta = patches_meta[j:j + p_bs]  # x_start and y_start

                # 使用已训练好的local权重提取特征
                # fea_l_neck = self.extract_feat(patch)

                with torch.no_grad():
                    patch=patch.to(device) # 转移回gpu上 
                    # print('patch device:',patch.device)
                    fea_l_neck = self.extract_feat(patch)
                    # 这里输出Local的预测结果
                    outs_local = self.bbox_head(fea_l_neck)
                    # local的meta设置为1,因为这里未经过缩放
                    # print('outs_local',outs_local)
                    local_bbox_list = self.bbox_head.get_bboxes(*outs_local,
                                                                patch_meta,
                                                                rescale=rescale)

                    # print('local_bbox_list:',local_bbox_list)                                                    
                    # 将每个patch的local boxes放置到大图上对应的位置
                    for idx, res_list in enumerate(local_bbox_list):
                        det_bboxes, det_labels = res_list
                        relocate(idx, det_bboxes, patch_meta)
                    local_bboxes_lists.append(local_bbox_list)

                j = j+p_bs
            # fusion_out = self.fusion_fea(fea_g_neck, fusion_feature)
        local_bboxes_list = merge_results([local_bboxes_lists],iou_thr=0.4)
        #  加入底层结果
        all_bboxes_lists.append(local_bboxes_list)

        ## 进行NMS
        # bbox_list = [merge_results_tensor(all_bboxes_lists, iou_thr=0.5)]
        bbox_list = [merge_results(all_bboxes_lists, iou_thr=0.5)]

        # print(' bbox_list', bbox_list)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results



    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
