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
        inputs.append(img.cpu())  # 转移到cpu上,否则torch.stack内存不足
    inputs = torch.stack(inputs, dim=0)
    # inputs = torch.stack(inputs, dim=0).to(get_device())
    
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
    device = get_device()
    img_rate_thr = 0.6  # 图片与wins窗口的交并比阈值
    iof_thr = 0.1  # 裁剪后的标签占原标签的比值阈值

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
                                                    padding_value=[104, 116, 124])

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
                                                  padding_value=[104, 116, 124])

        # 对每张大图分解成的子图集合中的每张子图遍历
        for i, patch_info in enumerate(patch_infos):
            p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
                            'y_start': torch.tensor(patch_info['y_start'], device=device),
                            'shape': patch_shape,'img_shape':patch_shape, 'scale_factor':1})

            patch = patchs[i]
            p_imgs.append(patch.to(device))

        out_imgs.append(p_imgs)
        # out_bboxes.append(p_bboxes)
        # out_labels.append(p_labels)
        out_metas.append(p_metas)

        return out_imgs, out_metas


    return out_imgs, out_bboxes,out_labels, out_metas

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
class RotatedSingleStageDetector(RotatedBaseDetector):
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
        super(RotatedSingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.version='oc'


    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

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
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
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
        print('single stage infetence!!!!!!')
        gaps = [200]
        patch_shape = (1024, 1024)
        p_bs = 4  # patch batchsize

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
            p_imgs[i]=torch.stack(p_imgs[i], dim=0)
            patches=p_imgs[i]

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



                with torch.no_grad():
                    fea_l_neck = self.extract_feat(patch)
                    outs_local = self.bbox_head(fea_l_neck)
                    local_bbox_list = self.bbox_head.get_bboxes(*outs_local,
                                                                patch_meta,
                                                                rescale=rescale)
                                                 
                    # 将每个patch的local boxes放置到大图上对应的位置
                    for idx, res_list in enumerate(local_bbox_list):
                        det_bboxes, det_labels = res_list
                        relocate(idx, det_bboxes, patch_meta)
                    local_bboxes_lists.append(local_bbox_list)

                j = j+p_bs

            # fusion_out = self.fusion_fea(fea_g_neck, fusion_feature)
        
        bbox_list = [merge_results([local_bboxes_lists],iou_thr=0.4)]
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
