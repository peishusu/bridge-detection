# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadImageFromFile

from ..builder import ROTATED_PIPELINES
import re
import os
import os.path as osp
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from mmrotate.models.detectors.img_split_bridge_tools import *
import cv2
from random import randint


CLASSES = ('bridge', )

def load_annotations(ann_file,
                     version='oc',
                     difficulty=1):
    """
        Args:
            ann_folder: folder that contains DOTA v1 annotations txt files
    """
    cls_map = {c: i
               for i, c in enumerate(CLASSES)
               }  # in mmdet v2.0 label is 0-based
    gt_bboxes = []
    gt_labels = []
    with open(ann_file) as f:
        s = f.readlines()
        for si in s:
            bbox_info = si.split()
            poly = np.array(bbox_info[:8], dtype=np.float32)
            try:
                x, y, w, h, a = poly2obb_np(poly,version)  
            except:  # noqa: E722
                continue
            cls_name = bbox_info[8]
            difficult = int(bbox_info[9])
            label = cls_map[cls_name]
            if difficult > difficulty:
                pass
            else:
                # gt_bboxes.append([x, y, w, h, a])
                gt_bboxes.append(poly) 
                gt_labels.append(label)

        if gt_bboxes:
            gt_boxes_out = np.array(
                gt_bboxes, dtype=np.float32)
            gt_labels_out = np.array(
                gt_labels, dtype=np.int64)
        else:
            # gt_boxes_out = np.zeros((0, 5),dtype=np.float32)
            gt_boxes_out = np.zeros((0, 8), dtype=np.float32)  
            gt_labels_out = np.array([], dtype=np.int64)

    info=dict()
    info['gt_boxes'] = gt_boxes_out
    info['gt_labels'] = gt_labels_out
    return info


def crop_and_save_img_np(info, windows, window_anns, img, no_padding, padding_value):
    """

    Args:
        info (dict): Image's information.
        windows (np.array): information of sliding windows.
        window_anns (list[dict]): List of bbox annotations of every window.
        img (tensor): Full images.
        no_padding (bool): If True, no padding.
        padding_value (tuple[int|float]): Padding value.

    Returns:
        list[dict]: Information of paths.
    """

    patchs = []
    patch_infos = []
    for i in range(windows.shape[0]):
        patch_info = dict()
        for k, v in info.items():
            if k not in ['id', 'fileanme', 'width', 'height', 'ann']:
                patch_info[k] = v

        window = windows[i]
        x_start, y_start, x_stop, y_stop = window.tolist()

        patch_info['x_start'] = x_start
        patch_info['y_start'] = y_start
        ann = window_anns[i]
        ann['bboxes'] = translate(ann['bboxes'], -x_start, -y_start)
        patch_info['ann'] = ann
        x_0,y_0,x_1,y_1=np.clip(np.array(window.tolist()),0,img.shape[0])
        patch = img[y_0:y_1, x_0:x_1]  # crop
        # patch = img[y_start:y_stop, x_start:x_stop,:]

        if not no_padding:
            height = y_stop - y_start
            width = x_stop - x_start

            if height > patch.shape[0] or width > patch.shape[1]:
                padding_patch = np.empty((height, width, patch.shape[-1]),
                                         dtype=np.uint8)
                if not isinstance(padding_value, (int, float)):
                    assert len(padding_value) == patch.shape[-1]
                padding_patch[...] = padding_value
                # padding_patch[:patch.shape[0], :patch.shape[1], ...] = patch
                deltax=0
                deltay=0
                if x_start<0:
                    deltax=-x_start
                if y_start<0:
                    deltay=-y_start
                padding_patch[deltay:deltay+patch.shape[0], deltax:deltax+patch.shape[1],...] = patch

                patch = padding_patch
        patch_info['height'] = patch.shape[0]
        patch_info['width'] = patch.shape[1]

        bboxes_num = patch_info['ann']['bboxes'].shape[0]
        patch_label = []

        if bboxes_num == 0:
            # patch_info['labels']=[-1]  # bkgd
            patch_label = [-1]
            pass
        else:
            for idx in range(bboxes_num):
                obj = patch_info['ann']
                patch_label.append(patch_info['labels'][idx])

        patch_info['labels'] = patch_label
        patch_infos.append(patch_info)
        patchs.append(patch)

    return patchs, patch_infos

def CropSingleWin(info_global,win_info,img,iof_thr,version):

    info = dict()
    info['labels'] = info_global['gt_labels']
    info['ann'] = {'bboxes': {}}
    info['width'] = img.shape[0]
    info['height'] = img.shape[1]

    info['ann']['bboxes'] = info_global['gt_boxes']
    win_x0= win_info['win_x0']
    win_y0 = win_info['win_y0']
    win_size = win_info['win_size']
    single_win = np.array([win_x0, win_y0, win_x0 + win_size, win_y0 + win_size]).reshape(-1, 4)
    window_anns_single = get_window_obj(info, single_win, iof_thr)
    patchs_single, patch_info_single = crop_and_save_img_np(info, single_win,
                                                            window_anns_single,
                                                            img, no_padding=False,
                                                            padding_value=[104, 116, 124])
    imgs_out = []
    gt_box_out = []
    label_out = []
    # for i, patch_info_single in enumerate(patch_infos):
    patch_info_single=patch_info_single[0]
    obj = patch_info_single['ann']
    tmp_boxes=[]
    if min(obj['bboxes'].shape) == 0: 
        tmp_boxes = np.zeros((0, 5), dtype=np.float32)
        # tmp_boxes = poly2obb_np(obj['bboxes'],'oc')  
    else:
        for poly in range(obj['bboxes'].shape[0]):
            tmp_box = poly2obb_np(obj['bboxes'][poly,:], version)
            tmp_boxes.append(tmp_box)
        tmp_boxes=np.array(tmp_boxes, dtype=np.float32)

    # p_trunc.append(torch.tensor(obj['trunc'],device=device))  # trunc
    single_gt_box_out = tmp_boxes
    single_labels_out = patch_info_single['labels']
    single_img = patchs_single

    imgs_out = single_img[0]
    gt_box_out = single_gt_box_out
    label_out = single_labels_out
    return imgs_out, gt_box_out, label_out,patch_info_single



@ROTATED_PIPELINES.register_module()
class LoadFPNImageFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 load_global_threshold=300,
                 version='oc',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.global_threshold=load_global_threshold
        self.version=version
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        print('------LoadFPNImageFromFile------')
    
    
    def poly2obb_list(self,gt_boxes_poly):
        gt_boxes_obb = []
        for poly in range(gt_boxes_poly.shape[0]):
            tmp_box = poly2obb_np(gt_boxes_poly[poly, :], self.version)  
            gt_boxes_obb.append(tmp_box)
        gt_boxes_obb = np.array(gt_boxes_obb, dtype=np.float32)
        return gt_boxes_obb

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        # img = mmcv.imfrombytes(img_bytes, flag=self.color_type, channel_order=self.channel_order, backend='tifffile')

        if self.to_float32:
            img = img.astype(np.float32) # ori img

        read_global = False
        boxes = results['ann_info']['bboxes']
        w_max = np.max(boxes[:,2])
        h_max = np.max(boxes[:,3])
        if w_max >= self.global_threshold or h_max >= self.global_threshold:
            read_global=True

        read_global=False # no training crop

        results['g_img_list'] = []
        results['g_img_infos'] = []
        if read_global:

            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, filename)
            x_y_2 = re.findall(r'\d+', x_y[0])
            crop_size = int(filename.split('__')[1])
            begin_x, begin_y = int(x_y_2[0]), int(x_y_2[1])

            global_lvl=0
            global_img_list=[]
            global_img_infos=[]
            imgname=str(filename).split('/')[-1].split('__')[0]+'.png'
            root_path = '../WorldBridge/train6/'
            down2_dir = root_path+'down2/images/'+imgname
            down2_img = self.file_client.get(down2_dir)
            down2_img = mmcv.imfrombytes(
                down2_img, flag=self.color_type, channel_order=self.channel_order)
            ori_shape=down2_img.shape[0]*2  
            ratio = 2
            size = int(ori_shape/ratio) # 第一层
            # size = int(ori_shape[0])
            # if down2_img.shape[0] <= 1024:
            if down2_img.shape[0] <= crop_size:
                g_img = down2_img
                coor_x0 = begin_x /ratio  
                coor_y0 = begin_y /ratio
                # winodw position
                lvl_label_dir = root_path + 'down2/labelTxt/' + imgname.split('.')[0] + '.txt'
                info_global = load_annotations(lvl_label_dir, version=self.version, difficulty=1)
                g_info = dict()
                g_info['gt_box'] = info_global['gt_boxes']

                # to obb form
                # g_info['gt_box'] = []
                # for poly in range(info_global['gt_boxes'].shape[0]):
                #     tmp_box = poly2obb_np(info_global['gt_boxes'][poly, :], self.version)  # 转化回5参数
                #     g_info['gt_box'].append(tmp_box)
                # g_info['gt_box'] = np.array(g_info['gt_box'], dtype=np.float32)
                g_info['gt_box'] =self.poly2obb_list(info_global['gt_boxes'])

                g_info['labels'] = info_global['gt_labels']
                g_info['down_ratio'] = 2
                g_info['ori_shape'] = np.array(g_img.shape)
                rel_left = coor_x0/ g_img.shape[0] 
                rel_top = coor_y0 / g_img.shape[1]
                rel_right = (coor_x0 + crop_size / ratio) / g_img.shape[0]
                rel_down = (coor_y0 + crop_size / ratio) / g_img.shape[1]
                g_info['rel_x0y0x1y1'] = np.array([[rel_left, rel_top, rel_right, rel_down]])
                global_img_list.append(g_img)
                global_img_infos.append(g_info)

            else:
                while True:
                    global_lvl+=1
                    down_ratio = int(ratio**global_lvl)  # 计算当前下采样倍率
                    # print('当前下采样倍率:',down_ratio)

                    size = ori_shape/down_ratio
                    if size < 1024:  # 如果下采样后的大小小于切块patch,则跳过
                        break

                    if global_lvl==1:
                        g_img=down2_img
                    else:
                        lvl_dir=root_path+'down'+str(down_ratio)+'/images/'+imgname
                        # print('lvl_dir:',lvl_dir)
                        # assert os.path.exists(lvl_dir)
                        g_img_bytes = self.file_client.get(lvl_dir)
                        g_img = mmcv.imfrombytes(
                            g_img_bytes, flag=self.color_type, channel_order=self.channel_order)

                    lvl_label_dir = root_path + 'down' + str(down_ratio) + '/labelTxt/' + imgname.split('.')[
                        0] + '.txt'

                    ## crop and padding
                    coor_x0 = begin_x / down_ratio  # cood
                    coor_y0 = begin_y / down_ratio
                    padding_size = (crop_size-(crop_size/down_ratio))/2  # /padding siz
                    max_x,max_y=g_img.shape[:2]
                    
                    win_x0 = int(coor_x0-padding_size)
                    win_y0 = int(coor_y0-padding_size)
                    if max_x >= crop_size and max_y >= crop_size: 
                        win_x0 = max(0, min(win_x0, max_x - crop_size))
                        win_y0 = max(0, min(win_y0, max_y - crop_size))
                    # get window position
                    g_win = {'win_x0': int(win_x0), 'win_y0': int(win_y0), 'win_size': crop_size}

                    # gt_boxes and gt_labels
                    info_global = load_annotations(lvl_label_dir,version=self.version,difficulty=1)
                    tmp_img, tmp_box, tmp_label,patch_info_single\
                        = CropSingleWin(info_global, g_win, g_img, iof_thr=0.1, version=self.version)

                    # cv2.imwrite('/scratch/luojunwei/WorldBridge/tmp_vis/' + str(down_ratio) + '-ori-' + imgname,
                    #             g_img)
                    # cv2.imwrite('/scratch/luojunwei/WorldBridge/tmp_vis/'+str(down_ratio)+'-'+imgname,tmp_img)

                    global_img_list.append(tmp_img)
                    g_info = dict()
                    g_info['gt_box'] = tmp_box
                    g_info['labels'] =np.array(tmp_label, dtype=np.int64)
                    # g_info['labels'] = tmp_label
                    g_info['down_ratio'] = down_ratio
                    g_info['ori_shape'] = np.array(g_img.shape)
                    # 得到底层patch子图块的左上和右下在切出的global图块中的相对位置
                    rel_left = (coor_x0 - patch_info_single['x_start'])/tmp_img.shape[0]
                    rel_top = (coor_y0 - patch_info_single['y_start'])/tmp_img.shape[1]
                    rel_right = (coor_x0 - patch_info_single['x_start']+crop_size/down_ratio) / tmp_img.shape[0]
                    rel_down = (coor_y0 - patch_info_single['y_start']+crop_size/down_ratio) / tmp_img.shape[1]
                    g_info['rel_x0y0x1y1'] = np.array([[rel_left, rel_top, rel_right, rel_down]])
                    global_img_infos.append(g_info)
            # index_num = randint(0, len(global_img_list)-1)
            # results['g_img_list'] = [global_img_list[index_num]]
            # results['g_img_infos'] = [global_img_infos[index_num]]
            results['g_img_list'] = global_img_list
            results['g_img_infos'] = global_img_infos
        
        
        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape

        results['img_fields'] = ['img']
        return results


@ROTATED_PIPELINES.register_module()
class LoadPatchFromImage(LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but only reserve a patch of
    ``results['img']`` according to ``results['win']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with image in ``results['img']``.

        Returns:
            dict: The dict contains the loaded patch and meta information.
        """

        img = results['img']
        x_start, y_start, x_stop, y_stop = results['win']
        width = x_stop - x_start
        height = y_stop - y_start

        patch = img[y_start:y_stop, x_start:x_stop]
        if height > patch.shape[0] or width > patch.shape[1]:
            patch = mmcv.impad(patch, shape=(height, width))

        if self.to_float32:
            patch = patch.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = patch  
        results['img_shape'] = patch.shape
        results['ori_shape'] = patch.shape
        results['img_fields'] = ['img']
        return results

