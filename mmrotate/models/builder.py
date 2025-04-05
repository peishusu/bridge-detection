# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.models.builder import MODELS

ROTATED_BACKBONES = MODELS
ROTATED_LOSSES = MODELS
ROTATED_DETECTORS = MODELS
ROTATED_ROI_EXTRACTORS = MODELS
ROTATED_HEADS = MODELS
ROTATED_NECKS = MODELS
ROTATED_SHARED_HEADS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return ROTATED_BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return ROTATED_NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROTATED_ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head."""
    return ROTATED_SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    return ROTATED_HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return ROTATED_LOSSES.build(cfg)

'''
    是 OpenMMLab 框架中用于构建旋转目标检测器（Rotated Detector）的核心函数
'''
def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    # ROTATED_DETECTORS：旋转检测器的注册器（Registry）对象
    '''
        构建过程：
            从 cfg 中读取 type 字段（如 type='RotatedTwoStageDetectorImgFPN2'）
            在 ROTATED_DETECTORS 注册器中查找对应的类                
            实例化该类，传入：               
                主配置 cfg               
                默认参数 default_args（包含训练/测试配置）
    '''
    return ROTATED_DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
