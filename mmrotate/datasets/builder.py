# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform

from mmcv.utils import build_from_cfg
from mmdet.datasets import DATASETS, PIPELINES
from mmdet.datasets.builder import _concat_dataset

ROTATED_DATASETS = DATASETS
ROTATED_PIPELINES = PIPELINES

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

'''
是 OpenMMLab 框架中用于动态构建数据集的核心工厂函数，它通过递归解析配置字典来创建复杂的数据集组合。
'''
def build_dataset(cfg, default_args=None):
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                 ConcatDataset,
                                                 MultiImageMixDataset,
                                                 RepeatDataset)
    # 多配置列表处理
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        dataset = MultiImageMixDataset(**cp_cfg)
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        # ROTATED_DATASETS 表示注册器
        dataset = build_from_cfg(cfg, ROTATED_DATASETS, default_args)

    return dataset
