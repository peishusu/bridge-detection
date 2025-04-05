# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed

from mmrotate.apis import train_detector
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import collect_env, get_root_logger, setup_multi_processes


def parse_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Train a detector')

    # 解析出来得到 "configs/bridge_benchmark/oriented_rcnn_r50_fpn_2x_ImgFPN_oc.py"
    parser.add_argument('config',
                        default='/scratch/luojunwei/WorldBridge/Code/mmrotate_bridge/configs/bridge_benchmark/',
                        help='train config file path')
    # 解析出来得到：bridge_train/oriented_rcnn_r50_fpn_2x_ImgFPN_dist
    parser.add_argument('--work-dir',
                        default='',
                        help='the dir to save logs and models')

    # 这些参数都没有传递
    #args.resume_from 变量会是 None，训练会从 头开始，不会加载任何之前的 checkpoint。
    parser.add_argument(
        '--resume-from',
        help='the checkpoint file to resume from')
    # 默认是 False，表示 不会自动寻找最近的 checkpoint 进行续训。
    parser.add_argument(
        '--auto-resume',
        # default=True, #打开自动续训
        action='store_true',
        help='resume from the latest checkpoint automatically')
    # args.no_validate 默认是 False，表示 训练过程中会定期在验证集上评估模型。
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')

    #  Python argparse 模块中的一个方法，用于创建一组互斥的命令行参数，确保用户在运行程序时只能选择该组中的一个参数，而不能同时使用多个
        # 允许不传递任何组内参数（args.gpus 和 args.gpu_ids 均为 None）。
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        # 参数数量，表示接受 1个或者 多个整数
        nargs='+',
        # default='1',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')

    # 传递过来默认是 0
    parser.add_argument('--seed', type=int, default=None, help='random seed') 
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    # 指定任务启动方式，默认使用单机非分布式模式
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    # local_rank用于标识当前进程在单个计算节点（机器）中的GPU编号
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


'''
    该代码是 OpenMMLab 框架（如 MMDetection/MMRotate）的标准训练流程：
'''
def main():
    # 解析命令行参数
    args = parse_args()

    # 是 MMDetection/MMRotate 等 OpenMMLab 框架中用于加载配置文件的核心方法，其作用是将 Python 格式的配置文件解析为可操作的配置对象
    # 加载配置文件
    cfg = Config.fromfile(args.config) # # 从.py文件加载配置对象

    # 命令行覆盖，
    if args.cfg_options is not None:
        # # 深度合并字典
        cfg.merge_from_dict(args.cfg_options)

    # # 配置多进程环境变量
    # 是 OpenMMLab 框架（如 MMDetection/MMRotate）中用于 配置多进程训练环境 的工具方法，主要解决分布式训练和数据处理中的并行化问题
    setup_multi_processes(cfg)

    # set cudnn_benchmark，启用cudnn加速
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename 工作目录优先级：命令行 > 配置文件 > 默认（基于配置文件名）
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # 恢复训练设置
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    # GPU ID处理
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute training time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir，创建工作目录
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config，保存配置文件副本
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps，初始化日志系统
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    # 作用：收集环境信息
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)


    # 构建检测模型，根据配置构建旋转目标检测模型
    # bulid_detector()
    model = build_detector(
        cfg.model, # 模型配置
        train_cfg=cfg.get('train_cfg'), # 训练配置
        test_cfg=cfg.get('test_cfg')) # 测试配置
    # 权重初始化
    model.init_weights()


    # 构建训练集
    datasets = [build_dataset(cfg.data.train)]

    # 如果配置中定义了两阶段工作流（train+val），构建验证集
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    # 在chekpoint中保存元数据
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],# 版本信息
            CLASSES=datasets[0].CLASSES) # 类别名称

    # add an attribute for visualization convenience
    # 为模型添加类别属性（便于可视化）
    model.CLASSES = datasets[0].CLASSES

    # 开始训练
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
