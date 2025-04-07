'''
    配置文件定义了一个基于旋转目标检测的两阶段检测器（RotatedTwoStageDetectorImgFPN2），主要用于处理DOTA数据集中的旋转目标检测任务
'''


'''
继承了三个基础配置文件：
    datasets/dotav1.py：DOTA数据集的配置   
    schedules/schedule_2x.py：训练计划（2倍schedule）
    default_runtime.py：默认运行时配置
'''
_base_ = [
    '../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

# 角度编码方式，'oc'表示OpenCV风格的编码
angle_version = 'oc'


'''
模型定义：
'''
model = dict(
    # 自定义的两阶段旋转检测器
    type='RotatedTwoStageDetectorImgFPN2',
    # 骨干网络：ResNet50
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    # FPN 特征金字塔网络
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048], # 输入通道数(对应ResNet4个阶段)
        out_channels=256,# 输出通道数
        num_outs=5),# 输出5个尺度的特征图

    #检测头部部分
    # OrientedRPNHead（用于生成旋转建议框）
    rpn_head=dict(
        type='OrientedRPNHead',# 旋转RPN头部
        in_channels=256, # 输入通道
        feat_channels=256, # 特征通道
        version=angle_version, # 角度编码版本
        anchor_generator=dict(  # 锚框生成器
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict( # # 分类损失
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict( # 边界框回归损失
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),

    # 标准ROI Head（OrientedStandardRoIHead）
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            finest_scale=16,  
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            # num_classes=15,
            num_classes=1, 
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    # 全局ROI Head（额外添加的，用于处理全局信息）
    global_roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            finest_scale=56,  
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            use_reweight=True,
            # num_classes=15,
            num_classes=1, 
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    # 训练配置
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    # 测试配置
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=1000,
            min_bbox_size=0, 
            score_thr=0.05, 
            nms=dict(iou_thr=0.4),
            max_per_img=400)))

# DATASET
# 使用DOTADataset数据集类型
dataset_type = 'DOTADataset'

# 图像归一化配置
img_norm_cfg = dict(
    # 均值 标准差 是否转化为rgb
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# data_root_test='/project/luojunwei/tmp_scratch_envs/WorldBridge/test2/test_tmp/'
data_root_test='/project/luojunwei/tmp_scratch_envs/WorldBridge/test2/'

# 训练流水线
train_pipeline = [
    # 加载标注
    dict(type='LoadAnnotations', with_bbox=True),
    # 加载FPN图像
    dict(type='LoadFPNImageFromFile',load_global_threshold=300,version='oc'),
    # 调整大小
    dict(type='RResize', img_scale=(1024, 1024)),
    # 随机翻转（水平垂直对角线）
    dict(
        type='RandomFlipImgFPN',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    # 归一化
    dict(type='NormalizeImgFPN', **img_norm_cfg),
    # 填充到32的倍数
    dict(type='Pad', size_divisor=32),
    # 默认格式化转换
    dict(type='DefaultFormatBundle'),
    # 收集数据
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels','g_img_list','g_img_infos'])
]

# 验证流水线
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024), # 将图像resize到1024x1024分辨率
        flip=False, # 禁用水平翻转（如果设为True，会额外测试翻转后的图像）
        transforms=[ # 定义具体的转换操作列表
            # dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg), # 归一化
            dict(type='Pad', size_divisor=32), # 将图像填充到32的倍数
            dict(type='DefaultFormatBundle'), # 将数据转换为模型输入的标准化格式
            dict(type='Collect', keys=['img']) # 筛选需要传递给模型的数据键
        ])
]

# 训练数据路径
data_root_train='Bridge/train6/split_1024_gap200_iof03/'
# 不同下采样级别的数据（down2, down4, down8, down16）
data_root_train_down2='Bridge/train6/global_split_1024/down2/'
data_root_train_down4='Bridge/train6/global_split_1024/down4/'
data_root_train_down8='Bridge/train6/global_split_1024/down8/'
data_root_train_down16='Bridge/train6/global_split_1024/down16/'

# load_from='xxx/oriented_rcnn_r50_fpn_2x_ImgFPN_orires/epoch_18.pth'
# load_from='xxx/oriented_rcnn_r50_fpn_2x_ImgFPN_down2_loadfrom/epoch_12.pth'
# load_from='xxx/oriented_rcnn_r50_fpn_2x_ImgFPN_down4_loadfrom_down2_loadfrom/epoch_6.pth'
# load_from='xxx/oriented_rcnn_r50_fpn_2x_ImgFPN_down4_loadfrom-down2-ep12_loadfrom_lr001/epoch_6.pth'
load_from='xxx/oriented_rcnn_r50_fpn_2x_ImgFPN_down8_loadfrom-down4-ep6-loadfrom-down2-ep12_loadfrom_lr001/epoch_4.pth'


##### Raw large image
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            # dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1, # 每个cpu的样本数
    workers_per_gpu=4, # 每个gpu的数据加载线程数
    train=dict(
        type=dataset_type,
        # img_prefix=data_root_train +'images/', 
        # ann_file=data_root_train +'labelTxt/', 
        # img_prefix=data_root_train_down2 +'images/', 
        # ann_file=data_root_train_down2 +'labelTxt/',         
        # img_prefix=data_root_train_down4 +'images/', 
        # ann_file=data_root_train_down4 +'labelTxt/', 
        # img_prefix=data_root_train_down8 +'images/', 
        # ann_file=data_root_train_down8 +'labelTxt/', 
        img_prefix=data_root_train_down16 +'images/',  # 图像路径
        ann_file=data_root_train_down16 +'labelTxt/',  # 标注路径
        pipeline=train_pipeline), # 数据预处理流程
    val=dict(
        type=dataset_type,
        # pipeline=val_pipeline), 
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root_test + 'images/',  # vis
        ann_file=data_root_test + 'labelTxt/',
        pipeline=test_pipeline))

# 训练倍数
training_times = 2

# 优化器学习率
optimizer = dict(lr=0.0025) 
auto_scale_lr = dict(enable=True, base_batch_size=4)
# learning policy
lr_config = dict( # 学习率策略
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8*training_times, 11*training_times])
# 运行器配置
runner = dict(type='EpochBasedRunner', max_epochs=training_times*12)
# 每2个epoch保存一次模型
checkpoint_config = dict(interval=2)
# 每2个epoch评估一次
evaluation = dict(interval=2)
