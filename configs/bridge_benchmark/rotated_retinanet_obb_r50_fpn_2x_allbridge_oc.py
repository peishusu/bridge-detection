_base_ = [
    '../_base_/datasets/dotav1.py', #'../_base_/schedules/schedule_1x.py', 
    '../_base_/schedules/schedule_2x.py',  #改为2×
    '../_base_/default_runtime.py'
]

angle_version = 'oc'
model = dict(
    type='RotatedRetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RotatedRetinaHead',
        #num_classes=15,
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=False,
            proj_xy=False,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))


# DATASET
dataset_type = 'DOTADataset'

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

# data_root_train='/scratch/luojunwei/WorldBridge/train6/split_1024_gap200_iof03/'  # local
data_root_train='/scratch/luojunwei/WorldBridge/train6/global_split_1024/'  # global

data_root_val='/scratch/luojunwei/WorldBridge/val2/split_1024_gap0_iof03/'
data_root_test='/scratch/luojunwei/WorldBridge/test2/'
# data_root_test='/scratch/luojunwei/WorldBridge/test2/test_tmp/'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_prefix=data_root_train +'images/', 
        ann_file=data_root_train +'labelTxt/', 
        pipeline=train_pipeline, 
        version=angle_version),
    val=dict(
        type=dataset_type,
        img_prefix=data_root_test + 'images/', 
        ann_file=data_root_test + 'labelTxt/',
        pipeline=test_pipeline, 
        version=angle_version),
    test=dict(
        type=dataset_type,
        img_prefix=data_root_test + 'images/', 
        ann_file=data_root_test + 'labelTxt/',
        pipeline=test_pipeline, 
        version=angle_version))


training_times = 2
auto_scale_lr = dict(enable=True, base_batch_size=4)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8*training_times, 11*training_times])

runner = dict(type='EpochBasedRunner', max_epochs=training_times*12)
checkpoint_config = dict(interval=2)
evaluation = dict(interval=4)
