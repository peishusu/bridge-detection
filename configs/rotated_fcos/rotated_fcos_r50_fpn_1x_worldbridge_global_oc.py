_base_ = [
    '../_base_/datasets/dotav1.py', #'../_base_/schedules/schedule_1x.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
#angle_version = 'le90'
angle_version = 'oc'

# model settings
model = dict(
    # type='RotatedFCOS',
    type='RotatedBridge',
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
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RotatedFCOSHead',
        #num_classes=15,
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    crop_cfg=dict(
        local_cfg='/project/luojunwei/test4/mmrotate/configs/oriented_reppoints/oriented_qua_reppoints_r50_fpn_1x_bridgelocal_oc.py',
        local_ckpt='/project/luojunwei/test4/mmrotate/tools/World_bridge/oriented_reppoints_r50_fpn_2x_worldbridge_oc_local/epoch_24.pth',
        global_shape = (1024, 1024),
        # global_shape = (2048, 2048),
        patch_shape = (1024, 1024),
        patch_gaps=[0],
        patch_bs = 4  # patch batchsize
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0, 
        score_thr=0.05, 
        nms=dict(iou_thr=0.4),
        max_per_img=400))
    # test_cfg=dict(
    #     nms_pre=2000,
    #     min_bbox_size=0,
    #     score_thr=0.05,
    #     nms=dict(iou_thr=0.1),
    #     max_per_img=2000))

# DATASET
dataset_type = 'DOTADataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='RResize', img_scale=(4096, 4096)), 
    # dict(type='RResize', img_scale=(2048, 2048)), 
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

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # dict(type='RResize', img_scale=(4096, 4096)), 
        img_scale=(4096, 4096),
        # img_scale=(8192, 8192),
        flip=False,
        transforms=[
            # dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # dict(type='RResize', img_scale=(4096, 4096)), 
        img_scale=(4096, 4096), #(2048)
        # img_scale=(8192, 8192),
        flip=False,
        transforms=[
            # dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data_root_train='/scratch/luojunwei/WorldBridge/17/train8/split_4096_iof01/'
data_root_val='/scratch/luojunwei/WorldBridge/17/val2/split_4096_iof01/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        img_prefix=data_root_train +'images/', 
        ann_file=data_root_train +'labelTxt/', 
        # img_prefix=data_root +'train/images/',
        # ann_file=data_root +'train/labelTxt/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_prefix=data_root_val +'images/', 
        ann_file=data_root_val +'labelTxt/', 
        # img_prefix=data_root + 'val/images/',
        # ann_file=data_root + 'val/labelTxt/',
        pipeline=val_pipeline), 
        # pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root_val +'images/', 
        ann_file=data_root_val +'labelTxt/', 
        pipeline=test_pipeline))


optimizer = dict(lr=0.008)
# optimizer = dict(lr=0.002)
auto_scale_lr = dict(enable=True, base_batch_size=4)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    #warmup_ratio=0.001,
    #step=[8, 11])
    #step=[12, 18])
    step=[14, 18])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=2)
evaluation = dict(interval=1)
