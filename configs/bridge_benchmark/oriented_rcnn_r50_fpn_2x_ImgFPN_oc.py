_base_ = [
    '../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'oc'
model = dict(
    type='RotatedTwoStageDetectorImgFPN2',
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
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
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
dataset_type = 'DOTADataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# data_root_test='/project/luojunwei/tmp_scratch_envs/WorldBridge/test2/test_tmp/'
data_root_test='/project/luojunwei/tmp_scratch_envs/WorldBridge/test2/'

train_pipeline = [
    dict(type='LoadAnnotations', with_bbox=True), 
    dict(type='LoadFPNImageFromFile',load_global_threshold=300,version='oc'),
    dict(type='RResize', img_scale=(1024, 1024)),  
    dict(
        type='RandomFlipImgFPN',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),   
    dict(type='NormalizeImgFPN', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels','g_img_list','g_img_infos'])
]


val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            # dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data_root_train='Bridge/train6/split_1024_gap200_iof03/'
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
    samples_per_gpu=1,
    workers_per_gpu=4,
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
        img_prefix=data_root_train_down16 +'images/', 
        ann_file=data_root_train_down16 +'labelTxt/', 
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # pipeline=val_pipeline), 
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        img_prefix=data_root_test + 'images/',  # vis
        ann_file=data_root_test + 'labelTxt/',
        pipeline=test_pipeline))


training_times = 2

optimizer = dict(lr=0.0025) 
auto_scale_lr = dict(enable=True, base_batch_size=4)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8*training_times, 11*training_times])

runner = dict(type='EpochBasedRunner', max_epochs=training_times*12)
checkpoint_config = dict(interval=2)
evaluation = dict(interval=2)
