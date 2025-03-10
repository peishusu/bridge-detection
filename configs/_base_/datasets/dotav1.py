# dataset settings
dataset_type = 'DOTADataset'

data_root_train='/scratch/luojunwei/WorldBridge/train6/split_1024_gap200_iof03/'
data_root_val='/scratch/luojunwei/WorldBridge/val2/split_1024_gap0_iof03/'
data_root_test='/scratch/luojunwei/WorldBridge/test2/'


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024), 
        flip=False,
        transforms=[
            dict(type='RResize'),
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
        img_scale=(1024, 1024), 
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_prefix=data_root_train +'images/', 
        ann_file=data_root_train +'labelTxt/', 
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_prefix=data_root_val +'images/', 
        ann_file=data_root_val +'labelTxt/', 
        pipeline=val_pipeline), 
    test=dict(
        type=dataset_type,
        img_prefix=data_root_test + 'images/', 
        ann_file=data_root_test + 'labelTxt/',
        pipeline=test_pipeline))
