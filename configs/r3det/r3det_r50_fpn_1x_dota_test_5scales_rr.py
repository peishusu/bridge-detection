_base_ = [
    '../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'oc'   #旋转定义方式
model = dict(
    type='R3Det',   #检测器名称
    backbone=dict(  #主干网络的配置文件
        type='ResNet',  #主干网络类别
        depth=50, #网络层数（深度）
        num_stages=4, #res_net的stage数量
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,  #第一阶段权重被冻结
        zero_init_residual=False,  # 是否对残差块(resblocks)中的最后一个归一化层使用零初始化(zero init)让它们表现为自身
        norm_cfg=dict(type='BN', requires_grad=True),  #归一化层配置项
        norm_eval=True,
        style='pytorch',   #pytorch或caffe
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),  #加载预训练模型
    neck=dict(
        type='FPN',  #neck类型
        in_channels=[256, 512, 1024, 2048],  #输入的各个stage的通道数，与主干网络一致
        out_channels=256,   #金字塔特征图每层输出通道
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),   #输出多少尺度(scales)特征图
    bbox_head=dict(
        type='RotatedRetinaHead',  #类型
        num_classes=15,
        in_channels=256,
        stacked_convs=4,     ##head卷积层的层数，其中r3det_tiny在这一项为2
        feat_channels=256,
        anchor_generator=dict(   #锚点(Anchor)生成器的配置
            type='RotatedAnchorGenerator',  #类别
            octave_base_scale=4,   #RetinaNet用于生成锚点的超参数，也是特征图anchor的基本尺度
            #octave_base_scale其值越大，所有anchor的尺度都会变大
            scales_per_octave=3,   #生成锚点的超参数，表示每个特征图有3个尺度
            ratios=[1.0, 0.5, 0.25,2.0,4.0], #高度与宽度的比例
            strides=[8, 16, 32, 64, 128]), # 锚生成器的步幅。这与 FPN 特征步幅一致。如果未设置 base_sizes，则当前步幅值将被视为 base_sizes。
        bbox_coder=dict(   ## 在训练和测试期间对框进行编码和解码 什么意思？？？
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
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    frm_cfgs=[dict(in_channels=256, featmap_strides=[8, 16, 32, 64, 128])],
    num_refine_stages=1,
    refine_heads=[  #refine head
        dict(
            type='RotatedRetinaRefineHead',
            num_classes=15,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            assign_by_circumhbbox=None,
            anchor_generator=dict(
                type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=False,
                proj_xy=False,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0))
    ],
    train_cfg=dict(  #训练超参数的配置
        s0=dict(
            assigner=dict( #分配器
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,  #IoU>=0.5被视为正样本
                neg_iou_thr=0.4,  #IoU<0.4被视为负样本
                min_pos_iou=0,  #将框作为正样本的最小IoU阈值
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        sr=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.5,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        stage_loss_weights=[1.0]),
    test_cfg=dict(
        nms_pre=2000,   #NMS前的box数
        min_bbox_size=0,  #bbox允许的最小尺寸
        score_thr=0.05,   #bbox的分数阈值
        nms=dict(iou_thr=0.1),  #NMS的阈值
        max_per_img=2000))   #每张图的最大检测次数

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
    #尝试增加RR
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=[9,11],
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
