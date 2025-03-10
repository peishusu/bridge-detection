# evaluation
evaluation = dict(interval=1, metric='mAP') 
#evaluation = dict(interval=1, metric='recall',iou_thr=[0.5,0.75])
# evaluation=dict(interval=1,metric='mAP',iou_thr=0.75) 

# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=2)
