_base_ = ['./rotated_retinanet_obb_r50_fpn_2x_allbridge_le90.py']

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0)))
