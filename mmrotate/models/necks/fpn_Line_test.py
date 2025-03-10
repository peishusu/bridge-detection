# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import ROTATED_NECKS
# from ..builder import NECKS
import torch
import cv2
import numpy as np
import math

# class Line_Fusion(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  line_thre=50
#                  ):
#         super(Line_Fusion, self).__init__()
#         self.line_thre=line_thre
#
#         self.conv1=ConvModule(
#             in_channels,
#             out_channels,
#             1,
#             inplace=False
#         )
#
#
#     def forward(self,input):
#         '''
#         Args:
#             input: mask_lines from extract_Lines(),(batch,1,1024,1024)
#             output:features extracted from mask_lines
#         '''
#
#         output=input
#
#         return output



@ROTATED_NECKS.register_module()
class Line_FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale. 每个尺度输入的通道数
        out_channels (int): Number of output channels (used at each scale). 每个尺度的输出通道数
        num_outs (int): Number of output scales. 输出尺度(特征图)数量，如faster-rcnn中为5个特征图
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0. 从输入特征图的哪一层开始
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level. 结束
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(Line_FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels

        print('----测试代码：------,fpn的out_channel为 ',out_channels)

        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i], # 第[i]个特征图的输入通道,faster-rcnn为[256,512,1024,2048]
                out_channels, # 256
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv) # 这里的lateral_convs横向卷积与fpn_convs有什么区别？？？答：前者1×1卷积核，后者3×3
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1] # 如果只有一层extra_convs，从输入的倒数第二层获取
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        
        ######
        # line feature fusion conv, from 512 to 256
        self.line_fus_conv=ConvModule(
            2*out_channels,
            out_channels,
            kernel_size=7,
            padding=7//2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False
        )

        self.sigmoid=nn.Sigmoid()


    @auto_fp16()
    def forward(self, inputs):
        """Forward function.

        Args:
            inputs: Two kind of features after backbone: img features and line-mask features

        """

        # 这里的inputs，如果backbone为resnet，那么内容为4个tuple
        # inputs:(1,256,256,256)(1,512,128,128)(1,1024,64,64)(1,2048,32,32)

        ### inputs and lines
        l_feats=[]
        if len(inputs)>1:
            l_feats = inputs[1]  # Lines直线图经过backbone提取的特征
            inputs = inputs[0]  # Img经过backbone提取的特征

        


        assert len(inputs) == len(self.in_channels)

        # build laterals
        # laterals = [
        #     lateral_conv(inputs[i + self.start_level])+
        #     lateral_conv(l_feats[i + self.start_level]) # 直接concat线特征进行融合
        #     for i, lateral_conv in enumerate(self.lateral_convs)
        # ]
        laterals=[  # channel维度上concat后进行融合
            self.line_fus_conv(torch.cat([lateral_conv(inputs[i+self.start_level]), 
            lateral_conv(l_feats[i+self.start_level])],1))
            for i,lateral_conv in enumerate(self.lateral_convs)
        ] 

        
        # laterals=list(4),每层分别为(1,256,256,256)(1,256,128,128)(1,256,64,64)(1,256,32,32)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            # 这里就是FPN模块的上采样操作,每个特征图进行上采样和融合
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)  #上采样+融合相加操作,74行upsample_cfg=dict(mode='nearest'),
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:  # 添加额外来源卷积层
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]   #即为3*3卷积后的laterals[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
