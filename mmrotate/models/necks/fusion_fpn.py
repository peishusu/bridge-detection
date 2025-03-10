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
from mmdet.utils import get_device


# Channel Attention in BAM
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        gate_channels = [channel]
        gate_channels += [channel // reduction] * num_layers
        gate_channels += [channel]

        self.ca = nn.Sequential()
        self.ca.add_module('flatten', Flatten())
        for i in range(len(gate_channels) - 2):
            self.ca.add_module('fc%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.ca.add_module('bn%d' % i, nn.BatchNorm1d(gate_channels[i + 1]))
            self.ca.add_module('relu%d' % i, nn.ReLU())
        self.ca.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        res = self.avgpool(x)
        res = self.ca(res)
        res = res.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return res

# Spatial Attention in BAM
class SpatialAttention(nn.Module):
    # def __init__(self, channel, reduction=16, num_layers=3, dia_val=2):
    def __init__(self, channel, reduction=16, num_layers=2, dia_val=4):
        super().__init__()
        self.sa = nn.Sequential()
        # 1*1 conv reduce C
        self.sa.add_module('conv_reduce1', nn.Conv2d(kernel_size=1,
                                                     in_channels=channel,
                                                     out_channels=channel // reduction)) # 1*1 conv first
        # BN+Relu
        self.sa.add_module('bn_reduce1', nn.BatchNorm2d(channel // reduction))
        self.sa.add_module('relu_reduce1', nn.ReLU())

        for i in range(num_layers):
            self.sa.add_module('conv_%d' % i, nn.Conv2d(kernel_size=3,
                                                        in_channels=channel // reduction,
                                                        out_channels=channel // reduction,
                                                        padding=dia_val,    # 加上padding
                                                        dilation=dia_val))  # 膨胀卷积
            self.sa.add_module('bn_%d' % i, nn.BatchNorm2d(channel // reduction))
            self.sa.add_module('relu_%d' % i, nn.ReLU())
        self.sa.add_module('last_conv', nn.Conv2d(channel // reduction, 1, kernel_size=1))

    def forward(self, x):
        res = self.sa(x)
        res = res.expand_as(x)
        return res

class BAMBlock(nn.Module):

    def __init__(self, channel=256,reduction=16,dia_val=2):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(channel=channel,reduction=reduction,dia_val=dia_val)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        if b==1:  # 如果batchsize=1，ca里的bn会报错
            return x

        ca_out=self.ca(x)
        sa_out = self.sa(x)
        weight = self.sigmoid(sa_out * ca_out)
        out=(1+weight)*x
        return out


## 先验融合模块
class Prior_Attention(BaseModule):
    def __init__(self,
                 k_size=5,
                 levels=5,
                 prior_r=None,
                 prior_w=None,
                 channel=256,
                 reduction=16,
                 ):
        super(Prior_Attention, self).__init__()
        self.levels = levels
        self.prior_r = prior_r
        self.prior_w = prior_w
        self.fus_conv = ConvModule(2, 1, kernel_size=k_size, padding=k_size // 2,
                                   conv_cfg=None,
                                   norm_cfg=None,
                                   act_cfg=None,
                                   inplace=False)
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.bam=BAMBlock(channel=channel, reduction=reduction,dia_val=2)

    def forward(self, Inputs):
        '''
        Args:
            Inputs:multi features after ResNet
            Outs: fusion features
        '''
        Outs = []

        for level in range(self.levels):
            # 对每层特征使用先验进行注意力汇聚
            input = Inputs[level]
            pr_r = self.prior_r[level]  # road prior
            pr_w = self.prior_w[level]  # water prior
            # 求原特征图的max and mean
            max_fea, _ = torch.max(input, dim=1, keepdim=True)
            mean_fea = torch.mean(input, dim=1, keepdim=True)

            ## 点乘max_fea，将先验概率作为max feature的权重
            # 创建road、water的ones,分开运算
            device=get_device()
            pr_r_pos=pr_r
            r_ones=torch.ones(pr_r.shape).to(device)
            # max_fea_r = torch.mul(max_fea, pr_r) # 对道路 求max*(mask)的值 加大对道路区域关注

            w_ones=torch.ones(pr_w.shape).to(device)
            pr_w_neg = w_ones - pr_w     
            pr_w_pos=pr_w        
            # max_fea_w = torch.mul(mean_fea, pr_w_neg)# 对水体 求mean*(1-mask)的值 减少对水体区域关注

            fusion_r=torch.mul(input,pr_r_pos)  # 乘以(road)
            # fusion_w=torch.mul(input,pr_w_neg)  # 乘以(1-water)
            fusion_w=torch.mul(input,pr_w_pos)  # 乘以(water)
            lamda1=torch.tensor(0.8).to(device)
            lamda2 = torch.tensor(0.2).to(device)
            fusion_input = lamda1*input + lamda2*fusion_r + lamda2*fusion_w

            # ## fusion feature
            # # spatial fusion
            # # fus_fea = torch.cat([mean_fea, max_fea_r, max_fea_w], dim=1) # 融合
            # fus_fea = torch.cat([mean_fea, max_fea_w], dim=1) # 融合
            # fus_out = self.fus_conv(fus_fea)
            # fus_weight = torch.sigmoid(fus_out)  # activate
            # # fus_out = input * (fus_weight)  # 融合空间特征
            # fus_out = input * (fus_weight + 1)  # 融合空间特征

            # # channel attention
            # # ca_out = self.ca(fus_out)
            # # ca_fea = torch.sigmoid(ca_out)  # activate
            # # out = fus_out * (ca_fea + 1)  # 融合通道特征(BAM中的)
            # out = fus_out

            # BAM module 
            # out=self.bam(fusion_input)

            # output
            Outs.append(fusion_input)

        return Outs

# ## 只用到BAM
# class Prior_Attention(BaseModule):
#     def __init__(self,
#                  k_size=7,
#                  levels=5,
#                  prior_r=None,
#                  prior_w=None,
#                  channel=256,
#                  reduction=16,
#                  ):
#         super(Prior_Attention, self).__init__()
#         self.levels=levels
#         self.prior_r=prior_r
#         self.prior_w=prior_w
#         self.fus_conv = ConvModule(3,1,kernel_size=k_size,padding=k_size // 2,
#             conv_cfg=None,
#             norm_cfg=None,
#             act_cfg=None,
#             inplace=False)
#         self.ca = ChannelAttention(channel=channel, reduction=reduction)
#         self.bam=BAMBlock(channel=channel, reduction=reduction,dia_val=2)


#     def forward(self,Inputs):
#         '''
#         Args:
#             Inputs:multi features after ResNet
#             Outs: fusion features
#         '''
#         Outs=[]

#         for level in range(self.levels):
#             # 对每层特征使用BAM注意力模块
#             input=Inputs[level]
#             # BAM
#             out=self.bam(input)
#             # output
#             Outs.append(out)
#         return Outs





@ROTATED_NECKS.register_module()
class Fusion_FPN(BaseModule):
    r"""Fusion Feature Pyramid Network.


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
        super(Fusion_FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels

        print('----Fusion FPN测试代码------,fpn的out_channel为 ',out_channels)

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
                in_channels[i],  # 第[i]个特征图的输入通道,faster-rcnn为[256,512,1024,2048]
                out_channels,    # 256
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

        ## Mask downsample Module
        # from (800,800)

        ## Mask convolution Module
        self.mask_convs=nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            mask_conv = ConvModule(
                1,  # 单通道
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.mask_convs.append(mask_conv)


    # mask down sample
    def down_sample(self, x, size):
        f_mask=[]
        for i in range(self.num_outs):
            down_single=F.interpolate(x, size[i], mode='bilinear', align_corners=True)
            f_mask.append(down_single)
        return f_mask


    @auto_fp16()
    def forward(self, features_with_mask):
        """Forward function.

        Args:
            features: Two kind of features after backbone: img features and mask features,contains feature and mask
              feature: feature after Backbone+Neck (Resnet+FPN), multi scale,Tuple(5)
              mask: Tensor (batchsize,1,1024,1024)

        """
        # 这里的features，如果backbone为resnet，那么内容为4个tuple
        # features:(1,256,256,256)(1,512,128,128)(1,1024,64,64)(1,2048,32,32)

        # # (1) features and mask
        f_mask = []
        f_img = features_with_mask
        with_mask=False
        # print('len(features_with_mask)', len(features_with_mask))

        if len(features_with_mask)!= len(self.in_channels): # inchannels=4
            f_img = features_with_mask[0]
            mask_road = features_with_mask[1]
            mask_water = features_with_mask[2]
            with_mask = True
        else:
            f_img = features_with_mask
            with_mask = False

        # print('len(f_img)',len(f_img))
        # if len(f_img)==1:
        #     tmpp=0
        #     tmp+=1
        # print('len(self.in_channels)', len(self.in_channels))
        # print('shape f_img0',f_img[0].shape)

        assert len(f_img) == len(self.in_channels)

        # # (2) FPN laterals and build outputs
        # build laterals
        laterals = [
            lateral_conv(f_img[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
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
                    laterals[i], **self.upsample_cfg)  #上采样+融合相加操作
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
                    extra_source = f_img[self.backbone_end_level - 1]
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

        if with_mask:
            # # (3) 融合mask和feature
            ## 1)首先归一化至(0,1),得到概率图
            mask_road = mask_road / torch.tensor(255.)
            mask_water = mask_water / torch.tensor(255.)
            # 输入的prior mask大小为（b,1,h,w),而f_img为4个level feature maps
            sam_size = [(out.shape[2], out.shape[3]) for out in outs]
            ## 2)down sample to multi levels
            f_road = self.down_sample(mask_road, sam_size)
            f_water = self.down_sample(mask_water, sam_size)

            ## 3)进入融合网络
            ## fusion by attention
            device=get_device()
            fu = Prior_Attention(k_size=5,
                                 levels=self.num_outs,
                                 prior_r=f_road,
                                 prior_w=f_water).to(device)
            fusion_outs=fu(outs)
            outs = fusion_outs
        

        # ## BAM module
        # device=get_device()
        # BAM=BAMBlock(channel=256, reduction=16,dia_val=2).to(device)
        
        # final_outs=[]
        # for level in range(len(outs)):
        #     bam_out=BAM(outs[level])
        #     final_outs.append(bam_out)        

        return tuple(outs)
