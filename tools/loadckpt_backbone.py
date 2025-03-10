import torch
import numpy as np
from collections import OrderedDict
import math
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'


### 完成合并后,打印合并后的权重来检查
# state_dict_merge= torch.load('merged_weights.pth')
# print(state_dict_merge['optimizer']['state'])
# print(state_dict_merge['optimizer']['param_groups'])
# for key in state_dict_merge:
#     print(key)
#     print('==-=-=-=-=-==-===-=')
#     for keyy in state_dict_merge[key]:
#         print(keyy)

########


state_dict = torch.load('/media/dell/DATA/WLL/RSSGG/mmrotate/REsize_R50_1024/epoch_12.pth')
state_dict_d2 = torch.load('/media/dell/DATA/WLL/RSSGG/mmrotate/REsize_R50/oriented_rcnn_r50_fpn_1x_rsg_le90-0b66f6a4.pth')
# state_dict_d4 = torch.load('down4_noloadfrom_epoch_20.pth')
# state_dict_d8 = torch.load('down8_noloadfrom_epoch_18.pth')

# tmp_ori= torch.load('down2_loadfrom_epoch_18.pth')
# state_dict_d0_l = torch.load('down0_finestscale16_epoch_18.pth')
# state_dict_d2_l = torch.load('down2_loadfrom_epoch_18.pth')
# state_dict_d4_l = torch.load('down4_loadfrom-down2-ep12_loadfrom_lr001_epoch_6.pth')
# state_dict_d8_l = torch.load('down8_loadfrom-down4-ep6-loadfrom-down2-ep12_loadfrom_lr001_epoch_4.pth')
# state_dict_d16_l = torch.load('down16_loadfrom-down8-ep4_loadfrom-down4-ep6-loadfrom-down2-ep12_loadfrom_lr001_epoch_2.pth')

#variables_dict_list = [state_dict_d0_l, state_dict_d2, state_dict_d4, state_dict_d8, state_dict_d8]



# variables_dict_list = [state_dict_d0_l, state_dict_d2_l, state_dict_d4_l, state_dict_d8_l, state_dict_d16_l]

variables_dict_list = [state_dict, state_dict_d2]

# 创建一个新的空字典，用于存放合并后的权重
merged_state_dict = {}

# 定义命名规则的前缀
prefix = "backbone_d"

# 遍历每个权重的字典，并修改键名后存入新字典
for idx, variable_dict in enumerate(variables_dict_list):
    name = str(0) if idx == 0 else str(int(2**idx))
    for key in variable_dict['state_dict']:
        # 使用 split 方法将键名按照 '.' 分割成多个部分
        key_tmp=key
        
        parts = key.split('.')
        # 修改第一个部分（前缀）为新的命名规则
        parts[0] = parts[0]+"_d"+name
        # 使用 join 方法将修改后的部分重新连接成新的键名
        new_key = '.'.join(parts)

        if idx==0:
            new_key=key
        # 将对应的权重值保存到新字典中
        merged_state_dict[new_key] = variable_dict['state_dict'][key]

# 将第一份权重中的['meta']和['optimizer']部分直接保存到新字典中
# if 'meta' in variables_dict_list[0]:
#     merged_state_dict['meta'] = variables_dict_list[0]['meta']
# if 'optimizer' in variables_dict_list[0]:
#     merged_state_dict['optimizer'] = variables_dict_list[0]['optimizer']

# 将新合并的字典保存为一个新的权重文件

torch.save({'state_dict': merged_state_dict, 'meta': variables_dict_list[0]['meta'],
             'optimizer':variables_dict_list[0]['optimizer']}, '/media/dell/DATA/WLL/RSSGG/mmrotate/REsize_R50/CGL12_1024.pth')

for key in merged_state_dict:
    print(key)