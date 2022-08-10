import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

# 此文件用于定义网络
Basic_channel = 32  # 默认设置
with open('setting.txt', 'r') as f:
    lines = f.readlines()
    f.close()

for line in lines:
    if "basic_channel " in line:
        line = line.rstrip("\n")  # 去掉末尾\n
        line_split = line.split(' ')
        Basic_channel = int(line_split[2])  # 获取基本channel数

channels = [Basic_channel, Basic_channel * 2, Basic_channel * 4, Basic_channel * 8, Basic_channel * 16]


# 定义double vgg块 卷积-bn-relu-卷积-bn-relu 作相当于UNET++中的一个卷积模块部分
class Double_vgg(nn.Sequential):
    def __init__(self, input_channel, output_channel, middle_channel=None):
        if middle_channel == None:
            middle_channel = output_channel
        super(Double_vgg, self).__init__(
            nn.Conv2d(input_channel, middle_channel, padding=1, kernel_size=3),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel, output_channel, padding=1, kernel_size=3),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )


# 上采样过程 采用nn库中的上采样函数
class Up_sample(nn.Sequential):
    def __init__(self):
        super(Up_sample, self).__init__(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )


# 下采样过程 包含了一次Maxpool 和 一次Double vgg
class Down_sample(nn.Sequential):
    def __init__(self, input_channel, output_channel, middle_channel=None):
        if middle_channel is None:
            middle_channel = output_channel
        super(Down_sample, self).__init__(
            nn.MaxPool2d(2, stride=2),
            Double_vgg(input_channel, output_channel, middle_channel)
        )


# 跳跃连接过程 包含卷积过程 具体特征融合在Unet++网络中给定
class Skip_connection(nn.Sequential):
    def __init__(self, input_channel, output_channel, middle_channel=None):
        if middle_channel is None:
            middle_channel = output_channel
        super(Skip_connection, self).__init__(
            Double_vgg(input_channel, output_channel, middle_channel)
        )


# 网络主体架构
class Unet_plus_plus(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, deep_supervision=False, cut=False):
        super(Unet_plus_plus, self).__init__()
        self.input_channel = input_channel
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.cut = cut

        # 定义网络结构
        self.in_conv = Double_vgg(input_channel, channels[0])
        self.down1 = Down_sample(channels[0], channels[1])
        self.down2 = Down_sample(channels[1], channels[2])
        self.down3 = Down_sample(channels[2], channels[3])
        self.down4 = Down_sample(channels[3], channels[4])

        self.skip0_1 = Skip_connection(channels[0] + channels[1], channels[0])
        self.skip0_2 = Skip_connection(channels[0] * 2 + channels[1], channels[0])
        self.skip0_3 = Skip_connection(channels[0] * 3 + channels[1], channels[0])
        self.skip0_4 = Skip_connection(channels[0] * 4 + channels[1], channels[0])

        self.skip1_1 = Skip_connection(channels[1] + channels[2], channels[1])
        self.skip1_2 = Skip_connection(channels[1] * 2 + channels[2], channels[1])
        self.skip1_3 = Skip_connection(channels[1] * 3 + channels[2], channels[1])

        self.skip2_1 = Skip_connection(channels[2] + channels[3], channels[2])
        self.skip2_2 = Skip_connection(channels[2] * 2 + channels[3], channels[2])

        self.skip3_1 = Skip_connection(channels[3] + channels[4], channels[3])

        self.up1 = Up_sample()
        self.up2 = Up_sample()
        self.up3 = Up_sample()
        self.up4 = Up_sample()

        self.out_1 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.out_2 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.out_3 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.out_4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, input):
        # 第一条链
        # print('input:', input.shape)
        x_0 = self.in_conv(input)
        # print('x_0', x_0.shape)
        x_1 = self.down1(x_0)
        # print('x_1', x_1.shape)
        x_2 = self.down2(x_1)
        # print('x_2', x_2.shape)
        x_3 = self.down3(x_2)
        # print('x_3', x_3.shape)
        if not self.cut:
            x_4 = self.down4(x_3)  # 剪枝则去除
            # print('x_4', x_4.shape)

        # 第二条链
        x_0_1 = self.skip0_1(torch.cat([x_0, self.up1(x_1)], 1))
        # print('x_0_1', x_0_1.shape)
        x_1_1 = self.skip1_1(torch.cat([x_1, self.up2(x_2)], 1))
        # print('x_1_1', x_1_1.shape)
        x_2_1 = self.skip2_1(torch.cat([x_2, self.up2(x_3)], 1))
        # print('x_2_1', x_2_1.shape)
        if not self.cut:
            x_3_1 = self.skip3_1(torch.cat([x_3, self.up2(x_4)], 1))
            # print('x_3_1', x_3_1.shape)

        # 第三条链
        x_0_2 = self.skip0_2(torch.cat([x_0, x_0_1, self.up1(x_1_1)], 1))
        # print('x_0_2', x_0_2.shape)
        x_1_2 = self.skip1_2(torch.cat([x_1, x_1_1, self.up1(x_2_1)], 1))
        # print('x_1_2', x_1_2.shape)
        if not self.cut:
            x_2_2 = self.skip2_2(torch.cat([x_2, x_2_1, self.up1(x_3_1)], 1))
            # print('x_2_2', x_2_2.shape)

        # 第四条链
        x_0_3 = self.skip0_3(torch.cat([x_0, x_0_1, x_0_2, self.up1(x_1_2)], 1))
        # print('x_0_3', x_0_3.shape)
        if not self.cut:
            x_1_3 = self.skip1_3(torch.cat([x_1, x_1_1, x_1_2, self.up1(x_2_2)], 1))
            # print('x_1_3', x_1_3.shape)

        # 最后一个连接块
        if not self.cut:
            x_0_4 = self.skip0_4(torch.cat([x_0, x_0_1, x_0_2, x_0_3, self.up1(x_1_3)], 1))
            # print('x_0_4', x_0_4.shape)

        # 输出结果根据是否深监督取定
        if self.deep_supervision == True:
            output1 = self.out_1(x_0_1)
            output2 = self.out_2(x_0_2)
            output3 = self.out_3(x_0_3)
            if not self.cut:
                output4 = self.out_4(x_0_4)
                return [output1, output2, output3, output4]
            if self.cut:
                return [output1, output2, output3]
        else:
            if not self.cut:
                output = self.out_4(x_0_4)
            else:
                output = self.out_3(x_0_3)
            # print('output', output.shape)
            return output

# 如下为测试使用
# if __name__ == '__main__':
#     print("hi")
#     model = Unet_plus_plus(input_channel=3, num_classes=2)
#     model.train()
#     x = torch.randn(1,3,480,480)
#     y = model(x)
#     print(y)
