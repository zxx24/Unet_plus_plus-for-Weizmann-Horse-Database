# 导入相应库
import numpy as np
import os
import random
from net import Unet_plus_plus
from Dataset import HorseDataset
from torchvision import transforms
from my_augment import Transform_Compose, Train_Transform, Totensor, Test_Transform
from utils import calculate_iou, boundary_iou, bce_dice_loss, mask_to_boundary
from PIL import Image
import torch

# 判断GPU是否存在
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = 80  # 默认设置
with open('setting.txt', 'r') as f:
    lines = f.readlines()
    f.close()

for line in lines:

    if "root" in line:
        line = line.rstrip("\n")  # 去掉末尾\n
        line_split = line.split(' ')
        img_path = line_split[2]

    if "deep_supervision" in line:
        line = line.rstrip("\n")  # 去掉末尾\n
        line_split = line.split(' ')
        deep_supervision = line_split[2]
        if deep_supervision == 'False':
            deep_supervision = False
        else:
            deep_supervision = True
    if "image_size" in line:
        line = line.rstrip("\n")  # 去掉末尾\n
        line_split = line.split(' ')
        image_size = line_split[2]
        image_size = int(image_size)
    if "total_num" in line:
        line = line.rstrip("\n")  # 去掉末尾\n
        line_split = line.split(' ')
        total_num = line_split[2]
        total_num = int(total_num)
    if "train_size" in line:
        line = line.rstrip("\n")  # 去掉末尾\n
        line_split = line.split(' ')
        train_size = line_split[2]
        train_size = int(train_size)
    if "test_size" in line:
        line = line.rstrip("\n")  # 去掉末尾\n
        line_split = line.split(' ')
        test_size = line_split[2]
        test_size = int(test_size)
    if "shuffle" in line:
        line = line.rstrip("\n")  # 去掉末尾\n
        line_split = line.split(' ')
        shuffle = line_split[2]
        if shuffle == 'False':
            shuffle = False
        else:
            shuffle = True

# 划分训练集and 测试集
idx = np.arange(total_num)
if shuffle:
    np.random.shuffle(idx)
    print("数据打乱")
else:
    print("数据未打乱")
training_idx = idx[:train_size]
testing_idx = idx[train_size:train_size + test_size]

# 图像数据预处理  and 数据增强
train_transforms = Transform_Compose([Train_Transform(image_size=image_size), Totensor()])
test_transforms = Transform_Compose([Test_Transform(image_size=image_size), Totensor()])

# 载入数据
print("-" * 30)
print("载入数据...")
# 分别载入训练数据和测试数据
Train_data = HorseDataset(img_path, training_idx, train_transforms)
Test_data = HorseDataset(img_path, testing_idx, test_transforms)
train_data_loader = torch.utils.data.DataLoader(
    Train_data,
    batch_size=8,
    num_workers=0,
    shuffle=True,
    pin_memory=True,
    drop_last=True
)

# 为了能够复现模型结果，将测试集的batch_size设置1,并不打乱数据集
test_data_loader = torch.utils.data.DataLoader(
    Test_data,
    batch_size=1,
    num_workers=0,
    shuffle=False,
    pin_memory=True,
    drop_last=True
)

# 定义想要测试的数据
data_loader = test_data_loader

print("-" * 34)
print("开始预测")
# 参数可自行设置
best_model = Unet_plus_plus(deep_supervision=deep_supervision, cut=False)  # 默认在CPU上
# 以参数形式载入
state_dict = torch.load('best_model.pth', map_location=device)
best_model.load_state_dict(state_dict, strict=False)
best_model = best_model.to(device=device)

# 直接载入完整模型
# best_model = torch.load('Unet_plus_plus.pth')  #可完整载入模型
# best_model.eval()

print("testing...")
best_model.eval()
IOU = []
B_IOU = []
LOSS = []

with torch.no_grad():
    for test_data, test_mask in data_loader:
        test_data = test_data.to(device)
        test_mask = test_mask.to(device)

        loss = 0
        if deep_supervision:
            outputs = best_model(test_data)
            for output in outputs:
                loss += bce_dice_loss(output, test_mask)
            loss /= len(outputs)
            iou = calculate_iou(outputs[-1].squeeze(dim=1), test_mask)
            b_iou = boundary_iou(test_mask, outputs[-1].squeeze(dim=1))

        else:
            output = best_model(test_data)
            loss += bce_dice_loss(output, test_mask)
            iou = calculate_iou(output.squeeze(dim=1), test_mask)
            b_iou = boundary_iou(test_mask, output.squeeze(dim=1))

        IOU.append(iou)
        B_IOU.append(b_iou)
        LOSS.append(loss)
mean_iou = sum(IOU) / len(IOU)
mean_b_iou = sum(B_IOU) / len(B_IOU)
mean_loss = sum(LOSS) / len(LOSS)
print("测试数据 mean iou", mean_iou)
print("测试数据 mean boundary_iou", mean_b_iou)
print("测试数据 mean loss", mean_loss)
