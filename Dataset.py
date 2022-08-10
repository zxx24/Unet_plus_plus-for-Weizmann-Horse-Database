import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
import re
from my_augment import Transform_Compose, Train_Transform, Totensor, Test_Transform
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

# 此文件用于载入数据

# 数据载入
root = 'D:/Dataset/horse/archive/weizmann_horse_db'  # 默认设置
with open('setting.txt', 'r') as f:
    lines = f.readlines()
    f.close()

for line in lines:
    if "root" in line:
        line = line.rstrip("\n")  # 去掉末尾\n
        line_split = line.split(' ')
        root = line_split[2]


class HorseDataset(Dataset):
    def __init__(self, root: str, ID, transforms=None):
        super(HorseDataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self.ID = ID  # idx用于打乱样本集合 获取数据集中部分图片内容
        # 获取所有图像的文件名称，排序是为了保证样本图片和标记图片在列表中的位置一一对应
        self.imgs = list(np.array(list(sorted(os.listdir(os.path.join(root, "horse")))))[ID])
        self.masks = list(np.array(list(sorted(os.listdir(os.path.join(root, "mask")))))[ID])

    def __getitem__(self, idx):
        # 载入图片
        img_path = self.root + '/' + 'horse' + '/' + self.imgs[idx]
        mask_path = self.root + '/' + 'mask' + '/' + self.masks[idx]

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)

# 如下为测试时使用
# if __name__ == '__main__':
# idx = np.arange(327)
# np.random.shuffle(idx)
# training_idx = idx[:278]
# testing_idx = idx[278:]
#
# train_transforms = Transform_Compose([Train_Transform(image_size=100),Totensor()])
# test_transforms = Transform_Compose([Test_Transform(image_size=100),Totensor()])
# train_data = HorseDataset(root,training_idx,train_transforms)
# test_data =  HorseDataset(root,testing_idx,test_transforms)
#
# X,mask = train_data[100]
# show = ToPILImage()  # 可以把Tensor转成Image，方便可视化
# show(X).show()
# mask = np.array(mask)
# plt.imshow(mask, cmap="gray")
# plt.axis('off')
# plt.show()
