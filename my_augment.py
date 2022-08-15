from PIL import Image
import cv2
import torch
import numpy as np
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
from operator import methodcaller
from torchvision.transforms import ToPILImage


# 此文件为数据增强文件
# 在Augment_image类中定义了数种常见的数据增强增强方法，每次从中随机选用一种（含仅方式方式）
# 来处理图像数据

# 在此类中定义增强手段 当需要填充时，默认用0填充
# 因为0在图像中为黑色 在标签中代表背景
class Augment_image:
    def __init__(self, image_szie):
        self.image_size = image_szie

    # 翻转变换
    def flip(self, img, mask):
        if random.random() >= 0.5:
            img = transforms.RandomHorizontalFlip(1)(img)
            mask = transforms.RandomHorizontalFlip(1)(mask)
        elif random.random() < 0.5:
            img = transforms.RandomVerticalFlip(1)(img)
            mask = transforms.RandomVerticalFlip(1)(mask)

        img = transforms.Resize((self.image_size, self.image_size), interpolation=Image.BILINEAR)(img)
        mask = transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)(mask)
        return img, mask

    # 随机旋转  1~360°随机旋转
    def rotate(self, img, mask):
        random_angle = transforms.RandomRotation.get_params([-30, 30])
        img = TF.rotate(img, random_angle, fill=0)
        mask = TF.rotate(mask, random_angle, fill=0)

        img = transforms.Resize((self.image_size, self.image_size), interpolation=Image.BILINEAR)(img)
        mask = transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)(mask)
        return img, mask

    # 随机缩放 放缩范围为 0.6~1  w/H 比例为1：1
    def random_resize_crop(self, img, mask, scale=(0.6, 1.0), ratio=(1, 1)):
        top, left, height, width = transforms.RandomResizedCrop.get_params(img, scale=scale, ratio=ratio)
        img = TF.resized_crop(img, top, left, height, width, (self.image_size, self.image_size),
                              interpolation=Image.BILINEAR)
        mask = TF.resized_crop(mask, top, left, height, width, (self.image_size, self.image_size),
                               interpolation=Image.NEAREST)
        return img, mask

        # 中心裁剪  随机范围为0.6~1.0

    def centercrop(self, img, mask):
        size = np.random.randint(int(0.6 * self.image_size), self.image_size)
        img = TF.center_crop(img, size)
        mask = TF.center_crop(mask, size)
        img = transforms.Resize((self.image_size, self.image_size), interpolation=Image.BILINEAR)(img)
        mask = transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)(mask)
        return img, mask

    # 随机仿射
    def affine(self, img, mask):
        if random.random() >= 0.5:
            # 透视变换
            width, height = img.size
            startpoints, endpoints = transforms.RandomPerspective.get_params(width, height, 0.5)
            # 0值填充，原始图像大小
            img = TF.perspective(img, startpoints, endpoints, interpolation=Image.BICUBIC, fill=0)
            mask = TF.perspective(mask, startpoints, endpoints, interpolation=Image.NEAREST, fill=0)
        elif random.random() < 0.5:
            # 随机旋转-平移-缩放-错切 4种仿射变换 pytorch实现的是保持中心不变 不错切
            ret = transforms.RandomAffine.get_params(degrees=(-30, 30), translate=(0.3, 0.3), scale_ranges=(0.6, 1.4),
                                                     shears=None, img_size=img.size)
            # 0值填充，原始图像大小
            img = TF.affine(img, *ret, resample=0, fillcolor=0)
            mask = TF.affine(mask, *ret, resample=0, fillcolor=0)

        # 将图像处理成要求的大小
        img = TF.resize(img, (self.image_size, self.image_size), interpolation=Image.BILINEAR)
        mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=Image.NEAREST)
        return img, mask

    # 仅作Resize
    def resize(self, img, mask):
        img = transforms.Resize((self.image_size, self.image_size), interpolation=Image.BILINEAR)(img)
        mask = transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)(mask)
        return img, mask

    # 随机颜色增强    参数依次为亮度的偏移幅度、对比度偏移幅度、饱和度、 色相偏移幅度    均为最大幅值
    def colof_jitter(self, img, mask, brightness=0.4, contrast=0.3, saturation=0.2, hue=0.2):
        img = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)(img)

        img = transforms.Resize((self.image_size, self.image_size), interpolation=Image.BILINEAR)(img)
        mask = transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)(mask)
        return img, mask

    # 随机噪声  noise_sigma噪声幅值
    def noise(self, img, mask, noise_sigma=1):
        input_h_w = img.size[::-1] + (1,)
        # 生成正态分布的噪声
        noise = np.uint8(np.random.randn(*input_h_w) * noise_sigma)
        # 由于是uint8所以不需要截断
        img = np.array(img) + noise
        img = Image.fromarray(img, "RGB")

        img = transforms.Resize((self.image_size, self.image_size), interpolation=Image.BILINEAR)(img)
        mask = transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)(mask)
        return img, mask

    # 随机模糊  用卷积核来模糊
    def blur(self, img, mask, kernel_size=(5, 5)):
        img = cv2.GaussianBlur(np.array(img), ksize=kernel_size, sigmaX=0)
        img = Image.fromarray(img, "RGB")

        img = transforms.Resize((self.image_size, self.image_size), interpolation=Image.BILINEAR)(img)
        mask = transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)(mask)
        return img, mask


# 从数据增强方法中随机挑选
class Train_Transform(object):
    def __init__(self, image_size):
        self.image_size = image_size
        self.Augment = Augment_image(image_size)

    def __call__(self, img, mask):
        # 如下为数据增强方法列表，可自行挑选或者增减
        # transform_list = [methodcaller('flip', img, mask), methodcaller('rotate', img, mask),
        #                   methodcaller('random_resize_crop', img, mask)
        #     , methodcaller('centercrop', img, mask), methodcaller('affine', img, mask)
        #     , methodcaller('resize', img, mask), methodcaller('colof_jitter', img, mask)
        #     , methodcaller('noise', img, mask), methodcaller('blur', img, mask)]

        transform_list = [methodcaller('random_resize_crop', img, mask), methodcaller('resize', img, mask)
            , methodcaller('blur', img, mask), methodcaller('rotate', img, mask)
            , methodcaller('colof_jitter', img, mask), methodcaller('affine', img, mask)
            , methodcaller('random_resize_crop', img, mask)]

        # transform_list = [methodcaller('resize', img, mask)]   #只做resize

        img, mask = random.choice(transform_list)(self.Augment)

        return img, mask


# 测试样本只采取放缩变换
class Test_Transform(object):
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, img, mask):
        img = transforms.Resize((self.image_size, self.image_size), interpolation=Image.BILINEAR)(img)
        mask = transforms.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)(mask)
        return img, mask


# 像素归一化
class Totensor(object):
    def __call__(self, img, mask):
        img = transforms.ToTensor()(img)
        mask = torch.from_numpy(np.array(mask))
        if not isinstance(mask, torch.LongTensor):
            mask = mask.float()
        return img, mask


# 同时对图像数据和MASK进行transform处理
class Transform_Compose(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, mask):
        for trans in self.transform:
            img, mask = trans(img, mask)
        return img, mask

# 如下为测试使用
# if __name__ == '__main__':
#     path = 'D:/Dataset/pic.png'
#     image = Image.open(path).convert('RGB')
#     Image._show(image)
#     label = np.ones([200, 100], dtype=np.uint8)
#     label = Image.fromarray(label)
#     train_transforms = Transform_Compose([Train_Transform(image_size=100),
#                                           Totensor()])
#     test_transforms = Transform_Compose([
#         Test_Transform(image_size=100),
#         Totensor()])
#     x,y=train_transforms(image,label)
#     show = ToPILImage()  # 可以把Tensor转成Image，方便可视化
#     show(x).show()
#     print(x.shape)
