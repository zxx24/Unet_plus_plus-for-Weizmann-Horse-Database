# 导入相应库
import torch
import numpy as np
from net import Unet_plus_plus
from Dataset import HorseDataset
from torchvision import transforms
from my_augment import Transform_Compose, Train_Transform, Totensor, Test_Transform
from utils import mask_to_boundary
from PIL import Image

# 判断GPU是否存在
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 划分训练集and 测试集
idx = np.arange(327)
testing_idx = idx[278:327]

# 图像数据预处理  and 数据增强
train_transforms = Transform_Compose([Train_Transform(image_size=80), Totensor()])
test_transforms = Transform_Compose([Test_Transform(image_size=80), Totensor()])

root = './horse/archive/weizmann_horse_db'  # 默认设置
with open('setting.txt', 'r') as f:
    lines = f.readlines()
    f.close()

for line in lines:
    if "root" in line:
        line = line.rstrip("\n")  # 去掉末尾\n
        line_split = line.split(' ')
        root = line_split[2]

# 载入数据
print("-" * 30)
print("载入数据...")
# 载入测试数据
Test_data = HorseDataset(root, testing_idx, test_transforms)

test_data_loader = torch.utils.data.DataLoader(
    Test_data,
    batch_size=8,
    num_workers=0,
    shuffle=True,
    pin_memory=True,
    drop_last=True
)

if __name__ == '__main__':
    print("-" * 34)
    print("demo--strating")
    best_model = Unet_plus_plus(deep_supervision=True, cut=True)  # 默认在CPU上
    state_dict = torch.load('best_model.pth', map_location=device)
    best_model.load_state_dict(state_dict, strict=False)

    toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值

    for test_data, test_mask in test_data_loader:
        B, H, W = test_mask.shape
        pic1 = toPIL(test_data[0])
        pic2 = toPIL(test_mask[0])
        print("图像依次为原图像，mask标签,mask边界，预测分割图像和预测边界")
        pic1.show()
        pic2.show()
        outputs = best_model(test_data)
        # 获得边界
        mask_boundary = 255 * mask_to_boundary(test_mask, dilation_ratio=0.02, sign=1)

        print("采用深监督")
        out_boundary = 255 * mask_to_boundary(outputs[-1].squeeze(dim=1), dilation_ratio=0.02, sign=1)
        out = outputs[-1].squeeze(dim=1)

        Save_out = torch.sigmoid(out).data.cpu().numpy()
        Save_out[Save_out > 0.5] = 255
        Save_out[Save_out <= 0.5] = 0

        test_mask_ = torch.sigmoid(test_mask).data.cpu().numpy()
        test_mask_[test_mask_ > 0.5] = 255
        test_mask_[test_mask_ <= 0.5] = 0

        A = Image.fromarray(mask_boundary[0].astype('uint8'))
        B = Image.fromarray(out_boundary[0].astype('uint8'))
        A.show()

        X = Save_out[0].astype('uint8')
        Z = Image.fromarray(X)
        Z.show()
        B.show()
        print("预测结束")
        print("-" * 34)
        break
