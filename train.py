# 导入相应库
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from net import Unet_plus_plus
import torch.optim as optim
from torch.optim import lr_scheduler
from Dataset import HorseDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from my_augment import Transform_Compose, Train_Transform, Totensor, Test_Transform
from utils import calculate_iou, boundary_iou, bce_dice_loss, mask_to_boundary
from PIL import Image
import time

# 此文件为训练文件 参数已经配置好，如果您要自行训练，请在下列root中修改保存图片的路径,并且修改new_model属性为1
# 训练过程默认采用CPU 如果您要修改 还请自行修改model等置于GPU运算
# 此Unet++ 默认采用了深监督 您可自行去除 并采用全部网络结构用于预测 如果你想加快预测速度可将cut置为true 可以大幅提升预测速率
# 如果您仅仅想查看我的模型训练效果，可以将主函数中的main()注释掉，后运行即可，列表中save_num可以控制保存数目
# 如果您要新建模型并训练建议把学习率增大至1e-3 或者更换其他lr_schedule 微调模型则相应降低学习率
# best_model.pth为训练所得模型参数 Unet_plus_plus.pth为完整模型

# 参数列表   以下部分数据 1代表 是 ;0代表 否
# ！！！！！！！！！！！！！！！！！！！！！！！！！
# 有很多参数您可以默认使用 我把一些最关键的参数放在了参数列表的开头并加以标识
# ！！！！！！！！！！！！！！！！！！！！！！！！！

parser = argparse.ArgumentParser()
# ---------------------------------------------------------------------------
# 指向数据集储存位置
parser.add_argument('--root', default='D:/Dataset/horse/archive/weizmann_horse_db', help='folder to Dataset')
# epoch次数
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
# 是否新建模型    1-新建    0-使用历史最佳pth
parser.add_argument('--new_model', type=int, help='new model whether or not', default=0)
# 是否进行深监督
parser.add_argument('--deep_supervision', default=True)
# 是否进行减枝    剪枝操作只会对预测过程有用 设为True大幅度提升预测效率
parser.add_argument('--cut', default=False)
# 学习率
parser.add_argument('--lr', type=float, default=1e-4, help='the learning rate')
# 最小学习率
parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate')
# 保存模型条件    达到此要求会保存一个模型参数 best_biou.pth
parser.add_argument('--early_stop_b_iou', type=float, default=0.69, help='the minimal boundary iou')
# 保存模型条件    达到此要求会保存一个模型参数 best_iou.pth
parser.add_argument('--early_stop_iou', type=float, default=0.90, help='the minimal iou')
# 退出训练条件之一 达到此要求代表有机会可以停止训练
parser.add_argument('--signal_iou', type=float, default=0.90, help='the signal iou')
# ---------------------------------------------------------------------------
# 是否打乱数据集
parser.add_argument('--shuffle', type=bool, default=False, help='shuffle or not')
# 使用多少个子进程导入数据,0代表无子进程，1代表共两个进程，以此类推
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
# 是否使用gpu训练
parser.add_argument('--cuda', action='store_true', help='enables cuda')
# 网络基本通道数
parser.add_argument('--basic_channel', type=int, default=32, help='Batch size')
# batch大小
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
# 数据集个数
parser.add_argument('--total_num', type=int, default=327, help='Number of total images')
# 训练集样本个数   85%
parser.add_argument('--train_size', type=int, default=278, help='Number of training images')
# 测试集样本个数   15%
parser.add_argument('--test_size', type=int, default=49, help='Number of test images')
# image_size
parser.add_argument('--image_size', type=int, default=80, help='the height / width of the input image to network')
# 保存预测图片的batch数
parser.add_argument('--save_num', type=int, default=49, help='Number of predict images')
# SGD参数
parser.add_argument('--nesterov', type=bool, default=True, help='the momentum')
# 学习率衰减 及 参数控制 采用SGD时才有效
parser.add_argument('--momentum', type=float, default=0.9, help='the momentum')
# 学习率衰减 及 参数控制
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
# 早停标准
parser.add_argument('--early_stopping', default=10, type=int, help='early stopping')
# plot_all表示是否显示图片
parser.add_argument('--plot_all', type=int, default=0, help='1 == plot all images')
# 输出训练结果输出到目标目录
parser.add_argument('--outf', default='./logs', help='folder to train_log')
# 输出训练预测结果输出到目标目录
parser.add_argument('--predict', default='./predict', help='folder to predict')
# 设定随机种子
parser.add_argument('--manualSeed', type=int, default=11, help='manual seed')

opt = parser.parse_args()

argsDict = opt.__dict__
# 写入参数设置至setting.txt
with open('setting.txt', 'w') as f:
    f.writelines('------------------ start ------------------' + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ' : ' + str(value) + '\n')
    f.writelines('------------------- end -------------------')

# 构造输出目录
try:
    os.makedirs(opt.outf)
    os.makedirs(opt.predict)
except OSError:
    print("目录已存在或创建出错")

# 设置随机种子
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
# 载入种子
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
random.seed(opt.manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 载入参数
# 训练过程中参数赋值
workers = opt.workers
new_model = opt.new_model
shuffle = opt.shuffle
img_path = opt.root
save_num = opt.save_num
epoch_num = opt.epochs
batch_size = opt.batch_size
total_num = opt.total_num
early_stop_b_iou = opt.early_stop_b_iou
early_stop_iou = opt.early_stop_iou
signal_iou = opt.signal_iou
train_size = opt.train_size
test_size = opt.test_size
image_size = opt.image_size
learning_rate = opt.lr
min_learning_rate = opt.min_lr
# UNET++网络相关赋值
plot_all = opt.plot_all
deep_supervision = opt.deep_supervision
cut = opt.cut
weight_decay = opt.weight_decay
early_stopping = opt.early_stopping
momentum = opt.momentum
nesterov = opt.nesterov
# 训练集加测试集不能超过样本总数
assert train_size + test_size <= total_num, "Traing set size + Test set size > Total dataset size"

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
    batch_size=batch_size,
    num_workers=workers,
    shuffle=True,
    pin_memory=True,
    drop_last=True
)

# 为了能够复现模型结果，将测试集的batch_size设置1,并不打乱数据集
test_data_loader = torch.utils.data.DataLoader(
    Test_data,
    batch_size=1,
    num_workers=workers,
    shuffle=False,
    pin_memory=True,
    drop_last=True
)


# 训练过程
def train(model, data_loader, optimizer):
    print("training...")
    model.train()
    # 记录iou,boundary iou,loss
    IOU = []
    B_IOU = []
    LOSS = []
    num = 0
    # 一整个Epoch训练过程
    for image_batch, mask_batch in data_loader:
        num += 1
        loss = 0
        # 判断是否采用深监督  批次输入and输出
        if deep_supervision:
            outputs = model(image_batch)
            for output in outputs:
                loss += bce_dice_loss(output, mask_batch)
            loss /= len(outputs)
            iou = calculate_iou(outputs[-1].squeeze(dim=1), mask_batch)
            b_iou = boundary_iou(mask_batch, outputs[-1].squeeze(dim=1))
        else:
            output = model(image_batch)
            loss = bce_dice_loss(output, mask_batch)
            iou = calculate_iou(output.squeeze(dim=1), mask_batch)
            b_iou = boundary_iou(mask_batch, output.squeeze(dim=1))

        print(num, "iou:", iou, "boundary_iou:", b_iou, "loss:", loss.data)
        # 记录数据
        LOSS.append(loss)
        IOU.append(iou)
        B_IOU.append(b_iou)

        # 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_iou = sum(IOU) / len(IOU)
    mean_b_iou = sum(B_IOU) / len(B_IOU)
    mean_loss = sum(LOSS) / len(LOSS)
    print("训练集 mean iou", mean_iou)
    print("训练集 mean boundary_iou", mean_b_iou)
    print("训练集 mean loss", mean_loss)
    return mean_iou, mean_b_iou, mean_loss


# test类似于train 只是少了更新过程 不加以赘述
def test(model, test_data_loader):
    print("testing...")
    model.eval()
    IOU = []
    B_IOU = []
    LOSS = []

    with torch.no_grad():
        for test_data, test_mask in test_data_loader:

            loss = 0
            if deep_supervision:
                outputs = model(test_data)
                for output in outputs:
                    loss += bce_dice_loss(output, test_mask)
                loss /= len(outputs)
                iou = calculate_iou(outputs[-1].squeeze(dim=1), test_mask)
                b_iou = boundary_iou(test_mask, outputs[-1].squeeze(dim=1))

            else:
                output = model(test_data)
                loss += bce_dice_loss(output, test_mask)
                iou = calculate_iou(output.squeeze(dim=1), test_mask)
                b_iou = boundary_iou(test_mask, output.squeeze(dim=1))

            IOU.append(iou)
            B_IOU.append(b_iou)
            LOSS.append(loss)
    mean_iou = sum(IOU) / len(IOU)
    mean_b_iou = sum(B_IOU) / len(B_IOU)
    mean_loss = sum(LOSS) / len(LOSS)
    print("测试集 mean iou", mean_iou)
    print("测试集 mean boundary_iou", mean_b_iou)
    print("测试集 mean loss", mean_loss)
    return mean_iou, mean_b_iou, mean_loss


# 保存训练过程中得到数据图像
def save_fig(TRAIN_IOU, TRAIN_LOSS, TEST_IOU, TEST_LOSS, LR, EPOCH, TRAIN_B_IOU, TEST_B_IOU):
    plt.figure(figsize=(8, 8))
    plt.title('Mean Iou')
    plt.xlabel('EPOCH')
    plt.ylabel('Mean Iou')
    plt.plot(EPOCH, TRAIN_IOU, label='Train')
    plt.plot(EPOCH, TEST_IOU, label='Test')
    plt.legend(loc='upper left')
    if plot_all == 1:
        plt.show()
    else:
        plt.savefig(r"./%s/mean_iou.png" % (opt.outf))
    plt.clf()

    plt.figure(figsize=(8, 8))
    plt.title('Mean Boundary Iou')
    plt.xlabel('EPOCH')
    plt.ylabel('Mean Boundary Iou')
    plt.plot(EPOCH, TRAIN_B_IOU, label='Train')
    plt.plot(EPOCH, TEST_B_IOU, label='Test')
    plt.legend(loc='upper left')
    if plot_all == 1:
        plt.show()
    else:
        plt.savefig(r"./%s/mean_TEST_boundary_iou.png" % (opt.outf))
    plt.clf()

    plt.figure(figsize=(8, 8))
    plt.title('Mean Loss')
    plt.xlabel('EPOCH')
    plt.ylabel('Mean Loss')
    plt.plot(EPOCH, TRAIN_LOSS, label='Train')
    plt.plot(EPOCH, TEST_LOSS, label='Test')

    plt.legend(loc='upper left')
    if plot_all == 1:
        plt.show()
    else:
        plt.savefig(r"./%s/mean_loss.png" % (opt.outf))
    plt.clf()

    plt.figure(figsize=(8, 8))
    plt.title('learning rate')
    plt.xlabel('EPOCH')
    plt.ylabel('Learning rate')
    plt.plot(EPOCH, LR, label='lr')
    plt.legend(loc='upper left')
    if plot_all == 1:
        plt.show()
    else:
        plt.savefig(r"./%s/learning_rate.png" % (opt.outf))
    plt.clf()
    plt.close()


# 保存预测结果 包含语义分割预测结果，边界预测结果 和 实际图像分割标签和边界图像 以及 分割差值图像
def show_predict(test_data_loader, sign):
    print("-" * 34)
    print("开始预测")
    # 两种载入一种依据模型参数

    best_model = Unet_plus_plus(deep_supervision=deep_supervision, cut=cut)  # 默认在CPU上
    state_dict = torch.load('best_model.pth')
    best_model.load_state_dict(state_dict, strict=False)

    # 直接载入完整模型
    # best_model = torch.load('Unet_plus_plus.pth')  #可完整载入模型
    # best_model.eval()

    if sign == 1:
        if cut is True:
            print("减枝预测结果为:")
        elif cut is not True:
            print("完整结构预测结果为:")
        test_mean_iou, test_mean_b_iou, test_mean_loss = test(best_model, test_data_loader)
    cal = 0
    toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    with torch.no_grad():
        for test_data, test_mask in test_data_loader:
            cal += 1
            B, H, W = test_mask.shape
            for k in range(B):
                pic1 = toPIL(test_data[k])
                pic2 = toPIL(test_mask[k])

                pic1.save(r"./%s/%d_%d_ori.png" % (opt.predict, cal, k))
                pic2.save(r"./%s/%d_%d_mask.png" % (opt.predict, cal, k))

            outputs = best_model(test_data)
            # 获得边界
            mask_boundary = 255 * mask_to_boundary(test_mask, dilation_ratio=0.02, sign=1)
            if deep_supervision is True:
                # print("采用深监督")
                out_boundary = 255 * mask_to_boundary(outputs[-1].squeeze(dim=1), dilation_ratio=0.02, sign=1)
                out = outputs[-1].squeeze(dim=1)
            else:
                # print("无深监督")
                out_boundary = 255 * mask_to_boundary(outputs.squeeze(dim=1), dilation_ratio=0.02, sign=1)
                out = outputs.squeeze(dim=1)

            Save_out = torch.sigmoid(out).data.cpu().numpy()
            Save_out[Save_out > 0.5] = 255
            Save_out[Save_out <= 0.5] = 0

            test_mask_ = torch.sigmoid(test_mask).data.cpu().numpy()
            test_mask_[test_mask_ > 0.5] = 255
            test_mask_[test_mask_ <= 0.5] = 0
            for j in range(B):
                A = Image.fromarray(mask_boundary[j].astype('uint8'))
                B = Image.fromarray(out_boundary[j].astype('uint8'))
                A.save(r"./%s/%d_%d_mask_boundary.png" % (opt.predict, cal, j))
                B.save(r"./%s/%d_%d_predict_boundary.png" % (opt.predict, cal, j))
                Y = test_mask_[j].astype('uint8')
                X = Save_out[j].astype('uint8')
                sub = np.abs(X - Y)
                sub = Image.fromarray(sub)
                sub.save(r"./%s/%d_%d_sub.png" % (opt.predict, cal, j))
                Z = Image.fromarray(X)
                Z.save(r"./%s/%d_%d_predict.png" % (opt.predict, cal, j))
            if (cal == save_num):
                print("预测结束")
                print("-" * 34)
                break


def cal_time(data_loader):
    cut_model = Unet_plus_plus(deep_supervision=deep_supervision, cut=True)  # 默认在CPU上
    cut_state_dict = torch.load('best_model.pth')
    cut_model.load_state_dict(cut_state_dict, strict=False)
    cut_model.eval()

    time_start = time.time()  # 记录开始时间
    test(cut_model, data_loader)
    time_end = time.time()  # 记录结束时间
    time_gap = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print("剪枝时间间隔为", time_gap)

    not_cut_model = Unet_plus_plus(deep_supervision=deep_supervision, cut=False)  # 默认在CPU上
    not_cut_state_dict = torch.load('best_model.pth')
    not_cut_model.load_state_dict(not_cut_state_dict, strict=False)
    not_cut_model.eval()

    time_start = time.time()  # 记录开始时间
    test(not_cut_model, data_loader)
    time_end = time.time()  # 记录结束时间
    time_gap = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print("不剪枝时间间隔为", time_gap)


def main():
    # 载入模型 deep_supervision与否来构建模型  后剪枝只在预测过程中使用 对于创建模型不影响，默认为False
    print("载入Unet++模型...")
    if new_model == 1:
        print("新建模型")
        model = Unet_plus_plus(input_channel=3, num_classes=1, deep_supervision=deep_supervision, cut=False)  # 默认在cpu上

    elif new_model == 0:
        print("载入历史最佳模型")
        model = Unet_plus_plus(input_channel=3, num_classes=1, deep_supervision=deep_supervision, cut=False)
        state_dict = torch.load('best_model.pth')  # 载入参数
        model.load_state_dict(state_dict, strict=False)
    else:
        model = Unet_plus_plus(input_channel=3, num_classes=1, deep_supervision=deep_supervision, cut=False)

    # 定义需要更新的参数
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    # 定义优化器 经过实验Adam效果比较好
    print("定义优化器...")
    Optimizer = optim.Adam(params_to_optimize, lr=learning_rate, weight_decay=weight_decay)
    # Optimizer = optim.SGD(params_to_optimize, lr=learning_rate, momentum=momentum,nesterov=nesterov, weight_decay=weight_decay)

    # 定义学习率规划
    print("定义学习率规划...")
    # 采用模拟余弦退火规划学习率
    my_lr_scheduler = lr_scheduler.CosineAnnealingLR(Optimizer, T_max=epoch_num, eta_min=min_learning_rate)

    break_sign = 0  # 退出信号
    best_iou = early_stop_iou  # 用于记录最佳iou
    best_boundary_iou = early_stop_b_iou  # 用于记录最佳boundary iou

    # 用于记录
    TRAIN_ALL_IOU = []
    TRAIN_ALL_B_IOU = []
    TRAIN_ALL_LOSS = []
    TEST_ALL_LOSS = []
    TEST_ALL_IOU = []
    TEST_ALL_B_IOU = []
    LR = []
    EPOCH = []

    for epoch in range(epoch_num):
        print("_" * 15, epoch + 1, "_" * 15)
        print("learning_rate", Optimizer.state_dict()['param_groups'][0]['lr'])

        EPOCH.append(epoch + 1)
        LR.append(Optimizer.state_dict()['param_groups'][0]['lr'])
        # 训练过程
        train_mean_iou, train_mean_b_iou, train_mean_loss = train(model, train_data_loader, Optimizer)
        # 测试过程
        test_mean_iou, test_mean_b_iou, test_mean_loss = test(model, test_data_loader)
        # 记录数据
        TRAIN_ALL_IOU.append(train_mean_iou)
        TRAIN_ALL_B_IOU.append(train_mean_b_iou)
        TRAIN_ALL_LOSS.append(train_mean_loss)

        TEST_ALL_IOU.append(test_mean_iou)
        TEST_ALL_B_IOU.append(test_mean_b_iou)
        TEST_ALL_LOSS.append(test_mean_loss)

        # 保存最佳iou模型参数
        if test_mean_iou > best_iou:
            best_iou = test_mean_iou
            torch.save(model.state_dict(), 'best_model.pth')
            torch.save(model, 'Unet_plus_plus.pth')
            break_sign = 0
        # 保存最佳boundary iou模型参数
        if test_mean_b_iou > best_boundary_iou:
            best_boundary_iou = test_mean_b_iou
            torch.save(model.state_dict(), 'best_biou.pth')
            break_sign = 0

        break_sign += 1
        print("best_iou:", best_iou, "best boundary iou:", best_boundary_iou)
        print("-" * 34)

        # 停止训练条件
        if early_stopping > 0 and break_sign > early_stopping and best_iou > signal_iou:
            print("达到要求，停止训练")
            break

        # 更新学习率
        my_lr_scheduler.step()
        torch.cuda.empty_cache()

        # 输出记录到文件夹内
        if (epoch + 1) % 5 == 0:
            save_fig(TRAIN_ALL_IOU, TRAIN_ALL_LOSS, TEST_ALL_IOU, TEST_ALL_LOSS, LR, EPOCH, TRAIN_ALL_B_IOU,
                     TEST_ALL_B_IOU)
            show_predict(test_data_loader, 0)
        torch.cuda.empty_cache()

    save_fig(TRAIN_ALL_IOU, TRAIN_ALL_LOSS, TEST_ALL_IOU, TEST_ALL_LOSS, LR, EPOCH, TRAIN_ALL_B_IOU, TEST_ALL_B_IOU)
    show_predict(test_data_loader, 0)


if __name__ == '__main__':
    print("程序运行开始")
    print("-" * 33)

    # 包含训练、测试、预测等过程
    main()

    # 删除此注释并修改cut可以生成 剪枝或不剪枝 预测
    # show_predict(train_data_loader, 1)
    # show_predict(test_data_loader, 1)

    # 删除此注释可以用于测试时间
    # cal_time(test_data_loader)
