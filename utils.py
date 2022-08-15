import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F


# 由于数据进行了归一化
# 以0.5作为分界线 大于0.5的为前景 小于0.5的为背景
def calculate_iou(predict, mask):
    predict = torch.sigmoid(predict).cpu().data.numpy()
    mask = torch.sigmoid(mask).cpu().data.numpy()
    predict_ = predict > 0.5
    _predict = predict <= 0.5
    mask_ = mask > 0.5
    _mask = mask <= 0.5
    # 进行与或操作 获取交并集
    intersection = (predict_ & mask_).sum()
    union = (predict_ | mask_).sum()
    _intersection = (_predict & _mask).sum()
    _union = (_predict | _mask).sum()
    if union < 1e-5 or _union < 1e-5:
        return 0
    miou = (intersection / union) * 0.5 + 0.5 * (_intersection / _union)
    return miou


# 通过腐蚀操作获取边界
def mask_to_boundary(mask, dilation_ratio=0.02, sign=1):
    # 通过sign判断 来讲mask数值置为 0、1
    if sign == 1:
        mask = torch.sigmoid(mask).data.cpu().numpy()
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = mask.astype('uint8')
    elif sign == 0:
        mask = mask.cpu()
        mask = np.array(mask).astype('uint8')
        # mask = torch.tensor(mask).data.cpu.numpy().astype('uint8')

    b, h, w = mask.shape
    new_mask = np.zeros([b, h + 2, w + 2])
    mask_erode = np.zeros([b, h, w])
    img_diag = np.sqrt(h ** 2 + w ** 2)  # 计算图像对角线长度
    # 计算腐蚀的程度dilation
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    # 对一个batch中所有进行腐蚀操作
    for i in range(b):
        new_mask[i] = cv2.copyMakeBorder(mask[i], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  # 用0填充边框

    kernel = np.ones((3, 3), dtype=np.uint8)
    for j in range(b):
        # 腐蚀操作
        new_mask_erode = cv2.erode(new_mask[j], kernel, iterations=dilation)
        # 回填
        mask_erode[j] = new_mask_erode[1: h + 1, 1: w + 1]

    return mask - mask_erode


# 获取标签和预测的边界iou
def boundary_iou(gt, dt, dilation_ratio=0.02):
    dt_boundary = mask_to_boundary(dt, dilation_ratio, sign=1)
    gt_boundary = mask_to_boundary(gt, dilation_ratio, sign=0)
    B, H, W = dt_boundary.shape
    intersection = 0
    union = 0
    # 计算交并比
    for k in range(B):
        intersection += ((gt_boundary[k] * dt_boundary[k]) > 0).sum()
        union += ((gt_boundary[k] + dt_boundary[k]) > 0).sum()
    if union < 1:
        return 0
    boundary_iou = intersection / union

    return boundary_iou


# 损失函数
def bce_dice_loss(predict_batch, mask_batch):
    smooth = 1e-5
    # bce loss
    squeeze_predict_batch = predict_batch.squeeze(dim=1)
    bce = F.binary_cross_entropy_with_logits(squeeze_predict_batch, mask_batch)
    # dice loss
    torch_predict = torch.sigmoid(squeeze_predict_batch)
    num = mask_batch.size(0)
    torch_predict = torch_predict.view(num, -1)  # torch展平
    mask_batch = mask_batch.view(num, -1)  # torch展平
    intersection = (torch_predict * mask_batch)
    dice = (2. * intersection.sum(1) + smooth) / (torch_predict.sum(1) + mask_batch.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    return 0.5 * bce + dice
