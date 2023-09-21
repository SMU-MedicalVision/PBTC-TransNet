"""
author: Li Chuanpu
date: 2022_11_11 10:01
"""

import numpy as np
import SimpleITK as sitk
import os
import random
import torch
import albumentations
from monai.transforms import Compose, SpatialPad, CenterScaleCrop, SpatialCrop, RandCropByPosNegLabel, \
    NormalizeIntensity, RandRotate, RandFlip, RandAffine, Resize, ToTensor

def NiiDataRead(path, as_type=np.float32):
    nii = sitk.ReadImage(path)
    spacing = nii.GetSpacing()  # [x,y,z]
    volumn = sitk.GetArrayFromImage(nii)  # [z,y,x]
    origin = nii.GetOrigin()
    direction = nii.GetDirection()

    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2]

    spacing_ = np.array([spacing_z, spacing_y, spacing_x])
    return volumn.astype(as_type), spacing_.astype(np.float32), origin, direction

def NiiDataWrite(save_path, volumn, spacing, origin, direction, as_type=np.float32):
    spacing = np.array(spacing)
    spacing = spacing.astype(np.float64)
    raw = sitk.GetImageFromArray(volumn[:, :, :].astype(as_type))
    spacing_ = (spacing[2], spacing[1], spacing[0])
    raw.SetSpacing(spacing_)
    raw.SetOrigin(origin)
    raw.SetDirection(direction)
    sitk.WriteImage(raw, save_path)

def get_pad_roi(z_min, z_max, z_pad, img_max, resize_max):

    if (z_pad & 1) == 0:  # 双数
        z_min -= z_pad // 2
        z_max += z_pad // 2
    else:  # 单数，随机前后多补一层
        z_pad1 = z_pad // 2
        z_pad2 = z_pad - z_pad1
        if bool(random.getrandbits(1)):  # 随机返回一个布尔值
            z_min -= z_pad1
            z_max += z_pad2
        else:
            z_min -= z_pad2
            z_max += z_pad1

    # 调整：
    if z_min < 0:  # 下边界超出
        z_min = 0
        if z_max + z_pad < img_max:
            z_max = z_max + z_pad
        else:
            z_max = img_max

    if z_max > img_max:  # 上边界超出
        z_max = img_max
        if resize_max < img_max:
            z_min = img_max - resize_max
        else:
            z_min = 0

    return z_min, z_max

def get_roi3D(img, mask, box):
    # get roi
    z, x, y = mask.nonzero()
    z_min, z_max = z.min(), z.max()
    x_min, y_min = x.min(), y.min()
    x_max, y_max = x.max(), y.max()
    x_range = x_max - x_min + 1
    y_range = y_max - y_min + 1
    z_range = z_max - z_min + 1

    # z 处理
    if z_range < box[0] // 2:  # 层数小于一半，多补到一半
        z_pad = box[0] // 2 - z_range
        z_min, z_max = get_pad_roi(z_min, z_max, z_pad, img_max=img.shape[0], resize_max=box[0])
    elif z_range < box[0]:  # 层数大于一半但小于总数，补充到总数大小
        z_pad = box[0] - z_range
        z_min, z_max = get_pad_roi(z_min, z_max, z_pad, img_max=img.shape[0], resize_max=box[0])

    # x 处理
    if x_range < box[1] // 2:  # 小于一半，多补到一半
        x_pad = box[1] // 2 - x_range
        x_min, x_max = get_pad_roi(x_min, x_max, x_pad, img_max=img.shape[1], resize_max=box[1])
    elif x_range < box[1]:  # 大于一半但小于总数，补充到总数大小
        x_pad = box[1] - x_range
        x_min, x_max = get_pad_roi(x_min, x_max, x_pad, img_max=img.shape[1], resize_max=box[1])

    # y 处理
    if y_range < box[2] // 2:  # 小于一半，多补到一半
        y_pad = box[1] // 2 - y_range
        y_min, y_max = get_pad_roi(y_min, y_max, y_pad, img_max=img.shape[2], resize_max=box[2])
    elif x_range < box[1]:  # 大于一半但小于总数，补充到总数大小
        y_pad = box[1] - y_range
        y_min, y_max = get_pad_roi(y_min, y_max, y_pad, img_max=img.shape[2], resize_max=box[2])

    roi = img[z_min: z_max + 1, x_min: x_max + 1, y_min: y_max + 1]
    roi = torch.from_numpy(roi)
    roi = torch.unsqueeze(roi, dim=0)

    return roi

def get_roi2D(img, mask, box):
    # get roi
    z, x, y = mask.nonzero()
    x_min, y_min = x.min(), y.min()
    x_max, y_max = x.max(), y.max()
    x_range = x_max - x_min + 1
    y_range = y_max - y_min + 1

    # x 处理
    if x_range < box[1] // 2:  # 小于一半，多补到一半
        x_pad = box[1] // 2 - x_range
        x_min, x_max = get_pad_roi(x_min, x_max, x_pad, img_max=img.shape[1], resize_max=box[1])
    elif x_range < box[1]:  # 大于一半但小于总数，补充到总数大小
        x_pad = box[1] - x_range
        x_min, x_max = get_pad_roi(x_min, x_max, x_pad, img_max=img.shape[1], resize_max=box[1])

    # y 处理
    if y_range < box[2] // 2:  # 小于一半，多补到一半
        y_pad = box[1] // 2 - y_range
        y_min, y_max = get_pad_roi(y_min, y_max, y_pad, img_max=img.shape[2], resize_max=box[2])
    elif x_range < box[1]:  # 大于一半但小于总数，补充到总数大小
        y_pad = box[1] - y_range
        y_min, y_max = get_pad_roi(y_min, y_max, y_pad, img_max=img.shape[2], resize_max=box[2])

    roi = img[0, x_min: x_max + 1, y_min: y_max + 1]

    return roi

def get_rois_and_mask(name, ct_box, t1_box, t2_box, x_box):

    modality_type = name.split('/')[-2]  # 读取模态情况
    transformsCT = Compose([
              Resize(spatial_size=ct_box),    # resize到对应尺寸
              NormalizeIntensity(),
              ToTensor()
        ])

    transformsT1 = Compose([
              Resize(spatial_size=t1_box),    # resize到对应尺寸
              NormalizeIntensity(),
              ToTensor()
        ])

    transformsT2 = Compose([
              Resize(spatial_size=t2_box),    # resize到对应尺寸
              NormalizeIntensity(),
              ToTensor()
        ])

    transformsX = albumentations.Resize(height=x_box[1], width=x_box[2], always_apply=True)
    normX = NormalizeIntensity()


    # 根据模态情况构建数据
    if modality_type == 'second_full_modality':
        mask = torch.tensor([True, True, True])  # 把T1和T2数据concat, 用一个编码器进行编码， 第一个bool表示CT，第二个bool表示mr，第三个bool表示x

        # CT
        ct_roi = np.load(os.path.join(name, 'CT_crop.npy'))
        ct_roi = torch.from_numpy(ct_roi)
        ct_roi = torch.unsqueeze(ct_roi, dim=0)
        if transformsCT is not None:
            ct_roi = transformsCT(ct_roi)
        ct_roi = torch.unsqueeze(ct_roi, dim=0)

        # T1
        t1_roi = np.load(os.path.join(name, 'T1_crop.npy'))
        t1_roi = torch.from_numpy(t1_roi)
        t1_roi = torch.unsqueeze(t1_roi, dim=0)
        if transformsT1 is not None:
            t1_roi = transformsT1(t1_roi)
        # t1_roi = torch.unsqueeze(t1_roi, dim=0)

        # T2
        t2_roi = np.load(os.path.join(name, 'T2_crop.npy'))
        t2_roi = torch.from_numpy(t2_roi)
        t2_roi = torch.unsqueeze(t2_roi, dim=0)
        if transformsT2 is not None:
            t2_roi = transformsT2(t2_roi)
        # t2_roi = torch.unsqueeze(t2_roi, dim=0)

        # X
        x_roi = np.load(os.path.join(name, 'X_crop.npy'))
        if transformsX is not None:
            x_roi = transformsX(image=x_roi)['image']
        x_roi = normX(x_roi)
        x_roi = torch.unsqueeze(x_roi, dim=0)
        x_roi = torch.unsqueeze(x_roi, dim=0)

        rois = {'ct': ct_roi, 'mr': torch.unsqueeze(torch.cat((t1_roi, t2_roi), dim=0), 0), 'x': x_roi}

    elif modality_type == 'second_only_ct_and_mr':  # 没有的模态直接补0
        mask = torch.tensor([True, True, False])

        # CT
        ct_roi = np.load(os.path.join(name, 'CT_crop.npy'))
        ct_roi = torch.from_numpy(ct_roi)
        ct_roi = torch.unsqueeze(ct_roi, dim=0)
        if transformsCT is not None:
            ct_roi = transformsCT(ct_roi)
        ct_roi = torch.unsqueeze(ct_roi, dim=0)

        # T1
        t1_roi = np.load(os.path.join(name, 'T1_crop.npy'))
        t1_roi = torch.from_numpy(t1_roi)
        t1_roi = torch.unsqueeze(t1_roi, dim=0)
        if transformsT1 is not None:
            t1_roi = transformsT1(t1_roi)
        # t1_roi = torch.unsqueeze(t1_roi, dim=0)

        # T2
        t2_roi = np.load(os.path.join(name, 'T2_crop.npy'))
        t2_roi = torch.from_numpy(t2_roi)
        t2_roi = torch.unsqueeze(t2_roi, dim=0)
        if transformsT2 is not None:
            t2_roi = transformsT2(t2_roi)
        # t2_roi = torch.unsqueeze(t2_roi, dim=0)

        # X
        x_roi = torch.zeros(x_box)
        x_roi = torch.unsqueeze(x_roi, dim=0)

        rois = {'ct': ct_roi, 'mr': torch.unsqueeze(torch.cat((t1_roi, t2_roi), dim=0), 0), 'x': x_roi}

    elif modality_type == 'second_only_ct_and_x':  # 没有的模态直接补0
        mask = torch.tensor([True, False, True])

        # CT
        ct_roi = np.load(os.path.join(name, 'CT_crop.npy'))
        ct_roi = torch.from_numpy(ct_roi)
        ct_roi = torch.unsqueeze(ct_roi, dim=0)
        if transformsCT is not None:
            ct_roi = transformsCT(ct_roi)
        ct_roi = torch.unsqueeze(ct_roi, dim=0)

        # T1
        t1_roi = torch.unsqueeze(torch.zeros(t1_box), 0)
        # t1_roi = torch.unsqueeze(t1_roi, dim=0)

        # T2
        t2_roi = torch.unsqueeze(torch.zeros(t2_box), 0)
        # t2_roi = torch.unsqueeze(t2_roi, dim=0)

        # X
        x_roi = np.load(os.path.join(name, 'X_crop.npy'))
        if transformsX is not None:
            x_roi = transformsX(image=x_roi)['image']
        x_roi = normX(x_roi)
        x_roi = torch.unsqueeze(x_roi, dim=0)
        x_roi = torch.unsqueeze(x_roi, dim=0)

        rois = {'ct': ct_roi, 'mr': torch.unsqueeze(torch.cat((t1_roi, t2_roi), dim=0), 0), 'x': x_roi}

    elif modality_type == 'second_only_mr_and_x':  # 没有的模态直接补0
        mask = torch.tensor([False, True, True])

        # CT
        ct_roi = torch.unsqueeze(torch.zeros(ct_box), 0)
        ct_roi = torch.unsqueeze(ct_roi, dim=0)

        # T1
        t1_roi = np.load(os.path.join(name, 'T1_crop.npy'))
        t1_roi = torch.from_numpy(t1_roi)
        t1_roi = torch.unsqueeze(t1_roi, dim=0)
        if transformsT1 is not None:
            t1_roi = transformsT1(t1_roi)
        # t1_roi = torch.unsqueeze(t1_roi, dim=0)

        # T2
        t2_roi = np.load(os.path.join(name, 'T2_crop.npy'))
        t2_roi = torch.from_numpy(t2_roi)
        t2_roi = torch.unsqueeze(t2_roi, dim=0)
        if transformsT2 is not None:
            t2_roi = transformsT2(t2_roi)
        # t2_roi = torch.unsqueeze(t2_roi, dim=0)

        # X
        x_roi = np.load(os.path.join(name, 'X_crop.npy'))
        if transformsX is not None:
            x_roi = transformsX(image=x_roi)['image']
        x_roi = normX(x_roi)
        x_roi = torch.unsqueeze(x_roi, dim=0)
        x_roi = torch.unsqueeze(x_roi, dim=0)

        rois = {'ct': ct_roi, 'mr': torch.unsqueeze(torch.cat((t1_roi, t2_roi), dim=0), 0), 'x': x_roi}

    elif modality_type == 'second_only_ct':  # 没有的模态直接补0
        mask = torch.tensor([True, False, False])

        # CT
        ct_roi = np.load(os.path.join(name, 'CT_crop.npy'))
        ct_roi = torch.from_numpy(ct_roi)
        ct_roi = torch.unsqueeze(ct_roi, dim=0)
        if transformsCT is not None:
            ct_roi = transformsCT(ct_roi)
        ct_roi = torch.unsqueeze(ct_roi, dim=0)

        # T1
        t1_roi = torch.unsqueeze(torch.zeros(t1_box), 0)
        # t1_roi = torch.unsqueeze(t1_roi, dim=0)

        # T2
        t2_roi = torch.unsqueeze(torch.zeros(t2_box), 0)
        # t2_roi = torch.unsqueeze(t2_roi, dim=0)

        # X
        x_roi = torch.zeros(x_box)
        x_roi = torch.unsqueeze(x_roi, dim=0)

        rois = {'ct': ct_roi, 'mr': torch.unsqueeze(torch.cat((t1_roi, t2_roi), dim=0), 0), 'x': x_roi}

    elif modality_type == 'second_only_mr':  # 没有的模态直接补0
        mask = torch.tensor([False, True, False])

        # CT
        ct_roi = torch.unsqueeze(torch.zeros(ct_box), 0)
        ct_roi = torch.unsqueeze(ct_roi, dim=0)

        # T1
        t1_roi = np.load(os.path.join(name, 'T1_crop.npy'))
        t1_roi = torch.from_numpy(t1_roi)
        t1_roi = torch.unsqueeze(t1_roi, dim=0)
        if transformsT1 is not None:
            t1_roi = transformsT1(t1_roi)
        # t1_roi = torch.unsqueeze(t1_roi, dim=0)

        # T2
        t2_roi = np.load(os.path.join(name, 'T2_crop.npy'))
        t2_roi = torch.from_numpy(t2_roi)
        t2_roi = torch.unsqueeze(t2_roi, dim=0)
        if transformsT2 is not None:
            t2_roi = transformsT2(t2_roi)
        # t2_roi = torch.unsqueeze(t2_roi, dim=0)

        # X
        x_roi = torch.zeros(x_box)
        x_roi = torch.unsqueeze(x_roi, dim=0)

        rois = {'ct': ct_roi, 'mr': torch.unsqueeze(torch.cat((t1_roi, t2_roi), dim=0), 0), 'x': x_roi}

    elif modality_type == 'second_only_x':  # 没有的模态直接补0
        mask = torch.tensor([False, False, True])

        # CT
        ct_roi = torch.unsqueeze(torch.zeros(ct_box), 0)
        ct_roi = torch.unsqueeze(ct_roi, dim=0)

        # T1
        t1_roi = torch.unsqueeze(torch.zeros(t1_box), 0)
        # t1_roi = torch.unsqueeze(t1_roi, dim=0)

        # T2
        t2_roi = torch.unsqueeze(torch.zeros(t2_box), 0)
        # t2_roi = torch.unsqueeze(t2_roi, dim=0)

        # X
        x_roi = np.load(os.path.join(name, 'X_crop.npy'))
        if transformsX is not None:
            x_roi = transformsX(image=x_roi)['image']
        x_roi = normX(x_roi)
        x_roi = torch.unsqueeze(x_roi, dim=0)
        x_roi = torch.unsqueeze(x_roi, dim=0)

        rois = {'ct': ct_roi, 'mr': torch.unsqueeze(torch.cat((t1_roi, t2_roi), dim=0), 0), 'x': x_roi}

    else:
        raise ValueError('This modality type does not support!')

    return rois, mask











