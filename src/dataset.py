#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import pywt


class random_image_enhance(object):
    def __init__(self, methods=['contrast', 'brightness', 'sharpness']):
        self.enhance_method = []
        if 'contrast' in methods:
            self.enhance_method.append(ImageEnhance.Contrast)
        if 'brightness' in methods:
            self.enhance_method.append(ImageEnhance.Brightness)
        if 'sharpness' in methods:
            self.enhance_method.append(ImageEnhance.Sharpness)

    def __call__(self, image, mask):
        #image = Image.fromarray(image)
        np.random.shuffle(self.enhance_method)

        for method in self.enhance_method:
            if np.random.random() > 0.5:
                enhancer = method(image)
                factor = float(1 + np.random.random() / 10)
                image = enhancer.enhance(factor)
        image = image

        return image, mask


class random_gaussian_blur(object):
    def __init__(self):
        pass

    def __call__(self, image, mask):
        image = image
        if np.random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=np.random.random()))
        image = image

        return image, mask

########################### Wavelet Enhancement ###########################
class WaveletEnhance(object):
    """
    小波域自适应高频增强。
    对图像每个通道做一级 DWT，对 LH/HL/HH 三个高频子带进行自适应拉伸，
    再通过 IDWT 重建图像，增强病变边缘和纹理细节。

    参数:
        wavelet  : 小波基，默认 'haar'（速度最快，边缘响应好）
        hf_boost : 高频子带增强倍数，[1.0, 2.0] 合理范围
        prob     : 执行增强的概率（训练时随机应用，避免过拟合）
    """
    def __init__(self, wavelet='haar', hf_boost=1.3, prob=0.5):
        self.wavelet  = wavelet
        self.hf_boost = hf_boost
        self.prob     = prob

    def _enhance_channel(self, ch):
        # ch: float32 numpy array [H, W], 值域任意
        cA, (cH, cV, cD) = pywt.dwt2(ch, self.wavelet)

        # 自适应拉伸：高频系数 × boost，但限制幅值防止溢出
        cH = cH * self.hf_boost
        cV = cV * self.hf_boost
        cD = cD * self.hf_boost

        # 重建
        reconstructed = pywt.idwt2((cA, (cH, cV, cD)), self.wavelet)

        # 裁剪到与输入相同尺寸（小波重建可能多1像素）
        reconstructed = reconstructed[:ch.shape[0], :ch.shape[1]]
        return reconstructed

    def __call__(self, image, mask):
        """
        image: float32 numpy array [H, W, 3]，值域 [0, 255]
        mask : 不修改，直接透传
        """
        if np.random.random() > self.prob:
            return image, mask

        enhanced = np.empty_like(image)
        for c in range(image.shape[2]):
            enhanced[:, :, c] = self._enhance_channel(image[:, :, c])

        # 裁切回 [0, 255]，保持 float32
        enhanced = np.clip(enhanced, 0.0, 255.0)
        return enhanced, mask


########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image, mask):
        image = image / 255.0 
        image = (image - self.mean)/self.std
        mask /= 255
        return image, mask


class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2)==0:
            return image[:,::-1,:], mask[:, ::-1]
        else:
            return image, mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask

class RandomCrop(object):
    """Random crop augmentation (placeholder, not used in training)"""
    def __init__(self, crop_size=1024):
        self.crop_size = crop_size
    
    def __call__(self, image, mask):
        # Simple placeholder, just return original
        return image, mask

class Transformation(object):

    def __init__(self, resize=1024):
        self.transforms = A.Compose([
            # 空间增强（超声合理项）
            A.HorizontalFlip(p=0.5),
            # VerticalFlip 已删除：超声有近场/远场方向性，上下翻转破坏物理含义
            # RandomRotate90 已删除：90°旋转与实际采集方向不符
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            # ElasticTransform 已删除：会歪曲病灶轮廓（诊断依据），且是NaN不稳定来源
            # 随机裁剪后 resize：缩小裁剪范围，避免大幅丢失病灶
            A.RandomResizedCrop(height=resize, width=resize, scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.4),
            # 颜色 / 纹理增强
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            # HueSaturationValue 已删除：超声为灰度图，色相/饱和度调整无效
            A.Blur(blur_limit=3, p=0.2),
            # CoarseDropout 已删除：fill_mask=0 会人工制造假标注，严重干扰分割学习
            A.LongestMaxSize(1024, p=1),
        ])

    def __call__(self, image, mask):
        try:
            result = self.transforms(image=image, mask=mask)
            return result["image"], result["mask"]
        except Exception:
            # 增强失败时返回原图，不中断训练
            return image, mask

class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        return image, mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[0.485, 0.456, 0.406]]])
        self.std    = np.array([[[0.229, 0.224, 0.225]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg        = cfg
        self.samples    = []
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        #self.randomflip = RandomFlip()
        input_size = cfg.kwargs.get('resize', 1024)
        self.resize     = Resize(input_size, input_size)
        self.randomcrop = RandomCrop()
        self.totensor   = ToTensor()
        self.guassian   = random_gaussian_blur()
        self.use_wavelet_aug = cfg.kwargs.get('use_wavelet_aug', True)
        self.wavelet_aug_basis = cfg.kwargs.get('wavelet_aug_basis', 'haar')
        self.wavelet_aug_hf_boost = cfg.kwargs.get('wavelet_aug_hf_boost', 1.3)
        self.wavelet_aug_prob = cfg.kwargs.get('wavelet_aug_prob', 0.5)
        self.wavelet    = WaveletEnhance(
            wavelet=self.wavelet_aug_basis,
            hf_boost=self.wavelet_aug_hf_boost,
            prob=self.wavelet_aug_prob,
        )
        self.transform  = Transformation(resize=input_size)
        self.image_root = cfg.kwargs['datapath'] +'/images/'
        self.mask_root  = cfg.kwargs['datapath'] +'/masks/'
        self.label_root = cfg.kwargs.get('labelpath', None)
        image_lst = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.jpg') or f.endswith('.png')]
        
        for each in image_lst:
            img_name = each.split("/")[-1]
            img_name = img_name.split(".")[0]
            self.samples.append(img_name)


    def _find_mask_path(self, name):
        """支持多种 mask 命名格式：name.png / name_mask.png / name_mask_1.png"""
        candidates = [
            self.mask_root + name + '.png',
            self.mask_root + name + '_mask.png',
        ]
        # glob 兜底：匹配 name_mask*.png
        candidates += sorted(glob.glob(self.mask_root + name + '_mask*.png'))
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def __getitem__(self, idx):
        
        name  = self.samples[idx]
        image = cv2.imread(self.image_root + name+'.png').astype(np.float32)
        image = image[:,:,::-1].copy()
        mask_path = self._find_mask_path(name)
        if mask_path is None:
            raise FileNotFoundError(f'Cannot find mask for: {name}  (searched in {self.mask_root})')
        mask = cv2.imread(mask_path, 0).astype(np.float32)
        shape = mask.shape
        label = -1
        if self.label_root:
            label_path = os.path.join(self.label_root, name + '.txt')
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        label = int(f.read().strip())
                except Exception:
                    label = -1
        #print(shape)

        if self.cfg.mode=='train':
            image, mask = self.resize(image, mask)
            if self.use_wavelet_aug:
                image, mask = self.wavelet(image, mask)   # 小波增强（Normalize前，值域[0,255]）
            image, mask = self.transform(image, mask) # 几何+颜色数据增强
            image, mask = self.normalize(image, mask)
            #image, mask = self.enhance(image, mask)
            #image, mask = self.dilation(image, mask)
            #image, mask = self.guassian(image, mask)
            #image, mask = self.randomflip(image, mask)
            #image, mask = self.transform(image, mask)
            #image, mask = self.randomcrop(image, mask)
            return image.copy(), mask.copy(), label
        elif self.cfg.mode=='val':
            # 验证集必须保持确定性：不做小波增强、不做随机几何/颜色增强
            image, mask = self.resize(image, mask)
            image, mask = self.normalize(image, mask)
            return image.copy(), mask.copy(), label
        else:
            shape = mask.shape #
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            return image, mask, shape, name, label

    def collate(self, batch):
        size = self.cfg.kwargs.get('resize', 1024)
        image, mask, label = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0,3,1,2)
        mask  = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        label = torch.tensor(label, dtype=torch.long)
        return image, mask, label

    def __len__(self):
        return len(self.samples)

