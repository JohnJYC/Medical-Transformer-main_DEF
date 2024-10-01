import os
import numpy as np
import torch

from skimage import io, color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from typing import Callable
import cv2
import pandas as pd

from numbers import Number
from typing import Container
from collections import defaultdict


def to_long_tensor(pic):
    # 将 numpy 数组转换为长整型张量
    img = torch.from_numpy(np.array(pic, np.uint8))
    return img.long()


def correct_dims(*images):
    """
    确保图像具有正确的维度。
    """
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class JointTransform2D:
    """
    对图像和掩码同时进行数据增强变换。

    参数：
        crop: 随机裁剪的大小。如果为 False，则不进行裁剪。
        p_flip: 执行随机水平翻转的概率。
        color_jitter_params: torchvision.transforms.ColorJitter 的参数。
        p_random_affine: 执行随机仿射变换的概率。
        long_mask: 如果为 True，返回长整型的标签编码格式的掩码。
    """

    def __init__(self, crop=(32, 32), p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=0, long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        # 转换为 PIL 图像
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # 随机裁剪
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        # 仅对图像进行颜色变换
        if self.color_jitter_params:
            image = self.color_tf(image)

        # 随机仿射变换
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine.get_params(
                degrees=(-90, 90), translate=(0.1, 0.1), scale_ranges=(0.9, 1.1), shears=(-10, 10), img_size=self.crop)
            image = F.affine(image, *affine_params)
            mask = F.affine(mask, *affine_params)

        # 转换为张量
        image = F.to_tensor(image)
        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image, mask


class ImageToImage2D(Dataset):
    """
    读取图像和掩码，并对其应用数据增强变换。

    用法：
        1. 如果不使用 unet.model.Model 包装器，可以将该类的实例传递给 torch.utils.data.DataLoader。
           迭代时返回图像、掩码和图像文件名的元组。
        2. 使用 unet.model.Model 包装器时，可以将该类的实例作为训练或验证数据集传递。

    参数：
        dataset_path: 数据集的路径。数据集的结构应为：
            dataset_path
              |-- img
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- labelcol
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: 增强变换，一个 JointTransform2D 的实例。如果没有提供，则对图像和掩码使用 torchvision.transforms.ToTensor。
        one_hot_mask: bool，如果为 True，则返回 one-hot 编码形式的掩码。
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image_path = os.path.join(self.input_path, image_filename)

        # 获取不带扩展名的文件名
        base_filename = os.path.splitext(image_filename)[0]
        # 假设掩码文件名与图像文件名相同，扩展名为 .png
        mask_filename = base_filename + ".png"
        mask_path = os.path.join(self.output_path, mask_filename)

        # 读取图像和掩码
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 检查是否成功读取
        if image is None:
            raise FileNotFoundError(f"无法读取图像文件：{image_path}")
        if mask is None:
            raise FileNotFoundError(f"无法读取掩码文件：{mask_path}")

        # 调整维度（如果需要）
        image, mask = correct_dims(image, mask)

        # 二值化掩码
        mask = (mask >= 127).astype(np.uint8)

        # 应用联合变换
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        # 如果需要，转换为 one-hot 编码
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask 必须是正整数'
            mask = torch.nn.functional.one_hot(mask.squeeze().long(), num_classes=self.one_hot_mask)
            mask = mask.permute(2, 0, 1).float()

        return image, mask, image_filename


class Image2D(Dataset):
    """
    读取图像并对其应用数据增强变换。与 ImageToImage2D 相比，它只读取单个图像。

    参数：
        dataset_path: 数据集的路径。数据集的结构应为：
            dataset_path
              |-- img
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        transform: 增强变换。如果未提供，则使用 torchvision.transforms.ToTensor。
    """

    def __init__(self, dataset_path: str, transform: Callable = None):

        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'img')
        self.images_list = os.listdir(self.input_path)

        if transform:
            self.transform = transform
        else:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]
        image_path = os.path.join(self.input_path, image_filename)

        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 检查是否成功读取
        if image is None:
            raise FileNotFoundError(f"无法读取图像文件：{image_path}")

        # 调整维度（如果需要）
        image = correct_dims(image)

        # 应用变换
        image = self.transform(image)

        return image, image_filename


def chk_mkdir(*paths: Container) -> None:
    """
    如果文件夹不存在，则创建它们。

    参数：
        paths: 要创建的路径的容器。
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


class Logger:
    """
    用于记录训练或验证过程中日志的类。
    """

    def __init__(self, verbose=False):
        self.logs = defaultdict(list)
        self.verbose = verbose

    def log(self, logs):
        for key, value in logs.items():
            self.logs[key].append(value)

        if self.verbose:
            print(logs)

    def get_logs(self):
        return self.logs

    def to_csv(self, path):
        pd.DataFrame(self.logs).to_csv(path, index=None)


class MetricList:
    """
    用于计算和管理评估指标的类。
    """

    def __init__(self, metrics):
        assert isinstance(metrics, dict), '\'metrics\' 必须是一个包含可调用对象的字典'
        self.metrics = metrics
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def __call__(self, y_out, y_batch):
        for key, func in self.metrics.items():
            self.results[key] += func(y_out, y_batch)

    def reset(self):
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def get_results(self, normalize=False):
        assert isinstance(normalize, bool) or isinstance(normalize, Number), \
            '\'normalize\' 必须是布尔值或数字'
        if not normalize:
            return self.results
        else:
            return {key: value / normalize for key, value in self.results.items()}
