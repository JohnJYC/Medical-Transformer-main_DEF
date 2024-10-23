import os
import numpy as np
import torch
import random
from skimage import io, color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from typing import Callable
import os
import cv2
import pandas as pd

from numbers import Number
from typing import Container
from collections import defaultdict


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
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

    def __init__(self, crop=None, p_flip=0.5, color_jitter_params=None,
                 p_random_affine=0.0, long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.color_jitter_params = color_jitter_params
        if self.color_jitter_params:
            self.color_tf = T.ColorJitter(*self.color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask

    def __call__(self, image, mask):
        # 确保 image 和 mask 是 numpy 数组
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        elif not isinstance(image, np.ndarray):
            image = np.array(image)

        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        elif not isinstance(mask, np.ndarray):
            mask = np.array(mask)

        # 将数据类型转换为 np.uint8
        image = image.astype(np.uint8)
        mask = mask.astype(np.uint8)

        # 将图像和掩码转换为 PIL 图像
        image = F.to_pil_image(image)
        mask = F.to_pil_image(mask)

        # 随机裁剪
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image = F.crop(image, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)

        # 随机水平翻转
        if random.random() < self.p_flip:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # 颜色抖动（仅应用于图像）
        if self.color_jitter_params:
            image = self.color_tf(image)

        # 随机仿射变换
        if random.random() < self.p_random_affine:
            angle = random.uniform(-30, 30)
            translate = (random.uniform(-0.1, 0.1) * image.size[0], random.uniform(-0.1, 0.1) * image.size[1])
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-10, 10)
            image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=shear)
            mask = F.affine(mask, angle=angle, translate=translate, scale=scale, shear=shear, fill=0)

        # 将图像和掩码转换为张量
        image = F.to_tensor(image)
        if not self.long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image, mask

    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """


class ImageToImage2D(Dataset):
    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask

        self.joint_transform = joint_transform if joint_transform else self.default_transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]

        image = cv2.imread(os.path.join(self.input_path, image_filename))
        mask = cv2.imread(os.path.join(self.output_path, image_filename.replace('.tif', '_mask.tif')),
                          cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype(int)

        image, mask = self.correct_dims(image, mask)

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.nn.functional.one_hot(torch.tensor(mask), num_classes=self.one_hot_mask).permute(2, 0,
                                                                                                          1).float()

        return image, mask, image_filename

    def default_transform(self, image, mask):
        to_tensor = T.ToTensor()
        return to_tensor(image), to_tensor(mask)

    def correct_dims(self, image, mask):

        if len(image.shape) == 2:  # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if len(mask.shape) == 2:  # Expand dimensions for mask if needed
            mask = mask[..., None]
        return image, mask


class Image2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them. As opposed to ImageToImage2D, this
    reads a single image and requires a simple augmentation transform.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as a prediction
           dataset.

    Args:

        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        transform: augmentation transform. If bool(joint_transform) evaluates to False,
            torchvision.transforms.ToTensor will be used.
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
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]

        image = cv2.imread(os.path.join(self.input_path, image_filename))

        # image = np.transpose(image,(2,0,1))

        image = correct_dims(image)

        image = self.transform(image)

        # image = np.swapaxes(image,2,0)

        return image, image_filename


def chk_mkdir(*paths: Container) -> None:
    """
    Creates folders if they do not exist.

    Args:        
        paths: Container of paths to be created.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


class Logger:
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
    def __init__(self, metrics):
        assert isinstance(metrics, dict), '\'metrics\' must be a dictionary of callables'
        self.metrics = metrics
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def __call__(self, y_out, y_batch):
        for key, value in self.metrics.items():
            self.results[key] += value(y_out, y_batch)

    def reset(self):
        self.results = {key: 0.0 for key in self.metrics.keys()}

    def get_results(self, normalize=False):
        assert isinstance(normalize, bool) or isinstance(normalize, Number), '\'normalize\' must be boolean or a number'
        if not normalize:
            return self.results
        else:
            return {key: value / normalize for key, value in self.results.items()}
