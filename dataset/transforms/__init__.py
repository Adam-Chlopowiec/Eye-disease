from typing import Tuple, Dict
from torchvision.transforms import transforms, InterpolationMode

from PIL import Image
import torch
import numpy as np
import albumentations as A
import imgaug as ia
import imgaug.augmenters as iaa
import cv2


class ToNumpy:

    def __call__(self, x: Image) -> np.ndarray:
        return np.asarray(x)

class Albument:

    def __init__(self, augment) -> None:
        self.augment = augment

    def __call__(self, img: Image) -> np.ndarray:
        return self.augment(image=img)['image']

class Imgaugment:

    def __init__(self, augment) -> None:
        self.augment = augment

    def __call__(self, img: np.ndarray) -> np.ndarray:
        augmented_images = self.augment(images=[img])
        return augmented_images[0]


def train_transforms(
    target_size: Tuple[int, int],
    normalize: bool = True, 
    interpolation_mode = cv2.INTER_NEAREST) -> transforms.Compose:

    aug_A = A.Compose(
                    [
                        A.Resize(target_size[0], target_size[1], interpolation=interpolation_mode),
                        A.Rotate(limit=(-5, 5), p=0.3, interpolation=interpolation_mode),
                        A.HorizontalFlip(p=0.3),
                        A.VerticalFlip(p=0.3),
                        A.GaussianBlur(p=0.2),
                        A.Equalize(by_channels=False, p=0.2)
                    ])
    aug_ia = iaa.Sometimes(p=1, then_list=[
                                        iaa.Sometimes(p=0.2, then_list=[iaa.AdditiveGaussianNoise()]),
                                        iaa.Sometimes(p=0.3, then_list=[iaa.LinearContrast()]),
                                        iaa.Sometimes(p=0.3, then_list=[iaa.AddToBrightness()])
                                    ])

    albument = Albument(aug_A)
    imgaugment = Imgaugment(aug_ia)
    transforms_list = [
        ToNumpy(),
        albument,
        imgaugment,
        transforms.ToTensor()
    ]
    if normalize:
        transforms_list.append(transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ))
    return transforms.Compose(transforms_list)

def test_val_transforms(
    target_size: Tuple[int, int], 
    normalize: bool = True,
    interpolation_mode = cv2.INTER_NEAREST) -> transforms.Compose:
    
    aug_A = A.Resize(target_size[0], target_size[1], interpolation=interpolation_mode)
    albument = Albument(aug_A)
    transforms_list = [
        ToNumpy(),
        albument,
        transforms.ToTensor()
    ]
    if normalize:
        transforms_list.append(transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ))
    return transforms.Compose(transforms_list)

