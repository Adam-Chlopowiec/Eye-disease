from typing import Tuple
from torchvision.transforms import transforms


def train_transforms(target_size: Tuple[int, int], normalize: bool = True) -> transforms.Compose:
    transforms_list = [
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-5, 5)),
            transforms.ToTensor()
        ]
    if normalize:
        transforms_list.append(transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ))
    return transforms.Compose(transforms_list)

def test_val_transforms(target_size: Tuple[int, int], normalize: bool = True) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(target_size),
        transforms.ToTensor()
    ]
    if normalize:
        transforms_list.append(transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ))
    return transforms.Compose(transforms_list)