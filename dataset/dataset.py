from torch.utils.data import Dataset
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from typing import Tuple, List

import torch
import numpy as np
import os

class CrackSegDataset(Dataset):
    def __init__(
        self, 
        image_dir : str, 
        mask_dir : str, 
        transform : A.Compose | None = None,
        channel_means : List | None = None,
        channel_stds : List | None = None
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.general_transform = transform
        self.channel_normalizer = None

        if (channel_means is not None) and (channel_stds is not None):
            self.channel_normalizer = A.Compose([
                A.Normalize(mean=channel_means, std=channel_stds),
                ToTensorV2()
            ])

        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        return len(self.images)
    
    def get_image(self, index : int) -> List:
        img_path = os.path.join(self.image_dir, self.images[index])

        image = Image.open(img_path).convert('RGB')

        return np.array(image)

    def get_mask(self, index : int) -> List:
        mask_path = os.path.join(self.mask_dir, self.images[index])

        mask = Image.open(mask_path).convert('L')

        return np.array(mask)

    def __getitem__(
        self, 
        index : int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Getting full path
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # Reading Image & Path
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        # Apply augmentation
        if self.general_transform is not None:
            augmented = self.general_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Apply normalizer specifically on input image
        if self.channel_normalizer is not None:
            augmented = self.channel_normalizer(image=image)
            image = augmented['image']

        # Classify as 0 for background & 1 for target
        mask = (mask > 127).astype(float)

        return image, torch.from_numpy(mask)