from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import torch
import numpy as np
import os

class CrackSegDataset(Dataset):
    def __init__(
        self, 
        image_dir : str, 
        mask_dir : str, 
        transform : transforms.Compose | None = None
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = transform

        if self.image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.ToTensor()
            ])

        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x : x.squeeze(0)),
            transforms.Lambda(lambda x : (x == 1).long())
        ])

        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        return len(self.images)
    
    def get_image(self, index : int):
        img_path = os.path.join(self.image_dir, self.images[index])

        image = Image.open(img_path).convert('RGB')

        return np.array(image)

    def get_mask(self, index : int):
        mask_path = os.path.join(self.mask_dir, self.images[index])

        mask = Image.open(mask_path).convert('L')

        return np.array(mask)
    
    def __getitem__(self, index : int):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # Image handling
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        mask = (mask > 127).astype(int)

        image = self.image_transform(image)

        return image, torch.from_numpy(mask)
    
class CrackSegPatchedDataset(Dataset):
    def __init__(
        self, 
        image_dir : str, 
        mask_dir : str, 
        patch_size : int = 32
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size

        # Default Transform, FIX IT !
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x : self._transform_patches(x))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x : self._transform_patches(x)),
            transforms.Lambda(lambda x : x.squeeze(1)),
            transforms.Lambda(lambda x : (x == 1).long())
        ])

        self.images = os.listdir(image_dir)
    
    def _transform_patches(self, x : torch.Tensor):
        channel = x.shape[0]
        x = x.unfold(1, size=self.patch_size, step=self.patch_size).\
              unfold(2, size=self.patch_size, step=self.patch_size)
        
        x = x.permute(1, 2, 0, 3, 4).contiguous()

        return x.view(-1, channel, self.patch_size, self.patch_size)

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index : int):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # Image handling
        image = Image.open(img_path).convert('RGB')
        image = self.image_transform(image)

        # Mask handling
        mask = Image.open(mask_path).convert('L')
        mask = self.mask_transform(mask)

        return image, mask
    
    