from torch.utils.data import Dataset, DataLoader
import albumentations as A

from typing import Dict

class SegDatasetModule:
    def __init__(
        self, 
        root_dir : str, 
        batch_size : int, 
        num_workers : int,
        dataset_class : Dataset,
        transform : A.Compose | None = None
    ) -> None:
        
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_class = dataset_class
        self.transform = transform

        self.channel_means = [0.485, 0.456, 0.406]
        self.channel_stds  = [0.229, 0.224, 0.225]

        self.paths = {
            'train' : {
                'image_dir' : f'{root_dir}/Train/images',
                'mask_dir' : f'{root_dir}/Train/masks',
                'transform' : self.transform,
                'channel_means' : self.channel_means,
                'channel_stds' : self.channel_stds
            },
            'validation' : {
                'image_dir' : f'{root_dir}/Validation/images',
                'mask_dir' : f'{root_dir}/Validation/masks',
                'channel_means' : self.channel_means,
                'channel_stds' : self.channel_stds
            },
            'test' : {
                'image_dir' : f'{root_dir}/Test/images',
                'mask_dir' : f'{root_dir}/Test/masks',
                'channel_means' : self.channel_means,
                'channel_stds' : self.channel_stds
            }
        }
    
    def setup(self) -> None:
        self.train_dataset = self.dataset_class(**self.paths['train'])
        self.validation_dataset = self.dataset_class(**self.paths['validation'])
        self.test_dataset = self.dataset_class(**self.paths['test'])

    def get_loaders(self) -> Dict[str, DataLoader]:
        self.setup()

        return {
            'train' : DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers),
            'validation' : DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers),
            'test' : DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers),
        }
