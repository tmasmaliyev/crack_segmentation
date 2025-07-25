from model.unet16 import Unet16
from torchvision.models.vgg import VGG16_Weights
from torchvision import transforms

from dataset import CrackSegDataset, SegDatasetModule
from trainer import SegTrainer

import albumentations as A
import cv2

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',       type=str,   required=True)
    parser.add_argument('--model_save_dir', type=str,   required=True)
    parser.add_argument('--model_weight_path',  type=str)
    parser.add_argument('--batch_size',     type=int,   required=True)
    parser.add_argument('--num_workers',    type=int,   required=True)
    parser.add_argument('--learning_rate',  type=float, required=True)

    args = parser.parse_args()

    model = Unet16(
        num_classes = 2,
        pretrained = VGG16_Weights.DEFAULT
    )

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(channel_means, channel_stds)])
    
    # transform = A.Compose([
    #     A.HorizontalFlip(p=0.5),
    #     A.VerticalFlip(p=0.5),
    #     A.Rotate(limit=45, p=0.5, interpolation=cv2.INTER_NEAREST),
    #     A.RandomBrightnessContrast(p=0.2),
    #     A.GaussNoise(p=0.2)
    # ])

    seg_data_module = SegDatasetModule(
        root_dir = args.root_dir,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        dataset_class = CrackSegDataset,
        transform = transform
    )

    loaders = seg_data_module.get_loaders()

    train_loader = loaders['train']
    validation_loader = loaders['validation']
    test_loader = loaders['test']

    trainer = SegTrainer(
        model = model,
        train_loader = train_loader,
        validation_loader = validation_loader,
        test_loader = test_loader,
        learning_rate = args.learning_rate,
        model_weight_path = args.model_weight_path
    )

    # trainer.train(
    #     num_epochs=100,
    #     save_model_dir=args.model_save_dir,
    #     evaluate_validation = True
    # )
    trainer.evaluate()