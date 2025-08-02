from dataset import CrackSegDataset
from inference.draw_mask import draw_plot

from model.msunet import MSUNet
from model.unet16 import Unet16
from model.unet16att import AttUnet16

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import torch
from torchvision import transforms
from torchvision.models.vgg import VGG16_Weights

if __name__ == '__main__':
    model_weight_path = './model_weights/model_weights_final.pth'
    state_dict = torch.load(model_weight_path)

    model = AttUnet16(num_classes=2, pretrained=VGG16_Weights.DEFAULT)
    model.load_state_dict(state_dict)

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]

    transform = A.Compose([
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        # A.Rotate(limit=(-90, 90), p=0.2, interpolation=cv2.INTER_AREA),
        # A.RandomBrightnessContrast(p=0.2),
        # A.GaussNoise(p=0.1)
    ])

    channel_normalizer = A.Compose([
        A.Normalize(mean=channel_means, std=channel_stds),
        ToTensorV2()
    ])

    seg_train = CrackSegDataset(
        image_dir='./data/Validation/images',
        mask_dir='./data/Validation/masks',
        transform=transform,
        channel_means=channel_means,
        channel_stds=channel_stds
    )
    while True:
        draw_plot(seg_train, model, 5, transform, channel_normalizer)