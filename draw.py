from dataset import CrackSegDataset
from inference.draw_mask import draw_plot

from model.msunet import MSUNet
from model.unet16 import Unet16

import torch
from torchvision import transforms
from torchvision.models.vgg import VGG16_Weights

if __name__ == '__main__':
    model_weight_path = './model_weights/model_unet16best.pth'
    state_dict = torch.load(model_weight_path)

    model = Unet16(num_classes=2, pretrained=VGG16_Weights.DEFAULT)
    model.load_state_dict(state_dict)

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(channel_means, channel_stds)])

    seg_train = CrackSegDataset(
        image_dir='./data/Train/images',
        mask_dir='./data/Train/masks',
        transform=transform
    )

    draw_plot(seg_train, model, 15)