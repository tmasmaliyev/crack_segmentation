# from model.unet16 import Unet16
from model.unet16att import AttUnet16
from torchvision.models.vgg import VGG16_Weights

from dataset import CrackSegDataset, SegDatasetModule
from trainer import SegTrainer

import albumentations as A

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch',            type=int,   required=True)
    parser.add_argument('--root_dir',           type=str,   required=True)
    parser.add_argument('--model_save_dir',     type=str,   required=True)
    parser.add_argument('--model_weight_path',  type=str)
    parser.add_argument('--batch_size',         type=int,   required=True)
    parser.add_argument('--num_workers',        type=int,   required=True)
    parser.add_argument('--learning_rate',      type=float, required=True)

    args = parser.parse_args()

    model = AttUnet16(
        num_classes = 2,
        pretrained = VGG16_Weights.DEFAULT
    )

    transform = A.Compose([
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        # A.RandomBrightnessContrast(p=0.2),
        # A.GaussNoise(p=0.1)
    ])

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

    trainer.train(
        num_epochs=args.n_epoch,
        save_model_dir=args.model_save_dir,
        evaluate_validation = True
    )

    # trainer.evaluate()