from model.unet import SUNet
from model.msunet import MSUNet
from dataset import CrackSegDataset, CrackSegPatchedDataset, SegDatasetModule
from loss.diceloss import DiceLoss
from loss.focal import FocalLoss
from loss.iou import calculate_iou

from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

import torch
import torch.optim as optim
import torch.nn as nn

import argparse, os

class SegTrainer:
    def __init__(
        self,
        model : nn.Module,
        train_loader : DataLoader,
        validation_loader : DataLoader,
        test_loader : DataLoader,
        learning_rate : float = 1e-5,
        model_weight_path : str | None = None,
        device : str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        self.device = device
        self.model_weight_path = model_weight_path

        self.model = model.to(device)

        if model_weight_path:
            state_dict = torch.load(model_weight_path, map_location=self.device)

            model.load_state_dict(state_dict)
            print("Model weights loaded successfully!")

        self.criterion = FocalLoss(use_dice_loss=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.best_val_loss = float('inf')

    def train(
        self, 
        num_epochs : int,
        evaluate_validation : bool = False,
        save_model_dir : str | None = None
    ) -> None:
        for epoch in range(1, num_epochs + 1):
            self.model.train()

            total_loss = 0
            total_iou = 0
            loop = tqdm(self.train_loader, desc=f'Epoch [{epoch} / {num_epochs}]')

            for images, masks in loop:
                images = images.to(self.device).float()
                # images = images.view(-1, *images.shape[2:])

                masks = masks.to(self.device).long()
                # masks = masks.view(-1, *masks.shape[2:])

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                iou = calculate_iou(outputs, masks, 0.5) * 100

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_iou += iou.item()
                loop.set_postfix(
                    loss = loss.item(), 
                    iou = iou.item()
                )

            print(f'Epoch {epoch}, Loss : {total_loss / len(self.train_loader) : .4f} '
                  f'Accuracy : {total_iou / len(self.train_loader) : .4f}')

            
            if save_model_dir and evaluate_validation:
                val_loss, val_iou = self.evaluate()

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

                    torch.save(
                        self.model.state_dict(), 
                        os.path.join(save_model_dir, 'model_weights.pth')
                    )
            
            elif evaluate_validation:
                self.evaluate()

    def evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        total_iou = 0

        with torch.no_grad():
            loop = tqdm(self.validation_loader, desc='Evaluating validation')

            for images, masks in loop:
                images = images.to(self.device)
                # images = images.view(-1, *images.shape[2:])

                masks = masks.to(self.device).long()
                # masks = masks.view(-1, *masks.shape[2:])

                outputs = self.model(images)
                evaluation_loss = self.criterion(outputs, masks)
                evaluation_iou = calculate_iou(outputs, masks, 0.5) * 100

                total_loss += evaluation_loss.item()
                total_iou += evaluation_iou.item()

                loop.set_postfix(
                    loss = evaluation_loss.item(), 
                    iou = evaluation_iou.item()
                )
            
            print(f'Loss : {total_loss / len(self.validation_loader) : .4f} ' + 
                  f'Accuracy : {total_iou / len(self.validation_loader) : .4f}')
        
        return total_loss / len(self.validation_loader), \
               100 * (total_iou / len(self.validation_loader))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',       type=str,   required=True)
    parser.add_argument('--model_save_dir', type=str,   required=True)
    parser.add_argument('--model_weight_path',  type=str)
    parser.add_argument('--batch_size',     type=int,   required=True)
    parser.add_argument('--num_workers',    type=int,   required=True)
    parser.add_argument('--learning_rate',  type=float, required=True)

    args = parser.parse_args()

    seg_data_module = SegDatasetModule(
        root_dir = args.root_dir,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        dataset_class = CrackSegDataset
    )

    loaders = seg_data_module.get_loaders()

    train_loader = loaders['train']
    validation_loader = loaders['validation']
    test_loader = loaders['test']

        
    trainer = SegTrainer(
        model = MSUNet(),
        train_loader = train_loader,
        validation_loader = validation_loader,
        test_loader = test_loader,
        learning_rate = args.learning_rate,
        model_weight_path = args.model_weight_path
    )

    trainer.train(
        num_epochs=100,
        save_model_dir=args.model_save_dir,
        evaluate_validation = True
    )