import torch.nn as nn
import torch.nn.functional as F

from .diceloss import DiceLoss

import torch

class FocalLoss(nn.Module):
    def __init__(
        self, 
        alpha : int = 1,
        gamma : int = 2,
        use_dice_loss : bool = False
    ) -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_dice_loss = use_dice_loss

        self.dice = DiceLoss()

    def forward(
        self, 
        preds : torch.Tensor,
        targets : torch.Tensor   
    ) -> float:
        loss = 10 * self.calculate_focal(preds, targets)
        loss = 0

        if self.use_dice_loss:
            loss += self.dice(preds, targets)

        return loss
    
    def calculate_focal(
        self, 
        preds : torch.Tensor,
        targets : torch.Tensor  
    ) -> float:
        log_probs = F.log_softmax(preds, dim=1)
        probs = torch.exp(log_probs)

        targets_one_hot = F.one_hot(targets, num_classes=preds.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)

        pt = (probs * targets_one_hot).sum(1)
        log_pt = (log_probs * targets_one_hot).sum(1)

        focal_term = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_term * log_pt
        
        return loss.mean()
    
    def evaluate(        
        self, 
        preds : torch.Tensor,
        targets : torch.Tensor   
    ) -> float:
        loss = self.calculate_focal(preds, targets)

        if self.use_dice_loss:
            loss += self.dice(preds, targets)

        return loss
