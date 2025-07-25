import torch
import torch.nn as nn

from .diceloss import DiceLoss

class DiceBCELoss(nn.Module):
    def __init__(self) -> None:
        super(DiceBCELoss, self).__init__()

        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(
        self, 
        preds : torch.Tensor, 
        targets : torch.Tensor
    ):
        return self.ce(preds, targets) + 5 * self.dice(preds, targets)
    
    def evaluate(
        self,
        preds : torch.Tensor, 
        targets : torch.Tensor
    ) -> float:
        return self.ce(preds, targets) + 5 * self.dice(preds, targets)
    
    def get_iou(
        self,
        preds : torch.Tensor,
        targets : torch.Tensor
    ) -> float:
        return self.dice.evaluate(preds, targets)