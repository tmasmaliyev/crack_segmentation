import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps : float = 1e-7) -> None:
        super(DiceLoss, self).__init__()

        self.eps = eps
        self.ce = nn.CrossEntropyLoss()
    
    def forward(
        self, 
        preds : torch.Tensor, 
        targets : torch.Tensor
    ):
        return self.ce(preds, targets.long()) + 2 * (1 - self._get_dice_coef(preds, targets).mean())
    
    def evaluate(
        self,
        preds : torch.Tensor, 
        targets : torch.Tensor
    ) -> float:
        return self.ce(preds, targets.long()) + 2 * (1 - self._get_dice_coef(preds, targets).mean())

    def _get_dice_coef(
        self, 
        preds : torch.Tensor, 
        targets : torch.Tensor
    ) -> float:
        probs = F.softmax(preds, dim=1)

        probs_fg = probs[:, 1, :, :]

        probs_flat = probs_fg.contiguous().view(probs_fg.size(0), -1)
        targets_flat = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(1)
        union = probs_flat.sum(1) + targets_flat.sum(1)

        return (2 * intersection + self.eps) / (union + self.eps)