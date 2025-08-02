import torch.nn.functional as F
import torch

from typing import List

def calculate_iou(
    preds : torch.Tensor, 
    targets : torch.Tensor,
    thresholds : torch.Tensor
) -> float:
    predicted_prob = F.softmax(preds, dim=1)[:, 1, :, :]
    # thresholds = thresholds.view(-1, 1, 1)

    predicted_mask = (predicted_prob > thresholds)
    target_mask = targets.bool()

    intersection = (predicted_mask & target_mask).float().sum()
    union = (predicted_mask | target_mask).float().sum()

    return (intersection + 1) / (union + 1)

def calculate_recall(
    preds : torch.Tensor, 
    targets : torch.Tensor,
    thresholds : torch.Tensor
) -> float:
    predicted_prob = F.softmax(preds, dim=1)[:, 1, :, :]
    # thresholds = thresholds.view(-1, 1, 1)

    predicted_mask = (predicted_prob > thresholds)
    target_mask = targets.bool()

    tp = (predicted_mask & target_mask).float().sum()
    fn = ((~predicted_mask) & target_mask).float().sum()

    return (tp + 1) / (tp + fn + 1)

def calculate_precision(
    preds : torch.Tensor, 
    targets : torch.Tensor,
    thresholds : torch.Tensor
) -> float:
    predicted_prob = F.softmax(preds, dim=1)[:, 1, :, :]
    # thresholds = thresholds.view(-1, 1, 1)

    predicted_mask = (predicted_prob > thresholds)
    target_mask = targets.bool()

    tp = (predicted_mask & target_mask).float().sum()
    fp = (predicted_mask & (~target_mask)).float().sum()

    return (tp + 1) / (tp + fp + 1)