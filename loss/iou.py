import torch.nn.functional as F

import torch

def calculate_iou(
    preds : torch.Tensor, 
    targets : torch.Tensor,
    threshold : float
) -> float:
    predicted_prob = F.softmax(preds, dim=1)[:, 1, :, :]
    # predicted_prob = F.sigmoid(preds.squeeze(1))

    predicted_mask = (predicted_prob > threshold)
    target_mask = targets.bool()

    intersection = (predicted_mask & target_mask).float().sum()
    union = (predicted_mask | target_mask).float().sum()

    return union + 1 if union == 0 else intersection / union
    