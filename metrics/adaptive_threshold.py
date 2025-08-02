import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from typing import List

def calculate_adaptive_threshold(
    preds : torch.Tensor,   
) -> List[float]:
    preds = F.softmax(preds, dim=1)[:, 0, :, :]
    batch_size, height, width = preds.shape

    preds = preds.view(batch_size, height * width, 1)
    preds_cpu = preds.detach().cpu().numpy()

    thresholds = []

    for i in range(batch_size):
        pixel_data = preds_cpu[i]

        kmeans = KMeans(n_clusters=2, init='k-means++')
        kmeans.fit(pixel_data)

        centers = kmeans.cluster_centers_.flatten()

        thresholds.append(float(centers.mean()))
    
    return thresholds
