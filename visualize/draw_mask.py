import torch.nn.functional as F
import matplotlib.pyplot as plt

from metrics.adaptive_threshold import calculate_adaptive_threshold

import numpy as np
import torch

def _transform_patches(x: torch.Tensor, patch_size : int):
    c, h, w = x.shape

    x = x.unfold(1, patch_size, patch_size).\
          unfold(2, patch_size, patch_size)
    x = x.permute(1, 2, 0, 3, 4).contiguous() 
    x = x.view(-1, c, patch_size, patch_size) 

    resized = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)

    return resized

def combine_patches(patches: torch.Tensor, grid_size: int):
    """
    patches: [N, C, H, W] where N = grid_size x grid_size
    grid_size: number of patches per row/column (e.g., 7)

    Returns:
        image: [C, H_total, W_total]
    """
    N, C, H, W = patches.shape
    assert N == grid_size ** 2, "Patch count must match grid size"

    patches = patches.view(grid_size, grid_size, C, H, W)  # [grid_h, grid_w, C, H, W]
    patches = patches.permute(2, 0, 3, 1, 4)               # [C, grid_h, H, grid_w, W]
    patches = patches.contiguous().view(C, grid_size * H, grid_size * W)  # [C, H_total, W_total]

    return patches

def draw_plot(
    seg_dataset,
    model,
    ndata: int,
    general_transform,
    channel_normalizer
):
    fig, axs = plt.subplots(
        nrows=ndata, 
        ncols=3,
        figsize=(12, 5)
    )

    random_nums = np.random.choice(len(seg_dataset), ndata)

    for i in range(ndata):
        img_arr = seg_dataset.get_image(random_nums[i])
        mask_arr = seg_dataset.get_mask(random_nums[i])

        augmented = general_transform(image=img_arr, mask=mask_arr)
        image = augmented['image']
        mask = augmented['mask']

        axs[i][0].imshow(image)
        axs[i][1].imshow(mask)

        augmented = channel_normalizer(image=image, mask=mask)

        image = augmented['image']
        image = image.view(1, *image.shape)

        mask = (mask > 127).astype(float)
        mask = torch.from_numpy(mask)

        outputs = model(image)
        thresholds = calculate_adaptive_threshold(outputs)
        thresholds = torch.Tensor(thresholds).view(-1, 1, 1)

        predicted_prob = F.softmax(outputs, dim=1)[:, 1, :, :]
        predicted_mask = (predicted_prob > 0.5).int() * 255

        axs[i][2].imshow(predicted_mask.cpu().numpy().reshape(predicted_mask.shape[1:]))
    
    plt.show()