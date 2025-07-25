import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np

def draw_plot(
    seg_dataset,
    model,
    ndata: int
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

        axs[i][0].imshow(img_arr)
        axs[i][1].imshow(mask_arr)

        img, masks = seg_dataset[random_nums[i]]
        img = img.view(1, *img.shape)

        output = model(img)
        predicted_prob = F.softmax(output, dim=1)[:, 1, :, :]

        predicted_mask = (predicted_prob > 0.2).int() * 255

        axs[i][2].imshow(predicted_mask.cpu().numpy().reshape(predicted_mask.shape[1:]))
    
    plt.show()