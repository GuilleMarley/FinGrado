import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

labels = np.load("labels.npy")
data = torch.load("caracteres_segmentados.pt")


data = data.cpu()

for cluster_id in range(27):
    indices = np.where(labels == cluster_id)[0]
    if len(indices) == 0:
        continue

    print(f"Clúster {cluster_id} - muestras: {len(indices)}")

    images = data[indices][:10] 
    grid = make_grid(images, nrow=4, normalize=True, pad_value=1)

    plt.figure(figsize=(4, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title(f"Clúster {cluster_id}")
    plt.show()
