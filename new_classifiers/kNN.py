import gzip
import numpy as np
import struct
import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Your preloaded arrays
# X_tr, y_tr, X_te, y_te already defined
# X_tr shape: (60000, 784) → reshape to (60000, 1, 28, 28)

class FashionMNISTFromNumpy(Dataset):
    def __init__(self, images, labels):
        self.images = images.astype(np.float32) / 255.0  # Normalize to [0,1]
        self.labels = labels.astype(np.int64)
        self.images = self.images.reshape((-1, 1, 28, 28))  # PyTorch expects (N, C, H, W)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx])
        lbl = torch.tensor(self.labels[idx])
        return img, lbl

# Create datasets
train_dataset = FashionMNISTFromNumpy(X_tr, y_tr)
test_dataset = FashionMNISTFromNumpy(X_te, y_te)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)