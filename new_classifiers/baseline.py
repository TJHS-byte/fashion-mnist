import os
import gzip
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---------- Load Fashion MNIST ----------
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# ---------- Dataset ----------
class FashionMNISTFromNumpy(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.images[idx], self.labels[idx]

# ---------- CNN Model ----------
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ---------- Load Data ----------
dirname = os.path.dirname(__file__)
parent_dir = os.path.dirname(dirname)
train_images = load_mnist_images(os.path.join(parent_dir, "data/fashion/train-images-idx3-ubyte.gz"))
train_labels = load_mnist_labels(os.path.join(parent_dir, "data/fashion/train-labels-idx1-ubyte.gz"))
test_images = load_mnist_images(os.path.join(parent_dir, "data/fashion/t10k-images-idx3-ubyte.gz"))
test_labels = load_mnist_labels(os.path.join(parent_dir, "data/fashion/t10k-labels-idx1-ubyte.gz"))

X_tr = train_images.astype(np.float32) / 255.0
X_te = test_images.astype(np.float32) / 255.0

trainloader = DataLoader(FashionMNISTFromNumpy(X_tr, train_labels), batch_size=64, shuffle=True)
testloader = DataLoader(FashionMNISTFromNumpy(X_te, test_labels), batch_size=64, shuffle=False)

# ---------- Setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
f = CNNClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
opt_f = optim.Adam(f.parameters(), lr=0.001)

# ---------- Train ----------
print("🔧 Training CNN...")
f.train()
for epoch in range(30):  # Run longer for better benchmark
    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        opt_f.zero_grad()
        logits = f(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt_f.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}, Batch {batch_idx+1}] Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

# ---------- Evaluate ----------
f.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        pred = torch.argmax(f(x), dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"\n✅ Final Test Accuracy: {correct}/{total} = {100.0 * correct / total:.2f}%")
