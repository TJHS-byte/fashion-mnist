import os
import gzip
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

# ---------- Models ----------
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

class MLPForWrongSamples(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.model(x)

# ---------- Paths ----------
dirname = os.path.dirname(__file__)
parent_dir = os.path.dirname(dirname)
train_images = load_mnist_images(os.path.join(parent_dir, "data/fashion/train-images-idx3-ubyte.gz"))
train_labels = load_mnist_labels(os.path.join(parent_dir, "data/fashion/train-labels-idx1-ubyte.gz"))
test_images = load_mnist_images(os.path.join(parent_dir, "data/fashion/t10k-images-idx3-ubyte.gz"))
test_labels = load_mnist_labels(os.path.join(parent_dir, "data/fashion/t10k-labels-idx1-ubyte.gz"))

# ---------- Preprocessing ----------
X_tr = train_images.astype(np.float32) / 255.0
X_te = test_images.astype(np.float32) / 255.0

trainloader = DataLoader(FashionMNISTFromNumpy(X_tr, train_labels), batch_size=64, shuffle=True)
testloader = DataLoader(FashionMNISTFromNumpy(X_te, test_labels), batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
f = CNNClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
opt_f = optim.Adam(f.parameters(), lr=0.001)

# ---------- Train CNN ----------
print("🔧 Training CNN classifier...")
f.train()
for epoch in range(3):
    total_loss = 0
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        opt_f.zero_grad()
        logits = f(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt_f.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"[CNN] Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    print(f"[CNN] Epoch {epoch+1} Average Loss: {total_loss / len(trainloader):.4f}")

# ---------- Freeze CNN ----------
f.eval()
for p in f.parameters():
    p.requires_grad = False

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("🔎 Collecting training predictions...")
all_x, all_y, all_preds = [], [], []
with torch.no_grad():
    for x, y in trainloader:
        x = x.to(device)
        logits = f(x)
        pred = torch.argmax(logits, dim=1)
        all_x.append(x.cpu())
        all_y.append(y)
        all_preds.append(pred.cpu())

X_train = torch.cat(all_x)
Y_train = torch.cat(all_y)
Y_preds = torch.cat(all_preds)

# Compute CNN correctness flags
Y_np = Y_train.numpy()
Y_preds_np = Y_preds.numpy()
correct_flags = (Y_np == Y_preds_np).astype(np.uint8)

# Flatten image input
X_flat = X_train.view(X_train.size(0), -1).numpy()

# 🌐 Apply PCA
print("🔍 Performing PCA...")
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X_flat)

# 🌲 Train Random Forest on PCA features
print("🌲 Training Random Forest to predict CNN correctness...")
forest = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
forest.fit(X_reduced, correct_flags)

# 🧪 Evaluate performance on training set (for debugging)
predicted_flags = forest.predict(X_reduced)
train_acc = accuracy_score(correct_flags, predicted_flags)
print(f"✅ Random Forest Accuracy on training (predicting CNN correctness): {100 * train_acc:.2f}%")

# ---------- Train MLP on Incorrect Samples ----------
print("🧠 Training MLP on CNN-misclassified samples...")
X_incorrect = X_flat[correct_flags == 0]
Y_incorrect = Y_np[correct_flags == 0]

mlp = MLPForWrongSamples().to(device)
opt_mlp = optim.Adam(mlp.parameters(), lr=0.001)
loss_mlp = nn.CrossEntropyLoss()

wrong_dataset = FashionMNISTFromNumpy(X_incorrect.reshape(-1, 1, 28, 28), Y_incorrect)
wrong_loader = DataLoader(wrong_dataset, batch_size=64, shuffle=True)

mlp.train()
for epoch in range(20):
    total_loss = 0
    for batch_idx, (xb, yb) in enumerate(wrong_loader):
        xb, yb = xb.to(device), yb.to(device)
        pred = mlp(xb)
        loss = loss_mlp(pred, yb)
        opt_mlp.zero_grad()
        loss.backward()
        opt_mlp.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"[MLP] Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
    print(f"[MLP] Epoch {epoch+1} Average Loss: {total_loss / len(wrong_loader):.4f}")

# ---------- Evaluate DT on Training Set ----------
print("\n📈 Evaluating DT on CNN correctness prediction...")
dt_preds = forest.predict(X_reduced)
dt_acc = accuracy_score(correct_flags, dt_preds)
print(f"✅ DT Accuracy on training (predicting CNN correctness): {100.0 * dt_acc:.2f}%")

# ---------- Evaluate MLP on Incorrect Samples ----------
print("\n📈 Evaluating MLP on misclassified samples...")
mlp.eval()
mlp_correct = 0
mlp_total = len(X_incorrect)
with torch.no_grad():
    for i in range(mlp_total):
        x = torch.tensor(X_incorrect[i]).reshape(1, 1, 28, 28).to(device)
        y = Y_incorrect[i]
        pred = torch.argmax(mlp(x), dim=1).item()
        if pred == y:
            mlp_correct += 1
print(f"✅ MLP Accuracy on incorrect training samples: {mlp_correct}/{mlp_total} = {100.0 * mlp_correct / mlp_total:.2f}%")

# ---------- Evaluate CNN Alone ----------
print("\n🎯 Evaluating CNN Alone on Test Set...")
f.eval()
cnn_correct = 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        pred = torch.argmax(f(x), dim=1)
        cnn_correct += (pred == y).sum().item()
print(f"✅ CNN Native Test Accuracy: {cnn_correct}/{len(X_te)} = {100.0 * cnn_correct / len(X_te):.2f}%")

# ---------- Final Hybrid Inference ----------
print("\n📊 Final Hybrid Model Evaluation on Test Set...")
f.eval()
correct = 0
total = len(X_te)
with torch.no_grad():
    for i in range(total):
        x = torch.tensor(X_te[i]).unsqueeze(0).to(device)
        y = test_labels[i]
        x_np = x.view(1, -1).cpu().numpy()
        x_reduced = pca.transform(x_np)
        cnn_is_right = forest.predict(x_reduced)[0]
        if cnn_is_right:
            pred = torch.argmax(f(x), dim=1).item()
        else:
            pred = torch.argmax(mlp(x), dim=1).item()
        if pred == y:
            correct += 1
print(f"✅ Final Hybrid Model Accuracy: {correct}/{total} = {100.0 * correct / total:.2f}%")




