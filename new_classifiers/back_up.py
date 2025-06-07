import gzip
import os
import struct

import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binom
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch

CONFIDENCE_MODE = "SR"
ACCEPTED_RISK = 0.03
CONFIDENCE_LEVEL = 0.99
MC_DROPOUT_PASSES = 30
BATCH_SIZE = 64

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

class FashionMNISTFromNumpy(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def compute_vector_confidence(vec, mode):
    if mode == "SR": return max(vec)
    elif mode == "MC": return np.max(np.mean(vec, axis=0))
    else: raise ValueError(f"Unknown mode: {mode}")

def collect_softmax_vectors(model, dataloader, mode):
    model.eval()
    vectors = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            out = F.softmax(model(x), dim=1)
            vectors.extend(out.cpu().numpy())
    return vectors

def find_confidence_threshold(vecs, labels, preds, mode, accepted_risk, confidence_level):
    data = [(compute_vector_confidence(v, mode), p == y) for v, y, p in zip(vecs, labels, preds)]
    sorted_data = sorted(data, key=lambda x: -x[0])
    best_threshold = None
    delta = 1 - confidence_level
    for z in range(1, len(sorted_data)+1):
        errors = sum(1 for _, correct in sorted_data[:z] if not correct)
        emp_risk = errors / z
        bound = binom.ppf(1 - delta, z, emp_risk) / z if z > 0 else 1.0
        if bound <= accepted_risk:
            best_threshold = sorted_data[z - 1][0]
    return best_threshold if best_threshold else sorted_data[-1][0]

def hybrid_inference_with_knn_only(cnn, testloader, softmax_vectors, threshold, mode,
                                    train_images_raw, train_labels):
    cnn.eval()

    total = correct = 0
    accepted = rejected = 0
    accepted_but_incorrect = 0
    rejected_but_correct = 0

    disagreements_fixed = 0
    disagreements_harmed = 0
    disagreements_correctly_unchanged = 0
    disagreements_incorrectly_unchanged = 0

    # === Prepare PCA on raw image features ===
    flat_train_imgs = train_images_raw.reshape(len(train_images_raw), -1)
    norm_flat_train_imgs = flat_train_imgs / np.linalg.norm(flat_train_imgs, axis=1, keepdims=True)
    pca = PCA(n_components=175)
    train_pca_feats = pca.fit_transform(norm_flat_train_imgs)

    # === Fit kNN on PCA space ===
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(train_pca_feats)

    for i, (x, y) in enumerate(testloader):
        x = x.to(device)
        y = y.item()
        conf_score = compute_vector_confidence(softmax_vectors[i], mode)
        total += 1

        with torch.no_grad():
            cnn_logits = cnn(x)
            cnn_pred = torch.argmax(cnn_logits).item()

        if conf_score >= threshold:
            accepted += 1
            if cnn_pred == y:
                correct += 1
            else:
                accepted_but_incorrect += 1
        else:
            rejected += 1
            if cnn_pred == y:
                rejected_but_correct += 1

            # PCA + kNN classification on raw image
            with torch.no_grad():
                test_img = x.cpu().numpy().reshape(1, -1)
                norm_test_img = test_img / np.linalg.norm(test_img, axis=1, keepdims=True)
                test_pca_feat = pca.transform(norm_test_img)
                dists, indices = knn.kneighbors(test_pca_feat)
                neighbor_labels = train_labels[indices[0]]
                knn_pred = np.bincount(neighbor_labels).argmax()

            if knn_pred == cnn_pred:
                if cnn_pred == y:
                    disagreements_correctly_unchanged += 1
                    correct += 1
                else:
                    disagreements_incorrectly_unchanged += 1
            else:
                if knn_pred == y and cnn_pred != y:
                    disagreements_fixed += 1
                    correct += 1
                elif knn_pred != y and cnn_pred == y:
                    disagreements_harmed += 1
                elif knn_pred == y:
                    correct += 1

    # ======= Dump Statistics =======
    print("\n====================== Hybrid Model Performance Dump ======================")
    print(f"🔎 Rejection Threshold Used: {threshold:.4f} | Confidence Mode: {mode}")
    print(f"📌 Total Samples: {total}")

    print(f"\n✅ Accepted by Threshold: {accepted}")
    print(f"   ❌ Accepted but Incorrect: {accepted_but_incorrect}")

    print(f"\n🚫 Rejected by Threshold: {rejected}")
    print(f"   ✅ Rejected but Correct: {rejected_but_correct}")

    print(f"\n🧪 PCA-kNN Effect:")
    print(f"   🛠️ Fixed by PCA-kNN: {disagreements_fixed}")
    print(f"   ❌ Harmed by PCA-kNN: {disagreements_harmed}")
    print(f"   ➖ Correctly PCA-kNN: {disagreements_correctly_unchanged}")
    print(f"   🔄 Incorrectly PCA-kNN: {disagreements_incorrectly_unchanged}")

    print(f"\n📊 Final Accuracy:")
    print(f"   Correct: {correct}/{total} = {correct / total:.4f}")
    print("==========================================================================\n")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_images = load_mnist_images(os.path.join(parent_dir, "data/fashion/train-images-idx3-ubyte.gz"))
    train_labels = load_mnist_labels(os.path.join(parent_dir, "data/fashion/train-labels-idx1-ubyte.gz"))
    test_images = load_mnist_images(os.path.join(parent_dir, "data/fashion/t10k-images-idx3-ubyte.gz"))
    test_labels = load_mnist_labels(os.path.join(parent_dir, "data/fashion/t10k-labels-idx1-ubyte.gz"))

    X_tr = train_images.astype(np.float32) / 255.0
    X_te = test_images.astype(np.float32) / 255.0

    full_train_dataset = FashionMNISTFromNumpy(X_tr, train_labels)
    test_dataset = FashionMNISTFromNumpy(X_te, test_labels)

    train_dataset, val_dataset = random_split(full_train_dataset, [50000, 10000])
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    testloader = DataLoader(test_dataset, batch_size=1)

    cnn = CNN().to(device)
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(20):
        cnn.train()
        total_loss = 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(cnn(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[CNN Epoch {epoch+1}] Avg Loss: {total_loss/len(trainloader):.4f}")

    cnn.eval()
    val_probs, val_preds, val_labels = [], [], []
    with torch.no_grad():
        correct_val = total_val = 0
        for x, y in valloader:
            x = x.to(device)
            out = F.softmax(cnn(x), dim=1)
            pred = out.argmax(1).cpu().numpy()
            val_probs.extend(out.cpu().numpy())
            val_preds.extend(pred)
            val_labels.extend(y.numpy())
            correct_val += (pred == y.numpy()).sum()
            total_val += len(y)
    print(f"✅ CNN Validation Accuracy: {correct_val}/{total_val} = {correct_val / total_val:.4f}")

    # --------------------------
    # ✅ Standard CNN Test Accuracy
    # --------------------------
    cnn.eval()
    correct_test = total_test = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.item()
            logits = cnn(x)
            pred = torch.argmax(logits).item()
            correct_test += (pred == y)
            total_test += 1

    print(
        f"✅ CNN Standalone Test Accuracy: {correct_test}/{total_test} = {correct_test / total_test:.4f}")

    softmax_vectors = collect_softmax_vectors(cnn, valloader, CONFIDENCE_MODE)
    threshold = find_confidence_threshold(softmax_vectors, val_labels, val_preds, CONFIDENCE_MODE, ACCEPTED_RISK, CONFIDENCE_LEVEL)
    print(f"✅ Found Confidence Threshold: {threshold:.4f}")


    test_softmax = []
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)
            out = F.softmax(cnn(x), dim=1)
            test_softmax.append(out.squeeze().cpu().numpy())
    hybrid_inference_with_knn_only(
        cnn = cnn,
        testloader = testloader,
        softmax_vectors = test_softmax,
        threshold = threshold,
        mode = CONFIDENCE_MODE,
        train_images_raw = X_tr,
        train_labels = train_labels
    )