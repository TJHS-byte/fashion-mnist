import os
import gzip
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.neighbors import NearestNeighbors

CONFIDENCE_MODE = "SR"
ACCEPTED_RISK = 0.05
CONFIDENCE_LEVEL = 0.95
BATCH_SIZE = 64

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------- Model Definitions -----------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

class PunchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(16, 1, 3, padding=1), nn.Tanh()
        )

    def forward(self, x):
        perturb = self.net(x)
        return torch.clamp(perturb, -1, 1)

# ----------------- Dataset Loader -----------------
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def compute_vector_confidence(prob_vector, mode):
    if mode == "SR":
        return np.max(prob_vector)
    raise NotImplementedError()

def find_threshold(softmax_vectors, labels, mode, accepted_risk):
    sorted_confidences = sorted((compute_vector_confidence(p, mode), i) for i, p in enumerate(softmax_vectors))
    threshold_idx = int((1 - accepted_risk) * len(softmax_vectors))
    return sorted_confidences[threshold_idx][0]

# ----------------- CNN Training -----------------
def train_cnn(cnn, trainloader, valloader, num_epochs=20):
    cnn.train()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = cnn(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[CNN Epoch {epoch+1}] Avg Loss: {total_loss/len(trainloader):.4f}")

    cnn.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in valloader:
            x, y = x.to(device), y.to(device)
            pred = torch.argmax(cnn(x), dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"✅ CNN Validation Accuracy: {correct}/{total} = {correct / total:.4f}")

def evaluate_cnn_on_test(cnn, testloader):
    cnn.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            pred = torch.argmax(cnn(x), dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"✅ CNN Standalone Test Accuracy: {correct}/{total} = {correct / total:.4f}")

# ----------------- PunchNet Training -----------------
def train_punchnet(punchnet, cnn, images, labels, num_epochs=60):
    punchnet.train()
    cnn.eval()
    optimizer = torch.optim.Adam(punchnet.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        fixed = harmed = unchanged_correct = unchanged_incorrect = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                orig_logits = cnn(x)
                orig_pred = torch.argmax(orig_logits, dim=1)
            perturb = punchnet(x)
            x_pert = torch.clamp(x + perturb, 0, 1)
            logits = cnn(x_pert)
            pred = torch.argmax(logits, dim=1)
            loss = loss_fn(logits, y)

            cnn_wrong = (orig_pred != y)
            cnn_right = ~cnn_wrong
            unchanged = (pred == orig_pred)

            is_fixed = cnn_wrong & (pred == y)
            is_harmed = cnn_right & (pred != y)
            is_unchanged_correct = cnn_right & unchanged
            is_unchanged_wrong = cnn_wrong & unchanged

            fixed += is_fixed.sum().item()
            harmed += is_harmed.sum().item()
            unchanged_correct += is_unchanged_correct.sum().item()
            unchanged_incorrect += is_unchanged_wrong.sum().item()

            reward = (-1) * is_fixed.sum().float() / x.size(0)
            sub_reward = (-0.5) * is_unchanged_correct.sum().float() / x.size(0)
            harm_penalty = (+3) * is_harmed.sum().float() / x.size(0)
            unchanged_penalty = (+3) * is_unchanged_wrong.sum().float() / x.size(0)
            total = loss + harm_penalty + unchanged_penalty + reward + sub_reward

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            total_loss += total.item()

        if (epoch + 1) % 10 == 0:
            print(f"\n[PunchNet Epoch {epoch+1}] Loss: {total_loss / len(loader):.4f}")
            print(f"   🛠️ Fixed: {fixed}")
            print(f"   ❌ Harmed: {harmed}")
            print(f"   ➖ Correctly Unchanged: {unchanged_correct}")
            print(f"   🔄 Incorrectly Unchanged: {unchanged_incorrect}")

# ----------------- Hybrid Inference -----------------
def hybrid_inference_with_knn_and_punchnet(cnn, punchnet, testloader, test_probs, threshold, mode, train_logits, train_labels_np):
    cnn.eval()
    punchnet.eval()
    correct = accepted = accepted_wrong = rejected = rejected_correct = 0
    agreement_total = agreement_missed_error = fixed = harmed = unchanged_correct = unchanged_wrong = 0
    disagreement_samples = []
    nbrs = NearestNeighbors(n_neighbors=5).fit(train_logits)

    with torch.no_grad():
        for i, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            probs = F.softmax(cnn(x), dim=1)
            conf = compute_vector_confidence(probs.cpu().numpy()[0], mode)
            pred = probs.argmax(dim=1)
            true_label = y.item()

            if conf >= threshold:
                accepted += 1
                if pred.item() == true_label:
                    correct += 1
                else:
                    accepted_wrong += 1
                continue

            rejected += 1
            prob_vec = probs.cpu().numpy()[0]
            _, indices = nbrs.kneighbors(prob_vec.reshape(1, -1))
            knn_votes = train_labels_np[indices[0]]
            knn_pred = np.bincount(knn_votes).argmax()

            if knn_pred == pred.item():
                agreement_total += 1
                if pred.item() != true_label:
                    agreement_missed_error += 1
                if pred.item() == true_label:
                    correct += 1
                    rejected_correct += 1
                continue

            disagreement_samples.append((x, y, pred))
            x_punch = torch.clamp(x + punchnet(x), 0, 1)
            new_probs = F.softmax(cnn(x_punch), dim=1)
            new_pred = new_probs.argmax(dim=1)

            orig_correct = (pred.item() == true_label)
            new_correct = (new_pred.item() == true_label)

            if not orig_correct and new_correct:
                fixed += 1
                correct += 1
            elif orig_correct and not new_correct:
                harmed += 1
            elif orig_correct and new_correct:
                unchanged_correct += 1
                correct += 1
            else:
                unchanged_wrong += 1

    print("\n====================== Hybrid Model Performance Dump ======================")
    print(f"🔎 Rejection Threshold Used: {threshold:.4f} | Confidence Mode: {mode}")
    print(f"📌 Total Samples: {accepted + rejected}")
    print(f"✅ Accepted by CNN: {accepted}")
    print(f"   ❌ Accepted but Incorrect: {accepted_wrong}")
    print(f"🚫 Rejected by CNN: {rejected}")
    print(f"   ✅ Rejected but Correct: {rejected_correct}")
    print(f"📊 CNN–kNN Agreement: {agreement_total} (Missed errors: {agreement_missed_error})")
    print(f"⚡️ Disagreement Total (Sent to PunchNet): {len(disagreement_samples)}")
    print(f"🛠️ Fixed by PunchNet: {fixed}")
    print(f"❌ Harmed by PunchNet: {harmed}")
    print(f"➖ Correctly Unchanged: {unchanged_correct}")
    print(f"🔄 Incorrectly Unchanged: {unchanged_wrong}")
    print(f"📊 Final Accuracy: {correct}/{accepted + rejected} = {correct / (accepted + rejected):.4f}")
    print("==========================================================================")
if __name__ == '__main__':
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    train_images = load_mnist_images(os.path.join(parent_dir, "data/fashion/train-images-idx3-ubyte.gz"))
    train_labels = load_mnist_labels(os.path.join(parent_dir, "data/fashion/train-labels-idx1-ubyte.gz"))
    test_images = load_mnist_images(os.path.join(parent_dir, "data/fashion/t10k-images-idx3-ubyte.gz"))
    test_labels = load_mnist_labels(os.path.join(parent_dir, "data/fashion/t10k-labels-idx1-ubyte.gz"))

    X_tr = train_images.astype(np.float32) / 255.0
    X_te = test_images.astype(np.float32) / 255.0

    class FashionMNISTFromNumpy(Dataset):
        def __init__(self, images, labels):
            self.images = torch.tensor(images, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.int64)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    full_train_dataset = FashionMNISTFromNumpy(X_tr, train_labels)
    test_dataset = FashionMNISTFromNumpy(X_te, test_labels)
    train_dataset, val_dataset = random_split(full_train_dataset, [50000, 10000])
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    testloader = DataLoader(test_dataset, batch_size=1)

    cnn = CNN().to(device)
    train_cnn(cnn, trainloader, valloader, num_epochs=10)
    evaluate_cnn_on_test(cnn, testloader)

    val_probs = []
    cnn.eval()
    with torch.no_grad():
        for x, _ in valloader:
            x = x.to(device)
            probs = F.softmax(cnn(x), dim=1)
            val_probs.extend(probs.cpu().numpy())

    val_labels = [y for _, y in val_dataset]
    threshold = find_threshold(val_probs, val_labels, CONFIDENCE_MODE, ACCEPTED_RISK)
    print(f"🔎 Threshold selected: {threshold:.4f}")

    train_probs, train_preds = [], []
    cnn.eval()
    with torch.no_grad():
        for x, _ in DataLoader(full_train_dataset, batch_size=64):
            x = x.to(device)
            logits = cnn(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            train_probs.extend(probs)
            train_preds.extend(preds)

    confidences = [compute_vector_confidence(p, CONFIDENCE_MODE) for p in train_probs]
    rejected_idx = [i for i, c in enumerate(confidences) if c < threshold]
    train_logits = np.vstack(train_probs)
    nbrs = NearestNeighbors(n_neighbors=5).fit(train_logits)
    _, indices = nbrs.kneighbors(train_logits)
    knn_labels = []
    for idx_group in indices:
        neighbor_labels = [train_preds[i] for i in idx_group]
        knn_labels.append(np.bincount(neighbor_labels).argmax())

    disagreement_idx = [i for i in rejected_idx if train_preds[i] != knn_labels[i]]
    punch_images = torch.tensor(X_tr[disagreement_idx], dtype=torch.float32)
    punch_labels = torch.tensor(np.array(train_labels)[disagreement_idx], dtype=torch.int64)

    punchnet = PunchNet().to(device)
    train_punchnet(punchnet, cnn, punch_images, punch_labels, num_epochs=60)

    test_probs = []
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)
            out = F.softmax(cnn(x), dim=1)
            test_probs.append(out.squeeze().cpu().numpy())

    hybrid_inference_with_knn_and_punchnet(
        cnn, punchnet, testloader, test_probs, threshold,
        CONFIDENCE_MODE, train_logits, train_labels_np=np.array(train_labels)
    )







