import os
import gzip
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from scipy.stats import binom
from sklearn.neighbors import NearestNeighbors

CONFIDENCE_MODE = "SR"
ACCEPTED_RISK = 0.01
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

class PunchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),  # ← New layer
            nn.Conv2d(16, 1, kernel_size=3, padding=1), nn.Tanh()     # Output perturbation ∈ [-1, 1]
        )

    def forward(self, x):
        perturb = self.net(x)
        return torch.clamp(perturb, -0.5, 0.5)


def get_rejected_train_data(cnn, loader, threshold, mode = "SR"):
    rejected_images, rejected_labels = [], []
    cnn.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = cnn(x)
            probs = F.softmax(logits, dim = 1)
            conf = torch.max(probs, dim = 1).values

            for i in range(x.size(0)):
                if conf[i].item() < threshold:
                    rejected_images.append(x[i].cpu())
                    rejected_labels.append(y[i])

    return torch.stack(rejected_images), torch.tensor(rejected_labels)

def get_rejected_and_disagreeing_train_data(cnn, loader, threshold, train_logits, train_labels, mode="SR"):
    cnn.eval()
    rejected_imgs, rejected_lbls = [], []

    # Fit kNN on normalized logits
    norm_logits = train_logits / np.linalg.norm(train_logits, axis=1, keepdims=True)
    knn = NearestNeighbors(n_neighbors=15)
    knn.fit(norm_logits)
    count_selected = 0
    with torch.no_grad():
        idx_offset = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = cnn(x)
            probs = F.softmax(logits, dim=1)
            conf = torch.max(probs, dim=1).values
            preds = torch.argmax(probs, dim=1)

            for i in range(x.size(0)):
                conf_i = conf[i].item()
                pred_i = preds[i].item()
                y_i = y[i].item()

                logit_i = logits[i].cpu().numpy().reshape(1, -1)
                logit_i = logit_i / np.linalg.norm(logit_i)
                dists, indices = knn.kneighbors(logit_i)
                neighbor_labels = train_labels[indices[0]]
                knn_pred = np.bincount(neighbor_labels).argmax()

                if conf_i < threshold and knn_pred != pred_i:
                    rejected_imgs.append(x[i].cpu())
                    rejected_lbls.append(y[i].cpu())
                    count_selected += 1

            idx_offset += x.size(0)
        print(
            f"✅ Selected {count_selected} training samples for PunchNet (CNN rejected + kNN disagrees)")
    return torch.stack(rejected_imgs), torch.tensor(rejected_lbls)

def train_punchnet(punchnet, cnn, images, labels, num_epochs=20, learning_rate=1e-3):
    punchnet.train()
    cnn.eval()

    optimizer = torch.optim.Adam(punchnet.parameters(), lr=learning_rate, weight_decay=1e-4)
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
            x_perturbed = torch.clamp(x + perturb, 0, 1)
            logits = cnn(x_perturbed)
            pred = torch.argmax(logits, dim=1)

            # Main classification loss
            loss = loss_fn(logits, y)

            # Evaluation logic
            cnn_wrong = (orig_pred != y)
            unchanged = (pred == orig_pred)

            fixed += ((cnn_wrong) & (pred == y)).sum().item()
            harmed += ((~cnn_wrong) & (pred != y)).sum().item()
            unchanged_correct += ((~cnn_wrong) & (unchanged)).sum().item()
            unchanged_incorrect += ((cnn_wrong) & (unchanged)).sum().item()

            # Regularization penalties
            harm_penalty = harmed / x.size(0)
            unchanged_penalty = unchanged_incorrect / x.size(0)

            total = loss + 0.5 * harm_penalty + 0.5 * unchanged_penalty

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            total_loss += total.item()

        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            print(f"\n[PunchNet Epoch {epoch+1}] Loss: {total_loss / len(loader):.4f}")
            print(f"   🛠️ Fixed: {fixed}")
            print(f"   ❌ Harmed: {harmed}")
            print(f"   ➖ Correctly Unchanged: {unchanged_correct}")
            print(f"   🔄 Incorrectly Unchanged: {unchanged_incorrect}")

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

def hybrid_inference_with_knn_and_punchnet(cnn, punchnet, testloader, softmax_vectors, threshold, mode, train_logits, train_labels):
    cnn.eval()
    punchnet.eval()

    total = correct = 0
    accepted = rejected = punched = 0
    accepted_but_incorrect = 0
    rejected_but_correct = 0

    cnn_knn_agree_correct = 0
    cnn_knn_agree_incorrect = 0
    punched_data_points = 0

    disagreements_fixed = 0
    disagreements_harmed = 0
    disagreements_correctly_unchanged = 0
    disagreements_incorrectly_unchanged = 0

    # Prepare kNN
    train_logits = train_logits / np.linalg.norm(train_logits, axis=1, keepdims=True)
    knn = NearestNeighbors(n_neighbors=15)
    knn.fit(train_logits)

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

            # Run kNN
            with torch.no_grad():
                test_feat = cnn_logits.cpu().numpy().reshape(1, -1)
                test_feat = test_feat / np.linalg.norm(test_feat, axis=1, keepdims=True)
                dists, indices = knn.kneighbors(test_feat)
                neighbor_labels = train_labels[indices[0]]
                knn_pred = np.bincount(neighbor_labels).argmax()

            if knn_pred == cnn_pred:
                if cnn_pred == y:
                    cnn_knn_agree_correct += 1
                    correct += 1
                else:
                    cnn_knn_agree_incorrect += 1
            else:
                # CNN and kNN disagree → use PunchNet
                punched += 1
                punched_data_points += 1
                with torch.no_grad():
                    perturb = punchnet(x)
                    x_perturbed = torch.clamp(x + perturb, 0, 1)
                    punch_pred = torch.argmax(cnn(x_perturbed)).item()

                # Was CNN wrong before punching?
                cnn_wrong = (cnn_pred != y)
                # Is PunchNet prediction same as before?
                unchanged = (punch_pred == cnn_pred)

                if unchanged:
                    if cnn_pred == y:
                        disagreements_correctly_unchanged += 1
                        correct += 1
                    else:
                        disagreements_incorrectly_unchanged += 1
                else:
                    if cnn_wrong and punch_pred == y:
                        disagreements_fixed += 1
                        correct += 1
                    elif not cnn_wrong and punch_pred != y:
                        disagreements_harmed += 1
                    elif punch_pred == y:
                        correct += 1

    # ======= Dump Statistics =======
    print("\n====================== Hybrid Model Performance Dump ======================")
    print(f"🔎 Rejection Threshold Used: {threshold:.4f} | Confidence Mode: {mode}")
    print(f"📌 Total Samples: {total}")

    print(f"\n✅ Accepted by CNN: {accepted}")
    print(f"   ❌ Accepted but Incorrect: {accepted_but_incorrect}")

    print(f"\n🚫 Rejected by CNN: {rejected}")
    print(f"   ✅ Rejected but Correct: {rejected_but_correct}")

    print(f"\n📊 CNN–kNN Agreement vs. Disagreement (on rejected samples):")
    agreement_total = cnn_knn_agree_correct + cnn_knn_agree_incorrect
    print(f"   🤝 Agreement Total: {agreement_total}")
    print(f"      🔄 CNN Errors Missed by Agreement: {cnn_knn_agree_incorrect}")
    print(f"   ⚡️ Disagreement Total (Sent to PunchNet): {punched_data_points}")

    print(f"\n🧪 PunchNet Effect (on disagreements only):")
    print(f"   🛠️ Fixed by PunchNet: {disagreements_fixed}")
    print(f"   ❌ Harmed by PunchNet: {disagreements_harmed}")
    print(f"   ➖ Correctly Unchanged: {disagreements_correctly_unchanged}")
    print(f"   🔄 Incorrectly Unchanged: {disagreements_incorrectly_unchanged}")

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

    softmax_vectors = collect_softmax_vectors(cnn, valloader, CONFIDENCE_MODE)
    threshold = find_confidence_threshold(softmax_vectors, val_labels, val_preds, CONFIDENCE_MODE, ACCEPTED_RISK, CONFIDENCE_LEVEL)
    print(f"✅ Found Confidence Threshold: {threshold:.4f}")

    logits_train = []
    for x, _ in trainloader:
        x = x.to(device)
        with torch.no_grad():
            logits = cnn(x).cpu().numpy()
        logits_train.extend(logits)
    train_logits = np.array(logits_train)
    train_logits = train_logits / np.linalg.norm(train_logits, axis = 1, keepdims = True)
    train_labels_array = train_labels[:len(train_logits)]

    # Collect rejected training samples
    rejected_images, rejected_labels = get_rejected_and_disagreeing_train_data(
        cnn, trainloader, threshold, train_logits, train_labels_array, CONFIDENCE_MODE)
    # Create and train PunchNet
    punchnet = PunchNet().to(device)
    train_punchnet(punchnet, cnn, rejected_images, rejected_labels, num_epochs = 100)

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

    test_softmax = []
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)
            out = F.softmax(cnn(x), dim=1)
            test_softmax.append(out.squeeze().cpu().numpy())
    # Run the hybrid inference
    hybrid_inference_with_knn_and_punchnet(cnn, punchnet, testloader, test_softmax, threshold,
                                           CONFIDENCE_MODE, train_logits, train_labels_array)