import gzip
import os
import struct

import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binom
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.decomposition import PCA

CONFIDENCE_MODE = "SR"
ACCEPTED_RISK = 0.01
CONFIDENCE_LEVEL = 0.95
BATCH_SIZE = 64

import os
import random
import numpy as np
import torch

# ✅ Set a fixed seed
SEED = 42

# ✅ For `random`, `numpy`, and `torch`
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ✅ If using CUDA, ensure deterministic behavior
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

def hybrid_inference_with_mlp(cnn, testloader, softmax_vectors, threshold, mode,
                               pca, back_up_classifier, back_up_threshold):

    cnn.eval()

    total = correct = 0
    accepted = rejected = 0
    accepted_but_incorrect = 0
    rejected_but_correct = 0

    fixed = harmed = unchanged_correct = unchanged_incorrect = 0

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

            # MLP on PCA-transformed image
            test_img = x.cpu().numpy().reshape(1, -1)
            norm_test_img = test_img / np.linalg.norm(test_img, axis=1, keepdims=True)
            test_pca_feat = pca.transform(norm_test_img)
            probs = back_up_classifier.predict_proba(test_pca_feat)[0]
            pred = np.argmax(probs)
            conf = np.max(probs)

            if conf >= back_up_threshold:
                used_pred = pred
            else:
                used_pred = cnn_pred

            if used_pred == cnn_pred:
                if cnn_pred == y:
                    unchanged_correct += 1
                    correct += 1
                else:
                    unchanged_incorrect += 1
            else:
                if used_pred == y and cnn_pred != y:
                    fixed += 1
                    correct += 1
                elif used_pred != y and cnn_pred == y:
                    harmed += 1
                elif used_pred == y:
                    correct += 1

    # ======= Dump Statistics =======
    print("\n====================== Hybrid Model (MLP w/ Confidence Gate) ======================")
    print(f"🔎 Rejection Threshold Used: {threshold:.4f} | CNN Confidence Mode: {mode}")
    print(f"🔒 Backup Classifier Threshold: {back_up_threshold:.2f}")
    print(f"📌 Total Samples: {total}")

    print(f"\n✅ Accepted by Threshold: {accepted}")
    print(f"   ❌ Accepted but Incorrect: {accepted_but_incorrect}")

    print(f"\n🚫 Rejected by Threshold: {rejected}")
    print(f"   ✅ Rejected but Correct: {rejected_but_correct}")

    print(f"\n🧪 MLP Effect on Rejected:")
    print(f"   🛠️ Fixed by MLP: {fixed}")
    print(f"   ❌ Harmed by MLP: {harmed}")
    print(f"   ➖ Correctly Unchanged: {unchanged_correct}")
    print(f"   🔄 Incorrectly Unchanged: {unchanged_incorrect}")

    print(f"\n📊 Final Accuracy:")
    print(f"   Correct: {correct}/{total} = {correct / total:.4f}")
    print("==========================================================================\n")

#

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

    from torch.utils.data import random_split


    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [50000, 10000],
        generator = torch.Generator().manual_seed(SEED)  # ensure reproducible split
    )
    val_images_raw = X_tr[val_dataset.indices]
    val_labels = np.array(train_labels)[val_dataset.indices]
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, worker_init_fn=lambda _: np.random.seed(SEED))
    valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    testloader = DataLoader(test_dataset, batch_size=1)

    cnn = CNN().to(device)
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1):
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

    from sklearn.neural_network import MLPClassifier as SklearnMLP


    # === Flatten and normalize all training images ===
    flat_train_imgs = X_tr.reshape(len(X_tr), -1)
    norm_flat_train_imgs = flat_train_imgs / np.linalg.norm(flat_train_imgs, axis = 1,
                                                            keepdims = True)

    # === Fit PCA on full training data ===
    pca = PCA(n_components = 75)
    train_pca_feats = pca.fit_transform(norm_flat_train_imgs)

    # === Get CNN softmax vectors on training data ===
    trainloader_eval = DataLoader(train_dataset, batch_size = BATCH_SIZE)
    train_softmax_vectors = collect_softmax_vectors(cnn, trainloader_eval, CONFIDENCE_MODE)

    # === Select rejected training points based on threshold ===
    rejected_feats = []
    rejected_labels = []

    # Get corresponding labels from the original index (train_dataset.dataset = full_train_dataset)
    true_train_labels = np.array(train_labels)[train_dataset.indices]

    for v, conf_vec, label in zip(train_pca_feats[train_dataset.indices], train_softmax_vectors,
                                  true_train_labels):
        if compute_vector_confidence(conf_vec, CONFIDENCE_MODE) < threshold:
            rejected_feats.append(v)
            rejected_labels.append(label)

    rejected_feats = np.array(rejected_feats)
    rejected_labels = np.array(rejected_labels)

    # === Prepare PCA features for validation set ===
    flat_val_imgs = val_images_raw.reshape(len(val_images_raw), -1)
    norm_flat_val_imgs = flat_val_imgs / np.linalg.norm(flat_val_imgs, axis = 1, keepdims = True)
    val_pca_feats = pca.transform(norm_flat_val_imgs)

    from sklearn.ensemble import RandomForestClassifier
    print(f"Training Random Forest on {len(rejected_feats)} rejected samples")

    # === Train Random Forest Classifier ===
    rf_clf = RandomForestClassifier(
        n_estimators = 300,  # Number of trees
        max_depth = 45,  # Limit tree depth to avoid overfitting
        random_state = 42,
        n_jobs = -1  # Use all CPU cores
    )
    rf_clf.fit(rejected_feats, rejected_labels)



    # === Get RF predicted probabilities and confidences ===
    rf_val_probs = rf_clf.predict_proba(val_pca_feats)
    rf_val_preds = np.argmax(rf_val_probs, axis = 1)
    rf_val_confidences = np.max(rf_val_probs, axis = 1)

    # === Compute confidence threshold for RF using same method ===
    rf_conf_threshold = find_confidence_threshold(
        vecs = rf_val_probs,
        labels = val_labels,
        preds = rf_val_preds,
        mode = CONFIDENCE_MODE,
        accepted_risk = 0.03,
        confidence_level = 0.9
    )
    print(f"✅ Found Random Forest Confidence Threshold: {rf_conf_threshold:.4f}")

    test_softmax = []
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)
            out = F.softmax(cnn(x), dim=1)
            test_softmax.append(out.squeeze().cpu().numpy())
    hybrid_inference_with_mlp(
        cnn = cnn,
        testloader = testloader,
        softmax_vectors = test_softmax,
        threshold = threshold,
        mode = CONFIDENCE_MODE,
        pca = pca,
        back_up_classifier = rf_clf,
        back_up_threshold = rf_conf_threshold
    )
