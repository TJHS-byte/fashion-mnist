import gzip
import numpy as np
import struct
import os

# load data
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
        return images

dirname = os.path.dirname(__file__)
parent_dir = os.path.dirname(dirname)
train_images_filename = os.path.join(parent_dir, "data\\fashion\\train-images-idx3-ubyte.gz")
train_images = load_mnist_images(train_images_filename)
test_images_filename = os.path.join(parent_dir, "data\\fashion\\t10k-images-idx3-ubyte.gz")
test_images = load_mnist_images(test_images_filename)
def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

train_label_filename = os.path.join(parent_dir, "data\\fashion\\train-labels-idx1-ubyte.gz")
train_labels = load_mnist_labels(train_label_filename)
test_label_filename = os.path.join(parent_dir, "data\\fashion\\t10k-labels-idx1-ubyte.gz")
test_labels = load_mnist_labels(test_label_filename)

X_tr = train_images.reshape(train_images.shape[0], -1)
y_tr = train_labels

X_te = test_images.reshape(test_images.shape[0], -1)
y_te = test_labels

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}


grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    n_jobs=14,
    verbose=2
)

grid.fit(X_tr, y_tr)
chosen_model = grid.best_estimator_
y_pred_tr = chosen_model.predict(X_tr)
y_pred_te = chosen_model.predict(X_te)
print(f"best params = {grid.best_params_}")
print("Decision Tree Train Accuracy:", accuracy_score(y_tr, y_pred_tr))
print("Decision Tree Test Accuracy:", accuracy_score(y_te, y_pred_te))