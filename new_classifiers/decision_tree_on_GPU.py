import gzip
import numpy as np
import struct
import os
import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

# load data
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
        return images

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# load files
dirname = os.path.dirname(__file__)
parent_dir = os.path.dirname(dirname)
train_images_filename = os.path.join(parent_dir, "data\\fashion\\train-images-idx3-ubyte.gz")
test_images_filename = os.path.join(parent_dir, "data\\fashion\\t10k-images-idx3-ubyte.gz")
train_label_filename = os.path.join(parent_dir, "data\\fashion\\train-labels-idx1-ubyte.gz")
test_label_filename = os.path.join(parent_dir, "data\\fashion\\t10k-labels-idx1-ubyte.gz")

train_images = load_mnist_images(train_images_filename)
test_images = load_mnist_images(test_images_filename)
train_labels = load_mnist_labels(train_label_filename)
test_labels = load_mnist_labels(test_label_filename)

# reshape
X_tr = train_images.reshape(train_images.shape[0], -1).astype(np.float32)
y_tr = train_labels.astype(np.int32)
X_te = test_images.reshape(test_images.shape[0], -1).astype(np.float32)
y_te = test_labels.astype(np.int32)

# fix split seed
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED)

# fix Optuna randomness
sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
optuna.logging.set_verbosity(optuna.logging.INFO)

def objective(trial):
    params = {
        "tree_method": "gpu_hist",
        "gpu_id": 0,
        "booster": "gbtree",
        "n_estimators": 1,
        "learning_rate": 1.0,
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "objective": "multi:softmax",
        "num_class": 10,
        "random_state": SEED  # XGBoost seed
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return acc

# run tuning
study.optimize(objective, n_trials=30)

# best model
best_params = study.best_params
best_params.update({
    "tree_method": "gpu_hist",
    "gpu_id": 0,
    "booster": "gbtree",
    "n_estimators": 1,
    "learning_rate": 1.0,
    "objective": "multi:softmax",
    "num_class": 10,
    "random_state": SEED
})

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_tr, y_tr)
preds = final_model.predict(X_te)
print("Test Accuracy:", accuracy_score(y_te, preds))
print("Best Params:", study.best_params)
