import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# from utils.mnist_reader import load_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
    

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def knn_fashion():
    #load data
    train_images, train_labels = load_mnist("data/fashion", kind="train")
    test_images, test_labels = load_mnist("data/fashion", kind="t10k")
    
    #preprocess data
    X = train_images.reshape(-1, 28 * 28).astype(np.float32)
    X_test_final = test_images.reshape(-1, 28*28).astype(np.float32)
    y = train_labels
    y_test_final = test_labels

    # 归一化：KNN对特征尺度敏感
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test_final = scaler.transform(X_test_final)

    # test-split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=1234)

    k_values = [1, 3, 5, 7, 9]
    val_accuracies = []
    train_accuracies = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_pred = knn.predict(X_train)
        val_pred = knn.predict(X_val)
        acc_train = accuracy_score(y_train, train_pred)
        acc_val = accuracy_score(y_val, val_pred)
        val_accuracies.append(acc_val)
        train_accuracies.append(acc_train)
        print(f"K={k}, train accuracy: {acc_train:.4f}")
        print(f"K={k}, testset accuracy: {acc_val:.4f}")
    
    best_k = k_values[np.argmax(val_accuracies)]
    print(f"\n✅ 选择最佳K={best_k}，在最终测试集上评估：")
    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X, y)  # 用训练 + 验证集重新训练
    y_pred_test = knn_best.predict(X_test_final)

    test_acc = accuracy_score(y_test_final, y_pred_test)
    print(f"🎯 测试集准确率: {test_acc:.4f}")
    # print("\n分类报告L\n", classification_report(y_test_final, y_pred_test))

    train_sizes = [1000, 5000, 10000, 20000, 40000]
    train_scores = []

    for size in train_sizes:
        knn_lc = KNeighborsClassifier(n_neighbors=best_k)
        knn_lc.fit(X_train[:size], y_train[:size])
        val_pred = knn_lc.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        train_scores.append(acc)

    plt.plot(train_sizes, train_scores, marker='o')
    plt.title("KNN Learning Curve (validtion accuracy)")
    plt.xlabel("samples")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    knn_fashion()
