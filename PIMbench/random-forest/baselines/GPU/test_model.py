from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import time
import pickle

from cuml.ensemble import RandomForestClassifier as cuRF
# from cuml.tree import DecisionTreeClassifier as cuDT
import cupy as cp

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)  # Convert labels to int

X = X[:, :20]

# Normalize pixel values (optional, but improves performance)
X = X / 255.0  # Scale pixels between 0 and 1

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train + np.random.normal(0, 0.99, X_train.shape)

base_tree = cuRF(
    n_estimators=1000,  # Single tree
    max_depth=7,
    max_leaves=2**6,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
    random_state=42
)


base_tree.fit(X_train, y_train)

rf_model = base_tree

# Evaluate model
single_sample = cp.asarray(X_test[0].reshape(1, -1))  # move sample to GPU

start = cp.cuda.Event()
end = cp.cuda.Event()

start.record()

single_prediction = rf_model.predict(single_sample)
end.record()
end.synchronize()

elapsed_ms = cp.cuda.get_elapsed_time(start, end)
print(f"Inference Execution Time: {elapsed_ms:.4f} ms")

# cuML doesn't give access to individual trees

# leaf_counts = [estimator.get_n_leaves() for estimator in rf_model.estimators_]
# print("leng of sorted leafs: ", len(sorted(leaf_counts)))
# print(f"Number of leafs: {set(leaf_counts)}")

# if all(leaf_count == 2**6 for leaf_count in leaf_counts):
#     print("PASS: All trees have exactly 2**6 leaf nodes.")
# else:
#     print("FAIL: Some trees do not have exactly 2**6 leaf nodes.")