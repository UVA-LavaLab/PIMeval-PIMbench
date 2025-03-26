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

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)  # Convert labels to int

X = X[:, :20]

# Normalize pixel values (optional, but improves performance)
X = X / 255.0  # Scale pixels between 0 and 1

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pickle_name = "rf_cloned_tree.pkl"
with open(model_pickle_name, "rb") as file:
    loaded_model = pickle.load(file)


single_sample = X_test[0].reshape(1, -1)
start_time = time.time()
single_prediction = loaded_model.predict(single_sample)
end_time = time.time()
execution_time = (end_time - start_time) * 1000

# Evaluate model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Random Forest Model Accuracy: {accuracy:.2f}")
print(f"Inference Execution Time: {execution_time:.4f} ms")

# 3) Use the loaded model to make predictions


#single_prediction_loaded = loaded_model.predict(single_sample)
#print(f"Loaded model's prediction: {single_prediction_loaded}")

leaf_counts = [estimator.get_n_leaves() for estimator in loaded_model.estimators_]
print("leng of sorted leafs: ", len(sorted(leaf_counts)))
print(f"Number of leafs: {set(leaf_counts)}")

if all(leaf_count == 2**6 for leaf_count in leaf_counts):
    print("PASS: All trees have exactly 2**6 leaf nodes.")
else:
    print("FAIL: Some trees do not have exactly 2**6 leaf nodes.")