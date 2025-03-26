from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import copy

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# 2) Restrict to first 20 features
X = X[:, :20] / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train + np.random.normal(0, 0.99, X_train.shape)

# 4) Train a Single Decision Tree with Depth 8
base_tree = DecisionTreeClassifier(
    max_depth=None,
    max_leaf_nodes=2**6,
    min_samples_split=2,
    min_samples_leaf=1,
    criterion="gini",
    splitter="random", 
    random_state=42
)

print("options for a dt: ", dir(base_tree))


base_tree.fit(X_train, y_train)

# 5) Copy the trained tree 1000 times for a "Random Forest"
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)

# Manually set missing attributes required by scikit-learn
rf_model.n_classes_ = len(np.unique(y_train))
rf_model.classes_ = np.unique(y_train)
rf_model.n_outputs_ = 1

rf_model.estimators_ = [copy.deepcopy(base_tree) for _ in range(1000)]

# 6) Evaluate model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with replicated tree: {accuracy:.4f}")

# 7) Save the replicated model
with open("rf_cloned_tree.pkl", "wb") as file:
    pickle.dump(rf_model, file)
print("Model saved as rf_cloned_tree.pkl")
