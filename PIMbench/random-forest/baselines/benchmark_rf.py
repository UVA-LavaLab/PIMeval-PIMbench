import time
import copy
import argparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np


def train_cpu(X_train, y_train, num_trees, dt_height):
    print(f"[INFO] Started training Random Forest on CPU with {num_trees} trees, each with height {dt_height}")    
    # train a single decision tree
    base_tree = DecisionTreeClassifier(
        max_depth=None,
        max_leaf_nodes=2**dt_height,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
        splitter="random", 
        random_state=42
    )

    base_tree.fit(X_train, y_train)

    # put the same DT into the random forest to ensure consistency
    rf_model = RandomForestClassifier(n_estimators=num_trees, random_state=42)

    # Manually set missing attributes required by scikit-learn
    rf_model.n_classes_ = len(np.unique(y_train))
    rf_model.classes_ = np.unique(y_train)
    rf_model.n_outputs_ = 1

    rf_model.estimators_ = [copy.deepcopy(base_tree) for _ in range(num_trees)]
    print("[INFO] Finished training Random Forest on CPU")  
    return rf_model

def eval_cpu(model, input_sample):
    # Evaluate the model
    start_time = time.time()
    single_prediction = model.predict(input_sample)  # run and time inference
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000

    print(f"Inference Execution Time: {execution_time:.4f} ms")

    # ensure random forest actually creates the correct sized decision trees
    leaf_counts = [estimator.get_n_leaves() for estimator in model.estimators_]
    num_leaf_nodes = 2**args.dt_height
    if all(leaf_count == num_leaf_nodes for leaf_count in leaf_counts):
        print(f"PASS: All trees have exactly {num_leaf_nodes} leaf nodes.")
    else:
        print(f"FAIL: Some trees do not have exactly {num_leaf_nodes} leaf nodes.")


def train_gpu(X_train, y_train, num_trees, dt_height):
    print(f"[INFO] Started training Random Forest on GPU with {num_trees} trees, each with height {dt_height}")   
    rf_model = cuRF(
        n_estimators=num_trees,
        max_leaves=2**dt_height,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42
    )

    rf_model.fit(X_train, y_train)
    print("[INFO] Finished training Random Forest on GPU") 
    return rf_model

def eval_gpu(model, input_sample):
    single_gpu_sample = cp.asarray(input_sample)  # move sample to GPU
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()

    single_prediction = rf_model.predict(single_sample)
    end.record()
    end.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    print(f"Inference Execution Time: {elapsed_ms:.4f} ms")


def main(args):
    print("[INFO] Starting main function")

    # Handle data intilization 
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)  # Convert labels to int

    X = X[:, :args.input_dim]
    X = X / 255.0  # normalize pixels

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # make data completely random, so as to ensure we get enough decision splits during training
    X_train = X_train + np.random.normal(0, 0.99, X_train.shape)
    single_sample = X_test[0].reshape(1, -1)  # sample 1 input for evaluation
    
    if args.cuda:
        from cuml.ensemble import RandomForestClassifier as cuRF
        import cupy as cp
        
        rf_model = train_gpu(X_train, y_train, args.num_trees, args.dt_height)
        eval_gpu(rf_model, single_sample)
    else:
        # if cuda not set, train and eval on CPU
        rf_model = train_cpu(X_train, y_train, args.num_trees, args.dt_height)
        eval_cpu(rf_model, single_sample)
        


if __name__ == "__main__":
    print("[INFO] Parsing command line arguments")
    parser = argparse.ArgumentParser(description="Random Forest for single inference")
    parser.add_argument("-cuda", action='store_true', help="Use GPU if available")
    parser.add_argument("-input_dim", type=int, default=20, help="Dimension of input and training vectors")
    parser.add_argument("-num_trees", type=int, default=1000, help="Number of decision trees to put in the Random Forest ensemble")
    parser.add_argument("-dt_height", type=int, default=6, help="Height of the each decision tree. 2**dt_height is the number of leaf nodes")
    args = parser.parse_args()
    print("[INFO] Starting the process")
    main(args)
