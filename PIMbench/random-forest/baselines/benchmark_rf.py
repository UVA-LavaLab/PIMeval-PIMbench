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
import pynvml 

def _get_nvml_handle(idx: int = 0):
    """Return an nvmlDevice_t handle (assumes one GPU)."""
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(idx)

def _energy_joules(handle):
    """
    Energy since boot in joules if the counter is supported,
    else returns None.
    """
    try:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) #mJ
    except pynvml.NVMLError_FunctionNotFound:
        return None

def _power_watts(handle):
    # µW  →  W
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1_000

def train_model(X_train, y_train, num_trees, dt_height):
    # train the model on the CPU, since this time isn't taken into account for evaluation 
    print(f"[INFO] Started training Random Forest with {num_trees} trees, each with height {dt_height}") 

    rf_model = RandomForestClassifier(
        n_estimators=num_trees,
        max_leaf_nodes=2**dt_height,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    print("[INFO] Finished training Random Forest")
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

def eval_gpu(model, input_sample):
    import cupy as cp

    handle = _get_nvml_handle()
    start_energy = _energy_joules(handle)
    if start_energy is None:
        # fall back: record instantaneous power just before the kernel
        start_power = _power_watts(handle)

    single_gpu_sample = cp.asarray(input_sample)  # move sample to GPU
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()

    reps = 1000                           # 1000 × 0.7 ms ≈ 0.7 s
    start_energy = _energy_joules(handle)

    for _ in range(reps):
        single_prediction = model.predict(
            single_gpu_sample
        )

    end.record()
    end.synchronize()

    elapsed_ms = cp.cuda.get_elapsed_time(start, end) / reps
    end_energy = _energy_joules(handle)

    # make sure to average by the number of runs
    if end_energy is not None:
        energy_j = (end_energy - start_energy) / reps
        avg_power_w = (energy_j / (elapsed_ms / 1_000)) / reps
        print(f"Inference time   : {elapsed_ms:>8.4f} ms")
        print(f"Energy consumed : {energy_j:>8.4f} mJ")
        print(f"Avg GPU power   : {avg_power_w:>8.2f} W")
    else:
        # counter not supported → crude estimate
        avg_power_w = start_power / reps
        energy_j = (avg_power_w * (elapsed_ms / 1_000) ) / reps
        print(f"Inference time   : {elapsed_ms:>8.4f} ms")
        print("Energy counter   : not supported on this GPU")
        print(f"Est. energy      : {energy_j:>8.4f} mJ "
            f"(using {avg_power_w:.1f} W)")



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
    
    # train the same sklearn model for both CPU and GPU inference
    rf_model = train_model(X_train, y_train, args.num_trees, args.dt_height)
    
    if args.cuda:
        from cuml import ForestInference
        fil_model = ForestInference.load_from_sklearn(
            rf_model,
            output_class=True,
        )
        eval_gpu(fil_model, single_sample)
    else:
        # if cuda not set, train and eval on CPU
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
