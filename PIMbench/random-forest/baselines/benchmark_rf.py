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
    # mW  →  W
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1_000

def train_model(X_train, y_train, num_trees, dt_height):
    # train the model on the CPU, since this time isn't taken into account for evaluation 
    print(f"[INFO] Started training Random Forest with {num_trees} trees, each with height {dt_height}") 

    rf_model = RandomForestClassifier(
        n_estimators=num_trees,
        max_leaf_nodes=2**dt_height,
        criterion="gini",
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    print("[INFO] Finished training Random Forest")
    return rf_model

def eval_cpu(model, input_sample, dt_height):
    # Evaluate the model
    start_time = time.time()
    single_prediction = model.predict(input_sample)  # run and time inference
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000

    print(f"Inference Execution Time: {execution_time:.4f} ms")

    # ensure random forest actually creates the correct sized decision trees
    leaf_counts = [estimator.get_n_leaves() for estimator in model.estimators_]
    num_leaf_nodes = 2**dt_height
    if all(leaf_count == num_leaf_nodes for leaf_count in leaf_counts):
        print(f"PASS: All trees have exactly {num_leaf_nodes} leaf nodes.")
    else:
        print(f"FAIL: Some trees do not have exactly {num_leaf_nodes} leaf nodes.")

def eval_gpu(model, input_sample):
    import cupy as cp

    handle = _get_nvml_handle()
    n_samples = input_sample.shape[0]

    reps = 10
    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start_energy = _energy_joules(handle)
    if start_energy is None:
        start_power = _power_watts(handle)

    start.record()
    for _ in range(reps):
        _ = model.predict(input_sample)
    end.record()
    end.synchronize()

    total_elapsed_ms = cp.cuda.get_elapsed_time(start, end)  # ms over all reps
    avg_batch_time_ms = total_elapsed_ms / reps
    avg_sample_time_ms = avg_batch_time_ms / n_samples

    end_energy = _energy_joules(handle)

    print(f"[RESULT] Inference time per batch : {avg_batch_time_ms:.4f} ms")
    print(f"[RESULT] Inference time per sample: {avg_sample_time_ms:.6f} ms")

    if end_energy is not None:
        energy_per_batch_mJ = (end_energy - start_energy) / reps
        print(f"[RESULT] Energy per batch        : {energy_per_batch_mJ:.4f} mJ")
    else:
        avg_power_W = start_power / reps
        energy_per_batch_mJ = (avg_power_W * (avg_batch_time_ms / 1000)) * 1000  # W·s → mJ
        print("[RESULT] Energy counter not supported")
        print(f"[RESULT] Est. energy per batch   : {energy_per_batch_mJ:.4f} mJ")



def main(args):
    print("[INFO] Starting main function")

    # Handle data intilization 
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)  # Convert labels to int

    X = X[:, :args.input_dim]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # make data completely random, so as to ensure we get enough decision splits during training
    X_train = X_train + np.random.normal(0, 0.99, X_train.shape)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    batch_sample = X_test[:1].astype(np.float32)
    single_sample = np.array(batch_sample)  # sample 1 input for evaluation
    
    # train the same sklearn model for both CPU and GPU inference
    rf_model = train_model(X_train, y_train, args.num_trees, args.dt_height)
    
    if args.cuda:
        from cuml import RandomForestClassifier as cuRF
        import gc
        cu_rf_params = {
            'n_estimators': args.num_trees,
            'max_depth': 2**args.dt_height,
            }
        cu_rf = cuRF(**cu_rf_params)
        cu_rf.fit(X_train, y_train)
        print("[INFO] Finished training cuML Random Forest")
        # fil_model = ForestInference.load_from_sklearn(
        #     rf_model,
        #     output_class=True,
        # )
        eval_gpu(cu_rf, single_sample)
        del cu_rf
        gc.collect()
        pynvml.nvmlShutdown()
    else:
        # if cuda not set, train and eval on CPU
        eval_cpu(rf_model, single_sample, args.dt_height)
        


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
