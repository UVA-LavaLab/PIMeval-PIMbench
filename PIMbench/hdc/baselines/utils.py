# %%
import tqdm
import numpy as np
import torchhd
import torchvision
import os
import csv

DATASETS = ["MNIST", "ISOLET", "EMG_Hand", "UCIHAR", "OMS_iPRG_demo"]

hdc_query_list = [
    "b1906_293T_proteinID_01A_QE3_122212",
    "b1922_293T_proteinID_02A_QE3_122212",
    "b1923_293T_proteinID_03A_QE3_122212",
    "b1924_293T_proteinID_04A_QE3_122212",
    "b1925_293T_proteinID_05A_QE3_122212",
    "b1926_293T_proteinID_06A_QE3_122212",
    "b1927_293T_proteinID_07A_QE3_122212",
    "b1928_293T_proteinID_08A_QE3_122212",
    "b1929_293T_proteinID_09A_QE3_122212",
    "b1930_293T_proteinID_10A_QE3_122212",
    "b1931_293T_proteinID_11A_QE3_122212",
    "b1932_293T_proteinID_12A_QE3_122212",
    "b1937_293T_proteinID_01B_QE3_122212",
    "b1938_293T_proteinID_02B_QE3_122212",
    "b1939_293T_proteinID_03B_QE3_122212",
    "b1940_293T_proteinID_04B_QE3_122212",
    "b1941_293T_proteinID_05B_QE3_122212",
    "b1942_293T_proteinID_06B_QE3_122212",
    "b1943_293T_proteinID_07B_QE3_122212",
    "b1944_293T_proteinID_08B_QE3_122212",
    "b1945_293T_proteinID_09B_QE3_122212",
    "b1946_293T_proteinID_10B_QE3_122212",
    "b1947_293T_proteinID_11B_QE3_122212",
    "b1948_293T_proteinID_12B_QE3_122212",
]

def load_dataset(name, path=None):
    """
    Load dataset by name
    """
    if name in DATASETS:
        if name == "MNIST":
            train = torchvision.datasets.MNIST(
                root="./dataset", train=True, download=True
            )
            test = torchvision.datasets.MNIST(
                root="./dataset", train=False, download=True
            )
        elif name == "ISOLET":
            train = torchhd.datasets.ISOLET(root="./dataset", train=True, download=True)
            test = torchhd.datasets.ISOLET(root="./dataset", train=False, download=True)
        elif name == "EMG_Hand":
            data = torchhd.datasets.EMGHandGestures(root="./dataset", download=True)
            n_split = int(len(data) * 0.75)
            train, test = data[:n_split], data[n_split:]
        elif name == "UCIHAR":
            train = torchhd.datasets.UCIHAR(root="./dataset", train=True, download=True)
            test = torchhd.datasets.UCIHAR(root="./dataset", train=False, download=True)
        elif name == "OMS_iPRG_demo":
            train = np.load(path[0])
            test = np.load(path[1])
    else:
        raise NotImplementedError(f"Dataset {name} not implemented!")

    if name in ["EMG_Hand"]:
        data_train, data_test = (train[0].flatten(1).int(), train[1]), (
            test[0].flatten(1).int(),
            test[1],
        )
    elif name in ["MNIST", "ISOLET", "UCIHAR"]:
        data_train, data_test = (train.data.flatten(1), train.targets), (
            test.data.flatten(1),
            test.targets,
        )
    elif name in ["OMS_iPRG_demo"]:

        def convert_csr_to_dense(csr_info, spectra_idx, spectra_intensities):
            """
            convert data in csr format into dense 2d array
            """
            n_spec = len(csr_info) - 1
            n_pts = np.diff(csr_info).max()

            idxs = np.full((n_spec, n_pts), -1, dtype=np.int32)
            levels = np.full((n_spec, n_pts), -1, dtype=np.float32)
            for i in tqdm.tqdm(range(n_spec)):
                i_start, i_end = csr_info[i : i + 2]

                idxs[i][0 : i_end - i_start] = spectra_idx[i_start:i_end]
                levels[i][0 : i_end - i_start] = spectra_intensities[i_start:i_end]

            return idxs, levels

        train_idxs, train_levels = convert_csr_to_dense(
            train["csr_info"], train["spectra_idx"], train["spectra_intensities"]
        )
        data_train = {
            "idxs": train_idxs,
            "levels": train_levels,
            "pr_mzs": train["pr_mzs"],
        }

        test_idxs, test_levels = convert_csr_to_dense(
            test["csr_info"], test["spectra_idx"], test["spectra_intensities"]
        )
        data_test = {
            "idxs": test_idxs,
            "levels": test_levels,
            "pr_mzs": test["pr_mzs"],
        }

    return data_train, data_test

def write_tensor_to_csv(tensor, filename):
    dim1, dim2 = tensor.shape
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write dimensions
        writer.writerow([dim1, dim2])
        # Write data rows
        for row in tensor.tolist():
            writer.writerow(row)


