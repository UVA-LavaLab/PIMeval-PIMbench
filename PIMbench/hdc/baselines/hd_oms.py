import argparse
import math
import torch
import tqdm
import model
import utils
import time

# %% Parsing input arguments
parser = argparse.ArgumentParser(description="HDC-based database search")
parser.add_argument("--output-ref-file", type=str, default="../PIM/ref.csv", help="Output CSV file for the reference encoding")
parser.add_argument("--output-query-file", type=str, default="../PIM/query.csv", help="Output CSV file for the query encoding")
parser.add_argument("--warmup", type=int, default=1024, help="Warmup iterations")
parser.add_argument("--cuda", action="store_true", default=False, help="Enable GPU support for PyTorch")
parser.add_argument("--charge", type=int, default=2, help="Charge state (Load Mass Spec. Dataset)")
parser.add_argument("--ref-dataset", type=str, choices=["iprg", "hcd"], default="hcd", help="Reference dataset to benchmark (e.g., iPRG2012 or Massive Human HCD)")
parser.add_argument("--n-ref-test", type=int, default=8192, help="Number of reference samples for quick evaluation")
parser.add_argument("--n-query-test", type=int, default=8192, help="Number of query samples for quick evaluation")
parser.add_argument("--topk", type=int, default=5, help="Number of top-k results to search")
parser.add_argument("--n-lv", type=int, default=64, help="Quantization levels for level HVs")
parser.add_argument("--n-dim", type=int, default=8192, help="HV dimension (can range from 1k to 16k)")
parser.add_argument("--binary", action="store_true", default=True, help="Enable binary representation for HVs")
args = parser.parse_args()

# %% Load the dataset
if args.ref_dataset == "iprg":
    dim_spectra = 34976
    ref_fname = (
        f"../Dataset/human_yeast_targetdecoy_vec_{dim_spectra}.charge{args.charge}.npz"
    )
    query_fname = f"../Dataset/iPRG2012_vec_{dim_spectra}.charge{args.charge}.npz"
elif args.ref_dataset == "hcd":
    dim_spectra = 27981
    query_idx = 0  # pick one of the query files to test
    ref_fname = f"../Dataset/oms/ref/massive_human_hcd_unique_targetdecoy_vec_{dim_spectra}.charge{args.charge}.npz"
    query_fname = f"../Dataset/oms/query/{utils.hdc_query_list[query_idx]}_vec_{dim_spectra}.charge{args.charge}.npz"
else:
    raise NotImplementedError("Dataset not implemented")
ds_ref, ds_query = utils.load_dataset(
    name="OMS_iPRG_demo", path=[ref_fname, query_fname]
)
n_ref, n_query = len(ds_ref["pr_mzs"]), len(ds_query["pr_mzs"])

# %% HDC Model
n_id = dim_spectra  # Num of id HVs
hdc_model = model.HDC_ID_LV(
    n_class=n_ref,
    n_lv=args.n_lv,
    n_id=n_id,
    n_dim=args.n_dim,
    method_id_lv="cyclic",
    binary=args.binary,
)

# %% Data Quantization
ds_ref["levels"][ds_ref["levels"] == -1] = 0
ds_ref_levels_quantized = model.min_max_quantize(
    torch.tensor(ds_ref["levels"]), int(math.log2(args.n_lv) - 1)
)
ds_ref_idxs = torch.tensor(ds_ref["idxs"])
ds_query["levels"][ds_query["levels"] == -1] = 0
ds_query_levels_quantized = model.min_max_quantize(
    torch.tensor(ds_query["levels"]), int(math.log2(args.n_lv) - 1)
)
ds_query_idxs = torch.tensor(ds_query["idxs"])

# %% HDC Encoding Step for Database Pre-building
ref_enc = hdc_model.encode(
    {"lv": ds_ref_levels_quantized[:args.n_ref_test], "idx": ds_ref_idxs[:args.n_ref_test]},
    dense=False,
)

# %% HDC Encoding Step for Querying
query_enc = hdc_model.encode(
    {
        "lv": ds_query_levels_quantized[:args.n_query_test],
        "idx": ds_query_idxs[:args.n_query_test],
    },
    dense=False,
)

# %% Write the query and reference tensors to csv file
utils.write_tensor_to_csv(ref_enc, args.output_ref_file)
utils.write_tensor_to_csv(query_enc, args.output_query_file)

# %% Move data to the device
device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
query_enc = query_enc.to(device).float()
ref_enc = ref_enc.to(device).float()

# %% Database Search
start_time = time.time()
for i in range(args.warmup):
    ip = torch.matmul(query_enc, ref_enc.T)
    if not hdc_model.binary:
        dist = ip / ref_enc.float().norm(dim=1)
    sim, pred = torch.topk(ip, k=args.topk, dim=-1)
end_time = time.time()
execution_time = (end_time - start_time) / args.warmup

# %% Print informational messages
print(f"[INFO]: device = {device}")
print(f"[INFO] Execution time: {execution_time*1.0e+3:.6f} milliseconds")
print(f"[INFO]: {args.n_ref_test} of {n_ref} references and {args.n_query_test} of {n_query} queries are used for testing")
print("[INFO]: Topk index results:\n", pred)

