# %%
import math
import torch
import numpy as np

import tqdm

torch.manual_seed(0)


def binarize(x):
    return torch.where(x > 0, 1, -1).int()


def min_max_quantize(inp, bits):
    assert bits >= 1, bits
    if bits == 1:
        return torch.sign(inp) - 1
    min_val, max_val = inp.min(), inp.max()

    input_rescale = (inp - min_val) / (max_val - min_val)

    n = math.pow(2.0, bits) - 1
    v = torch.floor(input_rescale * n + 0.5) / n
    v = v * (max_val - min_val) + min_val
    v = torch.round(v * n)
    v = v - v.min()
    return v.int()


def train_init(model, inp_enc, target):
    assert inp_enc.shape[0] == target.shape[0]

    for i in range(model.n_class):
        idx = target == i
        model.class_hvs[i] = (
            binarize(inp_enc[idx].sum(dim=0))
            if model.binary
            else inp_enc[idx].sum(dim=0)
        )


def test(model, inp_enc, target):
    assert inp_enc.shape[0] == target.shape[0]

    # Distance matching
    if model.binary:
        dist = torch.matmul(inp_enc, binarize(model.class_hvs).T)
    else:
        dist = torch.matmul(inp_enc, model.class_hvs.T)
        # dist = dist / model.class_hvs.float().norm(dim=1)

    acc = dist.argmax(dim=1) == target.long()
    acc = acc.float().mean()

    return acc


def train(model, inp_enc, target):
    assert inp_enc.shape[0] == target.shape[0]

    n_samples = inp_enc.shape[0]

    for j in range(n_samples):
        # Distance matching
        if model.binary:
            pred = torch.matmul(inp_enc[j], binarize(model.class_hvs).T).argmax()
        else:
            dist = torch.matmul(inp_enc[j], model.class_hvs.T)
            # dist = dist / model.class_hvs.float().norm(dim=1)
            pred = dist.argmax()

        # if model.binary:
        # pred = torch.matmul(inp_enc[j], model.class_hvs.T).argmax()

        if pred != target[j]:
            model.class_hvs[target[j]] += inp_enc[j]
            model.class_hvs[pred] -= inp_enc[j]


def generate_lv_id_hvs(n_lv, n_id, n_dim, method="random"):
    def gen_cyclic_lv_hvs(n_dim: int, n_lv: int):
        base = np.ones(n_dim)
        base[: n_dim // 2] = -1.0
        l0 = np.random.permutation(base)
        levels = np.zeros((n_lv, n_dim))
        for i in range(n_lv):
            flip = int(int(i / float(n_lv - 1) * n_dim) / 2)
            li = np.copy(l0)
            li[:flip] = l0[:flip] * -1
            levels[i, :] = li
        return levels

    def gen_cyclic_id_hvs(n_dim: int, n_id: int):
        import copy

        n_flip = int(n_dim // 2)
        base = np.random.randint(0, 2, size=n_dim) * 2 - 1

        ids = [copy.copy(base)]
        for _ in range(n_id - 1):
            idx_to_flip = np.random.randint(0, n_dim, size=n_flip)
            base[idx_to_flip] *= -1
            ids.append(copy.copy(base))
        return np.vstack(ids)

    if method == "random":
        hv_lv = torch.randint(0, 2, size=(n_lv, n_dim), dtype=torch.int) * 2 - 1
        hv_id = torch.randint(0, 2, size=(n_id, n_dim), dtype=torch.int) * 2 - 1
    elif method == "cyclic":
        hv_lv = gen_cyclic_lv_hvs(n_dim, n_lv)
        hv_lv = torch.from_numpy(hv_lv).int()

        hv_id = gen_cyclic_id_hvs(n_dim, n_id)
        hv_id = torch.from_numpy(hv_id).int()
    else:
        NotImplementedError(f"Unknown method {method} to generate level and ID HVs")

    return hv_lv, hv_id


class HDC_ID_LV:
    def __init__(
        self, n_class, n_lv, n_id, n_dim, method_id_lv="random", binary=True
    ) -> None:
        self.method_id_lv = method_id_lv
        self.n_class = n_class
        self.n_dim, self.binary = n_dim, binary
        self.n_lv, self.n_id = n_lv, n_id
        self.hv_lv, self.hv_id = generate_lv_id_hvs(
            n_lv, n_id, n_dim, method=method_id_lv
        )

        self.class_hvs = torch.zeros(n_class, n_dim, dtype=torch.int)

    def encode(self, inp, dense=True):
        # ID-LV encoding
        if dense:
            assert inp.shape[1] == self.n_id
            n_batch = inp.shape[0]
            inp_enc = torch.zeros(n_batch, self.n_dim, dtype=torch.int)
            for i in tqdm.tqdm(range(n_batch)):
                # Vectorized version
                inp_enc[i] = (self.hv_id * self.hv_lv[inp[i].long()]).sum(dim=0)

                # Serial version
                # tmp = torch.zeros(1, self.n_dim, dtype=torch.int)
                # for j in range(self.n_id):
                # tmp = tmp + (self.hv_id[j] * self.hv_lv[inp_quant[i][j]])
                # inp_enc[i] = tmp
        else:
            n_batch = inp["lv"].shape[0]
            inp_enc = torch.zeros(n_batch, self.n_dim, dtype=torch.int)
            for i in tqdm.tqdm(range(n_batch)):
                idx_effective = inp["idx"][i] != -1
                lv = inp["lv"][i, idx_effective].long()
                id = inp["idx"][i, idx_effective].long()
                inp_enc[i] = (self.hv_id[id] * self.hv_lv[lv]).sum(dim=0)
        return binarize(inp_enc).int() if self.binary else inp_enc


class HDC_RP:
    def __init__(self, n_class, n_feat, n_dim, binary=True) -> None:
        self.n_class = n_class
        self.n_dim, self.n_feat = n_dim, n_feat
        self.binary = binary
        self.rp = torch.randint(0, 2, size=(n_feat, n_dim), dtype=torch.int) * 2 - 1
        self.class_hvs = torch.zeros(n_class, n_dim, dtype=torch.int)

    def encode(self, inp):
        assert inp.shape[1] == self.n_feat

        inp_enc = torch.matmul(inp, self.rp)
        return binarize(inp_enc).int() if self.binary else inp_enc
