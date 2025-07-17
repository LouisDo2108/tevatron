import torch
from safetensors import safe_open
from pdb import set_trace as st

path1 = "/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/ts-retriever/contriever/naive_temporal_5epoch_temp0.05_lora/adapter_model.safetensors"

path2 = "/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/ts-retriever/contriever/naive_temporal_5epoch_temp0.05_lora2/adapter_model.safetensors"

tensors1 = {}
with safe_open(path1, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors1[key] = f.get_tensor(key)

tensors2 = {}
with safe_open(path2, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors2[key] = f.get_tensor(key)

tensors1['base_model.model.encoder.layer.9.attention.self.value.lora_A.weight'].sum()
st()

{
    "loss": 6.4897,
    "loss_temporal_512": 4.2336,
    "loss_semantic_512": 1.1993,
    "loss_semantic_768": 1.0569,
    "grad_norm": nan,
    "learning_rate": "0.000e+00",
    "epoch": 0.01,
}
{
    "loss": 6.9855,
    "loss_temporal_512": 4.3824,
    "loss_semantic_512": 1.4314,
    "loss_semantic_768": 1.1717,
    "grad_norm": 5.5348,
    "learning_rate": "0.000e+00",
    "epoch": 0.02,
}
{
    "loss": 6.8635,
    "loss_temporal_512": 4.3095,
    "loss_semantic_512": 1.3873,
    "loss_semantic_768": 1.1667,
    "grad_norm": 5.4408,
    "learning_rate": "7.692e-06",
    "epoch": 0.02,
}
{
    "loss": 6.7897,
    "loss_temporal_512": 4.3567,
    "loss_semantic_512": 1.3004,
    "loss_semantic_768": 1.1326,
    "grad_norm": 5.5258,
    "learning_rate": "1.538e-05",
    "epoch": 0.03,
}
{
    "loss": 6.2894,
    "loss_temporal_512": 4.3075,
    "loss_semantic_512": 1.1054,
    "loss_semantic_768": 0.8765,
    "grad_norm": 4.9082,
    "learning_rate": "2.308e-05",
    "epoch": 0.04,
}
{
    "loss": 7.2089,
    "loss_temporal_512": 4.4104,
    "loss_semantic_512": 1.484,
    "loss_semantic_768": 1.3144,
    "grad_norm": 5.4416,
    "learning_rate": "3.077e-05",
    "epoch": 0.05,
}
{
    "loss": 7.1513,
    "loss_temporal_512": 4.4744,
    "loss_semantic_512": 1.4388,
    "loss_semantic_768": 1.2381,
    "grad_norm": 5.3371,
    "learning_rate": "3.846e-05",
    "epoch": 0.06,
}
{
    "loss": 6.7369,
    "loss_temporal_512": 4.3865,
    "loss_semantic_512": 1.2884,
    "loss_semantic_768": 1.062,
    "grad_norm": 5.6792,
    "learning_rate": "4.615e-05",
    "epoch": 0.06,
}
{
    "loss": 7.0549,
    "loss_temporal_512": 4.3789,
    "loss_semantic_512": 1.4335,
    "loss_semantic_768": 1.2425,
    "grad_norm": 5.6763,
    "learning_rate": "5.385e-05",
    "epoch": 0.07,
}
{
    "loss": 7.6437,
    "loss_temporal_512": 4.2878,
    "loss_semantic_512": 1.7231,
    "loss_semantic_768": 1.6328,
    "grad_norm": 6.1715,
    "learning_rate": "6.154e-05",
    "epoch": 0.08,
}
{
    "loss": 7.0037,
    "loss_temporal_512": 4.2925,
    "loss_semantic_512": 1.4285,
    "loss_semantic_768": 1.2826,
    "grad_norm": 5.1057,
    "learning_rate": "6.923e-05",
    "epoch": 0.09,
}
