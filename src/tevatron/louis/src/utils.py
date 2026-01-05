import logging
import os
import sys
import re
import subprocess
from typing import Union, List, Optional

import msgspec
from pathlib import Path
from msgspec.json import format
from pprint import pformat, pprint

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

import torch
import torch.nn.functional as F
import numpy as np
import random

from transformers.utils.import_utils import is_torch_available
from transformers.hf_argparser import HfArgumentParser
from tevatron.retriever.arguments import DataArguments, ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments


logger = logging.getLogger(__name__)


temporal_query_type_class_id = {
    "Explicit": 0,
    "Implicit": 1,
    "TemporalAnswer": 2,
}

allen_relation_class_id = {
    "Before": 0,
    "After": 1,
    "Meets": 2,
    "MetBy": 3,
    "Overlaps": 4,
    "OverlappedBy": 5,
    "Starts": 6,
    "StartedBy": 7,
    "During": 8,
    "Contains": 9,
    "Finishes": 10,
    "FinishedBy": 11,
    "Equals": 12,
    "Empty": 13,
}



def norm(x):
    return F.normalize(x, p=2, dim=-1)


# Compile once at module load time for speed
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_cmd(cmd: str) -> str:
    """Normalize whitespace and remove newlines in a shell command."""
    cmd = _WHITESPACE_RE.sub(" ", cmd.strip())
    cmd = cmd.replace("\n", " ")
    return cmd


def run(
    cmd: Union[str, List[str]], env: Optional[dict] = None, dry_run: bool = False
) -> None:
    """
    Run one or multiple shell commands safely.

    Args:
        cmd: A single command string or a list of commands.
        env: Optional environment variables to pass to subprocess.
        dry_run: If True, prints commands instead of executing them.
    """
    if isinstance(cmd, list):
        cmd_list = [normalize_cmd(c) for c in cmd if c.strip()]
        joined_cmd = " && ".join(cmd_list)
    else:
        joined_cmd = normalize_cmd(cmd)

    # print(f"\n>>> Running:\n{joined_cmd}\n", flush=True)

    if dry_run:
        return

    try:
        # pprint(dict(os.environ))
        pprint(joined_cmd)
        subprocess.run(
            joined_cmd, 
            # timeout=15,
            shell=True, check=True, env=dict(os.environ)
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        raise


def set_seed(seed: int, deterministic: bool = True):
    # Copy from transformers.trainer_utilss.set_seed with some modifications
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
        if deterministic:
            # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to :16:8 (may limit overall performance) or :4096:8 (will increase library footprint in GPU memory by approximately 24MiB). From https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

            # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
            torch.use_deterministic_algorithms(True)

            # # Enable CUDNN deterministic mode
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False


def write_json(file_path, data, jsonl=False):
    with open(file_path, "wb") as file:
        if jsonl:
            file.write(encoder.encode_lines(data))
        else:
            file.write(format(encoder.encode(data)))
    print(f"The file contains {len(data)} items.")
    print("Saved to", file_path)


def read_json(file_path, jsonl=False):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError("filepath is not a file")
    # if not file_path.suffix == ".jsonl" and jsonl:
    #     raise ValueError("the file is not jsonl")
    # if not file_path.suffix == ".json" and not jsonl:
    #     raise ValueError("the file is not json")
    file_path = file_path.__str__()

    output = []

    with open(file_path, "rb") as file:
        data = file.read()
        if jsonl:
            output = decoder.decode_lines(data)
        else:
            output = decoder.decode(data)

    print(f"The file is of type: {type(output)}")
    print(f"The file contains {len(output)} items.")
    return output


def get_params_info(model):
    all_param = 0
    trainable_param = 0

    print("\nAll trainable parameters:")
    for name, param in model.named_parameters():
        
        if name.startswith("base_model."):
            # This is the duplicate of the base model for KL loss
           continue 
        all_param += param.numel()
        
        if param.requires_grad:
            trainable_param += param.numel()
            print(name, param.numel())
            
    print(f"trainable params: {trainable_param:,} || all params: {all_param:,} || trainable%: {trainable_param / all_param * 100:.2f}")


def init():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)) # type: ignore

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("\n##### Training/evaluation arguments #####\n %s\n", pformat(training_args))
    logger.info("\n ##### Model arguments #####\n %s\n", pformat(model_args))
    logger.info("\n ##### Data arguments #####\n %s\n", pformat(data_args))

    set_seed(training_args.seed)

    return model_args, data_args, training_args
