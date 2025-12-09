import os
import re
import argparse
import subprocess
from pathlib import Path
from pprint import pprint, pformat
from pdb import set_trace as st
from typing import Union, List, Optional

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
        # print("Running with the following environment variables:")
        # pprint({k: v for k, v in os.environ.items()})
        
        subprocess.run(joined_cmd, shell=True, check=True, env=dict(os.environ))
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        raise


configs = {
    "bge": dict(
        checkpoint="BAAI/bge-base-en-v1.5",
        pooling="cls",
        query_prefix="'Represent this sentence for searching relevant passages: '",
        passage_prefix="''",
        query_prompts="'Represent this sentence for searching relevant passages: '",
        corpus_prompts="''",
        normalize="--normalize",
        padding_side="right",
        matryoshka_dim="768",
        temperature="0.02",
    ),
    "bgem3": dict(
        checkpoint="BAAI/bge-m3",
        pooling="cls",
        query_prefix="''",
        passage_prefix="''",
        query_prompts="''",
        corpus_prompts="''",
        normalize="--normalize",
        padding_side="right",
        matryoshka_dim="1024",
    ),
    "contriever": dict(
        checkpoint="facebook/contriever",
        pooling="mean",
        query_prefix="''",
        passage_prefix="''",
        query_prompts="''",
        corpus_prompts="''",
        normalize="",
        padding_side="right",
        matryoshka_dim="768",
        temperature="0.05",
    ),
    "gte": dict(
        checkpoint="thenlper/gte-base",
        pooling="mean",
        query_prefix="''",
        passage_prefix="''",
        query_prompts="''",
        corpus_prompts="''",
        normalize="--normalize",
        padding_side="right",
        matryoshka_dim="768",
    ),
    "gte1.5": dict(
        checkpoint="Alibaba-NLP/gte-base-en-v1.5",
        pooling="cls",
        query_prefix="''",
        passage_prefix="''",
        query_prompts="''",
        corpus_prompts="''",
        normalize="--normalize",
        padding_side="right",
        matryoshka_dim="768",
        temperature="0.01",
    ),
    "nomic": dict(
        checkpoint="nomic-ai/nomic-embed-text-v1.5",
        pooling="mean",
        query_prefix="search_query:",
        passage_prefix="search_document:",
        query_prompts="search_query:",
        corpus_prompts="search_document:",
        normalize="--normalize",
        padding_side="right",
        matryoshka_dim="768",
        temperature="0.02",
    ),
    "qwen3": dict(
        checkpoint="Qwen/Qwen3-Embedding-0.6B",
        pooling="last",
        query_prefix="'Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery:'",
        passage_prefix="''",
        query_prompts="'Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery:'",
        corpus_prompts="''",
        normalize="--normalize",
        padding_side="left",
        matryoshka_dim="1024",
        temperature="0.02",
    ),
}


def main():
    """Main entry for temporal retriever training and evaluation."""
    parser = argparse.ArgumentParser(description="Temporal Retriever Training Pipeline")
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Model name, e.g. bge | bgem3 | contriever | gte | nomic | qwen3",
    )
    parser.add_argument("--exp_name", default="dev", type=str)
    parser.add_argument(
        "--data",
        default="temporal_nobel_prize",
        choices=[
            "temporal_nobel_prize",
            "time_sensitive_qa",
        ],
        help="Dataset name",
    )
    parser.add_argument(
        "--lora", action="store_true", help="Use LoRA for training.", default=False
    )
    parser.add_argument("--batch_size", default=3072, type=int)
    parser.add_argument("--dataset_shard_index", default=0, type=int)
    args = parser.parse_args()

    # ==== PATHS ====
    HOME = Path("/home/thuy0050")
    CODE_DIR = HOME / "code" / "tevatron"
    DATA_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "data" / "third_work"
    OUTPUT_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "exp" / "tevatron"
    DATA_NAME = args.data
    EXP_NAME = args.exp_name
    # MODEL_NAME = "zero-shot"
    # MODEL_NAME = "ts-retriever"
    MODEL_NAME = "temporal"

    # ==== MODEL CONFIGS ====
    # If you want to run all models in `configs`, loop below.
    # Otherwise, you could also just do `cfg = configs[args.model]`.
    # for model_key, cfg in configs.items():
    cfg = configs[args.model]
    backbone = cfg["checkpoint"]
    # output_dir = OUTPUT_ROOT / "beir" / "nq" / backbone
    # output_dir = OUTPUT_ROOT / DATA_NAME / MODEL_NAME / backbone
    output_dir = OUTPUT_ROOT / DATA_NAME / MODEL_NAME / backbone / EXP_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = args.batch_size
    dataset_shard_index = args.dataset_shard_index

    lora_eval = ""
    if args.lora:
        lora_eval = f"--lora_name_or_path {output_dir}"

    encode_corpus_cmd = f"""python {CODE_DIR}/src/tevatron/retriever/driver/{'encode' if args.model != 'tempretriever' else 'encode_tempretriever'}.py \
    --per_device_eval_batch_size {batch_size} \
    --passage_max_len 512 \
    --pooling {cfg['pooling']} \
    --bf16 \
    --normalize \
    --dataset_name Tevatron/beir-corpus \
    --dataset_config nq \
    --encode_output_path {output_dir}/corpus_emb_beir_nq.{dataset_shard_index}.pkl \
    --dataset_number_of_shards 8 \
    --dataset_shard_index {dataset_shard_index} \
    --dataloader_num_workers {8 if args.model != 'tempretriever' else 0} \
    --model_name_or_path {output_dir} \
    {lora_eval} \
    --overwrite_output_dir \
    --query_prefix {cfg['query_prefix']} \
    --passage_prefix {cfg['passage_prefix']} \
    --padding_side {cfg['padding_side']} \
    --matryoshka_dim {cfg['matryoshka_dim']}
    """
    # ==== EXECUTION ====
    run(encode_corpus_cmd)


if __name__ == "__main__":
    main()
