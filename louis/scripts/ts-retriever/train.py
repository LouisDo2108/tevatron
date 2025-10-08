import os
import re
import argparse
import subprocess
from pathlib import Path
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
        subprocess.run(joined_cmd, shell=True, check=True, env=env)
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
        temperature="0.02",
    ),
    "contriever": dict(
        checkpoint="facebook/contriever",
        pooling="mean",
        query_prefix="''",
        passage_prefix="''",
        query_prompts="''",
        corpus_prompts="''",
        normalize="''",
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
        temperature="0.02",
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
        corpus_prompts="search_document:",
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
        choices=["temporal_nobel_prize", "time_sensitive_qa"],
        help="Dataset name",
    )
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_neg", default=1, type=int)
    args = parser.parse_args()

    # ==== PATHS ====
    HOME = Path("/home/thuy0050")
    CODE_DIR = HOME / "code" / "tevatron"
    DATA_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "data" / "third_work"
    OUTPUT_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "exp" / "tevatron"
    MODEL_NAME = "ts-retriever"

    # ==== ARGUMENTS ====
    batch_size = args.batch_size
    epoch = args.epoch
    num_neg = args.num_neg + 1
    grad_accum = args.gradient_accumulation_steps
    grad_ckpt = "--gradient_checkpointing" if args.gradient_checkpointing else ""
    wandb = "wandb" if args.wandb else "none"

    data_name = args.data
    exp_name = args.exp_name
    model_name = args.model

    # ==== MODEL CONFIG ====
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}")
    cfg = configs[model_name]

    backbone = cfg["checkpoint"]
    output_dir = OUTPUT_ROOT / data_name / MODEL_NAME / backbone / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==== COMMANDS ====
    train_cmd = f"""
    python {CODE_DIR}/src/tevatron/retriever/driver/train.py \
        --do_train \
        --pooling {cfg['pooling']} \
        --bf16 \
        {cfg['normalize']} \
        --train_group_size {num_neg} \
        --per_device_train_batch_size {batch_size} \
        --learning_rate 1e-4 \
        --temperature {cfg['temperature']} \
        --logging_steps 10 \
        --num_train_epochs {epoch} \
        --gradient_accumulation_steps {grad_accum} \
        --lora \
        --lora_r 4 \
        --lora_alpha 16 \
        --lora_target_modules all-linear \
        --dataset_name {DATA_ROOT}/tevatron/Tevatron___msmarco-passage \
        --dataset_path {DATA_ROOT}/temporal/{data_name}/train/train.jsonl \
        --eval_dataset_path {DATA_ROOT}/temporal/{data_name}/train/dev.jsonl \
        --model_name_or_path {backbone} \
        --run_name {backbone}_{exp_name} \
        --output_dir {output_dir} \
        --report_to {wandb} \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {cfg['matryoshka_dim']} \
        {grad_ckpt}
    """

    encode_corpus_cmd = f"""
    python {CODE_DIR}/src/tevatron/retriever/driver/encode.py \
        --per_device_eval_batch_size 512 \
        --passage_max_len 512 \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --dataset_name LouisDo2108/temporal-nobel-prize \
        --dataset_config corpus \
        --dataset_path {DATA_ROOT}/temporal/{data_name}/test/corpus.jsonl \
        --encode_output_path {output_dir}/corpus_emb.pkl \
        --model_name_or_path {output_dir} \
        --lora_name_or_path {output_dir} \
        --overwrite_output_dir \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {cfg['matryoshka_dim']}
    """

    encode_query_cmd = f"""
    python {CODE_DIR}/src/tevatron/retriever/driver/encode.py \
        --per_device_eval_batch_size 512 \
        --query_max_len 512 \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --encode_is_query \
        --dataset_name LouisDo2108/temporal-nobel-prize \
        --dataset_config query \
        --dataset_path {DATA_ROOT}/temporal/{data_name}/test/query.jsonl \
        --model_name_or_path {output_dir} \
        --lora_name_or_path {output_dir} \
        --encode_output_path {output_dir}/queries_emb.pkl \
        --overwrite_output_dir \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {cfg['matryoshka_dim']}
    """

    retrieval_cmd = f"""
    set -f && OMP_NUM_THREADS=12 python -m tevatron.retriever.driver.search \
        --query_reps {output_dir}/queries_emb.pkl \
        --passage_reps {output_dir}/corpus_emb.pkl \
        --depth 100 \
        --batch_size 512 \
        --save_text \
        --save_ranking_to {output_dir}/rank.txt &&

    python -m tevatron.utils.format.convert_result_to_trec \
        --input {output_dir}/rank.txt \
        --output {output_dir}/rank.trec \
        --remove_query &&

    python -m pyserini.eval.trec_eval -c \
        -mP.10 -mrecall.10 -mndcg_cut.10 -M 10 -mrecip_rank -mmap \
        {DATA_ROOT}/temporal/{data_name}/test/qrel.txt \
        {output_dir}/rank.trec &&

    python {CODE_DIR}/louis/beir_scripts/eval_nanobeir_with_sbert.py \
        --model_name_or_path {backbone} \
        --nanobeir_datasets NQ \
        --pooling {cfg['pooling']} \
        --bf16 \
        --query_prompts {cfg['query_prefix']} \
        --corpus_prompts {cfg['passage_prefix']} \
        --matryoshka_dim {cfg['matryoshka_dim']}
    """

    # ==== EXECUTION ====
    run([train_cmd, encode_corpus_cmd, encode_query_cmd, retrieval_cmd])


if __name__ == "__main__":
    main()
