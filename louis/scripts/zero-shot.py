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
        normalize="''",
        padding_side="right",
        matryoshka_dim="768",
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
    ),
}


def main():
    """Run zero-shot encoding and retrieval for temporal datasets."""
    parser = argparse.ArgumentParser(
        description="Zero-Shot Temporal Retriever Evaluation"
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model name, e.g. bge | bgem3 | contriever | gte | nomic | qwen3",
    )
    parser.add_argument(
        "--data",
        default="temporal_nobel_prize",
        choices=["temporal_nobel_prize", "time_sensitive_qa"],
        help="Dataset name",
    )
    args = parser.parse_args()

    # ==== PATHS ====
    HOME = Path("/home/thuy0050")
    CODE_DIR = HOME / "code" / "tevatron"
    DATA_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "data" / "third_work"
    OUTPUT_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "exp" / "tevatron"

    DATA_NAME = args.data
    MODEL_NAME = "zero-shot"

    # ==== MODEL CONFIGS ====
    # If you want to run all models in `configs`, loop below.
    # Otherwise, you could also just do `cfg = configs[args.model]`.
    # for model_key, cfg in configs.items():
    cfg = configs[args.model]
    backbone = cfg["checkpoint"]
    output_dir = OUTPUT_ROOT / DATA_NAME / MODEL_NAME / backbone
    output_dir.mkdir(parents=True, exist_ok=True)

    # === COMMANDS ===
    encode_corpus_cmd = f"""
    python {CODE_DIR}/src/tevatron/retriever/driver/encode.py \
        --per_device_eval_batch_size 512 \
        --passage_max_len 512 \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --dataset_name LouisDo2108/temporal-nobel-prize \
        --dataset_config corpus \
        --dataset_path {DATA_ROOT}/temporal/{DATA_NAME}/test/corpus.jsonl \
        --encode_output_path {output_dir}/corpus_emb.pkl \
        --model_name_or_path {cfg['checkpoint']} \
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
        --dataset_path {DATA_ROOT}/temporal/{DATA_NAME}/test/query.jsonl \
        --model_name_or_path {cfg['checkpoint']} \
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
        {DATA_ROOT}/temporal/{DATA_NAME}/test/qrel.txt \
        {output_dir}/rank.trec &&

    python {CODE_DIR}/louis/beir_scripts/eval_nanobeir_with_sbert.py \
        --model_name_or_path {cfg['checkpoint']} \
        --nanobeir_datasets NQ \
        --pooling {cfg['pooling']} \
        --bf16 \
        --query_prompts {cfg['query_prefix']} \
        --corpus_prompts {cfg['passage_prefix']} \
        --matryoshka_dim {cfg['matryoshka_dim']}
    """

    # === EXECUTION ===
    run([encode_corpus_cmd, encode_query_cmd, retrieval_cmd])


if __name__ == "__main__":
    main()
