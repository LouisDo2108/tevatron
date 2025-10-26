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
    "tempretriever": dict(
        checkpoint="google-bert/bert-base-uncased",
        pooling="cls",
        query_prefix="''",
        passage_prefix="''",
        query_prompts="''",
        corpus_prompts="''",
        normalize="--normalize",
        padding_side="right",
        matryoshka_dim="1536",
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
    parser.add_argument("--method_name", default="ts-retriever", type=str)
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
    args = parser.parse_args()

    # ==== PATHS ====
    HOME = Path("/home/thuy0050")
    CODE_DIR = HOME / "code" / "tevatron"
    DATA_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "data" / "third_work"
    OUTPUT_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "exp" / "tevatron"
    METHOD_NAME = args.method_name

    data_name = args.data
    exp_name = args.exp_name

    # ==== MODEL CONFIG ====
    model_name = args.model
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}")
    cfg = configs[model_name]

    backbone = cfg["checkpoint"]
    output_dir = OUTPUT_ROOT / data_name / METHOD_NAME / backbone / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_eval = ""
    if args.lora:
        lora_eval = f"--lora_name_or_path {output_dir}"

    encode_corpus_cmd = [
        f"""python {CODE_DIR}/src/tevatron/retriever/driver/{'encode' if args.model != 'tempretriever' else 'encode_tempretriever'}.py \
        --per_device_eval_batch_size 3072 \
        --passage_max_len 512 \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --dataset_name Tevatron/beir-corpus \
        --dataset_config nq \
        --encode_output_path {output_dir}/corpus_emb_beir_nq.{i}.pkl \
        --dataset_number_of_shards 8 \
        --dataset_shard_index {i} \
        --model_name_or_path {output_dir} \
        {lora_eval} \
        --overwrite_output_dir \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {cfg['matryoshka_dim']}
      """
        for i in range(8)
    ]

    # --dataloader_num_workers {4 if args.model != 'tempretriever' else 0} \

    encode_query_cmd = f"""
    python {CODE_DIR}/src/tevatron/retriever/driver/{'encode' if args.model != 'tempretriever' else 'encode_tempretriever'}.py \
        --per_device_eval_batch_size 512 \
        --query_max_len 512 \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --encode_is_query \
        --dataset_name Tevatron/beir \
        --dataset_config nq \
        --dataset_split 'test' \
        --encode_output_path {output_dir}/queries_emb_beir_nq.pkl \
        --model_name_or_path {output_dir} \
        {lora_eval} \
        --overwrite_output_dir \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {cfg['matryoshka_dim']} \
        --dataloader_num_workers {4 if args.model != 'tempretriever' else 0} \
      """

    retrieval_cmd = f"""
    set -f && OMP_NUM_THREADS=12 python -m tevatron.retriever.driver.search \
        --query_reps {output_dir}/queries_emb_beir_nq.pkl \
        --passage_reps {output_dir}/corpus_emb_beir_nq.*.pkl \
        --depth 1000 \
        --batch_size 512 \
        --save_text \
        --save_ranking_to {output_dir}/rank_beir_nq.txt &&

    python -m tevatron.utils.format.convert_result_to_trec \
        --input {output_dir}/rank_beir_nq.txt \
        --output {output_dir}/rank_beir_nq.trec \
        --remove_query &&

    python -m pyserini.eval.trec_eval -c \
        -mrecall.100 -mndcg_cut.10 \
        beir-v1.0.0-nq-test \
        {output_dir}/rank_beir_nq.trec > {output_dir}/out_beir_nq.txt
    """

    all_cmds = encode_corpus_cmd + [encode_query_cmd, retrieval_cmd]

    # ==== EXECUTION ====
    run(all_cmds)

if __name__ == "__main__":
    main()
