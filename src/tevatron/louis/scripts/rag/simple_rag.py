import os
import re
import argparse
import subprocess
from pathlib import Path
from pprint import pprint, pformat
from pdb import set_trace as st
from typing import Union, List, Optional
from tevatron.louis.src.configs import configs
from tevatron.louis.src.utils import run


def main():
    """Main entry for temporal retriever training and evaluation."""
    parser = argparse.ArgumentParser(description="Temporal Retriever Training Pipeline")
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Model name, e.g. bge | bgem3 | contriever | gte | nomic | qwen3",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="Model name, e.g. bge | bgem3 | contriever | gte | nomic | qwen3",
    )
    parser.add_argument(
        "--flashrag_config",
        required=True,
        type=str,
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
    parser.add_argument("--lora", action="store_true", help="Use LoRA for training.", default=False)
    parser.add_argument(
        "--matryoshka_dim",
        type=int,
        default=-1,
    )

    args = parser.parse_args()

    # ==== PATHS ====
    HOME = Path("/home/thuy0050")
    CODE_DIR = HOME / "code" / "tevatron"
    DATA_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "data" / "third_work"
    OUTPUT_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "exp" / "tevatron"
    MODEL_NAME = args.model_name
    FLASHRAG_CONFIG = Path(args.flashrag_config)

    data_name = args.data
    exp_name = args.exp_name
    cfg = configs[args.model]
    BACKBONE = cfg["checkpoint"]
    
    if MODEL_NAME == "zero-shot":
        checkpoint_dir = OUTPUT_ROOT / data_name / MODEL_NAME / BACKBONE
        if args.matryoshka_dim != -1:
            OUTPUT_DIR = OUTPUT_ROOT / data_name / MODEL_NAME / BACKBONE / f"rag_{FLASHRAG_CONFIG.stem}_{args.matryoshka_dim}"
        else:
            OUTPUT_DIR = OUTPUT_ROOT / data_name / MODEL_NAME / BACKBONE / f"rag_{FLASHRAG_CONFIG.stem}"
    else:
        checkpoint_dir = OUTPUT_ROOT / data_name / MODEL_NAME / BACKBONE / exp_name
        if args.matryoshka_dim != -1:
            OUTPUT_DIR = OUTPUT_ROOT / data_name / MODEL_NAME / BACKBONE / exp_name / f"rag_{FLASHRAG_CONFIG.stem}_{args.matryoshka_dim}"
        else:
            OUTPUT_DIR = OUTPUT_ROOT / data_name / MODEL_NAME / BACKBONE / exp_name / f"rag_{FLASHRAG_CONFIG.stem}"
        
    
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==== COMMANDS ====
    index_cmd = f"""
    python /home/thuy0050/code/FlashRAG/louis/index_builder.py \
        --per_device_eval_batch_size 512 \
        --query_max_len 512 \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --attn_implementation sdpa \
        --corpus_path {DATA_ROOT}/temporal/{data_name}/test/corpus.{"jsonl" if "nobel" in data_name else "parquet"} \
        --save_dir {OUTPUT_DIR} \
        --model_name_or_path {BACKBONE if MODEL_NAME == 'zero-shot' else checkpoint_dir} \
        --passage_prefix {cfg['passage_prefix']} \
        --embedding_path {checkpoint_dir}/{"corpus_emb_" + str(args.matryoshka_dim) if args.matryoshka_dim != -1 else "corpus_emb"}.pkl 
    """
        # --lora_name_or_path {checkpoint_dir} \
        

    rag_cmd = f"""
    python /home/thuy0050/code/FlashRAG/louis/simple_rag.py \
        --flashrag_config_yaml_path {str(FLASHRAG_CONFIG)} \
        --per_device_eval_batch_size 512 \
        --query_max_len 512 \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --attn_implementation sdpa \
        --encode_is_query \
        --output_dir {OUTPUT_DIR} \
        --dataset_name LouisDo2108/temporal-nobel-prize \
        --corpus_path {DATA_ROOT}/temporal/{data_name}/test/corpus.{"jsonl" if "nobel" in data_name else "parquet"} \
        --dataset_path {DATA_ROOT}/temporal/{data_name}/test/query.{"jsonl" if "nobel" in data_name else "parquet"} \
        --dataset_config query \
        --model_name_or_path {BACKBONE if MODEL_NAME == 'zero-shot' else checkpoint_dir} \
        --encode_output_path {checkpoint_dir}/{"queries_emb_" + str(args.matryoshka_dim) if args.matryoshka_dim != -1 else "queries_emb"}.pkl \
        --overwrite_output_dir \
        --query_prefix {cfg['query_prefix']}
    """
    # --lora_name_or_path {checkpoint_dir}

    # ==== EXECUTION ====
    run([index_cmd])
    run([rag_cmd])


if __name__ == "__main__":
    main()
