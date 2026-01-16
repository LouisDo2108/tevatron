import argparse
from pathlib import Path
from pdb import set_trace as st
from typing import Union, List, Optional
from tevatron.louis.src.utils import run
from tevatron.louis.src.configs import configs

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
        "--method_name",
        default="temporal",
        type=str,
        help="Method name, e.g. temporal, madaptor, tempretriever, ts-retriever, zero-shot",
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
    METHOD_NAME = args.method_name

    # ==== MODEL CONFIGS ====
    # If you want to run all models in `configs`, loop below.
    # Otherwise, you could also just do `cfg = configs[args.model]`.
    # for model_key, cfg in configs.items():
    cfg = configs[args.model]
    backbone = str(cfg["checkpoint"])
    batch_size = args.batch_size
    dataset_shard_index = args.dataset_shard_index

    if METHOD_NAME == "zero-shot":    
        output_dir = OUTPUT_ROOT / "beir-nq-zero-shot" / backbone
    else:
        output_dir = OUTPUT_ROOT / DATA_NAME / METHOD_NAME / backbone / EXP_NAME
        output_dir.mkdir(parents=True, exist_ok=True)

    lora_eval = ""
    if args.lora:
        lora_eval = f"--lora_name_or_path {output_dir}"
    
    encode_file_name = "encode.py"
    adaptor_dim = ""

    if METHOD_NAME == "madaptor":
        adaptor_dim = f"--adaptor_dim {cfg['adaptor_dim']}"
        encode_file_name = "encode_madaptor.py"
    elif METHOD_NAME == "tempretriever":
        encode_file_name = "encode_tempretriever_no_temporal.py"
    else:
        encode_file_name = "encode.py"

    encode_corpus_cmd = f"""
    python {CODE_DIR}/src/tevatron/retriever/driver/{encode_file_name} \
        --per_device_eval_batch_size {batch_size} \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --dataset_name Tevatron/beir-corpus \
        --dataset_config nq \
        --encode_output_path {output_dir}/corpus_emb_beir_nq.{dataset_shard_index}.pkl \
        --dataset_number_of_shards 8 \
        --dataset_shard_index {dataset_shard_index} \
        --model_name_or_path {output_dir} \
        {lora_eval} \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {cfg['matryoshka_dim']} \
        {adaptor_dim}
    """
    # ==== EXECUTION ====
    run(encode_corpus_cmd)


if __name__ == "__main__":
    main()
