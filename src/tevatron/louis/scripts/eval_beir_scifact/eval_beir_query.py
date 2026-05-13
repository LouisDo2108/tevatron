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
        help="Method name, e.g. temporal, madaptor, tempretriever, ts-retriever",
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
    if METHOD_NAME == "zero-shot":    
        output_dir = OUTPUT_ROOT / "beir-scifact-zero-shot" / backbone
    else:
        output_dir = OUTPUT_ROOT / DATA_NAME / METHOD_NAME / backbone / EXP_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = args.batch_size

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

    encode_query_cmd = f"""
    python {CODE_DIR}/src/tevatron/retriever/driver/{encode_file_name} \
        --per_device_eval_batch_size {batch_size} \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --encode_is_query \
        --dataset_name Tevatron/beir \
        --dataset_config scifact \
        --dataset_split test \
        --encode_output_path {output_dir}/queries_emb_beir_scifact.pkl \
        --model_name_or_path {output_dir} \
        {lora_eval} \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {cfg['matryoshka_dim']} \
        {adaptor_dim}
    """

    retrieval_cmd = f"""
    set -f && OMP_NUM_THREADS=16 python -m tevatron.retriever.driver.search \
        --query_reps {output_dir}/queries_emb_beir_scifact.pkl \
        --passage_reps {output_dir}/corpus_emb_beir_scifact.*.pkl \
        --depth 100 \
        --batch_size 2048 \
        --save_text \
        --save_ranking_to {output_dir}/rank_beir_scifact.txt

    && python -m tevatron.utils.format.convert_result_to_trec \
        --input {output_dir}/rank_beir_scifact.txt \
        --output {output_dir}/rank_beir_scifact.trec \
        --remove_query

    && python -m pyserini.eval.trec_eval -c \
        -m recall.10,100 -m ndcg_cut.10 -M 100 \
        /home/thuy0050/code/TMRL/data/scifact_qrel.txt \
        {output_dir}/rank_beir_scifact.trec > {output_dir}/out_beir_scifact.txt
    """

    # ==== EXECUTION ====
    run(encode_query_cmd)
    run(retrieval_cmd)

if __name__ == "__main__":
    main()
