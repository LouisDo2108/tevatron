import argparse
from pathlib import Path
from pdb import set_trace as st
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
        "--method_name",
        default="tmrl",
        type=str,
        help="Method name, e.g. tmrl, madaptor, tempretriever, ts-retriever, zero-shot, mrl, lora",
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
    parser.add_argument("--enhanced_temporal", action="store_true", help="Use this flag only for temporal_nobel_prize dataset to train on our enhanced temporal queries.")
    parser.add_argument("--eval", action="store_true", help="Eval with the corresponding dev set and also save the best model with the eval loss.", default=False)
    parser.add_argument("--lora", action="store_true", help="Use LoRA for training.", default=False)
    parser.add_argument("--lora_r", default=4, type=int)
    parser.add_argument("--lora_alpha", default=4, type=int)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--eval_batch_size", default=512, type=int)
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--qt", default=0.0, type=float)
    parser.add_argument("--pt", default=0.0, type=float)
    parser.add_argument("--qt_recon", default=0.0, type=float)
    parser.add_argument("--pt_recon", default=0.0, type=float)
    parser.add_argument("--temporal_dim", default=64, type=int)
    parser.add_argument("--max_temporal_length", default=16, type=int)
    parser.add_argument("--temporal", action="store_true", default=False)
    parser.add_argument("--temporal_reconstruction", action="store_true", default=False)
    parser.add_argument("--filter_false_negatives", action="store_true", default=False)
    parser.add_argument("--kl_loss", action="store_true", default=False)
    parser.add_argument("--add_cls", action="store_true", default=False)
    parser.add_argument(
        "--matryoshka_dim_list",
        type=int,
        nargs="+",  # Accepts multiple integers
        help="List of dimensions for Matryoshka representation, e.g. --matryoshka_dim_list 256 512 768",
    )

    args = parser.parse_args()

    # ==== PATHS ====
    HOME = Path("/home/thuy0050")
    CODE_DIR = HOME / "code" / "tevatron" / "src" / "tevatron"
    DATA_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "data" / "third_work"
    OUTPUT_ROOT = HOME / "mg61_scratch2" / "thuy0050" / "exp" / "tevatron"
    METHOD_NAME = args.method_name

    data_name = args.data
    exp_name = args.exp_name

    # ==== ARGUMENTS ====
    eval_batch_size = args.eval_batch_size if "qwen" not in args.model else 128

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
        lora_train = f"--lora --lora_r {args.lora_r} --lora_alpha {args.lora_alpha} --lora_target_modules all-linear"
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

    matryoshka_dim_list = []
    if args.matryoshka_dim_list:
        matryoshka_dim_list = args.matryoshka_dim_list
    else:
        # Default to first to second last dimension
        matryoshka_dim_list = cfg["matryoshka_dim_list"] # [:-1]
        
    # for m in matryoshka_dim_list: 
    m = 64
    # ==== COMMANDS ====
    encode_corpus_cmd = f"""
    python {CODE_DIR}/retriever/driver/{encode_file_name} \
        --per_device_eval_batch_size {eval_batch_size} \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --dataset_name LouisDo2108/temporal-nobel-prize \
        --dataset_config corpus \
        --dataset_path {DATA_ROOT}/temporal/{data_name}/test/corpus.{"jsonl" if "nobel" in data_name else "parquet"} \
        --encode_output_path {output_dir}/corpus_emb_{m}.pkl \
        --model_name_or_path {output_dir} \
        {lora_eval} \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {m} \
        {adaptor_dim}
    """
    encode_query_cmd = f"""
    python {CODE_DIR}/retriever/driver/{encode_file_name} \
        --per_device_eval_batch_size {eval_batch_size} \
        --pooling {cfg['pooling']} \
        --bf16 \
        --normalize \
        --encode_is_query \
        --dataset_name LouisDo2108/temporal-nobel-prize \
        --dataset_config query \
        --dataset_path {DATA_ROOT}/temporal/{data_name}/test/query.{"jsonl" if "nobel" in data_name else "parquet"} \
        --model_name_or_path {output_dir} \
        {lora_eval} \
        --encode_output_path {output_dir}/queries_emb_{m}.pkl \
        --query_prefix {cfg['query_prefix']} \
        --passage_prefix {cfg['passage_prefix']} \
        --padding_side {cfg['padding_side']} \
        --matryoshka_dim {m} \
        {adaptor_dim}
    """
    retrieval_cmd = f"""
    set -f && OMP_NUM_THREADS=16 \
        python -m tevatron.retriever.driver.searchv2 \
        --query_reps {output_dir}/queries_emb_{m}.pkl \
        --passage_reps {output_dir}/corpus_emb_{m}.pkl \
        --depth 100 \
        --batch_size 2048 \
        --save_text \
        --save_ranking_to {output_dir}/rank_funnel_64-768.txt
    
    && python -m tevatron.utils.format.convert_result_to_trec \
        --input {output_dir}/rank_funnel_64-768.txt \
        --output {output_dir}/rank_funnel_64-768.trec \
        --remove_query

    && python -m pyserini.eval.trec_eval -c \
        -m recall.10,100 -m ndcg_cut.10 -M 100 \
        {DATA_ROOT}/temporal/{data_name}/test/qrel.txt \
        {output_dir}/rank_funnel_64-768.trec > {output_dir}/out_{m}_funnel_64-768.txt 
    """
    # ==== EXECUTION ====
    # run([encode_corpus_cmd, encode_query_cmd])
    run([retrieval_cmd])


if __name__ == "__main__":
    main()
