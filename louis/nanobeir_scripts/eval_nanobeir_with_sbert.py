# from sentence_transformers import models, SentenceTransformer
# from sentence_transformers.evaluation import NanoBEIREvaluator

# model = models.Transformer(
#     # "BAAI/bge-base-en-v1.5"
#     "/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/ts-retriever/contriever/naive_temporal_v2_filter-95%-inbatch-negatives_only_temporal",
#     # "/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/ts-retriever/contriever/ts-retriever_5epoch_temp0.05_lora_bf16",
# )
# pooling = models.Pooling(model.get_word_embedding_dimension(), pooling_mode="mean")

# normalize = models.Normalize()
# model = SentenceTransformer(
#     modules=[model, pooling, normalize],
#     model_kwargs={"torch_dtype": "bfloat16", "max_seq_length": 512},
# )

# datasets = ["NQ"]
# evaluator = NanoBEIREvaluator(
#     dataset_names=datasets,
# )

# results = evaluator(
#     model,
#     output_path="/home/thuy0050/code/tevatron/louis/nanobeir_scripts",
# )

# print(evaluator.primary_metric)
# # => "NanoBEIR_mean_cosine_ndcg@10"
# print(results[evaluator.primary_metric])
# # => 0.8084508771660436


import argparse
from sentence_transformers import models, SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator
from pdb import set_trace as st

def main():
    parser = argparse.ArgumentParser(description="Run NanoBEIR evaluation.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the model or model checkpoint.",
    )
    # parser.add_argument("--lora_name_or_path", default=None)
    parser.add_argument(
        "--nanobeir_datasets", type=str, nargs="+", default=["NQ"], help="List of dataset names."
    )
    parser.add_argument("--pooling", type=str, default="mean", help='Pooling can either be "cls", "lasttoken", "max", "mean"')
    parser.add_argument(
        "--query_prompts",
        type=str,
        default=None
    )
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--bf16", default=True, action="store_true")
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length."
    )

    args = parser.parse_args()

    torch_dtype = "bf16" if args.bf16 else "fp16"
    print(f"Inference in {torch_dtype}")

    if args.pooling == "avg":
        args.pooling = "mean"
    print(f"Pooling method: {args.pooling}")

    # Load the transformer and SentenceTransformer model
    transformer_model = models.Transformer.load(args.model_name_or_path, trust_remote_code=True)

    pooling_model = models.Pooling(
        transformer_model.get_word_embedding_dimension(), pooling_mode=args.pooling
    )
    normalize_model = models.Normalize()

    # Set up the SentenceTransformer model
    model = SentenceTransformer(
        modules=[transformer_model, pooling_model, normalize_model],
        model_kwargs={
            "torch_dtype": torch_dtype,
            "max_seq_length": args.max_seq_length,
        },
        trust_remote_code=True,
    )

    # Set up and run the evaluator
    evaluator = NanoBEIREvaluator(dataset_names=args.nanobeir_datasets, query_prompts=args.query_prompts if args.query_prompts else None)
    results = evaluator(model)

    print("Primary Metric:", evaluator.primary_metric)
    print("Score:", results[evaluator.primary_metric])


if __name__ == "__main__":
    main()
