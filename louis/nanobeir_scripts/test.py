from sentence_transformers import models, SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator
from pdb import set_trace as st

model = models.Transformer(
    # "BAAI/bge-base-en-v1.5"
    "/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/ts-retriever/contriever/naive_temporal_v2_filter-95%-inbatch-negatives_only_temporal",
    # "/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/ts-retriever/contriever/ts-retriever_5epoch_temp0.05_lora_bf16",
)
pooling = models.Pooling(model.get_word_embedding_dimension(), pooling_mode="mean")

normalize = models.Normalize()
model = SentenceTransformer(
    modules=[model, pooling, normalize],
    model_kwargs={"torch_dtype": "bfloat16", "max_seq_length": 512},
)

datasets = ["NQ"]
evaluator = NanoBEIREvaluator(
    dataset_names=datasets,
)

results = evaluator(
    model,
    output_path="/home/thuy0050/code/tevatron/louis/nanobeir_scripts",
)

print(evaluator.primary_metric)
# => "NanoBEIR_mean_cosine_ndcg@10"
print(results[evaluator.primary_metric])
# => 0.8084508771660436
