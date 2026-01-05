import msgspec
from pathlib import Path

from pdb import set_trace as st

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

import pandas as pd

queries = pd.read_parquet(
    "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nanobeir_nq/queries.parquet"
)
corpus = pd.read_parquet(
    "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nanobeir_nq/corpus.parquet"
)
qrels = pd.read_parquet(
    "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nanobeir_nq/qrels.parquet"
)

queries_jsonl = []
query_id_to_index_dict = {}
for index, row in queries.iterrows():
    queries_jsonl.append(
        {
            "id": index,
            "original_id": row["_id"],
            "text": row["text"],
        }
    )
    query_id_to_index_dict[row["_id"]] = index
write_json(
    "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nanobeir_nq/queries.jsonl",
    queries_jsonl,
    jsonl=True,
)

corpus_jsonl = []
corpus_id_to_index_dict = {}
for index, row in corpus.iterrows():
    corpus_jsonl.append(
        {
            "id": index,
            "original_id": row["_id"],
            "text": row["text"],
        }
    )
    corpus_id_to_index_dict[row["_id"]] = index
write_json(
    "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nanobeir_nq/corpus.jsonl",
    corpus_jsonl,
    jsonl=True,
)

with open(
    "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nanobeir_nq/qrel.txt",
    "w",
) as f:
    for index, row in qrels.iterrows():
        try:
            qid = query_id_to_index_dict[row["query-id"]]
            docid = corpus_id_to_index_dict[row["corpus-id"]]
            f.write(f"{qid} 0 {docid} 1\n")
        except Exception as e:
            print(e)
