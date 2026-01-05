import msgspec
from pathlib import Path
import pandas as pd

from pdb import set_trace as st

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

import bm25s
import Stemmer  # optional: for stemming

timer = bm25s.utils.benchmark.Timer("[BM25S]")


def read_json(file_path, jsonl=False):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError("filepath is not a file")
    # if not file_path.suffix == ".jsonl" and jsonl:
    #     raise ValueError("the file is not jsonl")
    # if not file_path.suffix == ".json" and not jsonl:
    #     raise ValueError("the file is not json")
    file_path = file_path.__str__()

    output = []

    with open(file_path, "rb") as file:
        data = file.read()
        if jsonl:
            output = decoder.decode_lines(data)
        else:
            output = decoder.decode(data)

    print(f"The file is of type: {type(output)}")
    print(f"The file contains {len(output)} items.")
    return output


def write_json(file_path, data, jsonl=False):
    with open(file_path, "wb") as file:
        if jsonl:
            file.write(encoder.encode_lines(data))
        else:
            file.write(encoder.encode(data))
    print(f"The file contains {len(data)} items.")
    print("Saved to", file_path)


# corpus = read_json(
#     "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time_sensitive_qa/test/corpus_v2.jsonl",
#     jsonl=True,
# )
corpus = pd.read_parquet("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time_sensitive_qa/test/corpus.parquet")


corpus_records = [{"id": x["docid"], "title": "", "text": x["text"]} for ix, x in corpus.iterrows()]
corpus_lst = [x["text"] for ix, x in corpus.iterrows()]


# optional: create a stemmer
stemmer = Stemmer.Stemmer("english")
tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer)
corpus_tokens = tokenizer.tokenize(corpus_lst, return_as="tuple")


retriever = bm25s.BM25(corpus=corpus_records, backend="numba")
retriever.index(corpus_tokens)

index_dir = "/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/time_sensitive_qa/bm25"

retriever.save(index_dir)
tokenizer.save_vocab(index_dir)
tokenizer.save_stopwords(index_dir)
print(f"Saved the index to {index_dir}.")

mem_use = bm25s.utils.benchmark.get_max_memory_usage()
print(f"Peak memory usage: {mem_use:.2f} GB")

queries = pd.read_parquet("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time_sensitive_qa/test/query.parquet")
queries_lst = [x["query"] for ix, x in queries.iterrows()]
query_id = [x['query_id'] for ix, x in queries.iterrows()]


# queries = read_json(
#     "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time_sensitive_qa/test/query.jsonl",
#     jsonl=True,
# )

# queries_lst = [x["query"] for x in queries]
print(f"Loaded {len(queries_lst)} queries.")

stemmer = Stemmer.Stemmer("english")

# Tokenize the queries
queries_tokenized = bm25s.tokenize(queries_lst, stemmer=stemmer, return_ids=False)

# retriever = bm25s.BM25.load(index_dir, mmap=True, load_corpus=True)
# retriever.backend = "numba"
num_docs = retriever.scores['num_docs']


print("Retrieving the top-k results...")
t = timer.start("Retrieving")
results, scores = retriever.retrieve(queries_tokenized, k=100) #, corpus=corpus)
timer.stop(t, show=True, n_total=len(queries_lst))

with open(str(Path(index_dir) / "rank.txt"), "w") as f:
    # for q, r, s in zip(queries, results, scores):
    #     qid = q["query_id"]
    for qid, r, s in zip(query_id, results, scores):
        docid_list = [x['id'] for x in r]
        for docid, _s in zip(docid_list, s):
            f.write(f"{qid}\t{docid}\t{_s}\n")
