import os
import subprocess
import argparse
import re
import msgspec
from pathlib import Path
from msgspec.json import format
from pprint import pformat, pprint
from typing import Union, List, Optional
import pandas as pd
import bm25s
import Stemmer  # optional: for stemming
from tevatron.louis.src.utils import read_json, run

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

timer = bm25s.utils.benchmark.Timer("[BM25S]")

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="temporal_nobel_prize",
        choices=[
            "temporal_nobel_prize",
            "time_sensitive_qa",
        ],
        help="Dataset name",
    )
    args = parser.parse_args()
    DATA_ROOT_DIR = "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal"
    
    if args.data == "temporal_nobel_prize":
        corpus_path = Path(f"{DATA_ROOT_DIR}/{args.data}/test/corpus.jsonl")
        query_path = Path(f"{DATA_ROOT_DIR}/{args.data}/test/query.jsonl")
    else:
        corpus_path = Path(f"{DATA_ROOT_DIR}/{args.data}/test/corpus.parquet")
        query_path = Path(f"{DATA_ROOT_DIR}/{args.data}/test/query.parquet")
    index_dir = Path(f"/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/{args.data}/bm25")
    index_dir.mkdir(parents=True, exist_ok=True)
    
    if corpus_path.suffix == ".jsonl":
        corpus = read_json(
            corpus_path,
            jsonl=True,
        )
        corpus_records = [
            {"id": x["docid"], "title": "", "text": x["text"]} for x in corpus
        ]
        corpus_lst = [x["text"] for x in corpus]
    elif corpus_path.suffix == ".parquet":
        corpus = pd.read_parquet(corpus_path)
        corpus_records = [
            {"id": x["docid"], "title": "", "text": x["text"]} for ix, x in corpus.iterrows()
        ]
        corpus_lst = [x["text"] for ix, x in corpus.iterrows()]
    
    if query_path.suffix == ".jsonl":
        queries = read_json(
            query_path, 
            jsonl=True
        )
        queries_lst = [x["query"] for x in queries]
        print(f"Loaded {len(queries_lst)} queries.")
    elif query_path.suffix == ".parquet":
        queries = pd.read_parquet(query_path)
        queries_lst = [x["query"] for ix, x in queries.iterrows()]
        query_id = [x['query_id'] for ix, x in queries.iterrows()]
        print(f"Loaded {len(queries_lst)} queries.")

    # optional: create a stemmer
    stemmer = Stemmer.Stemmer("english")
    tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer)
    corpus_tokens = tokenizer.tokenize(corpus_lst, return_as="tuple")

    retriever = bm25s.BM25(corpus=corpus_records, backend="numba")
    retriever.index(corpus_tokens)
    
    retriever.save(str(index_dir))
    tokenizer.save_vocab(str(index_dir))
    tokenizer.save_stopwords(str(index_dir))
    print(f"Saved the index to {str(index_dir)}.")

    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    print(f"Peak memory usage: {mem_use:.2f} GB")


    # Tokenize the queries
    queries_tokenized = bm25s.tokenize(queries_lst, stemmer=stemmer, return_ids=False)

    retriever = bm25s.BM25.load(index_dir, mmap=True, load_corpus=True)
    retriever.backend = "numba"
    num_docs = retriever.scores['num_docs']

    print("Retrieving the top-k results...")
    t = timer.start("Retrieving")
    results, scores = retriever.retrieve(
        queries_tokenized, k=100, # , corpus=corpus
    )
    timer.stop(t, show=True, n_total=len(queries_lst))

    with open(str(Path(index_dir) / "rank.txt"), "w") as f:
        if args.data == "temporal_nobel_prize":
            for q, r, s in zip(queries, results, scores):
                qid = q["query_id"]
                docid_list = [x['docid'] for x in r]
                for docid, _s in zip(docid_list, s):
                    f.write(f"{qid}\t{docid}\t{_s}\n")
        else:
            for qid, r, s in zip(query_id, results, scores):
                docid_list = [x['id'] for x in r]
                for docid, _s in zip(docid_list, s):
                    f.write(f"{qid}\t{docid}\t{_s}\n")

    retrieval_cmd = f"""
        mamba deactivate && 
        mamba activate tevatron &&
        python -m tevatron.utils.format.convert_result_to_trec \
            --input {index_dir}/rank.txt \
            --output {index_dir}/rank.trec \
            --remove_query

        && python -m pyserini.eval.trec_eval -c \
            -m recall.10,100 -m ndcg_cut.10 -M 100 \
            {DATA_ROOT_DIR}/{args.data}/test/qrel.txt \
            {index_dir}/rank.trec > {index_dir}/out.txt
    """
    run(retrieval_cmd)