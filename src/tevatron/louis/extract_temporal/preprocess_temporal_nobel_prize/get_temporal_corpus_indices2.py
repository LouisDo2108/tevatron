import sys

import multiprocess as mp
from multiprocess import Pool, cpu_count

from pprint import pprint
from collections import defaultdict

import msgspec
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
import re 
import numpy as np

from sutime import SUTime
sutime = SUTime(mark_time_ranges=True, include_range=True)

# from python_heideltime import Heideltime
# heideltime_parser = Heideltime()
# heideltime_parser.set_document_type("NEWS")

# del heideltime_parser

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

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

    print("Reading the json file...")
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

def read_tsv(file_path, row_names=None):
    file_path = Path(file_path)
    if file_path.is_file() and file_path.suffix == ".tsv" :
        temp = pd.read_csv(file_path, sep='\t', names=row_names)
    else:
        raise ValueError("filepath is not a file or it is not a tsv file.")
    return temp

# IMPORTANT: instantiate SUTime *once* per worker, not in the main process
sutime = None

def init_sutime():
    from sutime import SUTime
    # each worker gets its own instance
    global sutime
    sutime = SUTime(mark_time_ranges=True, include_range=True)

def has_time_expression(text):
    """Return True if sutime finds at least one temporal expression."""
    global sutime
    return len(sutime.parse(text)) > 0

def check_idx(idx_text):
    global sutime
    idx, text = idx_text
    return idx if len(sutime.parse(text)) > 0 else None

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        print("spawned")
    except RuntimeError:
        pass
    
    from wikimapper import WikiMapper
    mapper = WikiMapper("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/flashrag/index_enwiki-20220820.db")
    corpus_parquet = pd.read_parquet("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/flashrag/enwiki-20211220-pages-articles.parquet")
    # corpus_parquet["title"] = corpus_parquet["title"].str.lower().replace("u.s . ", "u.s. ")
    corpus_parquet["text"] = corpus_parquet["text"].str.lower().replace("u.s . ", "u.s. ")
    # corpus_parquet = pl.from_pandas(corpus_parquet)
    
    texts = corpus_parquet['text'].to_list()[5000000:]

    with Pool(60, initializer=init_sutime) as pool:
        results = list(
            tqdm(pool.imap(check_idx, enumerate(texts)), total=len(texts))
        )

    # filter out None
    indices = [i for i in results if i is not None]
    np.save("/home/thuy0050/code/tevatron/louis/extract_temporal/indices2.npy", np.array(indices, dtype=int))

    