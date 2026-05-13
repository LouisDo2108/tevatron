import msgspec
import multiprocess as mp
import numpy as np
import pandas as pd
from sutime import SUTime
from tqdm.auto import tqdm
from tevatron.louis.src.utils import read_json, write_json

sutime = SUTime(mark_time_ranges=True, include_range=True)

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

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
    corpus_parquet["title"] = corpus_parquet["title"].str.lower().replace("u.s . ", "u.s. ")
    corpus_parquet["text"] = corpus_parquet["text"].str.lower().replace("u.s . ", "u.s. ")
    
    texts = corpus_parquet['text'].to_list()[:5000000]

    with mp.Pool(60, initializer=init_sutime) as pool:
        results = list(
            tqdm(pool.imap(check_idx, enumerate(texts)), total=len(texts))
        )

    # filter out None
    indices = [i for i in results if i is not None]
    np.save("/home/thuy0050/code/tevatron/louis/extract_temporal/indices1.npy", np.array(indices, dtype=int))

        