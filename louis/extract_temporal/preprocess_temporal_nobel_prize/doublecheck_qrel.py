import re
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import multiprocess as mp
from multiprocess import Pool
from sutime import SUTime

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



# -----------------------------
# 1. SUTime initializer per worker
# -----------------------------
sutime = None

def init_sutime():
    """
    Fork-safe SUTime initialization.
    Each worker gets its own instance.
    """
    global sutime
    sutime = SUTime(mark_time_ranges=True, include_range=True)


# -----------------------------
# 2. Helper functions
# -----------------------------
def detect_operator(query: str):
    q = query.lower()
    if re.search(r"\bfrom\b.*\bto\b", q):
        return "from_to"
    if "before" in q:
        return "before"
    if "after" in q:
        return "after"
    if "between" in q:
        return "between"
    if re.search(r"\bin\s+", q):
        return "in"
    return None


def parse_value_to_interval(v: str):
    if v is None:
        return None
    # Single year
    if re.fullmatch(r"\d{4}", v):
        y = int(v)
        return (y, y)
    # Year-month/day
    if re.fullmatch(r"\d{4}-\d{2}(-\d{2}|-XX)?", v):
        y = int(v[:4])
        return (y, y)
    # SUTime range
    if re.fullmatch(r"\d{4}/\d{4}", v):
        a, b = v.split("/")
        return (int(a), int(b))
    # Fallback: extract any 4-digit years
    years = re.findall(r"\d{4}", v)
    if len(years) == 1:
        y = int(years[0])
        return (y, y)
    if len(years) >= 2:
        return (int(years[0]), int(years[1]))
    return None


def get_query_interval(query: str, qt_list):
    op = detect_operator(query)
    years_in_query = re.findall(r"\d{4}", query)
    if op is None:
        return None

    if op == "in":
        if qt_list:
            return parse_value_to_interval(qt_list[0]["value"])
        elif years_in_query:
            y = int(years_in_query[0])
            return (y, y)
        return None

    if op in ["between", "from_to"]:
        if len(years_in_query) >= 2:
            return (int(years_in_query[0]), int(years_in_query[1]))
        if len(qt_list) >= 2:
            v0 = parse_value_to_interval(qt_list[0]["value"])
            v1 = parse_value_to_interval(qt_list[1]["value"])
            if v0 and v1:
                return (v0[0], v1[1])
        return None

    if op == "before":
        y = int(years_in_query[0]) if years_in_query else int(re.findall(r"\d{4}", str(qt_list[0]["value"]))[0])
        return (-float("inf"), y - 1)

    if op == "after":
        y = int(years_in_query[0]) if years_in_query else int(re.findall(r"\d{4}", str(qt_list[0]["value"]))[0])
        return (y + 1, float("inf"))

    return None


def get_passage_intervals(pt_list):
    intervals = []
    for pt in pt_list:
        v = pt.get("value", None)
        if isinstance(v, str):
            iv = parse_value_to_interval(v)
            if iv:
                intervals.append(iv)
        elif isinstance(v, dict):
            for vv in v.values():
                iv = parse_value_to_interval(vv)
                if iv:
                    intervals.append(iv)
    return intervals


def intervals_overlap(a, b):
    a_start, a_end = a
    b_start, b_end = b
    return not (a_end < b_start or b_end < a_start)


def is_temporally_relevant(query, qt_list, doc_text, pt_list):
    query_interval = get_query_interval(query, qt_list)
    if query_interval is None:
        return False
    passage_intervals = get_passage_intervals(pt_list)
    for p in passage_intervals:
        if intervals_overlap(query_interval, p):
            return True
    return False


# -----------------------------
# 3. Worker function
# -----------------------------
def build_temporal_qrels(args):
    """
    args: tuple (line, query_jsonl, corpus_parquet)
    """
    line, query_jsonl, corpus_parquet = args
    global sutime

    qid, _, docid, _ = line.split()
    qid = int(qid)
    docid = int(docid)

    query = query_jsonl[qid]["query"]
    expected_wikiid = query_jsonl[qid]["docid"]

    doc_entry = corpus_parquet[corpus_parquet["docid"] == docid]
    if doc_entry.empty:
        return None

    doc_text = doc_entry["text"].values[0]
    wiki_id = doc_entry["id"].values[0]
    if wiki_id != expected_wikiid:
        return None

    qt_list = sutime.parse(query)
    pt_list = sutime.parse(doc_text)

    if is_temporally_relevant(query, qt_list, doc_text, pt_list):
        return f"{qid} 0 {docid} 1\n"
    return None

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        print("spawned")
    except RuntimeError:
        pass
    
    # worker inherits these from fork, but we must load here
    query_jsonl = read_json(
        "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time_sensitive_qa/test/query.jsonl",
        jsonl=True
    )

    new_query_jsonl = {}
    for i in query_jsonl:
        qid = i['query_id']
        new_query_jsonl[qid] = {
            "query": i['query'],
            "docid": i['docid']
        }
    query_jsonl = new_query_jsonl

    corpus_parquet = pd.read_parquet(
        "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time_sensitive_qa/wiki_corpus/corpus_with_gt_answers.parquet"
    )
    corpus_parquet['docid'] = np.arange(len(corpus_parquet))
    
    print("Loading qrel in main...")
    with open("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time_sensitive_qa/test/qrel.txt") as f:
        qrel = f.readlines()
        
     # Prepare args for workers
    task_args = ((line, query_jsonl, corpus_parquet) for line in qrel)

    # Run multiprocessing
    with Pool(63, initializer=init_sutime) as pool:
        results = list(tqdm(pool.imap(build_temporal_qrels, task_args), total=len(qrel)))

    # Filter None
    results = [r for r in results if r is not None]

    with open("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time_sensitive_qa/test/qrel_temp.txt", 'w') as f:
        for i in results:
            f.writelines(i)
    

    