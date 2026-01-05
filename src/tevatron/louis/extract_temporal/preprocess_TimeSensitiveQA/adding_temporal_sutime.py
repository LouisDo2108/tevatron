from multiprocessing import Pool, cpu_count

from wikimapper import WikiMapper

from pdb import set_trace as st
from pprint import pprint
from collections import defaultdict
import pandas as pd

import msgspec
from tqdm import tqdm
from pathlib import Path

from sutime import SUTime
sutime_instance = SUTime(mark_time_ranges=True, include_range=True)

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


if __name__ == '__main__':
    
    train_jsonl = read_json("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time_sensitive_qa/train/train.jsonl", jsonl=True)
    
    for ix, sample in enumerate(train_jsonl):
        query = sample['query']
        query_temporal = sutime_instance.parse(query)  
        query_temporal = ", ".join([t["text"] for t in query_temporal])
        sample['temporal'] = [query_temporal]
        
        positive_passages = sample['positive_passages']
        
        for passage in positive_passages:
            passage_text = passage['text']
            passage_temporal = sutime_instance.parse(passage_text)  
            passage_temporal = ", ".join([t["text"] for t in passage_temporal])
            passage['temporal'] = [passage_temporal]
            
        # break
    write_json("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time_sensitive_qa/train/train_with_sutime_temporal.jsonl", train_jsonl, jsonl=True)
