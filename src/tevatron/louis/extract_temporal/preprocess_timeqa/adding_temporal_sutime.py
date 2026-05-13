from pathlib import Path
from pdb import set_trace as st

import msgspec
from sutime import SUTime

from tevatron.louis.src.utils import read_json, write_json

sutime_instance = SUTime(mark_time_ranges=True, include_range=True)

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()


if __name__ == '__main__':
    
    train_jsonl = read_json("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time_sensitive_qa/train/modified_train.jsonl", jsonl=True)
    
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

    write_json("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/time_sensitive_qa/train/modified_train_with_sutime_temporal.jsonl", train_jsonl, jsonl=True)
