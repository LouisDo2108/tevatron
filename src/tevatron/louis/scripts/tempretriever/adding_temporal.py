from pdb import set_trace as st
from tqdm.auto import tqdm
from tevatron.louis.src.utils import read_json, write_json

from sutime import SUTime
sutime = SUTime(mark_time_ranges=True, include_range=True)


def main():
    train_jsonl = read_json(
        "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/temporal_nobel_prize/train/backup/train_temporal_v2.jsonl",
        jsonl=True
    )
    
    for item in tqdm(train_jsonl):
        item["temporal"] = [x['text'] for x in sutime.parse(item["query"])]

    write_json(
        "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/temporal_nobel_prize/train/backup/train_temporal_v3.jsonl",
        train_jsonl,
        jsonl=True
    )


if __name__ == "__main__":
    main()
