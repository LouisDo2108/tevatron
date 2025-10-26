from pdb import set_trace as st
from sutime import SUTime
sutime = SUTime(mark_time_ranges=True, include_range=True)

from pathlib import Path
import msgspec
from tqdm import tqdm
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


def main():
    train_jsonl = read_json(
        "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/temporal_nobel_prize/train/backup/train_temporal_v2.jsonl",
        jsonl=True
    )
    
    # new_train_jsonl = []
    
    for item in tqdm(train_jsonl):
        item["temporal"] = [x['text'] for x in sutime.parse(item["query"])]

    write_json(
        "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/temporal_nobel_prize/train/backup/train_temporal_v3.jsonl",
        train_jsonl,
        jsonl=True
    )

if __name__ == "__main__":
    main()
