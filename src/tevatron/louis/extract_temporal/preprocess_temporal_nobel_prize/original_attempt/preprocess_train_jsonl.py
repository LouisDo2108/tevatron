
import msgspec
from tqdm.auto import tqdm
import pandas as pd
from pathlib import Path
import re

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from sutime import SUTime

global sutime, heideltime_parser
sutime = SUTime(mark_time_ranges=True, include_range=True)

from python_heideltime import Heideltime

heideltime_parser = Heideltime()
heideltime_parser.set_document_type("NEWS")

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

timex_text_pattern = re.compile(r"<TIMEX3[^>]*>(.*?)</TIMEX3>")
timex_value_pattern = re.compile(r'<TIMEX3[^>]*\bvalue="([^"]+)"')
timex_pattern = re.compile(r'<TIMEX3[^>]*\bvalue="([^"]+)"[^>]*>(.*?)</TIMEX3>')


def read_json(file_path, jsonl=False):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError("filepath is not a file")
    # if not file_path.suffix == ".jsonl" and jsonl:
    #     raise ValueError("the file is not jsonl")
    # if not file_path.suffix == ".json" and not jsonl:
    #     raise ValueError("the file is not json")
    file_path = file_path.__str__()

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
    if file_path.is_file() and file_path.suffix == ".tsv":
        temp = pd.read_csv(file_path, sep="\t", names=row_names)
    else:
        raise ValueError("filepath is not a file or it is not a tsv file.")
    return temp


def parse_heideltime_result(heideltime_result, flatten=False):
    # root = etree.fromstring(heideltime_result, parser=parser)

    # values = []
    # for el in root.iter():
    #     if el.tag == "TIMEX3":
    #         value = el.attrib.get("value", "")
    #         if value and "2025" not in value and not value.startswith("P"):
    #             values.append({"text": el.text or "", "value": value})

    heideltime_result = heideltime_result.split("\n")[3]
    values = timex_text_pattern.findall(heideltime_result)

    # if flatten:
    #     return [field for item in values for field in (item["text"], item["value"])]

    return values


def parse_sutime_result(sutime_result, flatten=False):
    values = [x.get("text") for x in sutime_result if x.get("text")]
    # for x in sutime_result:
    #     text = x.get("text", "")
    #     # value = x.get("value", "")

    #     if isinstance(value, dict):
    #         value = ""
    #     elif not value or ("2025" not in value and value.startswith("P")):
    #         value = ""

    #     if flatten:
    #         values.append(text)
    #         if value:
    #             values.append(value)
    #     else:
    #         values.append({"text": text, "value": value})

    return values

DATA_PATH = Path(
    "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/temporal_nobel_prize"
)
SPLIT = "train"

TRAIN_JSONL_PATH = DATA_PATH / SPLIT / "train.jsonl"
train_jsonl = read_json(TRAIN_JSONL_PATH, jsonl=True)


def process_passage(item):
    text = item["text"]

    sutime_result = parse_sutime_result(sutime.parse(text))
    heideltime_result = parse_heideltime_result(heideltime_parser.parse(text))

    item["temporal"] = list(set(sutime_result + heideltime_result))
    return item  # Optional: useful if you're returning to a new structure

all_passages = []

for line in train_jsonl:
    all_passages.extend(line["positive_passages"])
    all_passages.extend(line["negative_passages"])

with ThreadPoolExecutor(max_workers=12) as executor:
    futures = [executor.submit(process_passage, item) for item in all_passages]
    for _ in tqdm(as_completed(futures), total=len(futures)):
        pass  # We just wait for all tasks to complete, since results are stored in-place


write_json(
    "/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/temporal_nobel_prize/train/train_temporal.jsonl",
    train_jsonl,
    jsonl=True,
)
