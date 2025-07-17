import os

from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/temporal/nobel_prize/hf",
    repo_id="LouisDo2108/temporal-nobel-prize",
    repo_type="dataset",
)