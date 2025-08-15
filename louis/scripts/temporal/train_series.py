import subprocess, os
from pdb import set_trace as st
from pprint import pprint 
base_env = os.environ.copy()

experiments = [
    {
        "EXP_NAME": "dev1", 
        "TRAIN_GROUP_SIZE": "2",
        "PER_DEVICE_TRAIN_BATCH_SIZE": "64",
        "LEARNING_RATE": "1e-4",
        "TEMPERATURE": "0.05",
        "GRADIENT_ACCUMULATION_STEPS": "1",
        "NUM_TRAIN_EPOCHS": "1",
        "LOGGING_STEPS": "10",
        "REPORT_TO": "none",
        "MATRYOSHKA": "--matryoshka",
        "KL_LOSS": "",
        "TRUNCATED_NORMALIZE": "",
        "FILTER_FALSE_NEGATIVES": "",
        "TEMPORAL": "",
        "TEMPORAL_RECONSTRUCTION": "",
        "TEMPORAL_AS_SENTENCE": "",
        "EXTRACTED_TEMPORAL": "",
    },
    {
        "EXP_NAME": "dev2", 
        "TRAIN_GROUP_SIZE": "2",
        "PER_DEVICE_TRAIN_BATCH_SIZE": "64",
        "LEARNING_RATE": "1e-4",
        "TEMPERATURE": "0.05",
        "GRADIENT_ACCUMULATION_STEPS": "1",
        "NUM_TRAIN_EPOCHS": "1",
        "LOGGING_STEPS": "10",
        "REPORT_TO": "none",
        "MATRYOSHKA": "--matryoshka",
        "KL_LOSS": "--kl_loss",
        "TRUNCATED_NORMALIZE": "",
        "FILTER_FALSE_NEGATIVES": "",
        "TEMPORAL": "--temporal",
        "TEMPORAL_RECONSTRUCTION": "",
        "TEMPORAL_AS_SENTENCE": "",
        "EXTRACTED_TEMPORAL": "",
    },
]

for exp in experiments:
    env = {**base_env, **exp}
    subprocess.run(["bash", "/home/thuy0050/code/tevatron/louis/scripts/temporal/train_contriever_lora_series.sh"], env=env)
