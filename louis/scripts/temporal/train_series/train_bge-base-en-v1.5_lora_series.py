import subprocess, os
from pdb import set_trace as st
from pprint import pprint 

def main():
    base_env = os.environ.copy()

    experiments = [
        {
            "EXP_NAME": "baseline", 
            "TRAIN_GROUP_SIZE": "2",
            "PER_DEVICE_TRAIN_BATCH_SIZE": "128",
            "LEARNING_RATE": "1e-4",
            "TEMPERATURE": "0.02",
            "GRADIENT_ACCUMULATION_STEPS": "4",
            "NUM_TRAIN_EPOCHS": "5",
            "LOGGING_STEPS": "10",
            "REPORT_TO": "wandb",
            "FILTER_FALSE_NEGATIVES": "--filter_false_negatives",
        },
        {
            "EXP_NAME": "semantic_matryoshka", 
            "TRAIN_GROUP_SIZE": "2",
            "PER_DEVICE_TRAIN_BATCH_SIZE": "128",
            "LEARNING_RATE": "1e-4",
            "TEMPERATURE": "0.02",
            "GRADIENT_ACCUMULATION_STEPS": "4",
            "NUM_TRAIN_EPOCHS": "5",
            "LOGGING_STEPS": "10",
            "REPORT_TO": "wandb",
            "MATRYOSHKA": "--matryoshka",
            "FILTER_FALSE_NEGATIVES": "--filter_false_negatives",
        },
        {
            "EXP_NAME": "temporal", 
            "TRAIN_GROUP_SIZE": "2",
            "PER_DEVICE_TRAIN_BATCH_SIZE": "128",
            "LEARNING_RATE": "1e-4",
            "TEMPERATURE": "0.02",
            "GRADIENT_ACCUMULATION_STEPS": "4",
            "NUM_TRAIN_EPOCHS": "5",
            "LOGGING_STEPS": "10",
            "REPORT_TO": "wandb",
            "MATRYOSHKA": "--matryoshka",
            "TEMPORAL": "--temporal",
            "TEMPORAL_AS_SENTENCE": "--temporal_as_sentence",
            "FILTER_FALSE_NEGATIVES": "--filter_false_negatives",
        },
        {
            "EXP_NAME": "temporal_projector", 
            "TRAIN_GROUP_SIZE": "2",
            "PER_DEVICE_TRAIN_BATCH_SIZE": "128",
            "LEARNING_RATE": "1e-4",
            "TEMPERATURE": "0.02",
            "GRADIENT_ACCUMULATION_STEPS": "4",
            "NUM_TRAIN_EPOCHS": "5",
            "LOGGING_STEPS": "10",
            "REPORT_TO": "wandb",
            "MATRYOSHKA": "--matryoshka",
            "KL_LOSS": "",
            "TRUNCATED_NORMALIZE": "",
            "FILTER_FALSE_NEGATIVES": "",
            "TEMPORAL": "--temporal",
            "TEMPORAL_RECONSTRUCTION": "",
            "TEMPORAL_AS_SENTENCE": "",
            "EXTRACTED_TEMPORAL": "--extracted_temporal",
            "FILTER_FALSE_NEGATIVES": "--filter_false_negatives",
        },
        {
            "EXP_NAME": "temporal_projector_reconstruction", 
            "TRAIN_GROUP_SIZE": "2",
            "PER_DEVICE_TRAIN_BATCH_SIZE": "128",
            "LEARNING_RATE": "1e-4",
            "TEMPERATURE": "0.02",
            "GRADIENT_ACCUMULATION_STEPS": "4",
            "NUM_TRAIN_EPOCHS": "5",
            "LOGGING_STEPS": "10",
            "REPORT_TO": "wandb",
            "MATRYOSHKA": "--matryoshka",
            "TEMPORAL": "--temporal",
            "TEMPORAL_RECONSTRUCTION": "--temporal_reconstruction",
            "FILTER_FALSE_NEGATIVES": "--filter_false_negatives",
        },
        {
            "EXP_NAME": "temporal_projector_reconstruction_kl_loss", 
            "TRAIN_GROUP_SIZE": "2",
            "PER_DEVICE_TRAIN_BATCH_SIZE": "128",
            "LEARNING_RATE": "1e-4",
            "TEMPERATURE": "0.02",
            "GRADIENT_ACCUMULATION_STEPS": "4",
            "NUM_TRAIN_EPOCHS": "5",
            "LOGGING_STEPS": "10",
            "REPORT_TO": "wandb",
            "MATRYOSHKA": "--matryoshka",
            "KL_LOSS": "--kl_loss",
            "TEMPORAL": "--temporal",
            "TEMPORAL_RECONSTRUCTION": "--temporal_reconstruction",
            "FILTER_FALSE_NEGATIVES": "--filter_false_negatives",
        },
    ]

    for exp in experiments:
        env = {**base_env, **exp}
        subprocess.run(["sbatch", "/home/thuy0050/code/tevatron/louis/scripts/temporal/train_bge-base-en-v1.5_lora_series.sh"], env=env)


if __name__ == "__main__":
    main()
