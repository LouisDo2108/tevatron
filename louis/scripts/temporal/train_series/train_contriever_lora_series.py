import subprocess, os
from pdb import set_trace as st
from pprint import pprint 

def main():
    base_env = os.environ.copy()

    # Batch size 64
    experiments = [
        # {
        #     "EXP_NAME": "baseline_10epochs", 
        #     "TRAIN_GROUP_SIZE": "8",
        #     "PER_DEVICE_TRAIN_BATCH_SIZE": "64",
        #     "LEARNING_RATE": "1e-4",
        #     "TEMPERATURE": "0.05",
        #     "GRADIENT_ACCUMULATION_STEPS": "1",
        #     "NUM_TRAIN_EPOCHS": "10",
        #     "LOGGING_STEPS": "10",
        #     "REPORT_TO": "wandb",
        #     "FILTER_FALSE_NEGATIVES": "--filter_false_negatives",
        # },
        # {
        #     "EXP_NAME": "semantic_matryoshka_10epochs", 
        #     "TRAIN_GROUP_SIZE": "8",
        #     "PER_DEVICE_TRAIN_BATCH_SIZE": "64",
        #     "LEARNING_RATE": "1e-4",
        #     "TEMPERATURE": "0.05",
        #     "GRADIENT_ACCUMULATION_STEPS": "1",
        #     "NUM_TRAIN_EPOCHS": "10",
        #     "LOGGING_STEPS": "10",
        #     "REPORT_TO": "wandb",
        #     "MATRYOSHKA": "--matryoshka",
        #     "FILTER_FALSE_NEGATIVES": "--filter_false_negatives",
        # },
        # {
        #     "EXP_NAME": "temporal_10epochs", 
        #     "TRAIN_GROUP_SIZE": "8",
        #     "PER_DEVICE_TRAIN_BATCH_SIZE": "64",
        #     "LEARNING_RATE": "1e-4",
        #     "TEMPERATURE": "0.05",
        #     "GRADIENT_ACCUMULATION_STEPS": "1",
        #     "NUM_TRAIN_EPOCHS": "10",
        #     "LOGGING_STEPS": "10",
        #     "REPORT_TO": "wandb",
        #     "MATRYOSHKA": "--matryoshka",
        #     "TEMPORAL": "--temporal",
        #     "TEMPORAL_AS_SENTENCE": "--temporal_as_sentence",
        #     "FILTER_FALSE_NEGATIVES": "--filter_false_negatives",
        # },
        {
            "EXP_NAME": "dev", # "temporal_projector_10epochs", 
            "TRAIN_GROUP_SIZE": "2",
            "PER_DEVICE_TRAIN_BATCH_SIZE": "64",
            "LEARNING_RATE": "1e-4",
            "TEMPERATURE": "0.05",
            "GRADIENT_ACCUMULATION_STEPS": "1",
            "NUM_TRAIN_EPOCHS": "10",
            "LOGGING_STEPS": "10",
            "REPORT_TO": "none",
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
        # {
        #     "EXP_NAME": "temporal_projector_reconstruction_10epochs", 
        #     "TRAIN_GROUP_SIZE": "8",
        #     "PER_DEVICE_TRAIN_BATCH_SIZE": "64",
        #     "LEARNING_RATE": "1e-4",
        #     "TEMPERATURE": "0.05",
        #     "GRADIENT_ACCUMULATION_STEPS": "1",
        #     "NUM_TRAIN_EPOCHS": "10",
        #     "LOGGING_STEPS": "10",
        #     "REPORT_TO": "wandb",
        #     "MATRYOSHKA": "--matryoshka",
        #     "TEMPORAL": "--temporal",
        #     "TEMPORAL_RECONSTRUCTION": "--temporal_reconstruction",
        #     "FILTER_FALSE_NEGATIVES": "--filter_false_negatives",
        # },
        # {
        #     "EXP_NAME": "temporal_projector_reconstruction_kl_loss_10epochs", 
        #     "TRAIN_GROUP_SIZE": "8",
        #     "PER_DEVICE_TRAIN_BATCH_SIZE": "64",
        #     "LEARNING_RATE": "1e-4",
        #     "TEMPERATURE": "0.05",
        #     "GRADIENT_ACCUMULATION_STEPS": "1",
        #     "NUM_TRAIN_EPOCHS": "10",
        #     "LOGGING_STEPS": "10",
        #     "REPORT_TO": "wandb",
        #     "MATRYOSHKA": "--matryoshka",
        #     "KL_LOSS": "--kl_loss",
        #     "TEMPORAL": "--temporal",
        #     "TEMPORAL_RECONSTRUCTION": "--temporal_reconstruction",
        #     "FILTER_FALSE_NEGATIVES": "--filter_false_negatives",
        # },
    ]

    for exp in experiments:
        env = {**base_env, **exp}
        subprocess.run(["bash", "/home/thuy0050/code/tevatron/louis/scripts/temporal/train_contriever_lora_series.sh"], env=env)


if __name__ == "__main__":
    main()
