import logging
import os
import pickle
import sys
from contextlib import nullcontext

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from transformers.utils.import_utils import is_torch_available

from tevatron.retriever.arguments import DataArguments, ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.collator import EncodeCollator
from tevatron.retriever.dataset import EncodeDataset
from tevatron.retriever.modeling import DenseModel, EncoderOutput
import safetensors.torch
from pdb import set_trace as st
from torch import Tensor
import torch.nn as nn

logger = logging.getLogger(__name__)

class Adaptor(nn.Module):
    def __init__(self, hidden_size):
        super(Adaptor, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        return inputs + self.dense(inputs)
    

class MAdaptor(DenseModel):
    def __init__(self, *args, **kwargs):
        super(MAdaptor, self).__init__(*args, **kwargs)
        self.matryoshka_dim = self.training_args.matryoshka_dim
        self.adaptor = Adaptor(hidden_size=self.training_args.adaptor_dim)

    def encode_query(self, qry):
        query_hidden_states = self.encoder(**qry, return_dict=True)
        query_hidden_states = query_hidden_states.last_hidden_state
        
        return self._pooling(self.adaptor(query_hidden_states), qry["attention_mask"])

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        train_args: TrainingArguments,
        **hf_kwargs,
    ):
        if not os.path.exists(model_args.model_name_or_path):
            model = super().build(model_args, train_args, **hf_kwargs)
        else:
            model = cls.load(
                model_args.model_name_or_path,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                lora_name_or_path=model_args.lora_name_or_path,
                cache_dir=model_args.cache_dir,
                # torch_dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
            )
        for name, param in model.encoder.named_parameters():
            param.requires_grad = False

        get_params_info(model)

        return model

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        pooling: str = "cls",
        normalize: bool = False,
        lora_name_or_path: str = None,
        training_args: TrainingArguments = None,
        **hf_kwargs,
    ):
        try:
            default_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        except Exception as e:
            import json
            with open(os.path.join(model_name_or_path, "full_config.json")) as f:
                default_config = json.load(f)
        try:
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path, **hf_kwargs
            )
        except Exception as e:
            logger.debug(e)
            logger.warning("Either your model is not a PEFT-model or you are missing the lora_name_or_path argument.")
            logger.info("Consider your model as a normal model.")
            
            base_model = cls.try_using_flash_attn(default_config['model_name_or_path'], hf_kwargs)

        model = cls(
            encoder=base_model,
            pooling=pooling,
            normalize=normalize,
            training_args=training_args,
        )

        if os.path.exists(os.path.join(model_name_or_path, "adaptor.safetensors")):
            adaptor_state_dict = safetensors.torch.load_file(
                os.path.join(model_name_or_path, "adaptor.safetensors"), device="cpu"
            )
            model.adaptor.load_state_dict(adaptor_state_dict)
            print("Loaded adaptor")
        else:
            print("There is no adaptor state dict")

        return model


def get_params_info(model):
    all_param = 0
    trainable_param = 0

    print("\nAll trainable parameters:")
    for name, param in model.named_parameters():
        
        if name.startswith("base_model."):
            # This is the duplicate of the base model for KL loss
           continue 
        all_param += param.numel()
        
        if param.requires_grad:
            trainable_param += param.numel()
            print(name, param.numel())
            
    print(f"trainable params: {trainable_param:,} || all params: {all_param:,} || trainable%: {trainable_param / all_param * 100:.2f}")


def set_seed(seed: int, deterministic: bool = True):
    # Copy from transformers.trainer_utilss.set_seed with some modifications
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
        if deterministic:
            # set a debug environment variable CUBLAS_WORKSPACE_CONFIG to :16:8 (may limit overall performance) or :4096:8 (will increase library footprint in GPU memory by approximately 24MiB). From https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

            # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
            torch.use_deterministic_algorithms(True)

            # # Enable CUDNN deterministic mode
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.padding_side == 'right':
        tokenizer.padding_side = 'right'
    else:
        tokenizer.padding_side = 'left'

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = MAdaptor.load(
        model_args.model_name_or_path,
        pooling=model_args.pooling,
        normalize=model_args.normalize,
        lora_name_or_path=model_args.lora_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        training_args=training_args,
    )

    encode_dataset = EncodeDataset(
        data_args=data_args,
    )

    encode_collator = EncodeCollator(
        data_args=data_args,
        tokenizer=tokenizer,
    )

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()

    print(
        f"Using {type(model)} from this pretrained path {model_args.model_name_or_path}."
    )
    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        with (
            torch.autocast(
                "cuda", dtype=torch.float16 if training_args.fp16 else torch.bfloat16
            )
            if training_args.fp16 or training_args.bf16
            else nullcontext()
        ):
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_query:
                    model_output: EncoderOutput = model(query=batch)
                    encoded.append(model_output.q_reps.cpu().detach().numpy())
                else:
                    model_output: EncoderOutput = model(passage=batch)
                    encoded.append(model_output.p_reps.cpu().detach().numpy())

    encoded = np.concatenate(encoded).astype(np.float16)

    with open(data_args.encode_output_path, "wb") as f:
        pickle.dump((encoded, lookup_indices), f)


if __name__ == "__main__":
    main()
