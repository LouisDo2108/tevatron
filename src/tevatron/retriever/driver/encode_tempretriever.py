import logging
import os
import pickle
import sys
from typing import Dict, List, Optional
from torch import Tensor
from contextlib import nullcontext
from datasets import load_dataset, load_from_disk

from transformers import PreTrainedTokenizer
from dataclasses import dataclass
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from transformers.utils.import_utils import is_torch_available

from tevatron.retriever.arguments import DataArguments, ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
# from tevatron.retriever.collator import EncodeCollator
from torch.utils.data import Dataset

# from tevatron.retriever.dataset import EncodeDataset
from tevatron.retriever.modeling import DenseModel
from transformers.file_utils import ModelOutput

from sutime import SUTime
# sutime = SUTime(mark_time_ranges=True, include_range=True)

def init_sutime():
    global sutime_instance
    sutime_instance = SUTime(mark_time_ranges=True, include_range=True)
    return sutime_instance

logger = logging.getLogger(__name__)

import torch.multiprocessing as mp
try:
    mp.set_start_method('fork', force=True) # spawn
except Exception as e:
    logger.info("forked")


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    qt_reps: Optional[Tensor] = None
    pt_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncodeDataset(Dataset):
    """
    Dataset for encoding.
    Loads data and optionally shards it for distributed processing.
    """

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        self.encode_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
            num_proc=self.data_args.num_proc,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item):
        content = self.encode_data[item]

        self.data_args.passage_prefix = (
            self.data_args.passage_prefix.replace("\\n", "\n").strip() + " "
        )
        self.data_args.query_prefix = (
            self.data_args.query_prefix.replace("\\n", "\n").strip() + " "
        )

        if self.data_args.encode_is_query:
            # content_id = content['query_id']
            content_id = content.get("query_id", "")
            if content_id == "":
                content_id = content.get("_id", "")  # For NanoNQ
            content_text = content.get("query_text", content.get("query", ""))
            if content_text == "":
                content_text = content.get("text", "")  # For NanoNQ
            content_text = self.data_args.query_prefix + content_text
            content_image = content.get("query_image", None)
            content_video = content.get("query_video", None)
            content_audio = content.get("query_audio", None)
        else:
            # content_id = content['docid']
            content_id = content.get("docid", "")
            if content_id == "":
                content_id = content.get("_id", "")  # For NanoNQ
            content_text = content.get("text", "")
            if "title" in content:
                content_text = content["title"] + " " + content_text
            content_text = self.data_args.passage_prefix + content_text.strip()
            content_image = content.get("image", None)
            content_video = content.get("video", None)
            content_audio = content.get("audio", None)

        if content_video is not None and self.data_args.encode_video:
            content_video = os.path.join(self.data_args.assets_path, content_video)
            # check if the file exists
            if not os.path.exists(content_video):
                logger.warning(f"Video file {content_video} does not exist.")
                content_video = None

        if (
            content_audio is not None
        ):  # either an dict with 'array' key or a string .mp3 path
            if isinstance(content_audio, dict) and "array" in content_audio:
                content_audio = content_audio["array"]
            else:
                assert isinstance(content_audio, str) and content_audio.endswith(".mp3")
                content_audio = os.path.join(self.data_args.assets_path, content_audio)
                # check if the file exists
                if not os.path.exists(content_audio):
                    logger.warning(f"Audio file {content_audio} does not exist.")
                    content_audio = None

        if not self.data_args.encode_text:
            content_text = None
        if not self.data_args.encode_image:
            content_image = None
        if not self.data_args.encode_video:
            content_video = None
        if not self.data_args.encode_audio:
            content_audio = None

        content_temporal = sutime_instance.parse(content_text)
        content_temporal = ", ".join([t["text"] for t in content_temporal])

        return content_id, content_text, content_temporal


@dataclass
class EncodeCollator:
    """
    simple collator for text only data.
    """

    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features):
        """
        Collate function for encoding.
        :param features: list of (id, text, image) tuples
        but in this case, it's just image is None
        """
        content_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        temporal = [x[2] for x in features]

        max_length = (
            self.data_args.query_max_len
            if self.data_args.encode_is_query
            else self.data_args.passage_max_len
        )
        collated_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=(
                max_length - 1 if self.data_args.append_eos_token else max_length
            ),
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            padding_side=self.data_args.padding_side,
        )
        collated_temporal_inputs = self.tokenizer(
            temporal,
            padding=True,
            truncation=True,
            max_length=(
                max_length - 1 if self.data_args.append_eos_token else max_length
            ),
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
            padding_side=self.data_args.padding_side,
        )
        return content_ids, collated_inputs, collated_temporal_inputs


class TempRetriever(DenseModel):
    def __init__(self, *args, **kwargs):
        super(TempRetriever, self).__init__(*args, **kwargs)

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        query_temporal: Dict[str, Tensor] = None,
        passage: Dict[str, Tensor] = None,
        passage_temporal: Dict[str, Tensor] = None,
    ):

        # Copy from EncoderModel's forward
        q_reps = self.encode_query(query) if query else None
        qt_reps = self.encode_query(query_temporal) if query_temporal else None

        p_reps = self.encode_query(passage) if passage else None
        pt_reps = self.encode_query(passage_temporal) if passage_temporal else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(q_reps=q_reps, qt_reps=qt_reps, p_reps=p_reps, pt_reps=pt_reps)

        # for eval
        scores = self.compute_similarity(q_reps, p_reps)
        loss = None

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            qt_reps=qt_reps,
            p_reps=p_reps,
            pt_reps=pt_reps,
        )


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

    model = TempRetriever.load(
        model_args.model_name_or_path,
        pooling=model_args.pooling,
        normalize=model_args.normalize,
        lora_name_or_path=model_args.lora_name_or_path,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )
    model.matryoshka_dim = training_args.matryoshka_dim

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
        worker_init_fn=lambda _: init_sutime(),
    )

    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    model.eval()

    print(
        f"Using {type(model)} from this pretrained path {model_args.model_name_or_path}."
    )
    for (batch_ids, batch, batch_temporal) in tqdm(encode_loader):
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
                    batch_temporal[k] = v.to(training_args.device)
                if data_args.encode_is_query:
                    model_output: EncoderOutput = model(query=batch, query_temporal=batch_temporal)
                    q_reps = model_output.q_reps.cpu().detach().numpy()
                    qt_reps = model_output.qt_reps.cpu().detach().numpy()
                    vectors = np.concatenate([q_reps, qt_reps], axis=1)
                    print(vectors.shape)
                    encoded.append(vectors)
                else:
                    model_output: EncoderOutput = model(passage=batch, passage_temporal=batch_temporal)
                    p_reps = model_output.p_reps.cpu().detach().numpy()
                    pt_reps = model_output.pt_reps.cpu().detach().numpy()
                    vectors = np.concatenate([p_reps, pt_reps], axis=1)
                    print(vectors.shape)
                    encoded.append(vectors)

    encoded = np.concatenate(encoded)

    with open(data_args.encode_output_path, "wb") as f:
        pickle.dump((encoded, lookup_indices), f)


if __name__ == "__main__":
    main()
