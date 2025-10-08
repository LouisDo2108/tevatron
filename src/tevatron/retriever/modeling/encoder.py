import logging
from dataclasses import dataclass
from pprint import pprint
from typing import Dict, Optional
from copy import deepcopy

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, cast_mixed_precision_params

from torch import Tensor, nn
from transformers import AutoModel, PreTrainedModel
from transformers.file_utils import ModelOutput

from tevatron.retriever.arguments import ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments

from pdb import set_trace as st 

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 training_args: TrainingArguments = None,
                 base_model: PreTrainedModel = None,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        self._keys_to_ignore_on_save = None
        self.training_args = training_args
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.matryoshka_dim = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # For KL loss
        self.base_model = base_model
        if self.base_model is not None:
            for name, param in self.base_model.named_parameters():
                param.requires_grad = False

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None): 
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores / self.temperature, target)
            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.model.gradient_checkpointing_enable()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):  
        base_model = cls.try_using_flash_attn(model_args.model_name_or_path, hf_kwargs)

        if train_args.kl_loss:
            kl_loss_base_model = deepcopy(base_model)
        else:
            kl_loss_base_model = None

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path, is_trainable=True)
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=(
                        "all-linear"
                        if model_args.lora_target_modules == "all-linear"
                        else model_args.lora_target_modules.split(",")
                    ),
                    inference_mode=False,
                    use_rslora=False,
                    modules_to_save=train_args.modules_to_save if train_args.modules_to_save else None,
                )
                lora_model = get_peft_model(base_model, lora_config)

            cast_mixed_precision_params(
                lora_model, dtype=torch.float16 if train_args.fp16 else torch.bfloat16 if train_args.bf16 else torch.float32
            )

            # print(lora_model.get_layer_status())
            # print(lora_model.get_model_status())
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                training_args=train_args,
                base_model=kl_loss_base_model if train_args.kl_loss else None
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                training_args=train_args,
                base_model=kl_loss_base_model if train_args.kl_loss else None,
            )
        return model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             **hf_kwargs):
        """
        Slightly modify how to load the LoRA fine-tuned model to disable this warning: 
        UserWarning: Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!
        This is because the base_model will load a checkpoint that already has peft_config attribute in it.
        """

        # if lora_name_or_path:
        try:
            # Always try to load as a PEFT model first
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
        except Exception as e:
            logger.debug(e)
            logger.warning("Either your model is not a PEFT-model or you are missing the lora_name_or_path argument.")
            logger.info("Consider your model as a normal model.")
            lora_config = None

        if lora_config is not None:
            logger.info(" Loaded LORA config!!! ")
            base_model = cls.try_using_flash_attn(lora_config.base_model_name_or_path, hf_kwargs)
            lora_model = PeftModel.from_pretrained(
                base_model,
                model_name_or_path,
                config=lora_config,
            )
            # lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize
            )
        else:
            logger.info(" This is not a PEFT model!!! ")
            base_model = cls.try_using_flash_attn(model_name_or_path, hf_kwargs)
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize
            )
        # print("Please provide lora_name_or_path to load the PEFT model correctly!!!")
        return model

    @classmethod
    def try_using_flash_attn(cls, model_name_or_path, hf_kwargs):
        try:
            hf_kwargs["attn_implementation"] = "flash_attention_2"
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                    model_name_or_path, 
                    trust_remote_code=True,
                    weights_only=False,
                    **hf_kwargs
                )
            logger.info("Using flash attention 2!")
        except Exception as e:
            # logger.exception(e)
            hf_kwargs["attn_implementation"] = "sdpa"
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                    model_name_or_path, 
                    weights_only=False,
                    trust_remote_code=True,
                    **hf_kwargs
                )
            logger.info(f"Fall back to use {hf_kwargs['attn_implementation']}")

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        return base_model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
