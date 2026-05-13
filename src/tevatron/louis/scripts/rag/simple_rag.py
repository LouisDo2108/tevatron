import datetime
import json
import logging
import os
import pickle
import random
import re
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from pdb import set_trace as st
from time import perf_counter
from typing import Any, Dict, Generator, List, Optional, Union

import faiss
import numpy as np
import torch
import yaml
from datasets import load_dataset
from flashrag.evaluator import Evaluator
from flashrag.generator import BaseGenerator
from flashrag.generator.utils import resolve_max_tokens
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
from flashrag.retriever import BaseTextRetriever, load_corpus, load_docs
from flashrag.utils import get_refiner
from peft import LoraConfig, PeftModel
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.hf_argparser import HfArgumentParser
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.import_utils import is_torch_available
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

from tevatron.louis.src.utils import set_seed
from tevatron.retriever.arguments import DataArguments, ModelArguments
from tevatron.retriever.arguments import TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.modeling import EncoderModel

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VLLMGenerator(BaseGenerator):
    """Class for decoder-only generator, based on vllm."""

    def __init__(self, config):
        super().__init__(config)
        self.model = None

        # if self.model_path == "":
        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_batched_tokens=config["max_num_batched_tokens"],
            max_num_seqs=config["max_num_seqs"],  # Limit batch size
            max_model_len=config["max_model_len"],  # Limit context window
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            generation_config="auto",
            disable_cascade_attn=True,  # Avoid gibberish output due to batch inference
            seed=config["seed"],
            enforce_eager=config["enforce_eager"], # disable graph capturing completely if True
        )
        # else:
        #     self.model = LLM(
        #         self.model_path,
        #         tensor_parallel_size=self.tensor_parallel_size,
        #         gpu_memory_utilization=self.gpu_memory_utilization,
        #         max_logprobs=32016,
        #         max_model_len=self.max_model_len,
        #     )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def update_additional_setting(self):
        if "gpu_memory_utilization" not in self._config:
            self.gpu_memory_utilization = 0.85
        else:
            self.gpu_memory_utilization = self._config["gpu_memory_utilization"]
        if self.gpu_num != 1 and self.gpu_num % 2 != 0:
            self.tensor_parallel_size = self.gpu_num - 1
        else:
            self.tensor_parallel_size = self.gpu_num

        self.lora_path = (
            None
            if "generator_lora_path" not in self._config
            else self._config["generator_lora_path"]
        )
        self.use_lora = False
        if self.lora_path is not None:
            self.use_lora = True
        self.max_model_len = self._config["generator_max_input_len"]

    def generate(
        self,
        input_list: List[str],
        return_raw_output=False,
        return_scores=False,
        **params,
    ):

        if isinstance(input_list, str):
            input_list = [input_list]

        # Applying Qwen's chat tempolate, disabling thinking, according to https://qwen.readthedocs.io/en/latest/deployment/vllm.html#python-library
        messages = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": x}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._config["enable_thinking"],
            )
            for x in input_list
        ]

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            do_sample_flag = generation_params.pop("do_sample")
            if not do_sample_flag:
                generation_params["temperature"] = 0
        generation_params["seed"] = self._config["seed"]

        # handle param conflict
        generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=False)

        # # fix for llama3
        # if "stop" in generation_params:
        #     generation_params["stop"].append("<|eot_id|>")
        #     generation_params["include_stop_str_in_output"] = True
        # else:
        #     generation_params["stop"] = ["<|eot_id|>"]

        if return_scores:
            if "logprobs" not in generation_params:
                generation_params["logprobs"] = 100

        sampling_params = SamplingParams(**generation_params)

        if self.use_lora:
            from vllm.lora.request import LoRARequest

            outputs = self.model.generate(
                input_list,
                sampling_params,
                lora_request=LoRARequest("lora_module", 1, self.lora_path),
            )
        else:
            outputs = self.model.generate(messages, sampling_params)

        if return_raw_output:
            base_output = outputs
        else:
            generated_texts = [output.outputs[0].text for output in outputs]
            base_output = generated_texts
        if return_scores:
            scores = []
            for output in outputs:
                logprobs = output.outputs[0].logprobs
                scores.append(
                    [
                        np.exp(list(score_dict.values())[0].logprob)
                        for score_dict in logprobs
                    ]
                )
            return base_output, scores
        else:
            return base_output


class Encoder(EncoderModel):

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 instruction: str = "",
                 data_args=None,
                 ):
        super().__init__(encoder, pooling, normalize, temperature)
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.data_args = data_args

    def encode_query(self, qry):
        if self.encoder.name_or_path != "jinaai/jina-embeddings-v3":
            query_hidden_states = self.encoder(**qry, return_dict=True)
            query_hidden_states = query_hidden_states.last_hidden_state
            return self._pooling(query_hidden_states, qry["attention_mask"])
        else:
            task = 'retrieval.query'
            task_id = self.encoder._adaptation_map[task]
            adapter_mask = torch.full((qry['input_ids'].size(0),), task_id, dtype=torch.int32, device=qry['input_ids'].device)
            query_hidden_states = self.encoder(
                **qry, return_dict=True, adapter_mask=adapter_mask,
            )
            query_hidden_states = query_hidden_states.last_hidden_state[:, :, :768]
            return self._pooling(query_hidden_states, qry["attention_mask"])

    def encode_passage(self, psg):
        # encode passage is the same as encode query
        if self.encoder.name_or_path != "jinaai/jina-embeddings-v3":
            return self.encode_query(psg)
        else:
            task = "retrieval.passage"
            task_id = self.encoder._adaptation_map[task]
            adapter_mask = torch.full((psg["input_ids"].size(0),),task_id,dtype=torch.int32,device=psg["input_ids"].device,)
            query_hidden_states = self.encoder(**psg, return_dict=True, adapter_mask=adapter_mask,)
            query_hidden_states = query_hidden_states.last_hidden_state[:, :, :768]
            return self._pooling(query_hidden_states, psg["attention_mask"])

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling in ['cls', 'first']:
            reps = last_hidden_state[:, 0]
        elif self.pooling in ['mean', 'avg', 'average']:
            masked_hiddens = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            reps = masked_hiddens.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pooling in ['last', 'eos']:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                reps = last_hidden_state[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_state.shape[0]
                reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise ValueError(f'unknown pooling method: {self.pooling}')
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def parse_query(self, query_list):
        """
        processing query for different encoders
        """
        global _has_printed_instruction

        if isinstance(query_list, str):
            query_list = [query_list]

        if self.instruction is not None:
            self.instruction = self.instruction.strip() + " " if self.instruction else ""
        else:
            self.instruction = self.set_default_instruction(is_zh=judge_zh(query_list[0]))

        if not _has_printed_instruction:
            if self.instruction == "":
                warnings.warn('Instruction is not set')
            else:
                print(f"Use `{self.instruction}` as retrieval instruction")
            _has_printed_instruction = True

        query_list = [self.instruction + query for query in query_list]

        return query_list

    def set_default_instruction(self, is_zh=False): 
        model_name = self.model_args.model_name_or_path.lower()
        instruction = ""  

        if "e5" in model_name:
            if self.data_args.encode_is_query:
                instruction = "query: "
            else:
                instruction = "passage: "

        if "bge" in model_name:
            if self.data_args.encode_is_query:
                if "zh" in model_name.lower() or is_zh:
                    instruction = "为这个句子生成表示以用于检索相关文章："
                else:
                    instruction = (
                        "Represent this sentence for searching relevant passages: "
                    )


        return instruction

    @torch.inference_mode()
    def single_batch_encode(
        self, query_list: Union[List[str], str], is_query=True
    ) -> np.ndarray:
        query_list = self.parse_query(query_list)

        inputs = self.tokenizer(
            query_list,
            padding=True,
            truncation=True,
            max_length=self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len,
            return_tensors="pt",
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        inputs = {k: v.cuda() for k, v in inputs.items()}

        # if "T5" in type(self.model).__name__ or (
        #     isinstance(self.model, torch.nn.DataParallel)
        #     and "T5" in type(self.model.module).__name__
        # ):
        #     # T5-based retrieval model
        #     decoder_input_ids = torch.zeros(
        #         (inputs["input_ids"].shape[0], 1), dtype=torch.long
        #     ).to(inputs["input_ids"].device)
        #     output = self.model(
        #         **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
        #     )
        #     query_emb = output.last_hidden_state[:, 0, :]

        # else:
        # output = self.model(**inputs, return_dict=True)
        # pooler_output = output.get("pooler_output", None)
        # last_hidden_state = output.get("last_hidden_state", None)
        # query_emb = pooling(
        #     pooler_output,
        #     last_hidden_state,
        #     inputs["attention_mask"],
        #     self.pooling_method,
        # )
        # if "dpr" not in self.model_name:
        #     query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        if is_query:
            emb = self.encode_query(inputs)
        else:
            emb = self.encode_passage(inputs)

        emb = emb.cpu().detach().numpy() # Cast from bf16/fp16 to fp32 .float()
        # emb = emb.astype(np.float32, order="C")
        return emb

    @torch.inference_mode()
    def encode(self, query_list: List[str], batch_size=64, is_query=True, dtype=torch.float32) -> np.ndarray:
        query_emb = []
        with (torch.autocast("cuda", dtype=dtype)):
            with torch.no_grad():
                for i in tqdm(range(0, len(query_list), batch_size), desc="Encoding process: "):
                    query_emb.append(
                        self.single_batch_encode(query_list[i : i + batch_size], is_query)
                    )
        # query_emb = np.concatenate(query_emb, axis=0)
        query_emb = np.concatenate(query_emb)
        return query_emb

    @torch.inference_mode()
    def multi_gpu_encode(
        self, query_list: Union[List[str], str], batch_size=64, is_query=True
    ) -> np.ndarray:
        if self.gpu_num > 1:
            self.model = torch.nn.DataParallel(self.model)
        query_emb = self.encode_query(query_list, batch_size, is_query)
        return query_emb
    
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
            try:
                hf_kwargs["attn_implementation"] = "sdpa"
                logger.info(f"Fall back to use {hf_kwargs['attn_implementation']}")
                base_model = cls.TRANSFORMER_CLS.from_pretrained(
                    model_name_or_path, 
                    weights_only=False,
                    trust_remote_code=True,
                    **hf_kwargs
                )
            except ValueError as e:
                hf_kwargs["attn_implementation"] = "eager"
                logger.info(f"Fall back to use {hf_kwargs['attn_implementation']}")
                base_model = cls.TRANSFORMER_CLS.from_pretrained(
                    model_name_or_path, 
                    weights_only=False,
                    trust_remote_code=True,
                    **hf_kwargs
                )
            
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
            
        return base_model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             instruction: str = "",
             data_args=None,
             tokenizer: AutoTokenizer = None,
             **hf_kwargs):

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
                normalize=normalize,
                tokenizer=tokenizer,
                data_args=data_args,
                instruction=instruction
            )
        else:
            logger.info(" This is not a PEFT model!!! ")
            base_model = cls.try_using_flash_attn(model_name_or_path, hf_kwargs)
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize,
                tokenizer=tokenizer,
                data_args=data_args,
                instruction=instruction
            )

        # print("Please provide lora_name_or_path to load the PEFT model correctly!!!")
        return model


class TempRetrieverEncoder(EncoderModel):

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 encoder: PreTrainedModel,
                 base_model: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 instruction: str = "",
                 data_args=None,
                 ):
        super().__init__(encoder, pooling, normalize, temperature)
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.data_args = data_args
        self.base_model = base_model

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             instruction: str = "",
             data_args=None,
             tokenizer: AutoTokenizer = None,
             **hf_kwargs):

        with open(os.path.join(model_name_or_path, "full_config.json")) as f:
            default_config = json.load(f)
        
        semantic_model = cls.try_using_flash_attn(default_config['model_name_or_path'], hf_kwargs)
        
        temporal_model = cls.try_using_flash_attn(default_config['model_name_or_path'], hf_kwargs)
        
        from safetensors import safe_open
        tensors = {}
        with safe_open(f"{model_name_or_path}/model.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
                
        model = cls(
            encoder=semantic_model,
            base_model=temporal_model,
            pooling=pooling,
            normalize=normalize,
            tokenizer=tokenizer,
            data_args=data_args,
            instruction=instruction
        )
        model.load_state_dict(tensors)

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
            try:
                hf_kwargs["attn_implementation"] = "sdpa"
                logger.info(f"Fall back to use {hf_kwargs['attn_implementation']}")
                base_model = cls.TRANSFORMER_CLS.from_pretrained(
                    model_name_or_path, 
                    weights_only=False,
                    trust_remote_code=True,
                    **hf_kwargs
                )
            except ValueError as e:
                hf_kwargs["attn_implementation"] = "eager"
                logger.info(f"Fall back to use {hf_kwargs['attn_implementation']}")
                base_model = cls.TRANSFORMER_CLS.from_pretrained(
                    model_name_or_path, 
                    weights_only=False,
                    trust_remote_code=True,
                    **hf_kwargs
                )
            
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
            
        return base_model

class DenseRetriever(BaseTextRetriever):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, training_args, model_args, data_args, config: dict, corpus=None):

        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args

        self._config = config
        self.update_base_setting()
        self.update_additional_setting()

        self.load_corpus(corpus)
        self.load_index()

        self.torch_dtype = torch.float32 # Default
        if training_args.bf16:
            self.torch_dtype = torch.bfloat16
        elif training_args.fp16:
            self.torch_dtype = torch.float16

        tokenizer = AutoTokenizer.from_pretrained(
            (
                model_args.tokenizer_name
                if model_args.tokenizer_name
                else model_args.model_name_or_path
            ),
            cache_dir=model_args.cache_dir,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if data_args.padding_side == "right":
            tokenizer.padding_side = "right"
        else:
            tokenizer.padding_side = "left"

        # self.encoder = Encoder.load(
        #     model_args.model_name_or_path,
        #     pooling=model_args.pooling,
        #     normalize=model_args.normalize,
        #     lora_name_or_path=model_args.lora_name_or_path,
        #     cache_dir=model_args.cache_dir,
        #     torch_dtype=self.torch_dtype,
        #     attn_implementation=model_args.attn_implementation,
        #     tokenizer=tokenizer,
        #     data_args=data_args,
        #     instruction=self.instruction,
        # )
        # self.encoder = self.encoder.to(training_args.device)
        # self.encoder.eval()

    def load_corpus(self, corpus):
        try:
            corpus = load_dataset(
                self.data_args.dataset_name,
                "corpus",
                # data_files=self.data_args.dataset_path,
                data_files=self.data_args.corpus_path,
                split=self.data_args.dataset_split,
                cache_dir=self.data_args.dataset_cache_dir,
                num_proc=self.data_args.num_proc,
            )
            if "contents" not in corpus.features:
                try:
                    print("No `contents` field found in corpus, using `text` instead.")
                    corpus = corpus.map(lambda x: {"contents": x["text"]})
                except:
                    warnings.warn("No `contents` & `text` field found in corpus.")
        except:
            corpus_path = self.corpus_path
            if corpus_path.endswith(".jsonl"):
                corpus = load_dataset('json', data_files=corpus_path, split="train")
            elif corpus_path.endswith(".parquet"):
                corpus = load_dataset('parquet', data_files=corpus_path, split="train")
                corpus = corpus.cast_column('image', Image())
            else:
                raise NotImplementedError("Corpus format not supported!")
            if 'contents' not in corpus.features:
                try:
                    print("No `contents` field found in corpus, using `text` instead.")
                    corpus = corpus.map(lambda x: {"contents": x["text"]})
                except:
                    warnings.warn("No `contents` & `text` field found in corpus.")

        self.corpus = corpus

    def load_index(self):
        # if self.index_path is None or not os.path.exists(self.index_path):
        #     raise Warning(f"Index file {self.index_path} does not exist!")
        self.index_path = Path(self.training_args.output_dir) /  "dense_Flat.index" # Hard coded
        print(self.index_path, self.index_path.is_file())
        self.index_path = str(self.index_path)
        
        self.index = faiss.read_index(self.index_path)
        if self.use_faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

    def update_additional_setting(self):
        self.query_max_length = self._config["retrieval_query_max_length"]
        self.pooling_method = self._config["retrieval_pooling_method"]
        self.use_fp16 = self._config["retrieval_use_fp16"]
        self.batch_size = self._config["retrieval_batch_size"]
        self.instruction = (
            self._config["instruction"]
            if self._config["instruction"]
            else self.data_args.query_prefix
        )

        self.retrieval_model_path = self._config["retrieval_model_path"]
        self.use_st = self._config["use_sentence_transformer"]
        self.use_faiss_gpu = self._config["faiss_gpu"]

    def _check_pooling_method(self, model_path, pooling_method):
        try:
            # read pooling method from 1_Pooling/config.json
            pooling_config = json.load(
                open(os.path.join(model_path, "1_Pooling/config.json"))
            )
            for k, v in pooling_config.items():
                if k.startswith("pooling_mode") and v == True:
                    detect_pooling_method = k.split("pooling_mode_")[-1]
                    if detect_pooling_method == "mean_tokens":
                        detect_pooling_method = "mean"
                    elif detect_pooling_method == "cls_token":
                        detect_pooling_method = "cls"
                    else:
                        # raise warning: not implemented pooling method
                        warnings.warn(
                            f"Pooling method {detect_pooling_method} is not implemented.",
                            UserWarning,
                        )
                        detect_pooling_method = "mean"
                    break
        except:
            detect_pooling_method = None

        if (
            detect_pooling_method is not None
            and detect_pooling_method != pooling_method
        ):
            warnings.warn(
                f"Pooling method in model config file is {detect_pooling_method}, but the input is {pooling_method}. Please check carefully."
            )

    def _search(self, query: str, num: int = None, return_score=False):
        if num is None:
            num = self.topk
            query_emb = self.encoder.encode(query, dtype=self.torch_dtype)
        scores, idxs = self.index.search(query_emb, k=num)
        scores = scores.tolist()
        idxs = idxs[0]
        scores = scores[0]

        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query: List[str], num: int = None, return_score=False, emb=None):
        if isinstance(query, str):
            query = [query]
        if num is None:
            num = self.topk
        batch_size = self.batch_size

        results = []
        scores = []
        
        if emb is None:
            emb = self.encoder.encode(query, batch_size=batch_size, is_query=True, dtype=self.torch_dtype)

        scores, idxs = self.index.search(emb, k=num)
        scores = scores.tolist()
        idxs = idxs.tolist()

        # flat_idxs = sum(idxs, [])
        flat_idxs = [item for sub in idxs for item in sub]
        results = load_docs(self.corpus, flat_idxs)
        
        results = [results[i * num : (i + 1) * num] for i in range(len(idxs))]

        if return_score:
            return results, scores
        else:
            return results

class TempRetriever(DenseRetriever):
    def __init__(self, training_args, model_args, data_args, config: dict, corpus=None):

        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args

        self._config = config
        self.update_base_setting()
        self.update_additional_setting()

        self.load_corpus(corpus)
        self.load_index()

        self.torch_dtype = torch.float32 # Default
        if training_args.bf16:
            self.torch_dtype = torch.bfloat16
        elif training_args.fp16:
            self.torch_dtype = torch.float16

        tokenizer = AutoTokenizer.from_pretrained(
            (
                model_args.tokenizer_name
                if model_args.tokenizer_name
                else model_args.model_name_or_path
            ),
            cache_dir=model_args.cache_dir,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if data_args.padding_side == "right":
            tokenizer.padding_side = "right"
        else:
            tokenizer.padding_side = "left"
            
        from safetensors import safe_open
        tensors = {}
        with safe_open(f"{model_args.model_name_or_path}/model.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        
        self.encoder = TempRetrieverEncoder.load(
            model_args.model_name_or_path,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            lora_name_or_path=model_args.lora_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=self.torch_dtype,
            attn_implementation=model_args.attn_implementation,
            tokenizer=tokenizer,
            data_args=data_args,
            instruction=self.instruction,
        )
        self.encoder = self.encoder.to(training_args.device)
        self.encoder.eval()
        
    def _batch_search(self, query: List[str], num: int = None, return_score=False, emb=None):
        if isinstance(query, str):
            query = [query]
        if num is None:
            num = self.topk
        batch_size = self.batch_size

        results = []
        scores = []
        
        if emb is None:
            emb = self.encoder.encode(query, batch_size=batch_size, is_query=True, dtype=self.torch_dtype)

        # Can potentially skip this step since tevatron already encodes the queries once.
        # with open(embedding_path, 'rb') as f:
        #     reps, lookup = pickle.load(f)
        #     all_embeddings = np.array(reps).reshape(-1, hidden_size)

        scores, idxs = self.index.search(emb, k=num)
        scores = scores.tolist()
        idxs = idxs.tolist()

        flat_idxs = sum(idxs, [])
        results = load_docs(self.corpus, flat_idxs)
        results = [results[i * num : (i + 1) * num] for i in range(len(idxs))]

        if return_score:
            return results, scores
        else:
            return results

class Config:
    def __init__(self, config_file_path=None, config_dict={}):

        self.yaml_loader = self._build_yaml_loader()
        self.file_config = self._load_file_config(config_file_path)
        self.variable_config = config_dict

        self.external_config = self._merge_external_config()

        self.internal_config = self._get_internal_config()

        self.final_config = self._get_final_config()

        self._check_final_config()
        self._set_additional_key()

        self._init_device()
        self._set_seed()
        # if not self.final_config.get('disable_save', False):
        #     self._prepare_dir()

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        return loader

    def _load_file_config(self, config_file_path: str):
        file_config = dict()
        if config_file_path:
            with open(config_file_path, "r", encoding="utf-8") as f:
                file_config.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return file_config

    @staticmethod
    def _update_dict(old_dict: dict, new_dict: dict):
        # Update the original update method of the dictionary:
        # If there is the same key in `old_dict` and `new_dict`, and value is of type dict, update the key in dict

        same_keys = []
        for key, value in new_dict.items():
            if key in old_dict and isinstance(value, dict):
                same_keys.append(key)
        for key in same_keys:
            old_item = old_dict[key]
            new_item = new_dict[key]
            old_item.update(new_item)
            new_dict[key] = old_item

        old_dict.update(new_dict)
        return old_dict

    def _merge_external_config(self):
        external_config = dict()
        external_config = self._update_dict(external_config, self.file_config)
        external_config = self._update_dict(external_config, self.variable_config)

        return external_config

    def _get_internal_config(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        # init_config_path = os.path.join(current_path, "basic_config.yaml")
        init_config_path = "/home/thuy0050/code/FlashRAG/flashrag/config/basic_config.yaml"
        internal_config = self._load_file_config(init_config_path)

        return internal_config

    def _get_final_config(self):
        final_config = dict()
        final_config = self._update_dict(final_config, self.internal_config)
        final_config = self._update_dict(final_config, self.external_config)

        return final_config

    def _check_final_config(self):
        # check split
        split = self.final_config["split"]
        if split is None:
            split = ["train", "dev", "test"]
        if isinstance(split, str):
            split = [split]
        self.final_config["split"] = split

    def _init_device(self):
        gpu_id = self.final_config["gpu_id"]
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            # import pynvml 
            # pynvml.nvmlInit()
            # gpu_num = pynvml.nvmlDeviceGetCount()
            import torch
            gpu_num = torch.cuda.device_count()
        except:
            gpu_num = 0
        self.final_config['gpu_num'] = gpu_num
        if gpu_num > 0:
            self.final_config["device"] = "cuda"
        else:
            self.final_config['device'] = 'cpu'

    def _set_additional_key(self):
        def set_pooling_method(method, model2pooling):
            for key, value in model2pooling.items():
                if key.lower() in method.lower():
                    return value
            return "mean"

        def set_retrieval_keys(model2path, model2pooling, method2index, config):
            retrieval_method = config["retrieval_method"]
            if config["index_path"] is None:
                try:
                    config["index_path"] = method2index[retrieval_method]
                except:
                    print("Index is empty!!")

            if config.get("retrieval_model_path") is None:
                config["retrieval_model_path"] = model2path.get(retrieval_method, retrieval_method)

            if config.get("retrieval_pooling_method") is None:
                config["retrieval_pooling_method"] = set_pooling_method(retrieval_method, model2pooling)

            rerank_model_name = config.get("rerank_model_name", None)
            if config.get("rerank_model_path", None) is None:
                if rerank_model_name is not None:
                    config["rerank_model_path"] = model2path.get(rerank_model_name, rerank_model_name)
            if config.get("rerank_pooling_method", None) is None:
                if rerank_model_name is not None:
                    config["rerank_pooling_method"] = set_pooling_method(rerank_model_name, model2pooling)
            return config

        # set dataset
        dataset_name = self.final_config["dataset_name"]
        data_dir = self.final_config["data_dir"]
        self.final_config["dataset_path"] = os.path.join(data_dir, dataset_name)

        # set retrieval-related keys
        model2path = self.final_config["model2path"]
        model2pooling = self.final_config["model2pooling"]
        method2index = self.final_config["method2index"]
        self.final_config = set_retrieval_keys(model2path, model2pooling, method2index, self.final_config)
        # set keys for multi retriever
        if "multi_retriever_setting" in self.final_config:
            multi_retriever_config = self.final_config["multi_retriever_setting"]
            retriever_config_list = multi_retriever_config.get("retriever_list", [])
            # set for reranker merge method
            assert multi_retriever_config['merge_method'] in ['concat', 'rrf', 'rerank', None]
            if multi_retriever_config['merge_method'] == 'rerank':
                rerank_model_name = multi_retriever_config.get("rerank_model_name", None)
                assert rerank_model_name is not None
                multi_retriever_config['rerank_max_length'] = multi_retriever_config.get("rerank_max_length", 512)
                multi_retriever_config['rerank_batch_size'] = multi_retriever_config.get("rerank_batch_size", 256)
                multi_retriever_config['rerank_use_fp16'] = multi_retriever_config.get("rerank_use_fp16", True)
                
                if multi_retriever_config.get("rerank_model_path", None) is None:
                    if rerank_model_name is not None:
                        multi_retriever_config["rerank_model_path"] = model2path.get(rerank_model_name, rerank_model_name)
                if multi_retriever_config.get("rerank_pooling_method", None) is None:
                    if rerank_model_name is not None:
                        multi_retriever_config["rerank_pooling_method"] = set_pooling_method(rerank_model_name, model2pooling)
            
            # set config for each retriever
            for retriever_config in retriever_config_list:
                if "instruction" not in retriever_config:
                    retriever_config["instruction"] = None
                if "bm25_backend" not in retriever_config:
                    retriever_config["bm25_backend"] = "bm25s"
                if "use_reranker" not in retriever_config:
                    retriever_config["use_reranker"] = False
                if "index_path" not in retriever_config:
                    retriever_config["index_path"] = None
                if "corpus_path" not in retriever_config:
                    retriever_config["corpus_path"] = None
                if "use_sentence_transformer" not in retriever_config:
                    retriever_config["use_sentence_transformer"] = False
                retriever_config = set_retrieval_keys(model2path, model2pooling, method2index, retriever_config)
                
                # set other necessary keys as base setting
                keys = [
                    "retrieval_use_fp16",
                    "retrieval_query_max_length",
                    "faiss_gpu",
                    "retrieval_topk",
                    "retrieval_batch_size",
                    "use_reranker",
                    "rerank_model_name",
                    "rerank_model_path",
                    "retrieval_cache_path",
                ]
                for key in keys:
                    if key not in retriever_config:
                        retriever_config[key] = self.final_config.get(key, None)
                retriever_config["save_retrieval_cache"] = False
                retriever_config["use_retrieval_cache"] = False
        
        # set model path
        generator_model = self.final_config["generator_model"]

        if self.final_config.get("generator_model_path") is None:
            self.final_config["generator_model_path"] = model2path.get(generator_model, generator_model)

        if "refiner_name" in self.final_config:
            refiner_model = self.final_config["refiner_name"]
            if "refiner_model_path" not in self.final_config or self.final_config["refiner_model_path"] is None:
                self.final_config["refiner_model_path"] = model2path.get(refiner_model, None)
        if "instruction" not in self.final_config:
            self.final_config["instruction"] = None

        # set model path in metric setting
        metric_setting = self.final_config["metric_setting"]
        metric_tokenizer_name = metric_setting.get("tokenizer_name", None)
        from flashrag.utils.constants import OPENAI_MODEL_DICT

        if metric_tokenizer_name not in OPENAI_MODEL_DICT:
            metric_tokenizer_name = model2path.get(metric_tokenizer_name, metric_tokenizer_name)
            metric_setting["tokenizer_name"] = metric_tokenizer_name
            self.final_config["metric_setting"] = metric_setting

    def _prepare_dir(self):
        # save_note = self.final_config["save_note"]
        # save_dir = self.final_config['save_dir']
        
        # self.final_config["save_dir"]
        
        # if not save_dir.endswith("/"):
        #     save_dir += "/"

        # current_time = datetime.datetime.now()

        # self.final_config["save_dir"] = os.path.join(
        #     save_dir,
        #     f"{self.final_config['dataset_name']}_{current_time.strftime('%Y_%m_%d_%H_%M')}_{save_note}",
        # )
        os.makedirs(self.final_config["save_dir"], exist_ok=True)

        # save config parameters
        config_save_path = os.path.join(self.final_config["save_dir"], "config.yaml")
        with open(config_save_path, "w") as f:
            yaml.dump(self.final_config, f, indent=4, sort_keys=False)

    def _set_seed(self):
        import numpy as np
        import torch
        seed = self.final_config['seed']
        try:
            seed = int(seed)
        except:
            seed = 42
        self.final_config['seed'] = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
        torch.use_deterministic_algorithms(True)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config[key] = value

    def __getattr__(self, item):
        if "final_config" not in self.__dict__:
            raise AttributeError("'Config' object has no attribute 'final_config'")
        if item in self.final_config:
            return self.final_config[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __getitem__(self, item):
        return self.final_config.get(item)

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config

    def __repr__(self):
        return self.final_config.__str__()


class Item:
    """A container class used to store and manipulate a sample within a dataset.
    Information related to this sample during training/inference will be stored in `self.output`.
    Each attribute of this class can be used like a dict key (also for key in `self.output`).
    """

    def __init__(self, item_dict: Dict[str, Any]) -> None:
        self.id: Optional[str] = item_dict.get("id", None)
        self.question: Optional[str] = item_dict.get("question", None)
        self.golden_answers: List[str] = item_dict.get("answers", []) # Change from "golden answers" to "answers"
        self.choices: List[str] = item_dict.get("choices", [])
        self.metadata: Dict[str, Any] = item_dict.get("metadata", {})
        self.output: Dict[str, Any] = item_dict.get("output", {})
        self.data: Dict[str, Any] = item_dict

    def update_output(self, key: str, value: Any) -> None:
        """Update the output dict and keep a key in self.output can be used as an attribute."""
        if key in ["id", "question", "golden_answers", "output", "choices"]:
            raise AttributeError(f"{key} should not be changed")
        else:
            self.output[key] = value

    def update_evaluation_score(self, metric_name: str, metric_score: float) -> None:
        """Update the evaluation score of this sample for a metric."""
        if "metric_score" not in self.output:
            self.output["metric_score"] = {}
        self.output["metric_score"][metric_name] = metric_score

    def __getattr__(self, attr_name: str) -> Any:
        predefined_attrs = [
            "id",
            "question",
            "golden_answers",
            "metadata",
            "output",
            "choices",
        ]
        if attr_name in predefined_attrs:
            return super().__getattribute__(attr_name)
        else:
            output = self.output
            if attr_name in output:
                return output[attr_name]
            else:
                try:
                    return self.data[attr_name]
                except AttributeError:
                    raise AttributeError(f"Attribute `{attr_name}` not found")

    def __setattr__(self, attr_name: str, value: Any) -> None:
        predefined_attrs = [
            "id",
            "question",
            "golden_answers",
            "metadata",
            "output",
            "choices",
            "data",
        ]
        if attr_name in predefined_attrs:
            super().__setattr__(attr_name, value)
        else:
            self.update_output(attr_name, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert all information within the data sample into a dict. Information generated
        during the inference will be saved into output field.
        """
        from flashrag.dataset.utils import (
            clean_prompt_image,
            convert_numpy,
            remove_images,
        )

        output = remove_images(self.data)

        # clean base64 image
        if "prompt" in self.output:
            self.output["prompt"] = clean_prompt_image(self.output["prompt"])

        output["output"] = remove_images(convert_numpy(self.output))
        if self.metadata:
            output["metadata"] = remove_images(self.metadata)

        return output

    def __str__(self) -> str:
        """Return a string representation of the item with its main attributes."""
        return json.dumps(self.to_dict(), indent=4, ensure_ascii=False)


class Dataset:
    """A container class used to store the whole dataset. Inside the class, each data sample will be stored
    in `Item` class. The properties of the dataset represent the list of attributes corresponding to each item in the dataset.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dataset_path: Optional[str] = None,
        data: Optional[List[Dict[str, Any]]] = None,
        sample_num: Optional[int] = None,
        random_sample: bool = False,
    ) -> None:
        if config is not None:
            self.config = config
            dataset_name = config['dataset_name'] if 'dataset_name' in config else 'defalut_dataset'
        else:
            self.config = None
            warnings.warn("dataset_name is not in config, set it as default.")
            dataset_name = "default_dataset"
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

        self.sample_num = sample_num
        self.random_sample = random_sample

        if data is None:
            self.data = self._load_data(self.dataset_name, self.dataset_path)
        else:
            print("Load data from provided data")
            if isinstance(data[0], dict):
                self.data = [Item(item_dict) for item_dict in data]
            else:
                assert isinstance(data[0], Item)
                self.data = data

    def _load_data(self, dataset_name: str, dataset_path: str) -> List[Item]:
        """Load data from the provided dataset_path or directly download the file(TODO)."""
        if not os.path.exists(dataset_path):
            # TODO: auto download: self._download(self.dataset_name, dataset_path)
            raise FileNotFoundError(f"Dataset file {dataset_path} not found.")

        data = []
        if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    item_dict = json.loads(line)
                    item = Item(item_dict)
                    data.append(item)
        elif dataset_path.endswith('parquet'):
            hf_data = load_dataset('parquet', data_files=dataset_path, split="train")
            hf_data = hf_data.cast_column('image', datasets.Image())
            for item in hf_data:
                item = Item(item)
                data.append(item)
        else:
            raise NotImplementedError

        if self.sample_num is not None:
            self.sample_num = int(self.sample_num)
            if self.random_sample:
                print(f"Random sample {self.sample_num} items in test set.")
                data = random.sample(data, self.sample_num)
            else:
                data = data[: self.sample_num]

        return data

    def update_output(self, key: str, value_list: List[Any]) -> None:
        """Update the overall output field for each sample in the dataset."""
        assert len(self.data) == len(value_list)
        for item, value in zip(self.data, value_list):
            item.update_output(key, value)

    @property
    def question(self) -> List[Optional[str]]:
        return [item.query for item in self.data]

    @property
    def golden_answers(self) -> List[List[str]]:
        return [item.golden_answers for item in self.data]

    @property
    def id(self) -> List[Optional[str]]:
        return [item.id for item in self.data]

    @property
    def output(self) -> List[Dict[str, Any]]:
        return [item.output for item in self.data]

    def get_batch_data(self, attr_name: str, batch_size: int) -> Generator[List[Any], None, None]:
        """Get an attribute of dataset items in batch."""
        for i in range(0, len(self.data), batch_size):
            batch_items = self.data[i : i + batch_size]
            yield [item[attr_name] for item in batch_items]

    def __getattr__(self, attr_name: str) -> List[Any]:
        return [item.__getattr__(attr_name) for item in self.data]

    def get_attr_data(self, attr_name: str) -> List[Any]:
        """For the attributes constructed later (not implemented using property),
        obtain a list of this attribute in the entire dataset.
        """
        return [item[attr_name] for item in self.data]

    def __getitem__(self, index: int) -> Item:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def save(self, save_path: str) -> None:
        """Save the dataset into the original format."""

        save_data = [item.to_dict() for item in self.data]
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)

    def __str__(self) -> str:
        """Return a string representation of the dataset with a summary of items."""
        return f"Dataset '{self.dataset_name}' with {len(self)} items"


class ToyPipeline(SequentialPipeline):
    def __init__(self, training_args, model_args, data_args, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        self.config = config
        self.device = config["device"]
        self.data_args = data_args
        self.training_args = training_args
        self.model_args = model_args
        self.generator = None

        self.retriever = None
        self.evaluator = Evaluator(config)
        
        self.save_retrieval_cache = config["save_retrieval_cache"]
        
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template

        if "tempretriever" in model_args.model_name_or_path:
            self.retriever = TempRetriever(
                training_args, model_args, data_args, config
            )
        else:
            self.retriever = DenseRetriever(
                training_args, model_args, data_args, config
            )

        # TODO: add rewriter module

        self.use_fid = config["use_fid"]

        if config["refiner_name"] is not None:
            self.refiner = get_refiner(config, self.retriever, self.generator)
        else:
            self.refiner = None

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_fun is not None:
            dataset = pred_process_fun(dataset)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            for metric_name, score in eval_result.items():
                eval_result[metric_name] = round(score, 3)
            print(
                f"Evaluation of {self.data_args.dataset_name}'s {self.data_args.dataset_config}"
            )
            for k, v in eval_result.items():
                print(f"{k}: {v}")
            return eval_result

        # # save retrieval cache
        # if self.save_retrieval_cache:
        #     self.retriever._save_cache()

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question
        
        emb = None

        embed_path = Path(self.data_args.encode_output_path)
        
        if embed_path.is_file():
            with open(embed_path, 'rb') as f:
                emb, lookup = pickle.load(f)

        retrieval_results = self.retriever.batch_search(input_query, emb=emb)
        dataset.update_output("retrieval_result", retrieval_results)
        
        if self.generator is None:
            self.generator = VLLMGenerator(self.config)

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if "llmlingua" in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                dataset.update_output("prompt", input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]

        else:
            if not self.use_fid:
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc["contents"] for doc in docs])
        dataset.update_output("prompt", input_prompts)

        # delete used refiner to release memory
        if self.refiner:
            del self.refiner

        pred_answer_list = self.generator.generate(input_prompts)
        
        if self.config.generator_model == "deepseek-r1-qwen32B":
            for i in range(len(pred_answer_list)):
                pred_answer_list[i] = pred_answer_list[i].split("\n</think>\n\n")[-1]

        dataset.update_output("pred", pred_answer_list)
        
        # with open("/home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/tmrl/facebook/contriever/t_64_alpha_0.1/rag_deepseek_r1_qwen32b_768/intermediate_data.json") as f:
        #     intermediate_data = json.load(f)
            
        # for i in intermediate_data:
        #     i['output']["pred"] = i['output']["pred"].split("\n</think>\n\n")[-1]
            
        # dataset.update_output("pred", [x['output']['pred'] for x in intermediate_data])

        dataset = self.evaluate(
            dataset, do_eval=do_eval, pred_process_fun=pred_process_fun
        )

        return dataset


def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser.add_argument("--flashrag_config_yaml_path", type=str, required=True)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, other_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
        
        # config_dict = {"data_dir": "dataset/"} # If you want to override something
        flashrag_config_yaml_path = other_args.flashrag_config_yaml_path
        flashrag_config = Config(
            config_file_path=flashrag_config_yaml_path,
            # config_dict=config_dict,
        )
        
        flashrag_config["save_dir"] = training_args.output_dir
        if not flashrag_config.final_config.get('disable_save', False):
            flashrag_config._prepare_dir()
        
        
    set_seed(42, deterministic=True) # Ensuring the seed is set properly

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError("Multi-GPU encoding is not supported.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    
    data = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config,
        data_files=data_args.dataset_path,
        split=data_args.dataset_split,
        cache_dir=data_args.dataset_cache_dir,
        num_proc=data_args.num_proc,
    )
    if data_args.dataset_number_of_shards > 1:
        data = data.shard(
            num_shards=data_args.dataset_number_of_shards,
            index=data_args.dataset_shard_index,
        )
    
    # data = data.select([0])
    test_data = Dataset(
        config=flashrag_config,
        data=data
    )
    

    # ### For quick debugging
    # temp_data = []
    # for ix, item in enumerate(data):
    #     if ix > 10:
    #         break
    #     temp_data.append(item)
    # test_data = Dataset(
    #     config=flashrag_config, # type: ignore
    #     data=temp_data
    # )
    # ### For quick debugging

    if flashrag_config.generator_model == "deepseek-r1-qwen32B":
        prompt_templete = PromptTemplate(
            flashrag_config,
            system_prompt="",
            user_prompt="Answer the question based on the given document. Only give me the answer and do not output any other words.\nThe following are given documents.\n\n{reference}. Question: {question}\nAnswer:",
        )
    else:
        prompt_templete = PromptTemplate(
            flashrag_config,
            system_prompt="Answer the question based on the given document. Only give me the answer and do not output any other words.\nThe following are given documents.\n\n{reference}",
            user_prompt="Question: {question}\nAnswer:",
        )

    pipeline = ToyPipeline(
        training_args,
        model_args,
        data_args,
        config=flashrag_config, 
        prompt_template=prompt_templete
    )

    output_dataset = pipeline.run(test_data, do_eval=True)

    if hasattr(pipeline, "model"):
        del pipeline.model
    cleanup_dist_env_and_memory()


if __name__ == '__main__':
    main()
