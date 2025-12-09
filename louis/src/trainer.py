from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
from pdb import set_trace as st
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import SaveStrategy, has_length
from transformers.trainer_pt_utils import find_batch_size

from tevatron.retriever.trainer import TevatronTrainer

logger = logging.getLogger(__name__)


class MAdaptorTrainer(TevatronTrainer):

    def __init__(self, *args, **kwargs):
        super(MAdaptorTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.other_losses = defaultdict(lambda: torch.tensor(0.0).to(self.args.device))
        self.madaptor = False
        # self.grad_cache = False
        # if args.grad_cache:
        #     try:
        #         from grad_cache import GradCache
        #         _grad_cache_available = True
        #     except ModuleNotFoundError:
        #         _grad_cache_available = False
        #     if not _grad_cache_available:
        #         raise ValueError(
        #             'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')

        #     self.grad_cache = True

        #     self.gc = GradCache(
        #         models=[self.model, self.model],
        #         chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
        #         loss_fn=loss_fn,
        #         split_input_fn=split_dense_inputs,
        #         get_rep_fn=get_dense_rep,
        #         fp16=self.args.fp16,
        #         scaler=self.scaler if self.args.fp16 else None
        #     )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()

            # Remove the base_model which is only used for KL loss
            model_state_dict = {
                k: v for k, v in state_dict.items() if k.startswith("base_model.")
            }

            # Remove the encoder of Tevatron's DenseModel wrapper.
            prefix = "encoder."
            model_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

            if self.madaptor:
                ### Enable this if use madaptor ###
                adapter_state_dict = {k[8:]:v for k, v in state_dict.items() if k.startswith("adaptor.")}
                safetensors.torch.save_file(
                    adapter_state_dict, os.path.join(output_dir, "adaptor.safetensors")
                )
            else:
                self.model.encoder.save_pretrained(
                    output_dir,
                    state_dict=model_state_dict,
                    safe_serialization=self.args.save_safetensors,
                )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            self.data_collator.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        query, doc = inputs  # query is a dummy list, doc contains (docid, doc)
        outputs = model(query=query, passage=doc)

        loss = outputs.pop("loss")

        for loss_name, some_loss in outputs.items():
            self.other_losses[loss_name] += some_loss / self.args.gradient_accumulation_steps

        return (loss, outputs) if return_outputs else loss

    def recall_at_k(self, scores, target, k=5):
        # scores: [num_queries, num_passages]
        # target: [num_queries] → index of positive passage per query
        topk = scores.topk(k, dim=1).indices  # [num_queries, k]
        correct = (topk == target.unsqueeze(1)).any(dim=1)
        return correct.float()

    def cosine_diagnostics(self, scores, target):
        pos_scores = scores[torch.arange(scores.size(0)), target]
        neg_mask = torch.ones_like(scores, dtype=torch.bool)
        neg_mask[torch.arange(scores.size(0)), target] = False
        neg_scores = scores[neg_mask].view(scores.size(0), -1).mean(dim=-1)

        return pos_scores, neg_scores

    def eval_step(self, model, inputs):
        query, passage = inputs

        if isinstance(model.adaptor, torch.nn.Module):
            _, q_reps = model.encode_query(query) if query else None
            _, p_reps = model.encode_passage(passage) if passage else None
        else:
            q_reps = model.encode_query(query) if query else None
            p_reps = model.encode_passage(passage) if passage else None
        
        
        metrics = {}
        for m in self.model.matryoshka_dim_list:

            scores_semantic = self.model.compute_similarity(q_reps[:, :m], p_reps[:, :m])

            num_neg = p_reps.size(0) // q_reps.size(0)
            target = torch.arange(
                scores_semantic.size(0),
                device=scores_semantic.device,
                dtype=torch.long,
            )
            target = target * num_neg

            # ---- Add diagnostics ----
            with torch.no_grad():
                recall1 = self.recall_at_k(scores_semantic, target, k=1)
                recall5 = self.recall_at_k(scores_semantic, target, k=5)
                pos_sim, neg_sim = self.cosine_diagnostics(scores_semantic, target)

                metrics.update({
                    f"loss_{m}": F.cross_entropy(scores_semantic / self.model.temperature, target, reduction="none"),
                    f"recall@1_{m}": recall1,
                    f"recall@5_{m}": recall5,
                    f"pos_sim_{m}": pos_sim,
                    f"neg_sim_{m}": neg_sim,
                })
        return metrics

        # loss = F.cross_entropy(scores_semantic / self.model.temperature, target, reduction="none")
        # # losses = {"loss": loss}
        # return loss

    def training_step(self, *args):
        return (
            super(TevatronTrainer, self).training_step(*args)
            / self._dist_loss_scale_factor
        )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        pass

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            # if is_torch_xla_available():
            #     xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                2,
            )
            for loss_name, some_loss in self.other_losses.items():
                some_loss_scalar = self._nested_gather(some_loss).mean().item()
                self.other_losses[loss_name] -= some_loss
                logs[loss_name] = round(
                    some_loss_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    2,
                )
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
                logs["grad_norm"] = round(logs["grad_norm"], 2)

            if learning_rate is not None:
                logs["lr"] = learning_rate
            else:
                logs["lr"] = self._get_learning_rate()
            logs["lr"] = f"{learning_rate:.2e}"

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(
                metrics=metrics, trial=trial
            )

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        start_time = time.time()

        args = self.args
        # with self.compute_loss_context_manager():
        #     for inputs in eval_dataloader:
        #         loss = self.compute_loss(model, inputs)

        # eval_loop = (
        #     self.prediction_loop
        #     if self.args.use_legacy_prediction_loop
        #     else self.evaluation_loop
        # )
        # output = eval_loop(
        #     eval_dataloader,
        #     description="Evaluation",
        #     # No point gathering the predictions if there are no metrics, otherwise we defer to
        #     # self.args.prediction_loss_only
        #     prediction_loss_only=True if self.compute_metrics is None else None,
        #     ignore_keys=ignore_keys,
        #     metric_key_prefix=metric_key_prefix,
        # )

        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                or (
                    self.is_fsdp_enabled
                    and self.accelerator.mixed_precision != "fp8"
                    and not self.args.torch_compile
                )
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 2)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        # logger.info(f"\n***** Running {description} *****")
        if has_length(eval_dataloader):
            logger.info(f"  Num examples = {self.num_examples(eval_dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        # # This is for NaiveTemporalv4
        # adapters = ["semantic", "temporal"]
        # weights = [1.0, 1.0]
        # adapter_name = "merge"
        # density = 0.2
        # model.encoder.add_weighted_adapter(
        #     adapters, weights, adapter_name, combination_type="dare_ties", density=density
        # )
        # model.encoder.set_adapter("merge")

        self.callback_handler.eval_dataloader = eval_dataloader
        # Do this before wrapping.
        eval_dataset = getattr(eval_dataloader, "dataset", None)

        metrics = {}
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        total_correct1 = {m: [] for m in self.model.matryoshka_dim_list}
        total_correct5 = {m: [] for m in self.model.matryoshka_dim_list}
        total_loss = {m: [] for m in self.model.matryoshka_dim_list}
        pos_sims = {m: [] for m in self.model.matryoshka_dim_list}
        neg_sims = {m: [] for m in self.model.matryoshka_dim_list}

        total_queries = 0
        observed_num_examples = 0

        with torch.no_grad():
            for step, inputs in enumerate(eval_dataloader):
                # Update observed examples
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                    if batch_size is None:
                        batch_size = observed_batch_size

                with self.compute_loss_context_manager():
                    metrics = self.eval_step(model, inputs)

                # Accumulate per-dimension metrics
                for m in self.model.matryoshka_dim_list:
                    total_loss[m].append(metrics.get(f"loss_{m}", torch.tensor(0.0)))
                    total_correct1[m].append(metrics.get(f"recall@1_{m}", torch.tensor(0.0)))
                    total_correct5[m].append(metrics.get(f"recall@5_{m}", torch.tensor(0.0)))
                    pos_sims[m].append(metrics.get(f"pos_sim_{m}", torch.tensor(0.0)))
                    neg_sims[m].append(metrics.get(f"neg_sim_{m}", torch.tensor(0.0)))

                total_queries += observed_batch_size

        # Aggregate
        eval_loss = {}
        for m in self.model.matryoshka_dim_list:
            # eval_loss[f"eval_loss_{m}"] = round(torch.stack(total_loss[m]).mean().item(), 2)
            eval_loss[f"eval_loss_{m}"] = round(torch.cat(total_loss[m]).mean().item(), 2)
            eval_loss[f"eval_recall@1_{m}"] = round(torch.cat(total_correct1[m]).mean().item(), 4)
            eval_loss[f"eval_recall@5_{m}"] = round(torch.cat(total_correct5[m]).mean().item(), 4)
            eval_loss[f"eval_pos_sim_{m}"] = round(torch.cat(pos_sims[m]).mean().item(), 4)
            eval_loss[f"eval_neg_sim_{m}"] = round(torch.cat(neg_sims[m]).mean().item(), 4)
            eval_loss[f"eval_margin_{m}"] = round(
                eval_loss[f"eval_pos_sim_{m}"] - eval_loss[f"eval_neg_sim_{m}"], 4
            )
            
            # avg_loss = round(torch.concat(total_loss).mean().item(), 2)
            # avg_recall1 = round(torch.concat(total_correct1).mean().item(), 4)
            # avg_recall5 = round(torch.concat(total_correct5).mean().item(), 4)
            # avg_pos = round(torch.concat(pos_sims).mean().item(), 4)
            # avg_neg = round(torch.concat(neg_sims).mean().item(), 4)

            # eval_loss = {
            #     "eval_loss": avg_loss,
            #     "eval_recall@1": avg_recall1,
            #     "eval_recall@5": avg_recall5,
            #     "eval_pos_sim": avg_pos,
            #     "eval_neg_sim": avg_neg,
            #     "eval_margin": round((avg_pos - avg_neg), 4),
            # }
        self.log(eval_loss)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, eval_loss
        )

        self._memory_tracker.stop_and_update_metrics(eval_loss)

        return eval_loss