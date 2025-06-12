import logging
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import safetensors.torch
import torch
import torch.distributed as dist
from transformers.trainer import TRAINING_ARGS_NAME

from tevatron.retriever.trainer import TevatronTrainer

logger = logging.getLogger(__name__)

class MAdaptorTrainer(TevatronTrainer):

    def __init__(self, *args, **kwargs):
        super(MAdaptorTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1
        self.other_losses = defaultdict(lambda: torch.tensor(0.0).to(self.args.device))

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()
            prefix = "encoder."
            model_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            adapter_state_dict = {k[8:]:v for k, v in state_dict.items() if k.startswith("adaptor.")}
            self.model.encoder.save_pretrained(
                output_dir,
                state_dict=model_state_dict,
                safe_serialization=self.args.save_safetensors,
            )
            safetensors.torch.save_file(
                adapter_state_dict, os.path.join(output_dir, "adaptor.safetensors")
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

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
                4,
            )
            for loss_name, some_loss in self.other_losses.items():
                some_loss_scalar = self._nested_gather(some_loss).mean().item()
                self.other_losses[loss_name] -= some_loss
                logs[loss_name] = round(
                    some_loss_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
                logs["grad_norm"] = round(logs["grad_norm"], 4)

            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()
            logs["learning_rate"] = f"{learning_rate:.3e}"

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
