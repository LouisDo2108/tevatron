from __future__ import annotations

import inspect
import logging
import os
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import safetensors.torch
import torch
import torch.distributed as dist
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import SaveStrategy

from tevatron.retriever.trainer import TevatronTrainer

from torch.utils.data import BatchSampler, ConcatDataset, DataLoader, RandomSampler
from sentence_transformers.sampler import (
    DefaultBatchSampler,
    GroupByLabelBatchSampler,
    MultiDatasetDefaultBatchSampler,
    NoDuplicatesBatchSampler,
    ProportionalBatchSampler,
    RoundRobinBatchSampler,
)
from sentence_transformers.training_args import (
    BatchSamplers,
    MultiDatasetBatchSamplers,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.util import (
    disable_logging,
    is_datasets_available,
    is_training_available,
)

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Value

logger = logging.getLogger(__name__)


class NoDuplicatesBatchSampler(DefaultBatchSampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 42,
    ) -> None:
        """
        This sampler creates batches such that each batch contains samples where the values are unique,
        even across columns. This is useful when losses consider other samples in a batch to be in-batch
        negatives, and you want to ensure that the negatives are not duplicates of the anchor/positive sample.

        Recommended for:
            - :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`
            - :class:`~sentence_transformers.losses.CachedMultipleNegativesRankingLoss`
            - :class:`~sentence_transformers.losses.MultipleNegativesSymmetricRankingLoss`
            - :class:`~sentence_transformers.losses.CachedMultipleNegativesSymmetricRankingLoss`
            - :class:`~sentence_transformers.losses.MegaBatchMarginLoss`
            - :class:`~sentence_transformers.losses.GISTEmbedLoss`
            - :class:`~sentence_transformers.losses.CachedGISTEmbedLoss`

        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int): Number of samples per batch.
            drop_last (bool): If True, drop the last incomplete batch if the dataset size
                is not divisible by the batch size.
            valid_label_columns (List[str], optional): List of column names to check for labels.
                The first column name from ``valid_label_columns`` found in the dataset will
                be used as the label column.
            generator (torch.Generator, optional): Optional random number generator for shuffling
                the indices.
            seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 0.
        """
        super().__init__(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )
        # if label_columns := set(dataset.column_names) & set(
        #     self.valid_label_columns or []
        # ):
        #     dataset = dataset.remove_columns(list(label_columns))
        self.dataset = dataset

    def __iter__(self) -> Iterator[list[int]]:
        """
        Iterate over the remaining non-yielded indices. For each index, check if the sample values are already in the
        batch. If not, add the sample values to the batch keep going until the batch is full. If the batch is full, yield
        the batch indices and continue with the next batch.
        """
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)

        # We create a dictionary to None because we need a data structure that:
        # 1. Allows for cheap removal of elements
        # 2. Preserves the order of elements, i.e. remains random
        remaining_indices = dict.fromkeys(
            torch.randperm(len(self.dataset), generator=self.generator).tolist()
        )
        while remaining_indices:
            batch_values = set()
            batch_indices = []
            for index in remaining_indices:
                
                sample_values = {value for value in self.dataset[index]}
                if sample_values & batch_values:
                    continue

                batch_indices.append(index)
                if len(batch_indices) == self.batch_size:
                    yield batch_indices
                    break

                batch_values.update(sample_values)

            else:
                # NOTE: some indices might still have been ignored here
                if not self.drop_last:
                    yield batch_indices

            for index in batch_indices:
                del remaining_indices[index]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


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


class SentenceTransformerStyleTrainer(MAdaptorTrainer):
    def get_multi_dataset_batch_sampler(
        self,
        dataset: ConcatDataset,
        batch_samplers: list[BatchSampler],
        generator: torch.Generator | None = None,
        seed: int | None = 0,
    ) -> BatchSampler:
        """
        Returns the appropriate multi-dataset batch sampler based on the ``multi_dataset_batch_sampler`` argument
        in ``self.args``. This batch sampler class supports ``__len__`` and ``__iter__`` methods, and is used as the
        ``batch_sampler`` to create the :class:`torch.utils.data.DataLoader`.

        .. note::
            Override this method to provide a custom multi-dataset batch sampler.

        Args:
            dataset (ConcatDataset): The concatenation of all datasets.
            batch_samplers (List[BatchSampler]): List of batch samplers for each dataset in the concatenated dataset.
            generator (torch.Generator, optional): Optional random number generator for shuffling the indices.
            seed (int, optional): Optional seed for the random number generator
        """

        multi_batch_sampler_kwargs = {
            "batch_samplers": batch_samplers,
            "generator": generator,
            "seed": seed,
        }
        # If the multi-dataset batch sampler is a DefaultBatchSampler subclass, initialize it
        if inspect.isclass(self.args.multi_dataset_batch_sampler) and issubclass(
            self.args.multi_dataset_batch_sampler, MultiDatasetDefaultBatchSampler
        ):
            return self.args.multi_dataset_batch_sampler(dataset, **multi_batch_sampler_kwargs)

        # If it's a callable, call it
        if callable(self.args.multi_dataset_batch_sampler):
            return self.args.multi_dataset_batch_sampler(dataset, **multi_batch_sampler_kwargs)

        # Otherwise, it's an MultiDatasetBatchSamplers instance and we use the samplers that match the enum values
        if self.args.multi_dataset_batch_sampler == MultiDatasetBatchSamplers.ROUND_ROBIN:
            return RoundRobinBatchSampler(dataset=dataset, **multi_batch_sampler_kwargs)

        if self.args.multi_dataset_batch_sampler == MultiDatasetBatchSamplers.PROPORTIONAL:
            return ProportionalBatchSampler(dataset=dataset, **multi_batch_sampler_kwargs)


    def get_batch_sampler(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> BatchSampler | None:
        """
        Returns the appropriate batch sampler based on the ``batch_sampler`` argument in ``self.args``.
        This batch sampler class supports ``__len__`` and ``__iter__`` methods, and is used as the ``batch_sampler``
        to create the :class:`torch.utils.data.DataLoader`.

        .. note::
            Override this method to provide a custom batch sampler.

        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int): Number of samples per batch.
            drop_last (bool): If True, drop the last incomplete batch if the dataset size
                is not divisible by the batch size.
            valid_label_columns (List[str]): List of column names to check for labels.
                The first column name from ``valid_label_columns`` found in the dataset will
                be used as the label column.
            generator (torch.Generator, optional): Optional random number generator for shuffling
                the indices.
            seed (int): Seed for the random number generator to ensure reproducibility. Defaults to 0.
        """

        batch_sampler_kwargs = {
            "batch_size": batch_size,
            "drop_last": drop_last,
            "valid_label_columns": valid_label_columns,
            "generator": generator,
            "seed": seed,
        }
        # If the batch sampler is a DefaultBatchSampler subclass, initialize it
        if inspect.isclass(self.args.batch_sampler) and issubclass(
            self.args.batch_sampler, DefaultBatchSampler
        ):
            return self.args.batch_sampler(dataset, **batch_sampler_kwargs)

        # If it's a callable, call it
        if callable(self.args.batch_sampler):
            return self.args.batch_sampler(dataset, **batch_sampler_kwargs)

        # Otherwise it's a BatchSamplers enum. None of those work with IterableDatasets, so we
        # don't use them in that case
        if isinstance(dataset, IterableDataset):
            if self.args.batch_sampler != BatchSamplers.BATCH_SAMPLER:
                logger.warning(
                    "When using an IterableDataset, you cannot specify a batch sampler."
                )
            return None

        # Lastly, use the samplers that match the enum values
        if self.args.batch_sampler == BatchSamplers.NO_DUPLICATES:
            return NoDuplicatesBatchSampler(dataset, **batch_sampler_kwargs)

        if self.args.batch_sampler == BatchSamplers.GROUP_BY_LABEL:
            return GroupByLabelBatchSampler(dataset, **batch_sampler_kwargs)

        if self.args.batch_sampler == BatchSamplers.BATCH_SAMPLER:
            return DefaultBatchSampler(
                RandomSampler(dataset, generator=generator), **batch_sampler_kwargs
            )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError(
                "Training requires specifying a train_dataset to the SentenceTransformerTrainer."
            )

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        generator = torch.Generator()
        if self.args.seed:
            generator.manual_seed(self.args.seed)

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        if isinstance(train_dataset, IterableDataset):
            dataloader_params.update(
                {
                    "batch_size": self.args.train_batch_size,
                    "drop_last": self.args.dataloader_drop_last,
                }
            )
            if self.args.batch_sampler != BatchSamplers.BATCH_SAMPLER:
                logger.warning(
                    "When using an IterableDataset, you cannot specify a batch sampler."
                )

        elif isinstance(train_dataset, IterableDatasetDict):
            raise ValueError(
                "Sentence Transformers is not compatible with IterableDatasetDict. Please use a DatasetDict instead."
            )

        elif isinstance(train_dataset, DatasetDict):
            for dataset in train_dataset.values():
                if isinstance(dataset, IterableDataset):
                    raise ValueError(
                        "Sentence Transformers is not compatible with a DatasetDict containing an IterableDataset."
                    )

            batch_samplers = [
                self.get_batch_sampler(
                    dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    valid_label_columns=data_collator.valid_label_columns,
                    generator=generator,
                )
                for dataset in train_dataset.values()
            ]

            train_dataset = ConcatDataset(train_dataset.values())
            batch_sampler = self.get_multi_dataset_batch_sampler(
                dataset=train_dataset,
                batch_samplers=batch_samplers,
                generator=generator,
                seed=self.args.seed,
            )
            dataloader_params["batch_sampler"] = batch_sampler

        # elif isinstance(train_dataset, Dataset):
        else:
            batch_sampler = self.get_batch_sampler(
                train_dataset,
                batch_size=self.args.train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                valid_label_columns=None, #data_collator.valid_label_columns,
                generator=generator,
            )
            dataloader_params["batch_sampler"] = batch_sampler
        # else:
        #     raise ValueError(
        #         "Unsupported `train_dataset` type. Use a Dataset, DatasetDict, or IterableDataset for training."
        #     )

        # If 'even_batches' is True, it will use the initial few samples to pad out the last sample. This can
        # cause issues with multi-dataset training, so we want to set this to False.
        # For evaluation, setting 'even_batches' to False results in hanging, so we keep it as True there.
        self.accelerator.even_batches = False
        self._train_dataloader = self.accelerator.prepare(
            DataLoader(train_dataset, **dataloader_params)
        )
        return self._train_dataloader
