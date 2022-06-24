# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A subclass of `Trainer` specific to Question-Answering tasks
"""
from typing import Dict, List, Optional

from torch.utils.data import Dataset

from transformers import Seq2SeqTrainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedTensorGatherer,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    # has_length,
    number_of_arguments,
    set_seed,
    speed_metrics,
)
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

from transformers.utils import logging
logger = logging.get_logger(__name__)

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

class QuestionAnsweringSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            if self.args.sec_lr == 'zero':
                sec_lr = 0
            elif self.args.sec_lr == '/10':
                sec_lr = self.args.learning_rate / 10
            elif self.args.sec_lr == '/100':
                sec_lr = self.args.learning_rate / 100
            elif self.args.sec_lr == '/1000':
                sec_lr = self.args.learning_rate / 1000
            else:
                raise ValueError(f"sec_lr should be one of ['zero', '/10', '/100', '/1000'], but it is {self.args.sec_lr} now")
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.optim_group == 'all':
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    },
                ]
            elif self.args.optim_group == 'ori':
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "ex" not in n],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "ex" not in n],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "ex" in n],
                        "weight_decay": self.args.weight_decay,
                        "lr": sec_lr,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "ex" in n],
                        "weight_decay": 0.0,
                        "lr": sec_lr,
                    },
                ]
            elif self.args.optim_group == 'ex':
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "ex" in n],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "ex" in n],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "ex" not in n],
                        "weight_decay": self.args.weight_decay,
                        "lr": sec_lr,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "ex" not in n],
                        "weight_decay": 0.0,
                        "lr": sec_lr,
                    },
                ]
            elif self.args.optim_group == 'ori-ada':

                def is_ori_ada(name):
                    if "ex" not in name:
                        return True
                    if "ada" in name:
                        return True
                    return False

                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and is_ori_ada(n)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and is_ori_ada(n)],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and not is_ori_ada(n)],
                        "weight_decay": self.args.weight_decay,
                        "lr": sec_lr,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and not is_ori_ada(n)],
                        "weight_decay": 0.0,
                        "lr": sec_lr,
                    },
                ]
            elif self.args.optim_group == 'ori-noemb':

                def is_ori_noemb(name):
                    if "ex" in name:
                        return False
                    if "shared" in name:
                        return False
                    return True

                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and is_ori_noemb(n)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and is_ori_noemb(n)],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and not is_ori_noemb(n)],
                        "weight_decay": self.args.weight_decay,
                        "lr": sec_lr,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and not is_ori_noemb(n)],
                        "weight_decay": 0.0,
                        "lr": sec_lr,
                    },
                ]
            else:
                raise ValueError(f"optim_group should be one of ['all', 'ori', 'ex', 'ori-ada', 'ori-noemb'], but it is {self.args.optim_group} now")

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples=None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            # elif args.bf16_full_eval:
            #     model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        # logger.info(f"***** Running {description} *****")
        # if has_length(dataloader.dataset):
        #     logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        # else:
        logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        # if has_length(eval_dataset):
        # num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        if hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)