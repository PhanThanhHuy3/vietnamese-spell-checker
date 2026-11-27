"""Shared AdamW fine-tuning loop with validation checkpoint selection."""
import torch
from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

from src.data.dataset import build_data_collator


class RecoveryCheckpointCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 25 == 0:
            control.should_save = True


class SpellingTrainer:
    def __init__(self, wrapper, train, val, config, smoke_steps=None):
        settings = config["training"]
        args = Seq2SeqTrainingArguments(
            output_dir=settings["output_dir"],
            eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, save_total_limit=2, save_only_model=smoke_steps is not None,
            restore_callback_states_from_checkpoint=True,
            optim="adamw_torch", learning_rate=float(settings["learning_rate"]),
            weight_decay=settings["weight_decay"], warmup_ratio=settings["warmup_ratio"],
            num_train_epochs=settings["num_train_epochs"],
            per_device_train_batch_size=settings["per_device_train_batch_size"],
            per_device_eval_batch_size=settings["per_device_eval_batch_size"],
            gradient_accumulation_steps=settings["gradient_accumulation_steps"],
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            seed=config["seed"], logging_steps=5, logging_first_step=True, report_to="none",
            max_steps=smoke_steps or -1, prediction_loss_only=True,
            dataloader_pin_memory=torch.cuda.is_available(),
        )
        callbacks = [EarlyStoppingCallback(early_stopping_patience=settings["patience"])]
        if smoke_steps is None:
            callbacks.append(RecoveryCheckpointCallback())
        self.runner = Seq2SeqTrainer(
            model=wrapper.model, args=args, train_dataset=train, eval_dataset=val,
            data_collator=build_data_collator(wrapper.tokenizer, wrapper.model),
            callbacks=callbacks,
        )

    def train(self, resume_from_checkpoint=None):
        return self.runner.train(resume_from_checkpoint=resume_from_checkpoint)
