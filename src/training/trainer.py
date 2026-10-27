from pathlib import Path
from typing import Any, Dict

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from ..data.dataset import SpellingDataset, build_data_collator
from ..models.spell_corrector import VietnameseSpellCorrector
from .metrics import build_compute_metrics_fn


class SpellingTrainer:
    """Class điều phối quá trình huấn luyện mô hình sửa lỗi chính tả Seq2Seq."""

    def __init__(
        self,
        model_wrapper: VietnameseSpellCorrector,
        train_dataset: SpellingDataset,
        eval_dataset: SpellingDataset | None = None,
        config: Dict[str, Any] | None = None,
    ):
        self.model_wrapper = model_wrapper
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or {}

        training_cfg = self.config.get("training", {})
        output_dir = training_cfg.get("output_dir", "checkpoints/vit5_spelling_corrector")

        # Cấu hình tham số huấn luyện của HuggingFace
        self.training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=float(training_cfg.get("learning_rate", 5e-5)),
            weight_decay=float(training_cfg.get("weight_decay", 0.01)),
            num_train_epochs=int(training_cfg.get("num_train_epochs", 5)),
            per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 16)),
            per_device_eval_batch_size=int(training_cfg.get("per_device_eval_batch_size", 16)),
            gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
            warmup_ratio=float(training_cfg.get("warmup_ratio", 0.1)),
            fp16=bool(training_cfg.get("fp16", False)),
            logging_steps=int(training_cfg.get("logging_steps", 50)),
            eval_strategy=training_cfg.get("eval_strategy", "steps"),
            eval_steps=int(training_cfg.get("eval_steps", 500)),
            save_strategy=training_cfg.get("save_strategy", "steps"),
            save_steps=int(training_cfg.get("save_steps", 500)),
            save_total_limit=int(training_cfg.get("save_total_limit", 2)),
            load_best_model_at_end=bool(training_cfg.get("load_best_model_at_end", True)),
            metric_for_best_model=training_cfg.get("metric_for_best_model", "exact_match"),
            greater_is_better=bool(training_cfg.get("greater_is_better", True)),
            predict_with_generate=True,
            generation_max_length=self.config.get("model", {}).get("max_target_length", 128),
            report_to="none",  # Có thể đổi sang 'tensorboard' hoặc 'wandb'
        )

        # Chuẩn bị Collator và Metrics Callback
        self.data_collator = build_data_collator(
            tokenizer=self.model_wrapper.tokenizer,
            model=self.model_wrapper.model,
        )
        self.compute_metrics_fn = build_compute_metrics_fn(self.model_wrapper.tokenizer)

        # Khởi tạo HF Seq2SeqTrainer
        self.hf_trainer = Seq2SeqTrainer(
            model=self.model_wrapper.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics_fn,
        )

    def train(self) -> Any:
        """Thực thi vòng lặp huấn luyện."""
        train_result = self.hf_trainer.train()

        # Lưu model tốt nhất
        best_model_dir = Path(self.training_args.output_dir) / "best_model"
        self.model_wrapper.save_pretrained(best_model_dir)

        return train_result

    def evaluate(self) -> Dict[str, float]:
        """Đánh giá mô hình trên tập validation/test."""
        return self.hf_trainer.evaluate()
