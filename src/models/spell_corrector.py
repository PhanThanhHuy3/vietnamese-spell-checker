"""Transformer inference and checkpoint persistence."""
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data.preprocess import normalize_vietnamese_text


class VietnameseSpellCorrector:
    def __init__(self, model_name_or_path, device=None, local_only=False):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_only)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, local_files_only=local_only)
        self.model.to(self.device).eval()

    def fits(self, text, source_limit, prefix="spelling: "):
        ids = self.tokenizer.encode(prefix + normalize_vietnamese_text(text), verbose=False)
        return len(ids) <= source_limit

    @torch.inference_mode()
    def generate(self, texts, source_limit, target_limit, beams=4, prefix="spelling: "):
        if not texts:
            return []
        encoded = self.tokenizer([prefix + normalize_vietnamese_text(t) for t in texts],
                                 padding=True, truncation=False, return_tensors="pt", verbose=False)
        if encoded.input_ids.shape[1] > source_limit:
            raise ValueError("Đầu vào vượt giới hạn token")
        generated = self.model.generate(
            **encoded.to(self.device), max_new_tokens=target_limit, num_beams=beams,
            no_repeat_ngram_size=0, repetition_penalty=1.0, early_stopping=beams > 1,
            return_dict_in_generate=True, output_scores=True,
        )
        scores = self.model.compute_transition_scores(
            generated.sequences, generated.scores, getattr(generated, "beam_indices", None),
            normalize_logits=True,
        )
        generated_ids = generated.sequences[:, -scores.shape[1]:]
        eos = self.model.generation_config.eos_token_id
        token_scores = []
        for ids, row in zip(generated_ids.tolist(), scores.tolist()):
            end = ids.index(eos) + 1 if eos in ids else 0
            token_scores.append(sum(row[:end]) / end if end else None)
        return list(zip(self.tokenizer.batch_decode(generated.sequences, skip_special_tokens=True),
                        token_scores))

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
