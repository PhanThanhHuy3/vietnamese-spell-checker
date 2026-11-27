"""The single correction pipeline used by CLI, evaluation and HTTP."""
import json
import logging
from pathlib import Path
from threading import Lock
from time import perf_counter

from src.data.context_corpus import context_sentences
from src.models.baseline import NgramBaseline
from src.models.accent_restorer import END, START, term
from src.postprocessing import WORD, apply_edits, document_chunks, word_suggestions, without_accents
from src.data.validate_split import validate

MODEL_PATHS = {
    "vit5": "checkpoints/vit5/best_model",
    "byt5": "checkpoints/byt5/best_model",
}


LEGACY_VIT5 = "checkpoints/vit5_spelling_corrector/best_model"
MODEL_DETAILS = {
    "baseline": {"role": "Baseline truyền thống", "architecture": "N-gram → xếp hạng ứng viên → căn chỉnh",
                 "tokenizer": "Tách âm tiết Unicode", "dataset": "VSEC Train và ngữ liệu tiếng Việt đã lọc trùng"},
    "vit5": {"role": "Mô hình Deep Learning chính", "architecture": "SentencePiece → Encoder ViT5 → Decoder → căn chỉnh",
             "pretrained": "VietAI/vit5-base", "url": "https://huggingface.co/VietAI/vit5-base",
             "framework": "PyTorch", "tokenizer": "SentencePiece", "objective": "Seq2Seq · Cross-Entropy · AdamW",
             "dataset": "VSEC Corpus"},
    "byt5": {"role": "Mô hình Transformer đối chứng", "architecture": "Byte UTF-8 → Encoder ByT5 → Decoder → căn chỉnh",
             "pretrained": "google/byt5-small", "url": "https://huggingface.co/google/byt5-small",
             "framework": "PyTorch", "tokenizer": "Byte UTF-8", "objective": "Seq2Seq · Cross-Entropy · AdamW",
             "dataset": "VSEC Corpus"},
}
MODEL_NAMES = {"baseline": "N-gram + Levenshtein", "vit5": "ViT5-base", "byt5": "ByT5-small"}


class ModelUnavailable(RuntimeError):
    pass


def checkpoint_metadata(path, manifest):
    try:
        meta = json.loads((path / "training_meta.json").read_text(encoding="utf-8"))
        if (not meta.get("completed") or not meta.get("training_id")
                or meta["raw_sha256"] != manifest["raw_sha256"]
                or any(meta["splits"][k] != manifest["splits"][k]["sha256"] for k in ("train", "val", "test"))
                or not (path / "config.json").is_file()
                or not any(path.glob("model*.safetensors"))):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ModelUnavailable(f"Checkpoint chưa được xác minh trên phân chia VSEC hiện tại: {path.name}") from exc
    return meta


class SpellInferencePipeline:
    def __init__(self, path, meta, conservative=True):
        from src.models.spell_corrector import VietnameseSpellCorrector

        self.corrector = VietnameseSpellCorrector(str(path), local_only=True)
        self.source_limit = meta["model"]["max_source_length"]
        self.target_limit = meta["model"]["max_target_length"]
        self.prefix = meta["model"].get("prefix", "spelling: ")
        self.beams = meta["generation"]["num_beams"]
        self.threshold = None
        if conservative:
            self.threshold = meta["generation"].get("min_log_probability")
        self.max_distance = meta["generation"].get("max_edit_distance", 2) if conservative else None

    def fits(self, text):
        return self.corrector.fits(text, self.source_limit, self.prefix)

    def scored_chunks(self, texts):
        spans = [(i, offset, chunk) for i, text in enumerate(texts)
                 for offset, chunk in document_chunks(text, self.fits)]
        chunks = [span for span in spans if self.fits(span[2])]
        self.skipped_input_chunks = len(spans) - len(chunks)
        self.scored_input_chunks = len(spans)
        predictions = []
        for start in range(0, len(chunks), 8):
            batch = chunks[start:start + 8]
            outputs = self.corrector.generate([chunk for _, _, chunk in batch], self.source_limit,
                                             self.target_limit, self.beams, self.prefix)
            predictions.extend((*item, *output) for item, output in zip(batch, outputs, strict=True))
        return predictions

    def filter_chunks(self, texts, predictions, threshold):
        results = [[] for _ in texts]
        for index, offset, chunk, candidate, score in predictions:
            if score is None or (threshold is not None and score < threshold):
                continue
            for edit in word_suggestions(chunk, candidate, self.max_distance):
                edit["start"] += offset
                edit["end"] += offset
                results[index].append(edit)
        return results

    def batch_edits(self, texts):
        chunks = self.scored_chunks(texts)
        self.incomplete_output_chunks = sum(score is None for _, _, _, _, score in chunks)
        return self.filter_chunks(texts, chunks, self.threshold)

    def correct_many(self, texts):
        return [apply_edits(text, edits) for text, edits in zip(texts, self.batch_edits(texts), strict=True)]


class CorrectionService:
    def __init__(self, root, model_paths=None):
        self.root = Path(root)
        self.paths = {name: self.root / path for name, path in (model_paths or MODEL_PATHS).items()}
        self.lock = Lock()
        self.engines = {}
        self.runtime_checks = {}
        self.splits, self.manifest = validate(self.root / "data/processed", self.root / "data/splits/split_manifest.json")
        self.baseline = NgramBaseline(self.splits["train"], context_sentences(self.splits, self.root))

    def deployment_metadata(self, name):
        path = self.paths[name]
        label = MODEL_NAMES[name]
        if not path.is_dir():
            raise ModelUnavailable(f"{label} chưa có checkpoint triển khai.")
        meta = checkpoint_metadata(path, self.manifest)
        if meta.get("model_key") != name or meta.get("smoke"):
            raise ModelUnavailable(f"Checkpoint {label} hiện chỉ dùng thử hoặc sai mô hình.")
        if not meta.get("calibration", {}).get("selected"):
            raise ModelUnavailable(f"Checkpoint {label} chưa được chọn trên tập validation.")
        return meta

    def inventory(self, name):
        if name not in MODEL_NAMES:
            return None
        info = {"name": MODEL_NAMES[name], **MODEL_DETAILS[name]}
        if name == "baseline":
            return {**info, "available": True, "state": "ready",
                    "reason": "Sửa chính tả và khôi phục dấu theo ngữ cảnh trên CPU."}
        path = self.paths[name]
        legacy = self.root / LEGACY_VIT5
        inspect = legacy if name == "vit5" and not path.is_dir() and legacy.is_dir() else path
        files = {file.name: file.stat().st_size for file in inspect.iterdir() if file.is_file()} if inspect.is_dir() else {}
        fields = {"checkpoint_path": str(inspect.relative_to(self.root)) if inspect.is_relative_to(self.root) else str(inspect),
                  "files": files, "weights_bytes": sum(size for filename, size in files.items()
                                                       if filename.endswith(".safetensors")),
                  "legacy": inspect != path}
        state, meta, available = "untrained", None, False
        try:
            meta = self.deployment_metadata(name)
            changed = (meta.get("training_id"), path.stat().st_mtime_ns)
            check = self.runtime_checks.get(name)
            if check and check[0] == changed:
                available, reason = check[1], check[2]
                state = "ready" if available else "incompatible"
            else:
                state, reason = "validated", "Metadata hợp lệ. Đang chờ kiểm tra nạp và chạy thử."
        except ModelUnavailable as exc:
            reason = str(exc)
            if files:
                state = "unverified"
                if fields["legacy"]:
                    reason = ("Đã tìm thấy checkpoint ViT5 cũ có trọng số và tokenizer. "
                              "Thiếu metadata xác minh phân chia hiện tại.")
        return {**info, **fields, "available": available, "state": state, "reason": reason, "metadata": meta}

    def check_checkpoint(self, name):
        info = self.inventory(name)
        if not info or name == "baseline":
            return info
        if info.get("legacy"):
            from src.models.spell_corrector import VietnameseSpellCorrector
            try:
                with self.lock:
                    wrapper = VietnameseSpellCorrector(str(self.root / LEGACY_VIT5), local_only=True)
                    example = "ban la ai vay"
                    output, score = wrapper.generate([example], 96, 96, 4)[0]
                info["diagnostic"] = {"loaded": True, "input": example, "prediction": output,
                                      "stopped_at_eos": score is not None,
                                      "device": str(wrapper.device),
                                      "status": "Chỉ đối chiếu khả năng nạp checkpoint cũ."}
            except (OSError, ImportError, RuntimeError, ValueError) as exc:
                info["diagnostic"] = {"loaded": False, "error": type(exc).__name__}
            return info
        if info["state"] not in ("validated", "ready", "incompatible"):
            return info
        path = self.paths[name]
        meta = self.deployment_metadata(name)
        signature = (meta.get("training_id"), path.stat().st_mtime_ns)
        with self.lock:
            try:
                engine = SpellInferencePipeline(path, meta)
                _, _, _, raw_prediction, sequence_score = engine.scored_chunks(["Hôm nai trời đẹp."])[0]
                if not raw_prediction.strip() or sequence_score is None:
                    raise ValueError("Đầu ra thử nghiệm rỗng hoặc vượt giới hạn sinh.")
                self.engines[name] = (meta, engine)
                self.runtime_checks[name] = (signature, True, "Đã nạp checkpoint và chạy thử inference.")
            except (OSError, IndexError, ImportError, RuntimeError, ValueError) as exc:
                self.engines.pop(name, None)
                self.runtime_checks[name] = (signature, False, f"Không tương thích hoặc không chạy được: {type(exc).__name__}.")
        return self.inventory(name)

    def status(self):
        result = {"baseline": self.inventory("baseline")}
        for name in self.paths:
            info = self.inventory(name)
            result[name] = self.check_checkpoint(name) if info["state"] == "validated" else info
        return result

    def recommendation(self, models=None):
        models = self.status() if models is None else models
        preferred = next((key for key in ("vit5", "byt5", "baseline") if models.get(key, {}).get("available")))
        if preferred == "baseline":
            reason = "ViT5 và ByT5 chưa có checkpoint đủ điều kiện triển khai."
        else:
            reason = f"{MODEL_NAMES[preferred]} đã có checkpoint được chọn trên tập validation."
        return {"model": preferred, "reason": reason}

    def model_edits(self, text, model):
        if model == "baseline":
            return self.baseline.suggest(text), None
        if model not in self.paths:
            raise ValueError("Mô hình không hợp lệ")
        with self.lock:
            meta = self.deployment_metadata(model)
            try:
                if model not in self.engines or self.engines[model][0] != meta:
                    self.engines[model] = (meta, SpellInferencePipeline(self.paths[model], meta))
                engine = self.engines[model][1]
                edits = engine.batch_edits([text])[0]
                incomplete = engine.skipped_input_chunks + engine.incomplete_output_chunks
                warning = f"{incomplete} đoạn vượt giới hạn mô hình và được giữ nguyên." if incomplete else None
                return edits, warning
            except (OSError, ImportError, RuntimeError, ValueError) as exc:
                logging.exception("Không chạy được mô hình %s", model)
                raise ModelUnavailable(f"Không chạy được {MODEL_NAMES[model]} trên máy chủ.") from exc

    def correct(self, text, model="auto"):
        started = perf_counter()
        requested = model
        if model == "auto":
            model = self.recommendation()["model"]
        elif model in self.paths and not self.status()[model]["available"]:
            raise ModelUnavailable(self.inventory(model)["reason"])
        edits, warning = self.model_edits(text, model)
        edits = [edit for edit in edits if edit["original_segment"] != edit["corrected_segment"]]
        positions = {match.start(): i for i, match in enumerate(WORD.finditer(text), 1)}
        for index, edit in enumerate(edits, 1):
            edit.update(id=index, position=positions[edit["start"]])
            edit.setdefault("alternatives", [])
            edit.setdefault("auto_apply", True)
        for edit in edits:
            edit["alternatives"] = list(dict.fromkeys(option for option in edit["alternatives"]
                                                     if option not in (edit["original_segment"], edit["corrected_segment"])))
        automatic = [edit for edit in edits if edit["auto_apply"]]
        return {"original_text": text, "corrected_text": apply_edits(text, edits),
                "auto_corrected_text": apply_edits(text, automatic), "total_syllables": len(positions),
                "total_errors": len(edits), "errors_found": edits, "has_error": bool(edits),
                "model": model, "requested_model": requested, "warning": warning,
                "latency_ms": round((perf_counter() - started) * 1000)}

    def accent_options(self, text, original, suggestion, start, end):
        language = self.baseline.language_model
        key = without_accents(term(original))
        if not key.isalpha() or without_accents(term(suggestion)) != key:
            return []
        left_parts = text[:start].rsplit(maxsplit=2)[-2:]
        right_parts = text[end:].split(maxsplit=2)[:2]
        left = [term(part) for part in WORD.findall(" ".join(left_parts))[-2:]]
        right = [term(part) for part in WORD.findall(" ".join(right_parts))[:2]]
        earlier, previous = ([START, START] + left)[-2:]
        following, later = (right + [END, END])[:2]

        def score(word):
            return (language.transition(earlier, previous, word)
                    + language.transition(previous, word, following)
                    + language.transition(word, following, later))

        options = [word for word in language.vocabulary.get(key, ())
                   if language.unigrams[word] >= 4 and word not in (term(original), term(suggestion))]
        options.sort(key=score, reverse=True)
        return [word.upper() if original.isupper() else word.capitalize() if original[0].isupper() else word
                for word in options[:5]]

    def check(self, text, model="auto", assist=True):
        result = self.correct(text, model)
        errors = []
        for edit in result["errors_found"]:
            original, suggestion = edit["original_segment"], edit["corrected_segment"]
            kind = "missing_diacritics" if original.isascii() and without_accents(suggestion) == original else "spelling"
            if edit["category"] == "Tách hoặc ghép âm tiết":
                kind = "word_boundary"
            alternatives = (self.accent_options(text, original, suggestion, edit["start"], edit["end"])
                            if assist and result["model"] != "baseline" else [])
            auto_apply = edit["auto_apply"]
            if assist and result["total_syllables"] == 1 and alternatives:
                auto_apply = False
            errors.append({**edit, "id": f"error-{edit['id']}", "original": original, "suggestion": suggestion,
                           "alternatives": list(dict.fromkeys([*edit["alternatives"], *alternatives])),
                           "auto_apply": auto_apply, "type": kind, "status": "pending"})
        if assist and result["model"] != "baseline" and not result["warning"]:
            occupied = {index for edit in errors for index in range(edit["start"], edit["end"])}
            for edit in self.baseline.suggest(text, include_alternatives=True):
                original, suggestion = edit["original_segment"], edit["corrected_segment"]
                if (not original.isascii() or not original.isalpha() or
                        without_accents(term(suggestion)) != term(original) or
                        not edit["auto_apply"] or any(i in occupied for i in range(edit["start"], edit["end"]))):
                    continue
                errors.append({**edit, "id": f"assist-{edit['start']}", "original": original, "suggestion": suggestion,
                               "auto_apply": False, "type": "missing_diacritics", "status": "pending",
                               "source": "accent_support"})
            errors.sort(key=lambda edit: edit["start"])
        predicted = apply_edits(text, errors)
        automatic = apply_edits(text, [edit for edit in errors if edit["auto_apply"]])
        return {**result, "corrected_text": predicted, "auto_corrected_text": automatic, "predicted_text": predicted,
                "total_errors": len(errors), "has_error": bool(errors), "errors_found": errors,
                "syllable_count": result["total_syllables"], "suggestion_count": len(errors), "errors": errors}
