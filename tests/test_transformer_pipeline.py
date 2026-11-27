import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

from src.data.annotation_references import best_reference
from src.data.dataset import SpellingDataset, build_data_collator
from src.inference import ModelUnavailable, checkpoint_metadata
from src.models.spell_corrector import VietnameseSpellCorrector


def test_transformer_padding_training_validation_and_reload(tmp_path):
    tokens = ["<pad>", "</s>", "<unk>", "spelling:", "Tôi", "học", "hoc"]
    backend = Tokenizer(WordLevel({token: index for index, token in enumerate(tokens)}, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>", model_input_names=["input_ids", "attention_mask"])
    model = T5ForConditionalGeneration(T5Config(
        vocab_size=len(tokenizer), d_model=16, d_ff=32, d_kv=8, num_layers=1, num_decoder_layers=1,
        num_heads=2, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0,
    ))
    rows = [{"id": "short", "input_text": "hoc", "target_text": "học"},
            {"id": "long", "input_text": "Tôi hoc", "target_text": "Tôi học"}]
    data = SpellingDataset(rows, tokenizer, 16, 16, "spelling: ")
    batch = build_data_collator(tokenizer, model)([data[0], data[1]])
    assert batch["labels"][0, -1].item() == -100
    assert batch["attention_mask"][0, -1].item() == 0

    original = model.shared.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert model.shared.weight.grad.abs().sum() > 0
    optimizer.step()
    trained = model.shared.weight.detach().clone()
    assert not torch.equal(original, trained)

    model.eval()
    with torch.no_grad():
        expected = model(**batch).logits
    assert torch.equal(model.shared.weight, trained)
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    restored = VietnameseSpellCorrector(str(tmp_path), device=torch.device("cpu"), local_only=True)
    with torch.no_grad():
        assert torch.allclose(restored.model(**batch).logits, expected)
    assert len(restored.generate(["Tôi hoc"], 16, 8, beams=1)) == 1


def test_checkpoint_rejects_unverified_split(tmp_path):
    (tmp_path / "training_meta.json").write_text(json.dumps(
        {"completed": True, "training_id": "demo", "raw_sha256": "old", "splits": {}}
    ), encoding="utf-8")
    with pytest.raises(ModelUnavailable):
        checkpoint_metadata(tmp_path, {"raw_sha256": "new", "splits": {}})


def test_alternative_reference_is_accepted():
    assert best_reference("Tôi học Toán.", "Tôi học toán.", ("Tôi học Toán.",)) == "Tôi học Toán."
    assert best_reference("Tôi làm bài.", "Tôi học toán.", ()) == "Tôi học toán."
