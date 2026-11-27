import pytest

from src.training.metrics import (
    compute_bleu,
    compute_cer,
    compute_detection_metrics,
    compute_exact_match,
    compute_token_accuracy,
    compute_token_edit_distance,
    compute_wer,
)


def test_exact_match_and_token_accuracy():
    preds = ["Hôm nay trời đẹp", "Tôi đi học"]
    refs = ["Hôm nay trời đẹp", "Tôi đi hoc"]
    assert compute_exact_match(preds, refs) == 0.5
    # Câu 1 khớp 4/4 token; câu 2 khớp 2/3 token ("học" != "hoc") => 6/7.
    assert compute_token_accuracy(preds, refs) == pytest.approx(6 / 7)


def test_cer_is_character_level():
    assert compute_cer(["abc"], ["abd"]) == pytest.approx(1 / 3)


def test_token_edit_distance_is_token_level():
    assert compute_token_edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert compute_token_edit_distance(["a", "b"], ["a", "b", "c"]) == 1
    assert compute_token_edit_distance(["a", "b", "c"], ["a", "c"]) == 1


def test_wer_uses_word_edits_not_joined_characters():
    # 1 từ sai / 2 từ tham chiếu = 50%, không phải khoảng cách ký tự / 2.
    assert compute_wer(["abc xyz"], ["abc def"]) == pytest.approx(0.5)
    # Một từ bị xóa / 3 từ = 33.33%.
    assert compute_wer(["a b"], ["a b c"]) == pytest.approx(1 / 3)
    # Một từ bị chèn / 2 từ = 50%.
    assert compute_wer(["a b c"], ["a b"]) == pytest.approx(0.5)


def test_detection_metrics_basic():
    # Vị trí 2 có lỗi và được sửa; vị trí 3 không lỗi nhưng bị sửa (FP).
    metrics = compute_detection_metrics(
        predictions=["a c x"],
        sources=["a b c"],
        references=["a c c"],
    )
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0
    assert metrics["precision"] == 50.0
    assert metrics["recall"] == 100.0


def test_bleu_non_negative():
    assert compute_bleu(["a b c d"], ["a b c d"]) > 0
