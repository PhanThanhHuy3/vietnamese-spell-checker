import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from app import make_server
from src.data.prepare_vsec import MANIFEST, PROCESSED, content_keys, read_rows
from src.data.validate_split import validate
from src.inference import CorrectionService
from src.postprocessing import apply_edits, word_suggestions
from src.training.metrics import evaluate_edits
from generate_noise_data import augment

ROOT = PROCESSED.parents[1]


@pytest.fixture(scope="module")
def service():
    if not MANIFEST.is_file() or not (PROCESSED / "train.jsonl").is_file():
        pytest.skip("Chạy python -m src.data.prepare_vsec trước")
    return CorrectionService(ROOT)


@pytest.mark.parametrize(("source", "expected"), [
    ("Tôi cảm thây vui.", "Tôi cảm thấy vui."),
    ("Suy nghỉ.", "Suy nghĩ."),
    ("Chường học.", "Trường học."),
    ("Học xinh.", "Học sinh."),
    ("Côn nghệ.", "Công nghệ."),
    ("Hoc sinh.", "Học sinh."),
    ("Hôm nai.", "Hôm nay."),
    ("Tooi đi học.", "Tôi đi học."),
    ("Họcc tập.", "Học tập."),
    ("Chươg trình.", "Chương trình."),
    ("Tôi đang sữa bài.", "Tôi đang sửa bài."),
    ("Hôm nai tôi đến chường để học lập chình.", "Hôm nay tôi đến trường để học lập trình."),
    ("Tôi đang uống sữa.", "Tôi đang uống sữa."),
    ("Bạn ấy rất xinh.", "Bạn ấy rất xinh."),
    ("Con nai đang chạy.", "Con nai đang chạy."),
    ("Tôi sẽ đến vào tuần tới.", "Tôi sẽ đến vào tuần tới."),
    ("Chươg trình họcc tập của tuần tới đã được cập nhật.", "Chương trình học tập của tuần tới đã được cập nhật."),
    ("🙂  Tooi đang hoc Python tại Đà Nẵng.\nEmail demo@example.org.",
     "🙂  Tôi đang học Python tại Đà Nẵng.\nEmail demo@example.org."),
])
def test_expected_correction(service, source, expected):
    result = service.correct(source, "baseline")
    assert result["corrected_text"] == expected
    for edit in result["errors_found"]:
        assert source[edit["start"]:edit["end"]] == edit["original_segment"]
    assert apply_edits(source, result["errors_found"]) == expected


@pytest.mark.parametrize("record", read_rows(ROOT / "data/practice.jsonl"), ids=lambda row: row["id"])
def test_practice_cases(service, record):
    result = service.correct(record["input_text"], "baseline")
    assert result["corrected_text"] == record["target_text"]


def test_unaccented_input_has_six_independent_edits(service):
    original = "toi khong muon di an dau"
    result = service.correct(original, "baseline")
    assert result["total_syllables"] == result["total_errors"] == 6
    assert result["auto_corrected_text"] == "tôi không muốn đi ăn đâu"
    edits = result["errors_found"]
    assert [edit["position"] for edit in edits] == list(range(1, 7))
    assert [edit["start"] for edit in edits] == [0, 4, 10, 15, 18, 21]
    assert apply_edits(original, [edits[2]]) == "toi khong muốn di an dau"


def test_ambiguous_accents_offer_a_choice(service):
    result = service.correct("toi muon an dau", "baseline")
    edit = result["errors_found"][-1]
    assert edit["original_segment"] == "dau"
    assert {"đậu", "đâu"} <= {*edit["alternatives"], edit["corrected_segment"]}


def test_transformer_spelling_suggestions_offer_accent_choices(service, monkeypatch):
    def model_edits(text, model):
        proposed = "bạn là ai vậy" if text == "ban la ai vay" else text
        return word_suggestions(text, proposed, 2), None

    monkeypatch.setattr(service, "model_edits", model_edits)
    monkeypatch.setattr(service, "status", lambda: {"vit5": {"available": True}})
    response = service.check("ban la ai vay", "vit5")
    vay = response["errors"][-1]
    assert vay["suggestion"] == "vậy" and "váy" in vay["alternatives"]
    assert service.check("ban la ai vay", "vit5", assist=False)["errors"][-1]["alternatives"] == []
    assert service.check("Tôi muốn vay tiền.", "vit5")["errors"] == []

    source = "Em mua một chiếc vay màu xanh."
    response = service.check(source, "vit5")
    assert response["predicted_text"] == "Em mua một chiếc váy màu xanh."
    assert response["auto_corrected_text"] == source
    assert response["errors"][0]["source"] == "accent_support"


def test_long_document_and_repeated_tokens_preserve_offsets(service):
    source = ("Bạn dang làm gì? Tôi đang hoc. 😊  \\n" * 40) + "Bạn dang làm gì?"
    expected = ("Bạn đang làm gì? Tôi đang học. 😊  \\n" * 40) + "Bạn đang làm gì?"
    result = service.correct(source, "baseline")
    assert len(source) > 1000
    assert result["corrected_text"] == expected
    last = result["errors_found"][-1]
    assert last["start"] == source.rindex("dang")
    assert apply_edits(source, [last]) == source[:-len("dang làm gì?")] + "đang làm gì?"


def test_alignment_handles_word_insertion_without_false_shift():
    metrics = evaluate_edits(["Tôi đang đi học."], ["Tôi đi học."], ["Tôi sẽ đi học."])
    assert metrics["detection"]["true_positive"] == 1
    assert metrics["correction"]["true_positive"] == 0
    assert metrics["correction"]["false_positive"] == 1
    assert metrics["correction"]["false_negative"] == 1


def test_alignment_preserves_formatting_and_accepts_plausible_merge():
    original = "Tất nh iên,  tôi đi học.\nĐà Nẵng."
    candidate = "Tất nhiên , tôi đi học . Đà Nẳng ."
    assert apply_edits(original, word_suggestions(original, candidate, 2)) == "Tất nhiên,  tôi đi học.\nĐà Nẵng."


def test_only_train_can_supply_augmented_pairs(service):
    splits = service.splits
    train_ids = {row["id"] for row in splits["train"]}
    held_out = set().union(*(content_keys(row) for name in ("val", "test") for row in splits[name]))
    pairs = augment(splits, 25, .2, 42, identities=False)
    assert len(pairs) == 25
    assert {row["noise_type"] for row in pairs} == {"full_diacritics", "partial_diacritics", "mixed"}
    assert all(row["parent_id"] in train_ids and not (content_keys(row) & held_out) for row in pairs)


def test_http_integration(service):
    with make_server(port=0, root=ROOT) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"

        def get(path):
            with urlopen(url + path, timeout=60) as response:
                return json.load(response)

        def post(path, body):
            request = Request(url + path, data=json.dumps(body).encode("utf-8"),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=60) as response:
                return json.load(response)

        try:
            text = "Chươg trình học tập tại Đà Nẵng."
            result = post("/api/correct_spelling", {"text": text, "model": "baseline"})
            assert result["corrected_text"] == "Chương trình học tập tại Đà Nẵng."
            card = get("/api/models")["models"]["baseline"]
            assert card["available"] and card["state"] == "ready"
            assert get("/api/models/baseline")["role"] == "Baseline truyền thống"
            checked = post("/api/check", {"text": "ban la ai vay", "model": "baseline"})
            assert checked["predicted_text"] == "bạn là ai vậy"
            assert checked["syllable_count"] == 4 and checked["suggestion_count"] == 3
            assert [edit["position"] for edit in checked["errors"]] == [1, 2, 4]
            assert all(edit["status"] == "pending" and edit["original"] not in
                       [edit["suggestion"], *edit["alternatives"]] for edit in checked["errors"])
            assert all(checked["original_text"][edit["start"]:edit["end"]] == edit["original"]
                       for edit in checked["errors"])
            comparison = post("/api/compare", {"text": text, "models": ["baseline"]})
            assert not comparison["comparison"] and comparison["message"]
            assert comparison["original_text"] == comparison["results"]["baseline"]["original_text"]
            assert "reference_metrics" not in comparison["results"]["baseline"]
            with_reference = post("/api/compare", {"text": "Hôm nai trời đẹp.", "models": ["baseline"],
                                                   "reference": "Hôm nay trời đẹp."})
            assert with_reference["results"]["baseline"]["reference_metrics"]["correction"]["f1"] == 100
            samples = get("/api/dataset/samples?split=test&limit=1")["samples"]
            next_page = get("/api/dataset/samples?split=test&limit=1&offset=1")
            assert next_page["offset"] == 1 and next_page["samples"][0]["id"] != samples[0]["id"]
            assert get("/api/dataset/samples?split=test&limit=1&offset=935")["samples"] == []
            for path in ("/editor", "/guide", "/about", "/lab", "/lab/compare", "/lab/dataset",
                         "/lab/augmentation", "/lab/experiments", "/lab/models"):
                with urlopen(url + path, timeout=60) as response:
                    assert response.headers.get_content_type() == "text/html"
            selected = get("/api/dataset/samples?split=test&id=" + samples[0]["id"])["samples"][0]
            assert samples[0]["input_text"] == selected["input_text"] and selected["annotation"]
            assert get("/api/dataset/integrity")["manifest_ok"]
            payload = {"text": "Tôi đang học tiếng Việt.", "type": "full_diacritics", "seed": 0, "count": 2}
            first = post("/api/augment", payload)
            assert first == post("/api/augment", payload)
            assert len(first["variants"]) == 2 and first["variants"][0]["input_text"] == "Toi dang hoc tieng Viet."
            assert post("/api/augment", {**payload, "noise_prob": 0})["variants"][0]["input_text"] == payload["text"]
            assert sum(item["samples"] for item in get("/api/dataset")["splits"].values()) == 9341
            assert post("/api/compare_engines", {"text": text})["results"]["baseline"]["corrected_text"] == result["corrected_text"]
            assert post("/api/generate_noise", {"text": text, "noise_prob": 0})["noisy_text"] == text
            with pytest.raises(HTTPError) as exc:
                post("/api/correct_spelling", {"text": "   "})
            assert exc.value.code == 400
            with pytest.raises(HTTPError) as exc:
                post("/api/correct_spelling", {"text": []})
            assert exc.value.code == 400
            with pytest.raises(HTTPError) as exc:
                post("/api/correct_spelling", {"text": text, "model": "legacy"})
            assert exc.value.code == 400
        finally:
            server.shutdown()
            thread.join()
