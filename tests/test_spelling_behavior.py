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
    result = service.correct(source)
    assert result["corrected_text"] == expected
    for edit in result["errors_found"]:
        assert source[edit["start"]:edit["end"]] == edit["original_segment"]
    assert apply_edits(source, result["errors_found"]) == expected


@pytest.mark.parametrize("record", read_rows(ROOT / "data/practice.jsonl"), ids=lambda row: row["id"])
def test_practice_cases(service, record):
    result = service.correct(record["input_text"])
    assert result["corrected_text"] == record["target_text"]


def test_unaccented_input_has_six_independent_edits(service):
    original = "toi khong muon di an dau"
    result = service.correct(original)
    assert result["total_syllables"] == result["total_errors"] == 6
    assert result["auto_corrected_text"] == "tôi không muốn đi ăn đâu"
    edits = result["errors_found"]
    assert [edit["position"] for edit in edits] == list(range(1, 7))
    assert [edit["start"] for edit in edits] == [0, 4, 10, 15, 18, 21]
    assert apply_edits(original, [edits[2]]) == "toi khong muốn di an dau"


def test_ambiguous_accents_offer_a_choice(service):
    result = service.correct("toi muon an dau")
    edit = result["errors_found"][-1]
    assert edit["original_segment"] == "dau"
    assert {"đậu", "đâu"} <= {*edit["alternatives"], edit["corrected_segment"]}


def test_long_document_and_repeated_tokens_preserve_offsets(service):
    source = ("Bạn dang làm gì? Tôi đang hoc. 😊  \\n" * 40) + "Bạn dang làm gì?"
    expected = ("Bạn đang làm gì? Tôi đang học. 😊  \\n" * 40) + "Bạn đang làm gì?"
    result = service.correct(source)
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
            with urlopen(url + path, timeout=4) as response:
                return json.load(response)

        def post(path, body):
            request = Request(url + path, data=json.dumps(body).encode("utf-8"),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=4) as response:
                return json.load(response)

        try:
            text = "Chươg trình học tập tại Đà Nẵng."
            result = post("/api/correct_spelling", {"text": text, "model": "baseline"})
            assert result["corrected_text"] == "Chương trình học tập tại Đà Nẵng."
            assert get("/api/status")["models"]["baseline"]["available"]
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
