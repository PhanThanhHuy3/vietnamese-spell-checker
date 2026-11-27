"""Corpus metrics based on Levenshtein alignment."""
from rapidfuzz.distance import Levenshtein
from sacrebleu.metrics import BLEU

from src.postprocessing import WORD, alignment


def compute_exact_match(predictions, references):
    return sum(p == r for p, r in zip(predictions, references, strict=True)) / max(len(references), 1)


def compute_levenshtein(left, right):
    return Levenshtein.distance(left, right)


def compute_token_edit_distance(left, right):
    return Levenshtein.distance(left, right)


def compute_cer(predictions, references):
    return sum(Levenshtein.distance(p, r) for p, r in zip(predictions, references, strict=True)) / max(
        sum(map(len, references)), 1
    )


def compute_wer(predictions, references):
    return sum(Levenshtein.distance(p.split(), r.split()) for p, r in zip(predictions, references, strict=True)) / max(
        sum(len(r.split()) for r in references), 1
    )


def compute_bleu(predictions, references):
    return float(BLEU(effective_order=True).corpus_score(predictions, [references]).score)


def compute_token_accuracy(predictions, references):
    correct = total = 0
    for prediction, reference in zip(predictions, references, strict=True):
        for op in alignment(prediction.split(), reference.split()):
            length = max(op.src_end - op.src_start, op.dest_end - op.dest_start)
            total += length
            if op.tag == "equal":
                correct += length
    return correct / max(total, 1)


def edits_for(source, output):
    base = WORD.findall(source)
    target = WORD.findall(output)
    edits = {}
    for op in alignment(base, target):
        if op.tag == "equal":
            continue
        if op.tag == "insert":
            edits[(op.src_start, "gap")] = tuple(target[op.dest_start:op.dest_end])
            continue
        for offset, index in enumerate(range(op.src_start, op.src_end)):
            edits[(index, "word")] = tuple(target[op.dest_start + offset : min(op.dest_start + offset + 1, op.dest_end)])
    return edits


def rates(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": round(100 * precision, 2), "recall": round(100 * recall, 2),
            "f1": round(200 * tp / max(2 * tp + fp + fn, 1), 2)}


def evaluate_edits(predictions, sources, references):
    detection = [0, 0, 0]
    correction = [0, 0, 0]
    clean_samples = clean_modified = 0
    failures = []
    for index, (prediction, source, reference) in enumerate(zip(predictions, sources, references, strict=True)):
        predicted = edits_for(source, prediction)
        gold = edits_for(source, reference)
        found = predicted.keys() & gold.keys()
        fixed = sum(predicted[position] == gold[position] for position in found)
        detection[0] += len(found)
        detection[1] += len(predicted) - len(found)
        detection[2] += len(gold) - len(found)
        correction[0] += fixed
        correction[1] += len(predicted) - fixed
        correction[2] += len(gold) - fixed
        if not gold:
            clean_samples += 1
            clean_modified += bool(predicted)
        if prediction != reference:
            failures.append(dict(index=index, missed=len(gold.keys() - predicted.keys()),
                                 overcorrected=len(predicted.keys() - gold.keys()), wrong_replacement=len(found) - fixed))
    return {"detection": rates(*detection), "correction": rates(*correction),
            "overcorrection": {"false_edits": detection[1], "clean_samples": clean_samples,
                               "clean_modified": clean_modified,
                               "clean_modified_rate": clean_modified / max(clean_samples, 1)},
            "failures": failures}


def compute_detection_metrics(predictions, sources, references):
    return evaluate_edits(predictions, sources, references)["detection"]


def summarize(predictions, sources, references):
    result = evaluate_edits(predictions, sources, references)
    result.update(num_samples=len(references), exact_match=compute_exact_match(predictions, references),
                  token_accuracy=compute_token_accuracy(predictions, references), bleu=compute_bleu(predictions, references),
                  cer=compute_cer(predictions, references), wer=compute_wer(predictions, references))
    return result
