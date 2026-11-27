"""Shared edit alignment and document-preserving spelling suggestions."""
import re
import unicodedata

import regex
from rapidfuzz.distance import Levenshtein

WORD = regex.compile(r"[\p{L}\p{M}\p{N}]+")
PROTECTED = regex.compile(
    r"https?://[^\s<>]+|www\.[^\s<>]+|[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}"
    r"|[A-Za-z0-9_]+(?:[./_+#-][A-Za-z0-9_]+)+|(?:C\+\+|C#)(?!\w)"
    r"|\b(?:Python|JavaScript|TypeScript|HTML|CSS|SQL|PyTorch|ViT5|ByT5|BARTpho|OpenAI)\b"
    r"|\b\p{Lu}[\p{L}\p{M}]+(?:[ \t]+\p{Lu}[\p{L}\p{M}]+)+"
)


def without_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
                   if not unicodedata.combining(c))


def alignment(source, target):
    return Levenshtein.opcodes(source, target)


def protected_spans(text):
    return [m.span() for m in PROTECTED.finditer(text)]


def is_protected(match, spans):
    word = match.group()
    return (word.isupper() or word.isnumeric()
            or any(match.start() < end and match.end() > start for start, end in spans))


def error_category(original, replacement):
    a, b = original.casefold(), replacement.casefold()
    if without_accents(a) == without_accents(b):
        return "Dấu tiếng Việt"
    if len(a) != len(b):
        return "Thừa hoặc thiếu ký tự"
    if sorted(a) == sorted(b):
        return "Đảo ký tự"
    return "Thay thế âm tiết"


def word_suggestions(source, candidate, max_distance=None):
    """Accept local word substitutions and retain the original layout."""
    words = list(WORD.finditer(source))
    output = [m.group() for m in WORD.finditer(candidate)]
    keys = [unicodedata.normalize("NFC", m.group()) for m in words]
    output = [unicodedata.normalize("NFC", word) for word in output]
    spans = protected_spans(source)
    edits = []
    for op in alignment(keys, output):
        if op.tag != "replace":
            continue
        for source_index, output_index in zip(range(op.src_start, op.src_end), range(op.dest_start, op.dest_end)):
            match = words[source_index]
            before, after = keys[source_index], output[output_index]
            if is_protected(match, spans) or before.casefold() == after.casefold():
                continue
            if max_distance is not None and Levenshtein.distance(before.casefold(), after.casefold()) > max_distance:
                continue
            after = after.capitalize() if before[0].isupper() else after.lower()
            edits.append(dict(start=match.start(), end=match.end(), original_segment=match.group(),
                              corrected_segment=after, category=error_category(before, after)))
    return edits


def apply_edits(text, edits):
    for edit in reversed(edits):
        text = text[:edit["start"]] + edit["corrected_segment"] + text[edit["end"]:]
    return text


def document_chunks(text, fits):
    """Yield original slices with no source token silently truncated."""
    start = end = 0
    for match in re.finditer(r"\S+", text):
        if end > start and ("\n" in text[end:match.start()] or not fits(text[start:match.end()])):
            yield start, text[start:end]
            start = end = match.start()
        if end == start:
            start = match.start()
        end = match.end()
    if end > start:
        yield start, text[start:end]
