"""Shared edit alignment and document-preserving spelling suggestions."""
from itertools import groupby
import re
import unicodedata

import regex
from rapidfuzz.distance import Levenshtein

WORD = regex.compile(r"[\p{L}\p{M}\p{N}]+")
PROTECTED = regex.compile(
    r"(?i:https?://|www\.)[^\s<>]+|[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}"
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


def sentence_word_groups(text, max_words=128):
    """Partition word spans at sentence boundaries outside protected regions."""
    spans = protected_spans(text)
    group = []
    for match in WORD.finditer(text):
        if group:
            boundary = group[-1].end()
            gap = text[boundary:match.start()]
            sentence_end = re.search(r"[.!?…;\r\n]", gap) and not any(
                start <= boundary and match.start() < end for start, end in spans
            )
            if len(group) >= max_words or sentence_end:
                yield group
                group = []
        group.append(match)
    if group:
        yield group


def error_category(original, replacement):
    a, b = original.casefold(), replacement.casefold()
    if without_accents(a) == without_accents(b):
        return "Thiếu dấu" if a.isascii() else "Sai dấu tiếng Việt"
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
    for unchanged, group in groupby(alignment(keys, output), key=lambda op: op.tag == "equal"):
        if unchanged:
            continue
        operations = list(group)
        first, last = operations[0], operations[-1]
        source_indices = range(first.src_start, last.src_end)
        output_indices = range(first.dest_start, last.dest_end)
        if len(source_indices) == len(output_indices):
            pairs = [(range(i, i + 1), range(j, j + 1)) for i, j in zip(source_indices, output_indices)]
        elif {len(source_indices), len(output_indices)} == {1, 2}:
            pairs = [(source_indices, output_indices)]
        else:
            continue
        for source_part, target_part in pairs:
            matches = [words[i] for i in source_part]
            if any(is_protected(match, spans) for match in matches):
                continue
            start, end = matches[0].start(), matches[-1].end()
            before = " ".join(keys[i] for i in source_part)
            after = " ".join(output[i] for i in target_part)
            if before.casefold() == after.casefold():
                continue
            if len(source_part) != len(target_part):
                if (len(matches) > 1 and not re.fullmatch(r"[ 	]+", source[matches[0].end():matches[-1].start()])):
                    continue
                joined = lambda text: without_accents(text.replace(" ", "").casefold())
                if Levenshtein.distance(joined(before), joined(after)) > 1:
                    continue
            elif (max_distance is not None and without_accents(before.casefold()) != without_accents(after.casefold())
                  and Levenshtein.distance(before.casefold(), after.casefold()) > max_distance):
                continue
            after = after.capitalize() if before[0].isupper() else after.lower()
            category = "Tách hoặc ghép âm tiết" if len(source_part) != len(target_part) else error_category(before, after)
            edits.append(dict(start=start, end=end, original_segment=source[start:end],
                              corrected_segment=after, category=category))
    return edits


def apply_edits(text, edits):
    for edit in reversed(edits):
        text = text[:edit["start"]] + edit["corrected_segment"] + text[edit["end"]:]
    return text


def document_chunks(text, fits):
    """Yield original slices with no source token silently truncated."""
    start = end = 0
    spans = protected_spans(text)
    for match in re.finditer(r"\S+", text):
        previous = text[start:end]
        terminal = (previous.endswith((".", "!", "?", "…", ";"))
                    and not any(a <= end - 1 < b for a, b in spans))
        if end > start and (terminal or "\n" in text[end:match.start()] or not fits(text[start:match.end()])):
            yield start, text[start:end]
            start = end = match.start()
        if end == start:
            start = match.start()
        end = match.end()
    if end > start:
        yield start, text[start:end]
