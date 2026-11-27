"""Train-only Vietnamese bigram and Levenshtein baseline."""
from collections import Counter, defaultdict
from functools import lru_cache
import math

from rapidfuzz.distance import Levenshtein

from src.postprocessing import WORD, alignment, is_protected, protected_spans, without_accents


def tokens(text):
    return [m.group().casefold() for m in WORD.finditer(text)]


def deletion_keys(word):
    return {word, *(word[:i] + word[i + 1:] for i in range(len(word)))}


class NgramBaseline:
    def __init__(self, train_rows):
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.confusions = defaultdict(Counter)
        for row in train_rows:
            target = tokens(row["target_text"])
            source = tokens(row["input_text"])
            self.unigrams.update(target)
            sequence = ["<s>", *target, "</s>"]
            self.bigrams.update(zip(sequence, sequence[1:]))
            for op in alignment(source, target):
                if op.tag == "replace":
                    for a, b in zip(source[op.src_start:op.src_end], target[op.dest_start:op.dest_end]):
                        self.confusions[a][b] += 1

        self.deletions, self.accents = defaultdict(set), defaultdict(set)
        for word in self.unigrams:
            if not word.isalpha() or len(word) < 2:
                continue
            self.accents[without_accents(word)].add(word)
            for key in deletion_keys(word):
                self.deletions[key].add(word)

    @lru_cache(maxsize=4096)
    def candidates(self, word):
        known = self.confusions.get(word, {})
        result = {word, *known}
        if word not in self.unigrams:
            result.update(self.accents.get(without_accents(word), ()))
            close = set().union(*(self.deletions.get(key, ()) for key in deletion_keys(word)))
            result.update(w for w in close if Levenshtein.distance(word, w) == 1)
        return tuple(sorted(result))

    def correct(self, text):
        words = list(WORD.finditer(text))
        terms = [word.group().casefold() for word in words]
        protected = protected_spans(text)
        replacements = []
        for i, match in enumerate(words):
            word = terms[i]
            if len(word) < 2 or len(word) > 24 or is_protected(match, protected):
                continue
            left = terms[i - 1] if i else "<s>"
            right = terms[i + 1] if i + 1 < len(terms) else "</s>"

            def score(candidate):
                context = (math.log1p(self.bigrams[left, candidate])
                           + math.log1p(self.bigrams[candidate, right]))
                prior = .16 * math.log1p(self.unigrams[candidate])
                evidence = .65 * math.log1p(self.confusions.get(word, {}).get(candidate, 0))
                penalty = .85 * Levenshtein.distance(word, candidate)
                return context + prior + evidence - penalty

            candidate = max(self.candidates(word), key=score)
            margin = 1.3 if word in self.unigrams else .5
            if candidate != word and score(candidate) - score(word) >= margin:
                replacement = candidate.capitalize() if match.group()[0].isupper() else candidate
                replacements.append((match.start(), match.end(), replacement))
        for start, end, replacement in reversed(replacements):
            text = text[:start] + replacement + text[end:]
        return text
