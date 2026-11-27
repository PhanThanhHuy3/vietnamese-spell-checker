"""Decode accent candidates jointly with a Vietnamese trigram language model."""
from collections import Counter, defaultdict
from functools import lru_cache
from heapq import nlargest
import math
import unicodedata

from src.postprocessing import without_accents

START = "<s>"
END = "</s>"
BEAM_WIDTH = 32
CONTEXT_WINDOW = 7
AUTO_MARGIN = .75


def term(text):
    return unicodedata.normalize("NFC", text).casefold()


class AccentRestorer:
    def __init__(self, sequences):
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.skip_counts = Counter()
        self.vocabulary = defaultdict(list)
        for terms in sequences:
            if not terms:
                continue
            sequence = [START, START, *terms, END]
            self.unigrams.update(terms)
            self.bigrams.update(zip(sequence[1:], sequence[2:]))
            self.trigrams.update(zip(sequence, sequence[1:], sequence[2:]))
            for i, word in enumerate(terms):
                for earlier in set(terms[max(0, i - CONTEXT_WINDOW - 1):i - 1]) if i > 1 else ():
                    self.skip_counts[earlier, word] += 1
        self.total = sum(self.unigrams.values())
        for word, _ in self.unigrams.most_common():
            if word.isalpha():
                self.vocabulary[without_accents(word)].append(word)

    @lru_cache(maxsize=8192)
    def association(self, left, word):
        count = self.skip_counts[left, word]
        if count < 3:
            return 0
        expected = self.unigrams[left] * self.unigrams[word] * 6
        return .13 * max(0, math.log((count + 1) * self.total / expected))

    @lru_cache(maxsize=32768)
    def transition(self, earlier, left, word):
        bigram = self.bigrams[left, word]
        trigram = self.trigrams[earlier, left, word]
        if word == END:
            return .20 * math.log1p(bigram) + .18 * math.log1p(trigram)
        strength = .7 * math.log1p(bigram) + .5 * math.log1p(trigram)
        if bigram and left != START:
            expected = self.unigrams[left] * self.unigrams[word] * 4
            strength += .25 * max(0, math.log((bigram + 1) * self.total / expected))
            if bigram >= 5:
                strength += .35 * math.log1p(8 * bigram / self.unigrams[left])
        return strength

    def candidate_words(self, word):
        if not word.isascii() or not word.isalpha():
            return [word]
        return [word, *(candidate for candidate in self.vocabulary.get(word, ()) if candidate != word)][:24]

    def step(self, path, candidate, original):
        score = self.transition(path[-2], path[-1], candidate)
        if candidate != END:
            score += .18 * math.log1p(self.unigrams[candidate]) - .15 * (candidate != original)
            score += sum(self.association(earlier, candidate) for earlier in set(path[-CONTEXT_WINDOW - 1:-1]))
        return score

    def rank(self, words, protected, include_alternatives=True):
        choices = [self.candidate_words(word) if not blocked else [word]
                   for word, blocked in zip(words, protected, strict=True)]
        # Multiple eligible positions indicate sentence-level or partial accent loss.
        eligible = sum(len(candidates) > 1 for candidates in choices)
        visible_words = sum(word.isalpha() and not blocked for word, blocked in zip(words, protected))
        if eligible < 2 or eligible < .30 * visible_words:
            return {}
        missing = sum(not self.unigrams[word] for word, candidates in zip(words, choices) if len(candidates) > 1)
        accented = any(not word.isascii() and not blocked for word, blocked in zip(words, protected))
        if not accented and not missing and len(words) < 3:
            return {}

        beam = [(0.0, (START, START))]
        for original, candidates in zip([*words, END], [*choices, [END]], strict=True):
            beam = nlargest(BEAM_WIDTH, (
                (score + self.step(path, candidate, original), (*path, candidate))
                for score, path in beam for candidate in candidates
            ))
        best = list(beam[0][1][2:-1])
        if not missing and not accented and sum(a != b for a, b in zip(words, best)) < 2:
            return {}

        results = {}
        for index, (word, replacement, candidates) in enumerate(zip(words, best, choices, strict=True)):
            if replacement == word or len(candidates) == 1:
                continue
            # Keep the surrounding decoded context fixed when comparing per-position alternatives.
            path = [START, START, *best, END]
            at = index + 2

            def local_score(candidate):
                path[at] = candidate
                end = min(len(path), at + CONTEXT_WINDOW + 2)
                return sum(self.step(path[max(0, i - CONTEXT_WINDOW - 1):i], path[i],
                                     words[i - 2] if i - 2 < len(words) else END)
                           for i in range(at, end))

            winner_score = local_score(replacement)
            original_score = local_score(word)
            minimum_gain = 1.1 if accented and self.unigrams[word] else 0.0
            if winner_score - original_score <= minimum_gain:
                continue
            if include_alternatives:
                others = sorted(((local_score(c), c) for c in candidates if c not in (word, replacement)),
                                reverse=True)
                alternative_score = max(original_score, others[0][0]) if others else original_score
                results[index] = (replacement, [candidate for _, candidate in others],
                                  winner_score - alternative_score >= AUTO_MARGIN)
            else:
                results[index] = (replacement, [], True)
        return results
