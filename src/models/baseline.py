"""Train-only Vietnamese n-gram and Levenshtein baseline."""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import chain
import math
import unicodedata

from rapidfuzz.distance import DamerauLevenshtein

from src.data.noise_synthesizer import FINALS, INITIALS, TONE_MARKS, to_telex
from src.models.accent_restorer import AccentRestorer, term
from src.postprocessing import (WORD, alignment, apply_edits, error_category, is_protected,
                                protected_spans, sentence_word_groups, without_accents)


VERB_MARKERS = frozenset({"đã", "đang", "sẽ", "vừa", "cần", "để", "phải", "muốn", "nên"})


def tokens(text):
    return [term(m.group()) for m in WORD.finditer(text)]


def tone_key(word):
    return "".join(char for char in unicodedata.normalize("NFD", word) if char not in TONE_MARKS)


def deletion_keys(word):
    return {word, *(word[:i] + word[i + 1:] for i in range(len(word)))}


class NgramBaseline:
    def __init__(self, train_rows, extra_context=()):
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.confusions = defaultdict(Counter)
        for row in train_rows:
            target = tokens(row["target_text"])
            source = tokens(row["input_text"])
            self.unigrams.update(target)
            sequence = ["<s>", *target, "</s>"]
            self.bigrams.update(zip(sequence, sequence[1:]))
            self.trigrams.update(zip(sequence, sequence[1:], sequence[2:]))
            for op in alignment(source, target):
                if op.tag == "replace":
                    for a, b in zip(source[op.src_start:op.src_end], target[op.dest_start:op.dest_end]):
                        self.confusions[a][b] += 1

        self.language_model = AccentRestorer(
            tokens(text) for text in chain((row["target_text"] for row in train_rows), extra_context)
        )
        self.deletions, self.accents, self.variants = defaultdict(set), defaultdict(set), defaultdict(set)
        for word in self.unigrams:
            if not word.isalpha() or len(word) < 2:
                continue
            self.accents[without_accents(word)].add(word)
            self.variants[to_telex(word)].add(word)
            for original, wrong in INITIALS.items():
                if word.startswith(original):
                    self.variants[wrong + word[len(original):]].add(word)
            for original, wrong in FINALS.items():
                if word.endswith(original):
                    self.variants[word[:-len(original)] + wrong].add(word)
            for key in deletion_keys(word):
                self.deletions[key].add(word)

    @lru_cache(maxsize=4096)
    def candidates(self, word):
        known = self.confusions.get(word, {})
        result = {word, *known, *self.variants.get(word, ())}
        result.update(self.accents.get(without_accents(word), ()))
        close = set().union(*(self.deletions.get(key, ()) for key in deletion_keys(word)))
        result.update(w for w in close if DamerauLevenshtein.distance(word, w) == 1)
        return tuple(sorted(result))

    def spelling_candidates(self, original_terms, protected, phrase_breaks):
        terms = original_terms.copy()
        replacements = {}
        context = self.language_model.bigrams
        for observed_only in (False, True):
            context_terms = terms.copy()
            for i, word in enumerate(original_terms):
                if bool(self.unigrams[word]) != observed_only:
                    continue
                if len(word) < 2 or len(word) > 24 or protected[i]:
                    continue
                left = context_terms[i - 1] if i and i not in phrase_breaks else "<s>"
                right = context_terms[i + 1] if i + 1 < len(terms) and i + 1 not in phrase_breaks else "</s>"
                earlier = context_terms[i - 2] if i > 1 and left != "<s>" and i - 1 not in phrase_breaks else "<s>"
                later = context_terms[i + 2] if i + 2 < len(terms) and right != "</s>" and i + 2 not in phrase_breaks else "</s>"
                language = self.language_model

                def context_counts(candidate):
                    left_count = self.bigrams[left, candidate] if left != "<s>" else 0
                    right_count = self.bigrams[candidate, right] if right != "</s>" else 0
                    return left_count, right_count

                actual_left, actual_right = context_counts(word)
                def reference_context(candidate):
                    incoming = context[left, candidate] if left != "<s>" else 0
                    outgoing = context[candidate, right] if right != "</s>" else 0
                    return incoming, outgoing

                reference_left, reference_right = reference_context(word)
                reference_support = reference_left + reference_right

                def surrounding_trigrams(candidate):
                    return (language.trigrams[earlier, left, candidate]
                            + language.trigrams[candidate, right, later])

                original_trigrams = surrounding_trigrams(word)
                known = self.confusions.get(word, {})

                def mistyped_telex(candidate):
                    return (not self.unigrams[word] and word.isascii()
                            and candidate != word and to_telex(candidate) == word
                            and DamerauLevenshtein.distance(candidate, word) > 1)

                def score(candidate):
                    incoming, outgoing = context_counts(candidate)
                    context_score = (math.log1p(incoming) + math.log1p(outgoing)
                                     + .85 * math.log1p(self.trigrams[left, candidate, right]))
                    ref_left, ref_right = reference_context(candidate)
                    if not observed:
                        context_score += .36 * math.log1p(ref_left + ref_right)
                    if left in VERB_MARKERS and (candidate == word or candidate in known):
                        family_count = sum(self.bigrams[marker, candidate] for marker in VERB_MARKERS)
                        context_score += .95 * math.log1p(family_count)
                    prior = .10 * math.log1p(self.unigrams[candidate])
                    evidence = .70 * math.log1p(known.get(candidate, 0))
                    if mistyped_telex(candidate):
                        evidence += .8
                    distance = DamerauLevenshtein.distance(word, candidate)
                    if candidate in self.variants.get(word, ()) and candidate != word:
                        distance = 1
                    penalty = .95 * distance
                    return context_score + prior + evidence - penalty

                observed = self.unigrams[word]

                def supported(candidate):
                    if candidate == word:
                        return True
                    incoming, outgoing = context_counts(candidate)
                    similar = (candidate in self.variants.get(word, ())
                               or without_accents(candidate) == without_accents(word))
                    evidence = known.get(candidate, 0)
                    if candidate in (left, right) and language.trigrams[left, candidate, right] < 3:
                        return False
                    if mistyped_telex(candidate) and (incoming or outgoing):
                        return True
                    if not observed and similar and sum(reference_context(candidate)) >= 8:
                        return True
                    if observed >= 100 and not similar:
                        ref_left, ref_right = reference_context(candidate)
                        if (not (ref_left and ref_right) and not language.trigrams[left, candidate, right]
                                or min(ref_left, ref_right) < min(reference_left, reference_right)
                                or (reference_left > 3 * ref_left and reference_left >= 5)
                                or (reference_right > 3 * ref_right and reference_right >= 5)):
                            return False
                    if evidence:
                        return (evidence >= 2 or observed < 5 or similar
                                or (incoming > 0 and outgoing > 0))
                    if actual_left and actual_right and observed:
                        return self.trigrams[left, candidate, right] >= 3 and similar
                    if incoming and outgoing:
                        return True
                    if not actual_left and not actual_right:
                        if observed and not similar:
                            support = max(incoming / max(self.unigrams[left], 1),
                                          outgoing / max(self.unigrams[right], 1))
                            return max(incoming, outgoing) >= 6 and support >= .10
                        return max(incoming, outgoing) >= (4 if similar else 6)
                    return max(incoming, outgoing) >= 8 and similar and not observed

                candidates = self.candidates(word)
                candidate = max((c for c in candidates if supported(c)), key=score)
                margin = (.45 + .28 * math.log1p(observed)) if observed else .4
                if observed:
                    margin += .35 * math.log1p(max(actual_left, actual_right))
                    if actual_left and actual_right:
                        margin += 1.25
                    if candidate not in known:
                        margin += .3
                    if without_accents(candidate) != without_accents(word) and reference_support >= 10:
                        margin += min(2.5, .55 * math.log1p(reference_support))
                if candidate == word or score(candidate) - score(word) < margin:
                    candidate = self.contextual_candidate(word, left, right, candidates)
                elif (self.trigrams[earlier, left, word] >= 2
                      and not self.trigrams[earlier, left, candidate]
                      and not language.trigrams[left, candidate, right]):
                    candidate = word
                elif (candidate not in self.variants.get(word, ()) and original_trigrams >= 2
                      and not surrounding_trigrams(candidate) and not language.trigrams[left, candidate, right]
                      and reference_left and reference_right == 0):
                    candidate = word
                if candidate != word:
                    ref_left, ref_right = reference_context(candidate)
                    verified = language.trigrams[left, candidate, right] or surrounding_trigrams(candidate)
                    close_tone = tone_key(candidate) == tone_key(word)
                    phonetic_variant = candidate in self.variants.get(word, ())
                    if (observed and len(candidate) > len(word) and not close_tone and not phonetic_variant
                            and not (min(ref_left, ref_right) >= 3 or verified >= 3)):
                        continue
                    if (language.unigrams[word] >= 200 and not ref_left and left != "<s>" and not phonetic_variant
                            and not language.trigrams[earlier, left, candidate]
                            and not language.trigrams[left, candidate, right]):
                        continue
                    if (language.unigrams[word] >= 500 and not close_tone
                            and not known.get(candidate) and not self.trigrams[earlier, left, candidate]
                            and not phonetic_variant):
                        continue
                    terms[i] = candidate
                    replacements[i] = candidate
        return replacements

    def contextual_candidate(self, word, left, right, candidates):
        language = self.language_model
        context = language.bigrams
        original_left = context[left, word] if left != "<s>" else 0
        original_right = context[word, right] if right != "</s>" else 0
        if original_left and original_right:
            return word
        ranked = []
        for candidate in candidates:
            if candidate in (word, left, right):
                continue
            same_family = (without_accents(word) == without_accents(candidate)
                           or candidate in self.variants.get(word, ()))
            incoming = context[left, candidate] if left != "<s>" else 0
            outgoing = context[candidate, right] if right != "</s>" else 0
            local = language.trigrams[left, candidate, right]
            close_form = tone_key(candidate) == tone_key(word) or candidate in self.variants.get(word, ())
            if language.unigrams[word] and not close_form and local < 3:
                continue
            bilateral = incoming >= 2 and outgoing >= 2
            accent_pair = (same_family and incoming >= 1 and outgoing >= 2
                           and (not (original_left or original_right) or local))
            strong_collocation = (same_family and max(incoming, outgoing) >= 7
                                  and not (original_left or original_right)
                                  and (not language.unigrams[word] or tone_key(candidate) == tone_key(word)
                                       or candidate in self.variants.get(word, ())))
            distant = (not language.unigrams[word] and same_family and not incoming and right == "</s>"
                       and language.skip_counts[left, candidate] >= 4)
            distance = DamerauLevenshtein.distance(word, candidate)
            mechanical = (not language.unigrams[word] and distance == 1
                          and (word.startswith(candidate) or sorted(word) == sorted(candidate))
                          and max(incoming, outgoing) >= 3)
            phonetic = (language.unigrams[word] <= 1 and candidate in self.variants.get(word, ())
                        and (incoming or outgoing))
            tone_typo = (not language.unigrams[word] and tone_key(candidate) == tone_key(word)
                         and language.unigrams[candidate] >= 30 and (incoming or outgoing))
            recorded = (candidate in self.confusions.get(word, {}) and candidate in self.variants.get(word, ())
                        and (max(incoming, outgoing) >= 6 or (incoming and outgoing and incoming + outgoing >= 6))
                        and incoming + outgoing >= 2 * max(original_left + original_right, 1))
            if not (bilateral or accent_pair or strong_collocation or distant or mechanical or phonetic or tone_typo or recorded):
                continue
            if ((original_left and incoming < original_left)
                    or (original_right and outgoing < original_right)):
                continue
            evidence = math.log1p(incoming) + math.log1p(outgoing) + .5 * math.log1p(local) - .65 * distance
            if distant:
                evidence += .75 * language.association(left, candidate)
            ranked.append((evidence, candidate))
        if not ranked:
            return word
        return max(ranked)[1]

    def spacing_suggestions(self, text, matches, words, blocked, position):
        """Recover a glued or split syllable when the neighboring phrase supports it."""
        context = self.language_model
        edits = []
        occupied = set()
        for index, (word, match) in enumerate(zip(words, matches, strict=True)):
            before = words[index - 1] if index else "<s>"
            after = words[index + 1] if index + 1 < len(words) else "</s>"
            fragment = (len(word) <= 2 and len(after) <= 2 and not context.unigrams[after]
                        and context.unigrams[word] <= 20)
            if (blocked[index] or index in occupied or not word.isalpha()
                    or (context.unigrams[word] and not fragment)):
                continue
            end = match.end()
            replacement = None
            candidates = []
            for split in range(1, len(word)):
                first, second = word[:split], word[split:]
                internal = context.bigrams[first, second]
                external = max(context.bigrams[before, first], context.bigrams[second, after])
                if internal >= 10 and external >= 5:
                    candidates.append((internal * external, first + " " + second))
            if candidates:
                replacement = max(candidates)[1]
            elif index + 1 < len(words) and not blocked[index + 1]:
                following = matches[index + 1]
                combined = word + after
                following_context = words[index + 2] if index + 2 < len(words) else "</s>"
                gap = text[end:following.start()]
                if (gap and all(char in " \t" for char in gap) and combined.isalpha()
                        and max(context.bigrams[before, combined], context.bigrams[combined, following_context]) >= 8):
                    replacement = combined
                    end = following.end()
                    occupied.add(index + 1)
            if replacement:
                occupied.add(index)
                if match.group()[0].isupper():
                    replacement = replacement.capitalize()
                edits.append(dict(start=match.start(), end=end, position=position + index + 1,
                                  original_segment=text[match.start():end], corrected_segment=replacement,
                                  category="Tách hoặc ghép âm tiết", alternatives=[], auto_apply=True))
        return edits, occupied

    def suggest(self, text, include_alternatives=True):
        spans = protected_spans(text)
        edits = []
        position = 0
        for matches in sentence_word_groups(text):
            words = [term(match.group()) for match in matches]
            blocked = [is_protected(match, spans) for match in matches]
            spacing, occupied = self.spacing_suggestions(text, matches, words, blocked, position)
            edits.extend(spacing)
            blocked = [value or index in occupied for index, value in enumerate(blocked)]
            restored = self.language_model.rank(words, blocked, include_alternatives)
            working = [restored[index][0] if index in restored else word for index, word in enumerate(words)]
            frozen = [protected or i in restored for i, protected in enumerate(blocked)]
            phrase_breaks = {index for index in range(1, len(matches))
                             if "," in text[matches[index - 1].end():matches[index].start()]}
            spelling = self.spelling_candidates(working, frozen, phrase_breaks)
            for index, match in enumerate(matches):
                suggestion = restored.get(index)
                if not suggestion and index not in spelling:
                    continue
                replacement, alternatives, automatic = suggestion or (spelling[index], [], True)
                if match.group()[0].isupper():
                    replacement = replacement.capitalize()
                    alternatives = [candidate.capitalize() for candidate in alternatives]
                edits.append(dict(
                    start=match.start(), end=match.end(), position=position + index + 1,
                    original_segment=match.group(), corrected_segment=replacement,
                    category=error_category(words[index], replacement),
                    alternatives=alternatives, auto_apply=automatic,
                ))
            position += len(matches)
        return sorted(edits, key=lambda edit: edit["start"])

    def correct(self, text):
        return apply_edits(text, self.suggest(text, include_alternatives=False))
