"""Find likely duplicate VSEC targets and keep their groups in a single split."""
from collections import defaultdict

from rapidfuzz.distance import Levenshtein


def near_pairs(rows, normalize):
    index = defaultdict(list)
    pairs = set()
    for row in rows:
        text = normalize(row["target_text"])
        words = text.split()
        if len(words) < 8:
            continue
        for edge in {tuple(words[:5]), tuple(words[-5:])}:
            for old_id, previous in index[edge]:
                if previous != text and Levenshtein.normalized_similarity(text, previous) >= .9:
                    pairs.add(tuple(sorted((old_id, row["id"]))))
            index[edge].append((row["id"], text))
    return sorted(pairs)


def group_near_splits(allocation, exact_groups, normalize):
    rows = [row for split in allocation.values() for row in split]
    membership = {row["id"]: name for name, split in allocation.items() for row in split}
    pairs = near_pairs(rows, normalize)
    owners = {row["id"]: index for index, group in enumerate(exact_groups) for row in group}
    parent = list(range(len(exact_groups)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in pairs:
        parent[root(owners[left])] = root(owners[right])

    merged = defaultdict(list)
    for index, group in enumerate(exact_groups):
        merged[root(index)].extend(group)
    original_sizes = {name: len(split) for name, split in allocation.items()}
    priority = tuple(allocation)
    final = {}
    for group in merged.values():
        split = min((membership[row["id"]] for row in group), key=priority.index)
        final.update((row["id"], split) for row in group)

    # Restore split sizes with independent singleton sentences where possible.
    sizes = {name: sum(split == name for split in final.values()) for name in priority}
    singles = [group[0]["id"] for group in reversed(list(merged.values())) if len(group) == 1]
    for destination in priority:
        for identifier in singles:
            if sizes[destination] >= original_sizes[destination]:
                break
            source = final[identifier]
            if source == destination or sizes[source] <= original_sizes[source]:
                continue
            final[identifier] = destination
            sizes[source] -= 1
            sizes[destination] += 1

    grouped = {name: [] for name in priority}
    for row in rows:
        grouped[final[row["id"]]].append(row)
    moved_pairs = [pair for pair in pairs if membership[pair[0]] != membership[pair[1]]]
    return grouped, moved_pairs
