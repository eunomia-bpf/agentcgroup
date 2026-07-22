"""Shared helpers for measuring concrete tool-execution intervals."""

from datetime import datetime
from collections import defaultdict
from typing import Dict, Hashable, Iterable, List, Tuple


Interval = Tuple[datetime, datetime]


def merge_intervals(intervals: Iterable[Interval]) -> List[Interval]:
    """Merge overlapping valid intervals in chronological order."""
    valid = sorted(
        (start, end)
        for start, end in intervals
        if start is not None and end is not None and end >= start
    )
    if not valid:
        return []

    merged = [valid[0]]
    for start, end in valid[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def union_duration_seconds(intervals: Iterable[Interval]) -> float:
    """Return elapsed seconds covered by the union of intervals."""
    return sum((end - start).total_seconds() for start, end in merge_intervals(intervals))


def attribute_union_seconds(
    labeled_intervals: Iterable[Tuple[Hashable, datetime, datetime]],
) -> Dict[Hashable, float]:
    """Split union time among categories active in each elapsed-time segment.

    Intervals within one category are first merged. If different categories
    overlap, the segment is divided evenly among the active categories. The
    returned category totals therefore sum exactly to the overall interval
    union instead of counting parallel tool calls more than once.
    """
    by_label = defaultdict(list)
    for label, start, end in labeled_intervals:
        if start is not None and end is not None and end >= start:
            by_label[label].append((start, end))

    events = defaultdict(list)
    for label, intervals in by_label.items():
        for start, end in merge_intervals(intervals):
            events[start].append((label, 1))
            events[end].append((label, -1))

    attributed = defaultdict(float)
    active = defaultdict(int)
    previous = None
    for timestamp in sorted(events):
        if previous is not None and timestamp > previous:
            active_labels = [label for label, count in active.items() if count > 0]
            if active_labels:
                share = (timestamp - previous).total_seconds() / len(active_labels)
                for label in active_labels:
                    attributed[label] += share
        for label, delta in events[timestamp]:
            active[label] += delta
        previous = timestamp

    return dict(attributed)
