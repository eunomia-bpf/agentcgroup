import unittest
from datetime import datetime, timedelta, timezone

from analysis.tool_timing import (
    attribute_union_seconds,
    merge_intervals,
    union_duration_seconds,
)


class ToolTimingTest(unittest.TestCase):
    def setUp(self):
        self.origin = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def interval(self, start, end):
        return (
            self.origin + timedelta(seconds=start),
            self.origin + timedelta(seconds=end),
        )

    def test_union_does_not_double_count_nested_or_overlapping_calls(self):
        intervals = [
            self.interval(0, 10),
            self.interval(2, 4),
            self.interval(8, 12),
            self.interval(20, 25),
        ]
        self.assertEqual(union_duration_seconds(intervals), 17.0)

    def test_merge_ignores_invalid_intervals(self):
        valid = self.interval(1, 3)
        invalid = self.interval(5, 4)
        self.assertEqual(merge_intervals([invalid, valid]), [valid])

    def test_category_attribution_sums_to_wall_clock_union(self):
        labeled = [
            ("Bash", *self.interval(0, 10)),
            ("Bash", *self.interval(2, 8)),
            ("Read", *self.interval(5, 15)),
        ]
        attributed = attribute_union_seconds(labeled)
        self.assertEqual(attributed, {"Bash": 7.5, "Read": 7.5})
        self.assertEqual(sum(attributed.values()), 15.0)


if __name__ == "__main__":
    unittest.main()
