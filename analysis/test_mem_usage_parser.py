import unittest

from analysis.analyze_new_insights import _parse_mem_usage


class MemUsageParsingTest(unittest.TestCase):
    def test_podman_decimal_suffixes_preserve_all_digits(self):
        cases = {
            "195.7MB / 134.5GB": 195.7,
            "193MB": 193.0,
            "226MB": 226.0,
            "1.5GB": 1536.0,
            "2.041GB": 2.041 * 1024,
            "4GB": 4096.0,
            "512.3kB": 512.3 / 1024,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertAlmostEqual(_parse_mem_usage(value), expected)

    def test_binary_suffixes_and_invalid_input(self):
        for value, expected in [("1KiB", 1 / 1024), ("2MiB", 2),
                                ("3GiB", 3072), ("12", 12),
                                ("", 0), ("n/a", 0), ("10XB", 0)]:
            with self.subTest(value=value):
                self.assertAlmostEqual(_parse_mem_usage(value), expected)


if __name__ == "__main__":
    unittest.main()
