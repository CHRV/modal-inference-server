"""argv <-> {flag: value}, with and without a flag catalog."""

import unittest

from modal_inference_server import catalog_for_image, parse_argv, to_argv
from modal_inference_server.sglang_args import normalize

CATALOG = catalog_for_image("lmsysorg/sglang:v0.5.19-cu129")


class TestParseWithCatalog(unittest.TestCase):
    def parse(self, *tokens):
        return parse_argv(list(tokens), CATALOG)

    def test_values_are_typed_by_the_catalog(self):
        self.assertEqual(
            self.parse("--model-path", "Qwen/Qwen3-8B", "--tp", "2", "--mem-fraction-static", "0.8"),
            {"model-path": "Qwen/Qwen3-8B", "tp-size": 2, "mem-fraction-static": 0.8},
        )

    def test_equals_form(self):
        self.assertEqual(self.parse("--tp=2"), {"tp-size": 2})

    def test_command_words_are_skipped(self):
        for prefix in (["serve"], ["sglang", "serve"], ["python", "-m", "sglang.launch_server"]):
            with self.subTest(prefix=prefix):
                self.assertEqual(self.parse(*prefix, "--model", "m"), {"model-path": "m"})

    def test_aliases_and_prefixes_resolve(self):
        self.assertEqual(self.parse("--tensor-parallel-size", "2"), {"tp-size": 2})
        self.assertEqual(self.parse("--speculative-algo", "NEXTN"),
                         {"speculative-algorithm": "NEXTN"})

    def test_switches(self):
        self.assertEqual(self.parse("--enable-metrics", "--trust-remote-code"),
                         {"enable-metrics": True, "trust-remote-code": True})

    def test_switch_given_a_value_is_rejected(self):
        with self.assertRaises(ValueError):
            self.parse("--enable-metrics", "yes")

    def test_no_prefix_switches_off(self):
        self.assertEqual(self.parse("--no-enable-metrics"), {"enable-metrics": False})

    def test_no_prefix_on_a_value_flag_drops_it(self):
        self.assertEqual(self.parse("--no-reasoning-parser"), {"reasoning-parser": False})

    def test_nargs_collects_several_values(self):
        self.assertEqual(self.parse("--lora-paths", "a", "b=c"), {"lora-paths": ["a", "b=c"]})

    def test_single_value_flag_rejects_several(self):
        with self.assertRaises(ValueError):
            self.parse("--tp", "2", "4")

    def test_missing_value_is_rejected(self):
        with self.assertRaises(ValueError):
            self.parse("--model-path")

    def test_later_spelling_wins(self):
        self.assertEqual(self.parse("--tp", "1", "--tp-size", "2"), {"tp-size": 2})

    def test_unknown_flags_survive_for_the_checker(self):
        self.assertEqual(self.parse("--not-a-flag", "3"), {"not-a-flag": 3})


class TestParseWithoutCatalog(unittest.TestCase):
    def test_tokens_are_read_literally(self):
        self.assertEqual(
            parse_argv(["--a", "1", "--b", "--c", "x", "y", "--d=0.5"]),
            {"a": 1, "b": True, "c": ["x", "y"], "d": 0.5},
        )

    def test_no_prefix(self):
        self.assertEqual(parse_argv(["--no-b"]), {"b": False})

    def test_stray_positional_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_argv(["model"])


class TestToArgv(unittest.TestCase):
    def test_round_trip(self):
        args = {"model-path": "m", "tp-size": 2, "enable-metrics": True,
                "lora-paths": ["a", "b"], "dtype": None, "trust-remote-code": False}
        argv = to_argv(args, CATALOG)
        self.assertEqual(argv, ["--model-path", "m", "--tp-size", "2", "--enable-metrics",
                                "--lora-paths", "a", "b"])
        expected = {k: v for k, v in args.items() if v not in (None, False)}
        self.assertEqual(parse_argv(argv, CATALOG), expected)

    def test_boolean_optional_false_is_spelled_out(self):
        """For `--x/--no-x` flags, false is a real choice, not an omission."""
        self.assertEqual(to_argv({"lora-strict-loading": False}, CATALOG),
                         ["--no-lora-strict-loading"])


class TestNormalize(unittest.TestCase):
    def test_canonical_names(self):
        self.assertEqual(normalize({"tp": 2, "model": "m", "ep": 2}, CATALOG),
                         {"tp-size": 2, "model-path": "m", "ep-size": 2})

    def test_underscores_are_accepted(self):
        self.assertEqual(normalize({"context_length": 8}, CATALOG), {"context-length": 8})


if __name__ == "__main__":
    unittest.main()
