"""CLI tests. Modal options are generated from SGLangConfig, so these guard the
generation rules rather than a hand-written flag list."""

import dataclasses
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import yaml

from modal_inference_server import SGLangConfig
from modal_inference_server.cli import build_parser, compose_from_args, main, split_argv

MODEL = ["--", "serve", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]


def parse(argv: list[str]) -> SGLangConfig:
    ours, sglang = split_argv(["show", *argv])
    return compose_from_args(build_parser().parse_args(ours), sglang).config


def run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class TestCoverage(unittest.TestCase):
    def test_every_modal_field_is_reachable(self):
        """The whole point of generating options: no field can be orphaned."""
        parser = build_parser()
        show = parser._subparsers._group_actions[0].choices["show"]  # noqa: SLF001
        destinations = {action.dest for action in show._actions}  # noqa: SLF001
        for field in dataclasses.fields(SGLangConfig):
            if field.name == "sglang":
                continue  # that is everything after `--`
            with self.subTest(field=field.name):
                self.assertIn(field.name, destinations)


class TestSplit(unittest.TestCase):
    def test_everything_after_the_first_double_dash_is_sglang(self):
        self.assertEqual(split_argv(["run", "--gpu", "L4", "--", "serve", "--", "x"]),
                         (["run", "--gpu", "L4"], ["serve", "--", "x"]))

    def test_sglang_flags_reach_the_config(self):
        config = parse(["--gpu", "L4:2", *MODEL, "--tp", "2", "--enable-metrics"])
        self.assertEqual(config.sglang, {"model-path": "Qwen/Qwen2.5-0.5B-Instruct",
                                         "tp-size": 2, "enable-metrics": True})
        self.assertEqual(config.gpu, "L4:2")

    def test_omitted_options_leave_dataclass_defaults_alone(self):
        self.assertEqual(parse(MODEL), SGLangConfig.for_model("Qwen/Qwen2.5-0.5B-Instruct"))

    def test_non_config_commands_reject_sglang_flags(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            main(["presets", "--", "--tp", "2"])


class TestModalOptions(unittest.TestCase):
    def test_scalars_are_typed(self):
        config = parse(["--gpu", "H100", "--min-containers", "2", *MODEL])
        self.assertEqual(config.gpu, "H100")
        self.assertEqual(config.min_containers, 2)

    def test_booleans_take_both_forms(self):
        self.assertFalse(parse(["--no-unauthenticated", *MODEL]).unauthenticated)
        self.assertTrue(parse(["--allow-unsupported-gpu", *MODEL]).allow_unsupported_gpu)

    def test_repeated_key_value_builds_a_dict_of_strings(self):
        config = parse(["--env", "A=1", "--env", "B=two", *MODEL])
        self.assertEqual(config.env, {"A": "1", "B": "two"})

    def test_repeated_plain_values_build_a_tuple(self):
        self.assertEqual(parse(["--secret", "hf", "--secret", "wandb", *MODEL]).secrets,
                         ("hf", "wandb"))

    def test_malformed_key_value_is_rejected(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            parse(["--env", "no-equals-sign"])


class TestFromAndOverrides(unittest.TestCase):
    def test_cli_wins_over_from(self):
        config = parse(["--from", "cookbook:Qwen3.6-35B-A3B@H100/speculative-mtp",
                        "--gpu", "H200", "--", "--context-length", "32768",
                        "--no-tool-call-parser"])
        self.assertEqual(config.gpu, "H200:1")
        self.assertEqual(config.sglang["context-length"], 32768)
        self.assertIs(config.sglang["tool-call-parser"], False)
        self.assertNotIn("--tool-call-parser", config.launch_command())
        self.assertEqual(config.sglang["speculative-algorithm"], "NEXTN")

    def test_unset(self):
        config = parse(["--from", "cookbook:Qwen3.6-35B-A3B@H100/speculative-mtp",
                        "--unset", "speculative-algorithm"])
        self.assertNotIn("speculative-algorithm", config.sglang)

    def test_show_yaml_reloads_identically(self):
        argv = ["--from", "cookbook:Qwen3.6-35B-A3B", "--gpu", "H200", "--env", "A=1",
                "--", "--context-length", "8192"]
        code, out, _ = run(["show", "--yaml", *argv])
        self.assertEqual(code, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.yaml"
            path.write_text(out)
            self.assertEqual(parse(["--from", str(path)]), parse(argv))

    def test_show_json(self):
        code, out, _ = run(["show", "--json", *MODEL])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["sglang"]["model-path"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(yaml.safe_load(run(["show", "--yaml", *MODEL])[1]), json.loads(out))


class TestShowCommand(unittest.TestCase):
    def test_show_reports_the_launch_command_and_provenance(self):
        code, out, _ = run(["show", *MODEL])
        self.assertEqual(code, 0)
        self.assertIn("sglang.launch_server", out)
        self.assertIn("<- cli", out)

    def test_show_suggests_a_fix_for_a_typo(self):
        code, _, err = run(["show", *MODEL, "--tp-sise", "2"])
        self.assertEqual(code, 1)
        self.assertIn("did you mean --tp-size", err)

    def test_show_fails_without_a_model(self):
        code, _, err = run(["show"])
        self.assertEqual(code, 1)
        self.assertIn("no model", err)

    def test_show_fails_on_a_gpu_without_kernels(self):
        code, _, _ = run(["show", "--gpu", "T4", *MODEL])
        self.assertEqual(code, 1)

    def test_show_accepts_t4_with_the_escape_hatch(self):
        code, out, _ = run(["show", "--gpu", "T4", "--allow-unsupported-gpu",
                            "--image", "lmsysorg/sglang:v0.4.6", *MODEL])
        self.assertEqual(code, 0)
        self.assertIn("--dtype float16", out)

    def test_bad_source_is_a_clean_error(self):
        code, _, err = run(["show", "--from", "cookbook:NoSuchModel"])
        self.assertEqual(code, 2)
        self.assertIn("no cookbook preset", err)


class TestCatalogCommands(unittest.TestCase):
    def test_flags_search(self):
        code, out, _ = run(["flags", "tensor", "parallelism"])
        self.assertEqual(code, 0)
        self.assertIn("--tp-size", out)

    def test_flags_json(self):
        code, out, _ = run(["flags", "--json", "kv", "cache", "dtype"])
        self.assertIn("choices", json.loads(out)["kv-cache-dtype"])

    def test_presets(self):
        code, out, _ = run(["presets", "Qwen3.6-35B-A3B"])
        self.assertEqual(code, 0)
        self.assertIn("cookbook:Qwen3.6-35B-A3B@H100/default", out)

    def test_schema_documents_sglang_flags(self):
        code, out, _ = run(["schema"])
        schema = json.loads(out)
        self.assertIn("tp-size", schema["properties"]["sglang"]["properties"])


if __name__ == "__main__":
    unittest.main()
