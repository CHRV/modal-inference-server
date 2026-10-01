"""Config sources (--from / extends) and the order layers stack in."""

import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from modal_inference_server import ConfigError, SGLangConfig
from modal_inference_server.sources import Layer, compose, load_source


def write(directory: str, name: str, text: str) -> str:
    path = Path(directory) / name
    path.write_text(textwrap.dedent(text))
    return str(path)


class TestMerging(unittest.TestCase):
    def test_later_layers_win_and_provenance_follows(self):
        composed = compose([
            Layer("a", modal={"gpu": "L4", "env": {"A": "1"}}, sglang={"model": "m", "tp": 1}),
            Layer("b", modal={"env": {"B": "2"}}, sglang={"tp-size": 2}),
        ])
        config = composed.config
        self.assertEqual(config.sglang, {"model-path": "m", "tp-size": 2})
        self.assertEqual(config.env, {"A": "1", "B": "2"})  # dicts merge key by key
        self.assertEqual(composed.provenance["sglang.tp-size"], "b")
        self.assertEqual(composed.provenance["sglang.model-path"], "a")

    def test_null_deletes_an_inherited_flag(self):
        config = compose([Layer("a", sglang={"model": "m", "dtype": "half"}),
                          Layer("b", sglang={"dtype": None})]).config
        self.assertNotIn("dtype", config.sglang)

    def test_gpu_without_count_keeps_the_earlier_count(self):
        config = compose([Layer("a", modal={"gpu": "H100:2"}), Layer("b", modal={"gpu": "H200"})])
        self.assertEqual(config.config.gpu, "H200:2")
        config = compose([Layer("a", modal={"gpu": "H100:2"}), Layer("b", modal={"gpu": "H200:1"})])
        self.assertEqual(config.config.gpu, "H200:1")

    def test_argv_layer_is_parsed_with_the_final_image_catalog(self):
        config = compose([Layer("cli", argv=["serve", "--model", "m", "--tp", "2"])]).config
        self.assertEqual(config.sglang, {"model-path": "m", "tp-size": 2})

    def test_unset(self):
        config = compose([Layer("a", modal={"min_containers": 2},
                                sglang={"model": "m", "speculative-algorithm": "EAGLE"})],
                         unset=["speculative-algo", "min-containers"]).config
        self.assertEqual(config.sglang, {"model-path": "m"})
        self.assertEqual(config.min_containers, 0)

    def test_unset_of_an_unknown_modal_field_fails(self):
        with self.assertRaises(ConfigError):
            compose([Layer("a")], unset=["modal.nope"])


class TestFiles(unittest.TestCase):
    def test_extends_is_relative_to_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            os.mkdir(Path(directory) / "base")
            write(directory, "base/common.yaml", """
                modal: {gpu: "H100:2", secrets: [hf]}
                sglang: {model-path: m, tp: 2, context-length: 4096}
            """)
            child = write(directory, "child.yaml", """
                extends: base/common.yaml
                sglang: {context-length: 8192}
            """)
            config = SGLangConfig.load(child)
        self.assertEqual(config.gpu, "H100:2")
        self.assertEqual(config.secrets, ("hf",))
        self.assertEqual(config.sglang["context-length"], 8192)

    def test_circular_extends_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            a = write(directory, "a.yaml", "extends: b.yaml\n")
            write(directory, "b.yaml", "extends: a.yaml\n")
            with self.assertRaises(ConfigError) as caught:
                SGLangConfig.load(a)
        self.assertIn("circular", str(caught.exception))

    def test_plain_sglang_config_file_is_accepted(self):
        """SGLang's own `--config` YAML has no sections: it is all flags."""
        with tempfile.TemporaryDirectory() as directory:
            path = write(directory, "sglang.yaml", "model-path: m\ntensor-parallel-size: 2\n")
            config = SGLangConfig.load(path, gpu="L4:2")
        self.assertEqual(config.sglang, {"model-path": "m", "tp-size": 2})

    def test_json_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write(directory, "c.json", '{"sglang": {"model-path": "m"}}')
            self.assertEqual(SGLangConfig.load(path).model, "m")

    def test_unknown_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write(directory, "c.yaml", "modal: {}\nsgl: {}\n")
            with self.assertRaises(ConfigError):
                SGLangConfig.load(path)

    def test_missing_file(self):
        with self.assertRaises(ConfigError):
            SGLangConfig.load("/nonexistent/config.yaml")

    def test_yaml_round_trip(self):
        config = SGLangConfig.for_model("m", gpu="H100:2", env={"A": "1"},
                                        sglang={"tp": 2, "lora-paths": ["a", "b"]})
        with tempfile.TemporaryDirectory() as directory:
            path = write(directory, "c.yaml", config.to_yaml())
            self.assertEqual(SGLangConfig.load(path).to_dict(), config.with_(
                sglang={"tp": None, "tp-size": 2}).to_dict())


class TestCookbookAndHub(unittest.TestCase):
    def test_cookbook_picks_the_cheapest_gpu_and_default_variant(self):
        data, label, _ = load_source("cookbook:Qwen3.6-35B-A3B")
        self.assertEqual(label, "cookbook:Qwen3.6-35B-A3B@H100/default")
        self.assertEqual(data["modal"]["gpu"], "H100:1")

    def test_cookbook_explicit_gpu_and_variant(self):
        data, label, _ = load_source("cookbook:qwen3.6-35b-a3b@H200/speculative-mtp")
        self.assertEqual(label, "cookbook:Qwen3.6-35B-A3B@H200/speculative-mtp")
        self.assertEqual(data["sglang"]["speculative-algorithm"], "NEXTN")

    def test_cookbook_follows_the_requested_gpu(self):
        config = SGLangConfig.load("cookbook:Qwen3.6-35B-A3B", gpu="B200")
        self.assertEqual(config.gpu, "B200:1")

    def test_cookbook_errors_list_the_choices(self):
        with self.assertRaises(ConfigError) as caught:
            load_source("cookbook:Qwen3.6-35B-A3B@H100/nope")
        self.assertIn("speculative-mtp", str(caught.exception))
        with self.assertRaises(ConfigError):
            load_source("cookbook:NoSuchModel")

    def test_hf_uses_the_cookbook_when_it_knows_the_model(self):
        _, label, _ = load_source("hf:Qwen/Qwen3.6-35B-A3B")
        self.assertTrue(label.startswith("cookbook:"))

    def test_hf_falls_back_to_the_bare_model(self):
        data, _, _ = load_source("hf:someone/unknown-model")
        self.assertEqual(data, {"sglang": {"model-path": "someone/unknown-model"}})

    def test_hf_file_and_github_urls(self):
        with mock.patch("modal_inference_server.sources._fetch",
                        return_value="sglang: {model-path: m}") as fetch:
            load_source("hf:org/repo/configs/a.yaml")
            self.assertEqual(fetch.call_args[0][0],
                             "https://huggingface.co/org/repo/resolve/main/configs/a.yaml")
            load_source("https://github.com/o/r/blob/main/c.yaml")
            self.assertEqual(fetch.call_args[0][0],
                             "https://raw.githubusercontent.com/o/r/main/c.yaml")

    def test_every_bundled_preset_is_valid(self):
        from modal_inference_server.presets import all_presets

        for preset in all_presets():
            for key, spec in preset.configs.items():
                with self.subTest(preset=preset.model, config=key):
                    config = SGLangConfig.from_dict({"modal": spec["modal"],
                                                     "sglang": spec["sglang"]})
                    errors, _ = config.check()
                    self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
