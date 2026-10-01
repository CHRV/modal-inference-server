"""Config/argv tests. Offline — run with `uv run python -m unittest discover -s tests`."""

import unittest

from modal_inference_server import ConfigError, SGLangConfig, UnsupportedGPUError

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def model(**fields) -> SGLangConfig:
    return SGLangConfig.for_model(fields.pop("model", "m"), **fields)


def flags(config: SGLangConfig) -> dict[str, str | bool]:
    """Parse `launch_command()` back into a {flag: value} mapping."""
    argv = config.launch_command()
    assert argv[:3] == ["python", "-m", "sglang.launch_server"]
    parsed: dict[str, str | bool] = {}
    rest = argv[3:]
    for index, token in enumerate(rest):
        if not token.startswith("--"):
            continue
        following = rest[index + 1] if index + 1 < len(rest) else None
        parsed[token[2:]] = (
            True if following is None or following.startswith("--") else following
        )
    return parsed


class TestLaunchCommand(unittest.TestCase):
    def test_minimal_config_serves_the_model(self):
        parsed = flags(model(model=MODEL))
        self.assertEqual(parsed["model-path"], MODEL)
        self.assertEqual(parsed["host"], "0.0.0.0")
        self.assertEqual(parsed["port"], "8000")

    def test_aliases_are_normalised(self):
        """`tp` and `model` are what people type; SGLang's first spelling is canonical."""
        config = SGLangConfig(sglang={"model": "m", "tp": 2}, gpu="L4:2")
        self.assertEqual(config.args, {"model-path": "m", "tp-size": 2})
        self.assertEqual(config.model, "m")

    def test_host_and_port_are_managed(self):
        parsed = flags(model(port=9000, sglang={"port": 1234, "host": "127.0.0.1"}))
        self.assertEqual(parsed["port"], "9000")
        self.assertEqual(parsed["host"], "0.0.0.0")

    def test_t4_gets_turing_safe_backends(self):
        parsed = flags(model(gpu="T4"))
        self.assertEqual(parsed["dtype"], "float16")
        self.assertEqual(parsed["attention-backend"], "triton")
        self.assertEqual(parsed["sampling-backend"], "pytorch")
        self.assertIs(parsed["disable-cuda-graph"], True)

    def test_ampere_and_later_keep_sglang_defaults(self):
        parsed = flags(model(gpu="H100"))
        for flag in ("dtype", "attention-backend", "sampling-backend", "disable-cuda-graph"):
            self.assertNotIn(flag, parsed)

    def test_explicit_value_overrides_the_gpu_default(self):
        self.assertEqual(flags(model(gpu="T4", sglang={"dtype": "bfloat16"}))["dtype"], "bfloat16")

    def test_explicit_false_beats_the_gpu_default(self):
        self.assertNotIn("disable-cuda-graph",
                         flags(model(gpu="T4", sglang={"disable-cuda-graph": False})))

    def test_false_and_none_are_dropped(self):
        parsed = flags(model(sglang={"enable-metrics": False, "dtype": None}))
        self.assertNotIn("enable-metrics", parsed)
        self.assertNotIn("dtype", parsed)

    def test_list_values_follow_nargs(self):
        argv = model(sglang={"lora-paths": ["a", "b"]}).launch_command()
        index = argv.index("--lora-paths")
        self.assertEqual(argv[index:index + 3], ["--lora-paths", "a", "b"])
        self.assertEqual(argv.count("--lora-paths"), 1)

    def test_tensor_parallelism_defaults_to_the_gpu_count(self):
        self.assertNotIn("tp-size", flags(model()))
        self.assertEqual(flags(model(gpu="H100:2"))["tp-size"], "2")

    def test_explicit_tp_wins_over_the_gpu_count(self):
        self.assertEqual(flags(model(gpu="H100:4", sglang={"tp": 2, "dp": 2}))["tp-size"], "2")

    def test_unknown_image_passes_flags_through(self):
        config = model(image="my/sglang:custom", sglang={"brand-new-flag": 3, "tp": 2})
        self.assertIsNone(config.catalog)
        parsed = flags(config)
        self.assertEqual(parsed["brand-new-flag"], "3")
        self.assertEqual(parsed["tp"], "2")


class TestGPUValidation(unittest.TestCase):
    """sgl-kernel 0.5.x ships no sm75 binaries, so a T4 dies at the first
    RMSNorm with "no kernel image is available". Catch it locally instead."""

    def test_turing_is_rejected(self):
        with self.assertRaises(UnsupportedGPUError):
            model(gpu="T4").validate()

    def test_error_names_the_gpu_and_a_way_out(self):
        with self.assertRaises(UnsupportedGPUError) as caught:
            model(gpu="T4").validate()
        message = str(caught.exception)
        self.assertIn("T4", message)
        self.assertIn("allow_unsupported_gpu", message)

    def test_escape_hatch_allows_pinning_an_old_image(self):
        model(gpu="T4", allow_unsupported_gpu=True, image="lmsysorg/sglang:v0.4.6").validate()

    def test_supported_gpus_pass(self):
        for gpu in ("L4", "A10G", "A100-80GB", "H100", "B200", "H100:1"):
            with self.subTest(gpu=gpu):
                model(gpu=gpu).validate()

    def test_unknown_gpu_is_not_blocked(self):
        model(gpu="SOME-FUTURE-GPU").validate()

    def test_default_gpu_is_usable(self):
        model().validate()


class TestCheck(unittest.TestCase):
    def test_model_is_required(self):
        with self.assertRaises(ConfigError):
            SGLangConfig().validate()

    def test_unknown_flag_is_an_error_with_a_suggestion(self):
        errors, _ = model(sglang={"tp-sise": 2}).check()
        self.assertEqual(len(errors), 1)
        self.assertIn("--tp-size", errors[0])

    def test_flag_errors_become_warnings_when_asked(self):
        errors, warnings = model(sglang={"tp-sise": 2}, validate_flags=False).check()
        self.assertEqual(errors, [])
        self.assertTrue(any("tp-sise" in w for w in warnings))

    def test_choices_are_checked(self):
        errors, _ = model(sglang={"kv-cache-dtype": "fp7"}).check()
        self.assertIn("kv-cache-dtype", errors[0])

    def test_gpu_count_mismatch_is_a_warning(self):
        _, warnings = model(gpu="H100:2", sglang={"tp": 4}).check()
        self.assertTrue(any("4 GPU(s)" in w for w in warnings))

    def test_dp_attention_reuses_the_tp_gpus(self):
        config = model(gpu="H200:8", sglang={"tp": 8, "dp": 8, "enable-dp-attention": True})
        _, warnings = config.check()
        self.assertFalse(any("GPU(s)" in w for w in warnings))

    def test_bad_gpu_spec(self):
        errors, _ = model(gpu="H100:zero").check()
        self.assertIn("bad GPU spec", errors[0])


class TestDerivedValues(unittest.TestCase):
    def test_app_name_is_slugified_from_the_model(self):
        self.assertEqual(model(model=MODEL).app_name, "sglang-qwen-qwen2-5-0-5b-instruct")

    def test_explicit_app_name_wins(self):
        self.assertEqual(model(name="prod").app_name, "prod")

    def test_served_model_name(self):
        self.assertEqual(model().model_name, "m")
        self.assertEqual(model(sglang={"served-model-name": "x"}).model_name, "x")

    def test_gpu_count(self):
        self.assertEqual(model().n_gpus, 1)
        self.assertEqual(model(gpu="H100:2").n_gpus, 2)
        self.assertEqual(model(gpu="H100:2").gpu_type, "H100")

    def test_gpu_defaults_ignore_count_and_strict_suffix(self):
        self.assertEqual(flags(model(gpu="T4!:2"))["dtype"], "float16")

    def test_with_derives_a_modified_copy(self):
        base = model(gpu="T4", sglang={"dtype": "float16", "tp": 1})
        derived = base.with_(gpu="A10G", min_containers=1,
                             sglang={"dtype": None, "context-length": 4096})
        self.assertEqual(base.gpu, "T4")
        self.assertEqual(base.sglang["dtype"], "float16")
        self.assertEqual(derived.gpu, "A10G")
        self.assertEqual(derived.min_containers, 1)
        self.assertEqual(derived.sglang, {"model-path": "m", "tp": 1, "context-length": 4096})


class TestGPUSnapshot(unittest.TestCase):
    def test_snapshot_enables_sglang_memory_release(self):
        parsed = flags(model(gpu_snapshot=True))
        self.assertIs(parsed["enable-memory-saver"], True)
        self.assertIs(parsed["enable-weights-cpu-backup"], True)
        self.assertNotIn("enable-draft-weights-cpu-backup", parsed)

    def test_draft_weights_are_backed_up_with_speculative_decoding(self):
        parsed = flags(model(gpu_snapshot=True, sglang={"speculative-draft-model-path": "d"}))
        self.assertIs(parsed["enable-draft-weights-cpu-backup"], True)

    def test_explicit_flag_wins(self):
        parsed = flags(model(gpu_snapshot=True, sglang={"enable-weights-cpu-backup": False}))
        self.assertNotIn("enable-weights-cpu-backup", parsed)

    def test_no_snapshot_flags_by_default(self):
        self.assertNotIn("enable-memory-saver", flags(model()))

    def test_single_gpu_only(self):
        for fields in ({"gpu": "H100:2"}, {"gpu": "H100", "sglang": {"tp": 2}}):
            with self.subTest(**fields):
                with self.assertRaises(ConfigError) as caught:
                    model(gpu_snapshot=True, **fields).validate()
                self.assertIn("single GPU", str(caught.exception))
        model(gpu_snapshot=True, gpu="H100:1").validate()


class TestDictRoundTrip(unittest.TestCase):
    def test_to_dict_from_dict(self):
        config = model(gpu="H100:2", secrets=("hf",), env={"A": "1"}, sglang={"tp": 2})
        self.assertEqual(SGLangConfig.from_dict(config.to_dict()), config)

    def test_defaults_are_omitted_except_gpu_and_image(self):
        self.assertEqual(set(model().to_dict()["modal"]), {"gpu", "image"})

    def test_unknown_modal_key_is_rejected_with_a_hint(self):
        with self.assertRaises(ConfigError) as caught:
            SGLangConfig.from_dict({"modal": {"gpus": "L4"}, "sglang": {}})
        self.assertIn("'gpu'", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
