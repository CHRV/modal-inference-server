"""The server class must cloudpickle by value.

`build_server` creates the class at runtime and ships it with
`serialized=True`, which is only safe if cloudpickle embeds the class instead
of storing an import path — the SGLang image does not have this library
installed. This test unpickles in a subprocess where importing
`modal_inference_server` raises, which is what the container looks like.
"""

import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest

# Modal vendors cloudpickle and uses it for `serialized=True`; testing with
# the same copy is what makes this test faithful to the real deploy path.
from modal._vendor import cloudpickle

from modal_inference_server import SGLangConfig, build_app, make_server_class

UNPICKLE_IN_A_CONTAINER_LIKE_PROCESS = textwrap.dedent(
    """
    import sys

    class Blocked:
        def find_module(self, name, path=None):
            return self if name.split(".")[0] == "modal_inference_server" else None

        def load_module(self, name):
            raise ImportError(f"{name} is not installed in the image")

    sys.meta_path.insert(0, Blocked())
    assert "modal_inference_server" not in sys.modules

    from modal._vendor import cloudpickle

    with open(sys.argv[1], "rb") as handle:
        cls = cloudpickle.load(handle)

    assert cls.__name__ == "SGLangServer", cls
    # `@modal.enter` leaves a descriptor that raises outside a container,
    # so check the class dict rather than getattr.
    assert {"start", "stop"} <= set(vars(cls)), sorted(vars(cls))
    print("OK")
    """
)


class TestServerClassSerialization(unittest.TestCase):
    def test_unpickles_without_this_library_installed(self):
        config = SGLangConfig.for_model("Qwen/Qwen2.5-0.5B-Instruct", gpu="T4")
        blob = cloudpickle.dumps(make_server_class(config))

        # A by-reference pickle would just name the module; a by-value pickle
        # carries the code, including the argv we baked in.
        self.assertIn(b"sglang.launch_server", blob)

        with tempfile.NamedTemporaryFile(suffix=".pkl") as handle:
            handle.write(blob)
            handle.flush()
            result = subprocess.run(
                [sys.executable, "-c", UNPICKLE_IN_A_CONTAINER_LIKE_PROCESS, handle.name],
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)

    def test_snapshot_variant_unpickles_too(self):
        config = SGLangConfig.for_model("m", gpu_snapshot=True)
        blob = cloudpickle.dumps(make_server_class(config))
        self.assertIn(b"release_memory_occupation", blob)
        with tempfile.NamedTemporaryFile(suffix=".pkl") as handle:
            handle.write(blob)
            handle.flush()
            result = subprocess.run(
                [sys.executable, "-c", UNPICKLE_IN_A_CONTAINER_LIKE_PROCESS, handle.name],
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_modal_accepts_both_server_variants(self):
        """`snap=True` without `enable_memory_snapshot` is an InvalidError at
        decoration time; build both variants to catch it offline."""
        for snapshot in (False, True):
            with self.subTest(gpu_snapshot=snapshot):
                build_app(SGLangConfig.for_model("m", gpu_snapshot=snapshot))

    def test_config_changes_reach_the_serialized_class(self):
        blob = cloudpickle.dumps(
            make_server_class(SGLangConfig.for_model("m", sglang={"context-length": 1234}))
        )
        self.assertIn(b"1234", blob)


if __name__ == "__main__":
    unittest.main()
