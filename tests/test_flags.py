"""The flag catalog: lookup, validation, (de)serialisation."""

import tempfile
import unittest
from pathlib import Path

from modal_inference_server.flags import Catalog, catalog_for_image, sglang_version

CATALOG = catalog_for_image("lmsysorg/sglang:v0.5.19-cu129")

# The shape `_EXTRACT_SCRIPT` prints, trimmed to a few representative actions.
EXTRACTION = {
    "version": "9.9.9",
    "flags": [
        {"options": ["--tp-size", "--tensor-parallel-size"], "dest": "tp_size",
         "action": "_StoreAction", "nargs": None, "type": "int", "choices": None,
         "default": 1, "help": "tp", "group": "options"},
        {"options": ["--enable-metrics"], "dest": "enable_metrics",
         "action": "_StoreTrueAction", "nargs": 0, "type": None, "choices": None,
         "default": False, "help": "metrics", "group": "options"},
        {"options": ["--strict", "--no-strict"], "dest": "strict",
         "action": "BooleanOptionalAction", "nargs": 0, "type": None, "choices": None,
         "default": True, "help": "strict", "group": "options"},
        {"options": ["--old"], "dest": "old", "action": "DeprecatedAliasStoreAction",
         "nargs": None, "type": "str", "choices": ["a", "b"], "default": None,
         "help": "use --new", "group": "options"},
    ],
}


class TestBundledCatalog(unittest.TestCase):
    def test_default_image_has_a_catalog(self):
        self.assertIsNotNone(CATALOG)
        self.assertGreater(len(CATALOG.flags), 300)

    def test_version_from_image_tag(self):
        self.assertEqual(sglang_version("lmsysorg/sglang:v0.5.19-cu129"), "0.5.19")
        self.assertEqual(sglang_version("lmsysorg/sglang:latest"), None)
        self.assertEqual(sglang_version("registry:5000/sglang"), None)

    def test_unknown_version_has_no_catalog(self):
        self.assertIsNone(catalog_for_image("lmsysorg/sglang:v0.0.1"))

    def test_suggestions(self):
        self.assertIn("tp-size", CATALOG.suggest("tp-sise"))

    def test_ambiguous_prefix_is_not_resolved(self):
        self.assertIsNone(CATALOG.resolve("speculative"))

    def test_search(self):
        names = [f.name for f in CATALOG.search("reasoning parser")]
        self.assertIn("reasoning-parser", names)


class TestExtraction(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog.from_extraction(EXTRACTION, image="x:v9.9.9")

    def test_kinds(self):
        self.assertEqual(self.catalog.flags["tp-size"].kind, "value")
        self.assertEqual(self.catalog.flags["enable-metrics"].kind, "flag")
        self.assertEqual(self.catalog.flags["strict"].kind, "bool")

    def test_negative_spelling_is_not_an_alias(self):
        self.assertEqual(self.catalog.flags["strict"].aliases, ())

    def test_check(self):
        errors, warnings = self.catalog.check(
            {"tp-size": "two", "enable-metrics": "yes", "old": "c", "nope": 1})
        self.assertEqual(len(errors), 4)
        self.assertTrue(any("deprecated" in w for w in warnings))

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.catalog.save(Path(directory))
            self.assertEqual(path.name, "sglang-9.9.9.yaml")
            import yaml

            loaded = Catalog.from_dict(yaml.safe_load(path.read_text()))
        self.assertEqual(loaded.flags, self.catalog.flags)
        self.assertEqual(loaded.resolve("tensor-parallel-size"), "tp-size")


if __name__ == "__main__":
    unittest.main()
