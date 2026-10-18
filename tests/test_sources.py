"""Standard-library-only fixed source catalog contract."""

import unittest

from bigbrain_scout import live, sources
from bigbrain_scout.core import ScoutError


class SourceCatalogTests(unittest.TestCase):
    def test_available_presets_are_fixed_registered_sections(self):
        self.assertEqual(sources.source_ids(), ("section-3905", "section-3954", "section-4003"))
        self.assertEqual(live.source_ids(), sources.source_ids())
        for record in sources.catalog():
            with self.subTest(source_id=record["id"]):
                self.assertEqual(record, sources.catalog(record["id"]))
                self.assertIn(f"B20_{record['section_number']}/precomputed", record["source_url"])
                self.assertIn(f"bigbrain_1um_{record['section_number']}.json", record["configuration_url"])
                self.assertEqual(live.defaults(record["id"])["roi_mm"], record["default_roi_mm"])
                self.assertEqual(live.defaults(record["id"])["scale"], "4um")

    def test_public_descriptors_and_defaults_are_independent(self):
        original = sources.catalog()
        modified = sources.catalog()
        modified[0]["source_url"] = "https://example.org/"
        modified[1]["default_roi_mm"][0][0] = 999
        self.assertEqual(sources.catalog(), original)
        for source_id in sources.source_ids():
            request = live.defaults(source_id)
            request["roi_mm"][0][0] = 999
            self.assertEqual(live.defaults(source_id)["roi_mm"][0][0], 7.6)

    def test_unknown_sources_are_rejected_by_local_helpers(self):
        for source_id in ("section-9999", "https://example.org/", [], {}, True):
            for helper in (sources.catalog, sources.artifact_names, live.defaults):
                with self.subTest(source_id=source_id, helper=helper.__name__):
                    with self.assertRaises(ScoutError):
                        helper(source_id)
        for helper in (sources.artifact_names, live.defaults):
            with self.assertRaises(ScoutError):
                helper(None)

    def test_legacy_constants_and_filenames_remain_compatible(self):
        record = sources.catalog("section-3905")
        self.assertEqual(live.SOURCE_ID, record["id"])
        self.assertEqual(live.SOURCE_URL, record["source_url"])
        self.assertEqual(live.CONFIG_URL, record["configuration_url"])
        self.assertEqual(live.ARTIFACT_NAMES, sources.artifact_names("section-3905"))
        for source_id in sources.source_ids():
            self.assertEqual(sources.artifact_names(source_id), {
                "data": f"{source_id}.nii", "preview": "preview.pgm",
                "metadata": "metadata.json", "reproduce": "README.md"})


if __name__ == "__main__":
    unittest.main()
