"""Fixed registered cellbody-section presets, not arbitrary remote URLs.

The provider URLs below were read from the official siibra configurations.
The new preset planes were checked against each provider's transform.json on
2026-09-29. Presets are small geometry examples, not anatomical selections.
"""

from __future__ import annotations

import copy

from .core import ScoutError

SOURCE_ID = "section-3905"
_SOURCES = {
    "section-3905": {
        "id": "section-3905", "section_number": 3905,
        "label": "Registered cellbody section #3905",
        "source_url": "https://1um.brainatlas.eu/registered_sections/bigbrain/B20_3905/precomputed",
        "configuration_url": "https://github.com/FZJ-INM1-BDA/siibra-configurations/blob/master/features/images/sections/cellbody/73c1fa55-d099-4854-8cda-c9a403c6080a_bigbrain_1um_3905.json",
        "default_roi_mm": [[7.6, 8, 7.6], [8, 10, 8]],
        "default_scale": "4um",
    },
    "section-3954": {
        "id": "section-3954", "section_number": 3954,
        "label": "Registered cellbody section #3954",
        "source_url": "https://1um.brainatlas.eu/registered_sections/bigbrain/B20_3954/precomputed",
        "configuration_url": "https://github.com/FZJ-INM1-BDA/siibra-configurations/blob/master/features/images/sections/cellbody/73c1fa55-d099-4854-8cda-c9a403c6080a_bigbrain_1um_3954.json",
        "default_roi_mm": [[7.6, 9.04, 7.6], [8, 9.08, 8]],
        "default_scale": "4um",
    },
    "section-4003": {
        "id": "section-4003", "section_number": 4003,
        "label": "Registered cellbody section #4003",
        "source_url": "https://1um.brainatlas.eu/registered_sections/bigbrain/B20_4003/precomputed",
        "configuration_url": "https://github.com/FZJ-INM1-BDA/siibra-configurations/blob/master/features/images/sections/cellbody/73c1fa55-d099-4854-8cda-c9a403c6080a_bigbrain_1um_4003.json",
        "default_roi_mm": [[7.6, 10.02, 7.6], [8, 10.06, 8]],
        "default_scale": "4um",
    },
}


def source_ids():
    """Return the fixed available IDs in display order, without network access."""
    return tuple(_SOURCES)


def catalog(source_id=None):
    """Return fresh public descriptors for all sources or one whitelisted ID."""
    if source_id is None:
        return copy.deepcopy(list(_SOURCES.values()))
    if not isinstance(source_id, str) or source_id not in _SOURCES:
        raise ScoutError("Live source must be a whitelisted registered section")
    return copy.deepcopy(_SOURCES[source_id])


def artifact_names(source_id=SOURCE_ID):
    """Return source-specific filenames, never paths supplied by a capsule."""
    if not isinstance(source_id, str):
        raise ScoutError("Live source must be a whitelisted registered section")
    source = catalog(source_id)
    return {"data": f"{source['id']}.nii", "preview": "preview.pgm",
            "metadata": "metadata.json", "reproduce": "README.md"}
