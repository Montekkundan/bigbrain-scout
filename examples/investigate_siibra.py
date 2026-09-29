"""Metadata-only evidence report for historical siibra #704–706 discussions.

This intentionally does not reproduce #704 by downloading a whole section.
It records the exact adapter version and scale keys for a future scoped test.
"""

import json

from fetch_tiny_roi import prepare


if __name__ == "__main__":
    _, _, metadata = prepare()
    print(json.dumps({
        "environment": metadata["environment"],
        "source": metadata["source_url"],
        "available_scales": metadata["available_scales"],
        "selected_scale": metadata["selected_scale_key"],
        "historical_reports": [
            {"issue": 704, "status": "Reported ROI problem; maintainer could not reproduce on newer versions. Not independently reproduced here."},
            {"issue": 705, "status": "Reported scale-key mismatch; maintainer states fixed. Inspect current keys rather than assume 1um naming."},
            {"issue": 706, "status": "Import-time configuration matters; missing warning reportedly remedied by a21. This script sets the limit before import."},
        ],
        "image_fetch_performed": False,
        "upstream": "https://github.com/FZJ-INM1-BDA/siibra-python/issues",
    }, indent=2))
