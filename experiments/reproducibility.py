"""Pinned identifiers shared by the tracked experiment entry points.

These identifiers select immutable upstream revisions. They do not make remote
model calls deterministic; see ``experiments/README.md`` for the remaining
limitations.
"""

from __future__ import annotations

RANDOM_SEED = 42

EVOEXPLORE_REVISION = "2b573b35b464870ab6f7f3105ee0b1704e0f4b2a"
ICEWS14_FILE_SHA256 = {
    "stat.txt": "679358d774d38fb15d17897673f553d7ba9b86cb68b1846bd9c22c8cdfeed928",
    "test.txt": "9e543f6ab6bac9cd1dc1e7429366d3da63cd1a5f967922c73d137062f0481667",
    "train.txt": "0a56c8a660a0b11ad950c0ff065fb334e8f1d9c7f4f4b537221e6fd2fd617af9",
    "valid.txt": "510ba87a63dcfa81461746765154da405935556b09d382a4167efa08dc07ea69",
}
TIME_LITE_REVISION = "cd93b48ffeb8491f6d46359a8e2bf79495cb0ad3"
GDELT_REVISION = "8ee57955b595b4edddf39b5bb93ea3309bc3233f"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MIMIC_CHECKSUM_MANIFEST_SHA256 = "19dfd55fac45035ba8e9f4c7e59180ee5184e877504f098e517a434d2646366a"

# This is a provider model identifier, not an immutable model revision.
GEMINI_MODEL_ID = "gemini-2.5-flash"
