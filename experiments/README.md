# Chronofy experiments

This directory keeps experiment **source and reproducibility metadata** under
version control while keeping acquired data, credentials, caches, and generated
outputs local. The numbered drivers are research programs, not part of the
supported `chronofy` package API.

The only CI-safe experiment run is the deterministic, synthetic validation:

```bash
poetry install --no-interaction
poetry run python -B experiments/validate.py
```

It makes no network calls and does not parse or print clinical/third-party
records. It compiles every tracked experiment source, validates the source/input
manifest, checks for literal credentials, hashes any already-present local
dataset bytes, and runs four synthetic temporal facts against explicit scores
and a `1e-12` tolerance.
The fixture contains invented identifiers and no person, patient, or clinical
content.

## Audited migration baseline

Before source was exposed for version control, `experiments/` contained exactly
109 files in 10 directories (28,199,015 bytes), all ignored by one blanket rule:

| Class | Files | Bytes | Treatment |
|---|---:|---:|---|
| Python source | 17 | 216,555 | Tracked after review and cleanup |
| Acquired dataset files/metadata | 39 | 18,514,823 | Still ignored |
| Generated results | 45 | 6,561,223 | Still ignored |
| Python bytecode | 6 | 150,268 | Still ignored |
| Embedding cache | 1 | 2,756,144 | Still ignored |
| Download marker | 1 | 2 | Still ignored |

The exact pre-migration tree digest is
`a139ecadd4ce4a7480b05b496d574486c5906ce2063397256cd8c8fab0502d98`.
It was calculated from sorted UTF-8 lines of
`relative/path<TAB>bytes<TAB>file_sha256`, joined with LF. The 53
result-like/generated files are precisely the 45 results, 6 bytecode files,
embedding cache, and download marker. Nothing in those classes was deleted or
moved during the migration.

## What is tracked

- The eight numbered experiment drivers, shared helpers, and opt-in inspection
  utilities.
- [`manifest.json`](manifest.json), which lists commands, seeds, input IDs,
  expected output contracts, tolerances, and limitations.
- [`datasets.json`](datasets.json), which pins upstream revisions and lawful
  acquisition/checksum information without embedding dataset records.
- [`reproducibility.py`](reproducibility.py), the constants actually consumed by
  the source entry points.
- [`requirements.txt`](requirements.txt), the additional experiment-only
  top-level dependencies.
- [`fixtures/synthetic_temporal_facts.json`](fixtures/synthetic_temporal_facts.json),
  the neutral smoke fixture.
- [`validate.py`](validate.py), the offline static/smoke gate.

## What remains intentionally ignored

`data/` (including the deidentified MIMIC-IV Demo), `results/`, archives,
embedding caches, checkpoints, logs, figures, tables, generated model files,
and `.env*` files remain ignored by granular root rules. The MIMIC demo includes
subject/admission-linked records and is never redistributed by this repository.
Do not put credentials in a command line or file; the remote-model experiments
read `GEMINI_API_KEY` or `GOOGLE_API_KEY` from the process environment.

Generated results stay outside Git by default even when they contain only
aggregates. An intentional research release should package reviewed outputs
separately and record their SHA-256 hashes, the source tag/commit, manifest,
environment, hardware, input revisions/checksums, and exact command. See the
[release policy](../RELEASING.md).

## Environment

The supported library and ML/graph extras are locked by the root
[`poetry.lock`](../poetry.lock). From the repository root:

```bash
poetry install --all-extras --no-interaction
poetry run python -m pip install --requirement experiments/requirements.txt
```

The second file pins the two additional top-level tools used by historical
experiment drivers (`datasets` and `google-generativeai`). It is not a complete
transitive lock. For a released result bundle, record `poetry --version`, Python
and OS/architecture, accelerator/driver details, and `poetry run python -m pip
freeze`; differences in the local accelerator stack can affect embeddings.

Run entry points as modules from the repository root so imports and pinned
configuration resolve consistently:

```bash
poetry run python -m experiments.exp1_tkg_decay --run --quick
```

## Experiment inventory

The manifest is the authoritative command/acceptance contract. This is the
short operational view:

| ID | Purpose | Inputs | Lightweight/manual command | Reproducibility boundary |
|---|---|---|---|---|
| exp1 | ICEWS14 decay-weighted TKG retrieval | Pinned EvoExplore files | `python -m experiments.exp1_tkg_decay --run --quick` | Deterministic with verified bytes and locked stack |
| exp2 | STL signal versus answer correctness | Pinned TIME-Lite + Gemini API | `python -m experiments.exp2_stl_time --run --subset news --limit 10` | Remote provider output is not immutable; schema/range checks only |
| exp3 | ICEWS14 beta sensitivity | Pinned EvoExplore files | `python -m experiments.exp3_beta_sensitivity --quick` | Relation categories are research proxies |
| exp3b | Beta sensitivity on the open MIMIC-IV Demo | Version 2.2 demo | `python -m experiments.exp3b_clinical_beta --quick` | Research only; no clinical validation or full MIMIC data |
| exp3c | Clinical-demo decay ablations | Version 2.2 demo | `python -m experiments.exp3c_clinical_ablations --quick` | Same clinical boundary; compute-intensive |
| exp4 | GDELT/TFLEX relation sensitivity | Pinned Hugging Face revision | `python -m experiments.exp4_gdelt_decay --run --limit 100` | Large download and manual compute |
| exp5 | TIME-Lite temporal RAG retrieval/LLM run | Pinned dataset/model + Gemini API | `python -m experiments.exp5_rag_endtoend --inspect 10` | GPU variance and mutable remote inference; cache is ignored |
| exp6 | Layer 1/2 retrieval ablation | Pinned dataset + exp5 cache | `python -m experiments.exp6_ablation` | Requires a separately checksummed ignored embedding cache |

Prefix the table's commands with `poetry run`. Full commands, output names,
numeric bounds, and tolerances are in `manifest.json`. The diagnostic utilities
are opt-in, are never imported by CI, and may download or display third-party
dataset excerpts when explicitly run. They must not be pointed at restricted
data.

## Input acquisition and provenance

### ICEWS14

`exp1 --download` reads four files from an immutable EvoExplore commit and
verifies each against the SHA-256 values in `datasets.json`. Existing files are
also verified before they are accepted. The downloader does not grant
redistribution rights; review the upstream dataset/repository terms for the
intended use.

### MIMIC-IV Demo

Only the openly available, deidentified **MIMIC-IV Clinical Database Demo
v2.2** is in scope, not credentialed MIMIC-IV. Acquire it from the
[official versioned PhysioNet record](https://physionet.org/content/mimic-iv-demo/2.2/)
and comply with the license shown there. `exp3b --download` now rejects unsafe
archive paths and verifies PhysioNet's `SHA256SUMS.txt` plus all 34 listed files
before writing its local completion marker. The data remain ignored even though
the demo is openly accessible.

### TIME-Lite, GDELT, and the embedding model

Hugging Face dataset/model revisions are pinned in `datasets.json` and imported
from `reproducibility.py`; downloads go to the user's Hugging Face cache, not the
repository. Review each upstream dataset card/license before use or
redistribution. A generated embedding `.npz` must be treated as a cache, and a
released result must record its checksum and producing environment.

## Seeds and limitations

The declared default seed is `42`. Existing clinical sampling retains that
seed, and embedding generation seeds NumPy and Torch. These controls do not make
GPU kernels, platform math, dependency resolution outside `poetry.lock`, or
provider-hosted LLMs bit-for-bit deterministic. `gemini-2.5-flash` is a mutable
provider identifier, and temperature zero is not a reproducibility guarantee.
For exp2/exp5, preserve provider response metadata locally and describe the run
as best-effort replay, never deterministic reproduction.

The historical result files are intentionally not treated as golden fixtures:
they lack a complete code/input/environment record. Full dataset, API, and GPU
runs are manual and non-CI. Passing `validate.py` establishes source integrity
and deterministic synthetic behavior only; it does not validate a paper result,
clinical performance, safety, regulatory compliance, or real-world RuView
integration.
