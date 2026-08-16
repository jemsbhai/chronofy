# Release and research-artifact policy

This policy is intentionally lightweight. It prevents the independent Python,
npm, paper, and experiment histories from being collapsed into one ambiguous
version. It does not authorize a publication, tag, commit, push, or release.

## Independent version authorities

| Component | Version authority | Required consistency |
|---|---|---|
| Python distribution | `pyproject.toml` | Must equal `chronofy.__version__` in `chronofy/__init__.py` |
| npm distribution | `npm/package.json` | Must equal top-level `version` and `packages[""].version` in `npm/package-lock.json` |
| IRI paper | Explicit research revision | Not tied automatically to either package version |
| Experiment snapshot | Explicit research revision | Must identify code/input/config/environment, not merely a package version |

Use semantic versioning independently for Python and npm. A release of one does
not require a cosmetic bump of the other. Because npm `0.1.7` already exists in
the registry, the locally stabilized npm source must be bumped before it can be
published; an existing registry version must never be replaced or
misrepresented.

## Changelog rules

- Keep pending work under `Unreleased`, split by Python, npm, and
  repository/research artifacts.
- Move only the relevant bullets into a versioned section when that component
  is actually released.
- Add a release date only when publication occurs; do not infer it from a commit
  timestamp or source version bump.
- Link detailed API, paper, and experiment instructions to their canonical
  tracked documents instead of copying them into the changelog.
- If a registry artifact predates local source changes at the same version,
  state that explicitly and bump before the next publication.

## Tags

Create immutable, annotated, namespaced tags only after the corresponding
artifact has passed its release gates:

- `python-vX.Y.Z`
- `npm-vX.Y.Z`
- `paper-iri-2026-rN`
- `experiments-iri-2026-rN`

Do not use an ambiguous shared `vX.Y.Z`, move/reuse a tag, or tag a different
commit from the one used to build the artifact. Package artifacts should be
built and tested from the tagged commit, and the changelog should link the tag.
There are currently no repository tags; this policy does not create any.

## Package release gates

Run the Python gates from the repository root with the supported Poetry
version and committed lock file:

```bash
poetry check --lock
poetry run ruff check chronofy tests
poetry run ruff check experiments
poetry run mypy --strict chronofy
poetry run pytest
poetry run python -B experiments/validate.py
poetry build --clean --format wheel
```

Verify the built wheel in a clean environment and confirm that its metadata
matches both Python version authorities. Mirror the clean-wheel check in
`.github/workflows/python.yml`.

Run npm gates from `npm/`:

```bash
npm ci
npm test -- --ci
npm audit --audit-level=low
npm pack --dry-run --json --ignore-scripts
```

Review the dry-run file list against `npm/package.json`'s allowlist and confirm
that no dependency tree, coverage, cache, credential, or unrelated root file is
included. The npm version must be new in the registry before publishing.

For either component, also run `git diff --check`, review workflow syntax/pinned
actions if workflow files changed, and resolve every relevant CI job before
tagging.

## Paper revisions

Paper snapshots are research artifacts, not Python or npm releases. Follow the
tracked [IRI build guide](papers/iri-2026/README.md) and the paper workflow for:

- canonical source structure;
- anonymous versus explicitly enabled camera-ready identity;
- repeated deterministic builds using a recorded `SOURCE_DATE_EPOCH`;
- reference/log checks; and
- PDF text/metadata anonymity checks.

A paper tag should identify the exact source commit and toolchain. Do not track
ordinary TeX build products. If a paper revision changes a reported number, it
must cite the experiment snapshot that produced it.

## Experiment revisions

The [experiment manifest](experiments/manifest.json), [input provenance](experiments/datasets.json),
and [experiment guide](experiments/README.md) are the minimum source record.
An intentional experiment release must additionally record:

- exact source commit/tag and package versions;
- command, config, deterministic seed, expected output contract, and numeric
  tolerance;
- Python/Poetry/dependency freeze, OS/architecture, and accelerator/driver;
- lawful dataset/model provenance, immutable revision or file checksum, and
  license/access review;
- checksums for released result, table, figure, embedding, or checkpoint
  artifacts; and
- a clear distinction between deterministic reproduction, tolerance-based
  replay, and non-deterministic remote-provider output.

Do not tag or archive credentials, `.env` files, caches, raw/restricted clinical
or CSI records, or bulky generated directories. Reviewed aggregate artifacts
may be attached separately with a checksum manifest when licensing, privacy,
and provenance allow it; they remain ignored in ordinary development.

Passing the synthetic experiment smoke check is not evidence of a paper result,
clinical performance, regulatory compliance, or real-world sensor validation.

## Stable citations and links

Relative links are appropriate for discoverability in the working tree. A
published package, paper, or experiment should cite immutable tag/commit
permalinks so later documentation changes cannot alter its record. Verify that
all citation/reproducibility links resolve at the tag before publication.

For registry history, use official registry metadata. Absence of a Git tag means
an exact source-to-registry mapping is unknown and must not be invented.

## Final release checklist

1. Select exactly one component/research artifact and confirm authority to
   release it.
2. Set and cross-check only that component's version/revision.
3. Update the matching changelog section without rewriting prior history.
4. Run its complete gates and inspect the built/reviewed artifact.
5. Record provenance, dependencies, inputs, tolerances, checksums, and stable
   links appropriate to that artifact.
6. Commit the reviewed release state, then create the matching annotated tag.
7. Publish only the artifact built from that tag; verify registry/release
   metadata afterward.
