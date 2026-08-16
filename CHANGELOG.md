# Changelog

Chronofy's Python and npm distributions are versioned independently.
**Unreleased** describes repository source, not a registry artifact. This
repository had no Git tags when this history was audited, so a source commit
cannot be inferred from a registry version unless a future release records that
mapping explicitly.

## Unreleased

No unreleased changes are currently recorded.

## [Python 0.1.9] — 2026-08-15

Python `0.1.9` is the first release built from a namespaced repository tag and
verified against the committed lock file and isolated wheel metadata tests.

- Ground temporal rule-graph ranking/results in caller-supplied triples,
  preserving timestamp, source quality, fact type, and provenance; avoid
  synthetic results when grounding is absent and reject invalid future or
  duplicate evidence.
- Make Subjective Logic fusion group by an explicit
  `metadata["proposition_key"]` or exact content by default instead of
  conflating changing values that share a decay category; validate custom
  grouping keys.
- Validate built-in decay parameters/time units and extreme numeric cases;
  correct power-scoring endpoint semantics; strengthen scalar/batch plugin
  consistency checks.
- Enforce temporal-embedding rank/dimension contracts and preserve the
  distinction between a missing optional dependency and an internal module
  defect.
- Correct jsonld-ex Byzantine-removal index handling and tighten Subjective
  Logic/ML typing and optional-import boundaries.
- Declare Poetry extras as optional runtime dependencies (including NetworkX
  for `sl`), commit the lock file, and verify wheel metadata/version consistency.
- Correct executable README examples and API descriptions for observation
  versus publication timestamps, beta-zero/source-quality behavior, pipeline
  construction, grounded graph requirements, Subjective Logic exports/grouping,
  and the paper citation. Detailed usage remains in the [main README](README.md)
  rather than being duplicated here.

## [JavaScript 0.1.8] — 2026-08-15

JavaScript `0.1.8` contains the repository fixes that postdate the published
`0.1.7` artifact. The npm package remains independently versioned from Python.

- Mirror built-in decay validation, extreme-value handling, power-scoring
  endpoint behavior, and scalar/batch plugin checks.
- Stop tracking the dependency tree; add an explicit package-file allowlist and
  npm-local MIT license; refresh the lock/dev-test toolchain; verify the dry-run
  tarball manifest.
- Document the npm package's Python-0.1.6-core scope and independent versioning;
  correct runnable examples, provenance/invariance language, and citation.

See the packaged [npm README](npm/README.md) for its supported API surface.

## Repository and research artifacts — 2026-08-15

These tracked sources accompany the package release commit. The paper and
experiment records are not independently tagged or published as research
artifacts by this release.

- Add independent least-privilege CI gates for Python 3.10–3.12, Node 22/24,
  package metadata/audit/manifests, and the manuscript.
- Restore one canonical IRI 2026 TeX source, preserve source-recovery
  provenance, make the anonymous build the default, and check deterministic PDF
  output and anonymity. See the [paper guide](papers/iri-2026/README.md).
- Track experiment drivers, input/command manifests, pinned revisions,
  dependency declarations, and a deterministic synthetic smoke fixture while
  keeping datasets, credentials, caches, and generated results ignored. See the
  [experiment guide](experiments/README.md).
- Replace the ignored, stale RuView draft with an evidence-labelled public
  [integration roadmap](improvements/RUVIEW_INTEGRATION_PLAN.md); all eight
  phases remain proposed—not started.
- Add the independent package/research [release policy](RELEASING.md).

## Published registry index

This index records only versions present in official registry metadata, checked
2026-08-15. Dates are registry publication dates in UTC, not reconstructed Git
release dates. No tag existed to prove an exact commit-to-artifact mapping.

| Distribution | Version | Registry publication date | Registry record |
|---|---:|---:|---|
| Python / PyPI | 0.1.0 | 2026-03-16 | [PyPI 0.1.0](https://pypi.org/project/chronofy/0.1.0/) |
| Python / PyPI | 0.1.1 | 2026-03-16 | [PyPI 0.1.1](https://pypi.org/project/chronofy/0.1.1/) |
| Python / PyPI | 0.1.6 | 2026-03-18 | [PyPI 0.1.6](https://pypi.org/project/chronofy/0.1.6/) |
| Python / PyPI | 0.1.8 | 2026-03-19 | [PyPI 0.1.8](https://pypi.org/project/chronofy/0.1.8/) |
| JavaScript / npm | 0.1.0 | 2026-03-16 | [npm 0.1.0](https://www.npmjs.com/package/chronofy/v/0.1.0) |
| JavaScript / npm | 0.1.1 | 2026-03-16 | [npm 0.1.1](https://www.npmjs.com/package/chronofy/v/0.1.1) |
| JavaScript / npm | 0.1.6 | 2026-03-18 | [npm 0.1.6](https://www.npmjs.com/package/chronofy/v/0.1.6) |
| JavaScript / npm | 0.1.7 | 2026-03-18 | [npm 0.1.7](https://www.npmjs.com/package/chronofy/v/0.1.7) |

The registry source data are the official
[PyPI JSON metadata](https://pypi.org/pypi/chronofy/json) and
[npm registry metadata](https://registry.npmjs.org/chronofy). Python repository
milestones `0.1.2`–`0.1.5` and `0.1.7` are intentionally not presented as
published releases because they do not appear in the PyPI release index.

[Python 0.1.9]: https://github.com/jemsbhai/chronofy/releases/tag/python-v0.1.9
[JavaScript 0.1.8]: https://github.com/jemsbhai/chronofy/releases/tag/npm-v0.1.8
