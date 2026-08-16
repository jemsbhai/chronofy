# Chronofy IRI 2026 manuscript

`chronofy-iri-2026.tex` is the canonical maintained source. Its default build is the double-blind submission: author names, affiliations, email addresses, and the public package URL are omitted. Define `\ChronofyCameraReady` before loading the source only for a camera-ready build.

Build the anonymous manuscript from this directory with:

```powershell
New-Item -ItemType Directory -Force build | Out-Null
1..3 | ForEach-Object {
    pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build chronofy-iri-2026.tex
}
```

For a camera-ready build, define the toggle before loading the same source:

```powershell
New-Item -ItemType Directory -Force build | Out-Null
1..3 | ForEach-Object {
    pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -jobname=chronofy-iri-2026-camera-ready -output-directory=build '\def\ChronofyCameraReady{1}\input{chronofy-iri-2026.tex}'
}
```

Generated files belong in `build/` and are not versioned.

## Source recovery policy

- Use Git history for source recovery; do not create manually synchronized `*-backup.tex` copies.
- Before replacing or deleting a draft, compare both its SHA-256 digest and its textual diff. Preserve any unique draft as a clearly named archival source or a documented patch before removal.
- The pre-repair source and its `-backup.tex` copy were byte-identical: both were 87,756 bytes with SHA-256 `7a014763742e01b17c4c3b240438e56ac70a5b176856c41488b5c287bbe6932c`. The backup therefore held no independent recovery data.
- The official arXiv v1 source for [arXiv:2607.20560](https://arxiv.org/abs/2607.20560) supplied the exact Experiment 3/4 transition. The downloaded source archive had SHA-256 `604e8c19faa6077c30487824ad04406df56273b64962636d91605340b23c27b3`; its TeX source had SHA-256 `85bd08a6ab9f7fe75f03ab92a678ed67455e4057976b4b622b79d3cc74bf7359`.

The corrupt file contained three appended Experiment 4--bibliography tails. No deleted tail contained a unique experiment or numerical result. Two older tails did contain superseded discussion prose, preserved verbatim here so it is not silently lost:

```tex
STL verification adds computational overhead per query. In practice, the Layer~3 computation reduces to a \texttt{min()} over $k$ precomputed validity scores (where $k$ is the retrieval set size), adding $<$1ms per query on our hardware. The dominant cost is fact-level validity annotation, which is computed during Layer~2 retrieval and reused.

Chronofy addresses a fundamental IRI challenge: determining \emph{when} previously captured information remains valid for reuse. The re-acquisition trigger directly addresses the IRI principle that systems should recognize insufficient information. The architecture is domain-agnostic: pluggable decay functions and learnable $\beta$ parameters allow adaptation to financial analytics, legal compliance, and enterprise knowledge management. The framework is released as an open-source Python package.\footnote{Package name reserved; release upon publication.}
```

The maintained source keeps the newer total-pipeline latency wording, stationary-Gaussian caveat, retrieval-applicability discussion, and live package link in camera-ready mode; these match the first legitimate document and the official arXiv revision direction.
