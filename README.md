# Chronofy

**Temporal validity framework for time-aware AI systems.**

Chronofy implements the Temporal-Logical Decay Architecture (TLDA), a three-layer neuro-symbolic framework that embeds temporal validity directly into the representation, retrieval, and reasoning layers of Retrieval-Augmented Generation (RAG) systems.

## The Problem

Current RAG systems treat all retrieved facts as equally valid regardless of when they were recorded. A clinical lab reading from yesterday and one from six months ago get weighted identically if they're semantically similar to the query. This causes **temporal hallucination** — plausible but factually obsolete outputs.

## The Solution

Chronofy provides three layers of temporal protection:

- **Layer 1 — Temporal Embeddings**: Reserves dedicated dimensions in embedding vectors for temporal information, making fact age structurally irremovable from the representation.
- **Layer 2 — Decay-Weighted Retrieval**: Integrates learnable exponential decay functions into graph-based retrieval, where decay rates are grounded in Bayesian decision theory.
- **Layer 3 — STL Verification**: Applies Signal Temporal Logic robustness functions to verify the temporal validity of retrieved knowledge throughout the reasoning chain.

## Installation

```bash
pip install chronofy
```

With optional dependencies:

```bash
pip install chronofy[graph]   # NetworkX for graph-based retrieval
pip install chronofy[ml]      # PyTorch + sentence-transformers for embeddings
pip install chronofy[all]     # Everything
```

## Quick Start

```python
from chronofy import TemporalFact, ExponentialDecay, EpistemicFilter
from datetime import datetime, timedelta

# Define facts with timestamps
facts = [
    TemporalFact(
        content="Patient serum potassium: 4.1 mEq/L",
        timestamp=datetime.now() - timedelta(days=1),
        fact_type="vital_sign",
        source_quality=0.95,
    ),
    TemporalFact(
        content="Patient serum potassium: 3.2 mEq/L",
        timestamp=datetime.now() - timedelta(days=180),
        fact_type="vital_sign",
        source_quality=0.95,
    ),
    TemporalFact(
        content="Patient blood type: O+",
        timestamp=datetime.now() - timedelta(days=3650),
        fact_type="demographic",
        source_quality=1.0,
    ),
]

# Configure decay rates per fact type (beta = 2 * kappa)
decay = ExponentialDecay(
    beta={
        "vital_sign": 5.0,      # High volatility — decays in days
        "medication": 1.0,       # Medium — decays in weeks
        "chronic_condition": 0.01,  # Low — stable over months
        "demographic": 0.0,      # Invariant — never decays
    }
)

# Filter facts by temporal validity
query_time = datetime.now()
ep_filter = EpistemicFilter(decay_fn=decay, threshold=0.1)
valid_facts = ep_filter.filter(facts, query_time)

for fact in valid_facts:
    score = decay.compute(fact, query_time)
    print(f"  [{score:.3f}] {fact.content}")
```

## Decision-Theoretic Grounding

The decay coefficient β is not an arbitrary hyperparameter. Under Gaussian latent dynamics (Ornstein-Uhlenbeck process), the information-theoretically optimal decay rate is:

**β = 2κ**

where κ is the mean-reversion rate of the underlying process. This means:
- High κ (volatile processes like vital signs) → high β → fast decay
- Low κ (stable processes like genetic markers) → low β → slow/no decay

See our paper for the full derivation: *Chronofy: A Temporal-Logical Decay Architecture for Information Validity in Time-Aware RAG* (IEEE IRI 2026).

## Architecture

```
┌─────────────────────────────────────────────┐
│              Chronofy Pipeline               │
├─────────────┬───────────────┬───────────────┤
│  Layer 1    │   Layer 2     │   Layer 3     │
│  Temporal   │   Decay-      │   STL         │
│  Embedding  │   Weighted    │   Robustness  │
│             │   Retrieval   │   Verification│
├─────────────┼───────────────┼───────────────┤
│ TMRL-style  │ exp(-β·Δt)    │ G[0,n](v≥γ)  │
│ subspace    │ graph walk    │ weakest-link  │
│ encoding    │ + epistemic   │ bound +       │
│             │ filter        │ re-acquisition│
└─────────────┴───────────────┴───────────────┘
```

## Citation

```bibtex
@inproceedings{syed2026chronofy,
  title={Chronofy: A Temporal-Logical Decay Architecture for Information Validity in Time-Aware Retrieval-Augmented Generation},
  author={Syed, Muntaser},
  booktitle={Proc. IEEE International Conference on Information Reuse and Integration (IRI)},
  year={2026}
}
```

## License

MIT
