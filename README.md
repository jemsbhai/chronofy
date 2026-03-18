# Chronofy

**Temporal validity framework for time-aware AI systems.**

[![PyPI version](https://img.shields.io/pypi/v/chronofy.svg)](https://pypi.org/project/chronofy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Chronofy implements the **Temporal-Logical Decay Architecture (TLDA)** — a three-layer neuro-symbolic framework that embeds temporal validity directly into the representation, retrieval, and reasoning layers of Retrieval-Augmented Generation (RAG) systems. It also provides a standalone suite of temporal statistical tools useful outside of RAG.

---

## The Problem

Current RAG systems treat all retrieved facts as equally valid regardless of when they were recorded. A clinical lab reading from yesterday and one from six months ago receive identical weight if they are semantically similar to the query. This causes **temporal hallucination** — plausible but factually obsolete outputs.

The same failure mode appears across domains:

| Domain | Stale fact | Consequence |
|---|---|---|
| Clinical | Serum potassium from 6 months ago | Wrong medication dosing |
| Financial | Market volatility from last quarter | Mispriced risk |
| Legal | Superseded regulatory ruling | Non-compliant advice |
| Geopolitical | Alliance structure from 2 years ago | Incorrect threat assessment |

---

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

---

## Architecture

Chronofy operates as three independent but composable layers:

```
┌──────────────────────────────────────────────────────────────┐
│                     Chronofy Pipeline                        │
├───────────────┬──────────────────────┬───────────────────────┤
│   Layer 1     │       Layer 2        │       Layer 3         │
│   Temporal    │   Decay-Weighted     │   STL Knowledge       │
│   Embedding   │     Retrieval        │     Verification      │
├───────────────┼──────────────────────┼───────────────────────┤
│ TMRL subspace │  w = q·c·exp(-β·Δt) │  φ = G[0,n](v ≥ γ)   │
│ preserves age │  epistemic filter    │  weakest-link bound   │
│ at all scales │  τ-threshold pruning │  re-acquisition alert │
└───────────────┴──────────────────────┴───────────────────────┘
```

Each layer is independently usable — you do not need all three.

---

## Quick Start

```python
from chronofy import (
    TemporalFact,
    ExponentialDecay,
    EpistemicFilter,
    STLVerifier,
    ReasoningStep,
    ReasoningTrace,
)
from datetime import datetime, timedelta

now = datetime.now()

# 1. Define timestamped facts
facts = [
    TemporalFact(
        content="Serum potassium: 4.1 mEq/L",
        timestamp=now - timedelta(days=1),
        fact_type="vital_sign",
        source_quality=0.95,
    ),
    TemporalFact(
        content="Serum potassium: 3.2 mEq/L",
        timestamp=now - timedelta(days=180),
        fact_type="vital_sign",
        source_quality=0.95,
    ),
    TemporalFact(
        content="Blood type: O+",
        timestamp=now - timedelta(days=3650),
        fact_type="demographic",
        source_quality=1.0,
    ),
]

# 2. Configure decay — β = 2κ (twice the latent process mean-reversion rate)
decay = ExponentialDecay(
    beta={
        "vital_sign":       5.0,   # volatile — decays in hours/days
        "lab_result":       2.0,   # decays in days/weeks
        "medication":       1.0,   # decays in weeks
        "chronic_condition":0.01,  # stable over months/years
        "demographic":      0.0,   # invariant — never decays
    }
)

# 3. Filter to temporally valid facts only
ep_filter = EpistemicFilter(decay_fn=decay, threshold=0.1)
valid_facts = ep_filter.filter(facts, query_time=now)

for fact in valid_facts:
    score = decay.compute(fact, now)
    print(f"[{score:.4f}] {fact.content}")
# [0.0067] Serum potassium: 4.1 mEq/L   ← 1 day old, high β
# [1.0000] Blood type: O+               ← 10 years old, β=0 (invariant)
# (6-month-old reading filtered out)

# 4. Verify reasoning chain with STL
step = ReasoningStep(step_index=0, content="assess arrhythmia risk", facts_used=valid_facts)
trace = ReasoningTrace(steps=[step], query_time=now)

verifier = STLVerifier(decay_fn=decay, threshold=0.05)
result = verifier.verify(trace)

print(f"STL satisfied:      {result.satisfied}")
print(f"Robustness score:   {result.robustness:.4f}")
print(f"Confidence bound:   {result.output_confidence_bound:.4f}")

if not result.satisfied:
    print(f"Re-acquisition needed for: {result.weakest_fact.content}")
```

---

## Core Concepts

### TemporalFact — the evidence tuple

Every piece of evidence is represented as a tuple `e = (c, t_e, q, m)`:

```python
from chronofy import TemporalFact

fact = TemporalFact(
    content="Patient serum potassium: 4.1 mEq/L",   # c: the claim
    timestamp=datetime(2024, 6, 1, 8, 30),           # t_e: observation time
    fact_type="lab_result",                           # determines decay rate β_j
    source_quality=0.95,                              # q ∈ (0, 1]: reliability weight
    publication_timestamp=datetime(2024, 6, 1, 9),   # optional: reporting time
    source="EPIC/LAB",                               # optional: provenance
    metadata={"units": "mEq/L", "method": "ISE"},   # optional: extra context
)

# Age in days at any reference time
age = fact.age_at(datetime.now())
```

`observation_timestamp` is used for age computation by default. Use `publication_timestamp` when the two differ (e.g. a paper published today reporting data from 3 years ago).

---

### Decay Functions

All decay functions implement the `DecayFunction` protocol and return a validity score `V(e, T_q) ∈ [0, 1]` — incorporating both temporal decay and source reliability.

#### ExponentialDecay (recommended default)

Grounded in Bayesian decision theory: for a latent state following an Ornstein-Uhlenbeck process with mean-reversion rate κ, the information-theoretically optimal decay rate is **β = 2κ** (Proposition 1, Chronofy paper).

```python
from chronofy import ExponentialDecay

decay = ExponentialDecay(
    beta={"vital_sign": 5.0, "demographic": 0.0},
    default_beta=0.5,
    time_unit="days",       # "days" | "hours" | "seconds"
)

# Construct directly from mean-reversion rates (β = 2κ)
decay = ExponentialDecay.from_mean_reversion_rate(
    kappa={"vital_sign": 2.5, "lab_result": 1.0, "demographic": 0.0}
)

# Inspect half-life
print(decay.half_life("vital_sign"))   # ln(2)/5.0 ≈ 0.139 days ≈ 3.3 hours
print(decay.half_life("demographic"))  # None (invariant)
```

#### Other built-in backends

```python
from chronofy import HalfLifeDecay, LinearDecay, PowerLawDecay, WeibullDecay

# HalfLifeDecay — same as exponential, parameterised by half-life in days
decay = HalfLifeDecay(half_life={"vital_sign": 3.0, "lab_result": 14.0})

# LinearDecay — hard expiry at 1/α days; useful for compliance deadlines
decay = LinearDecay(rate={"regulatory": 0.1}, default_rate=0.05)

# PowerLawDecay — heavy-tailed decay; matches citation/web freshness patterns
decay = PowerLawDecay(exponent={"news": 1.5})

# WeibullDecay — generalized exponential from survival analysis
# shape > 1: accelerating decay; shape < 1: decelerating decay
decay = WeibullDecay(scale={"lab_result": 7.0}, shape={"lab_result": 1.5})
```

---

### EpistemicFilter — Layer 2 retrieval gate

Structurally excludes stale evidence from the context window before it reaches the LLM. Facts below the validity threshold `τ` are pruned.

```python
from chronofy import EpistemicFilter, ExponentialDecay

decay = ExponentialDecay(beta={"general": 1.0})
ep_filter = EpistemicFilter(decay_fn=decay, threshold=0.1)

query_time = datetime.now()
valid_facts = ep_filter.filter(facts, query_time)

# Partition into valid/stale
fresh, stale = ep_filter.partition(facts, query_time)

# Check if any fact type needs re-acquisition
needs_refresh = ep_filter.needs_reacquisition(facts, query_time)
```

**Property (Retrieval Completeness):** Under threshold `τ`, the retrieved context excludes all facts `e` satisfying `q_e · c(tr_j) · exp(-β_j · Δt) < τ`, guaranteeing the LLM never receives evidence below the minimum validity floor.

---

### STLVerifier — Layer 3 reasoning guard

Applies Signal Temporal Logic robustness to the **temporal validity of retrieved knowledge** at each reasoning step — not to the LLM's output confidence. This is the headline novelty: the LLM can be *confidently wrong* when it retrieves plausible but stale facts. STL over knowledge validity catches this.

**STL specification:** `φ_valid = G[0,n]( v(s_i) ≥ γ )`

where `v(s_i) = min over facts used at step i of V(e, T_q)`.

**Robustness score:** `ρ = min_i (v(s_i) − γ)`

- `ρ ≥ 0`: reasoning satisfies the temporal validity constraint
- `ρ < 0`: at least one step relies on insufficiently fresh evidence

```python
from chronofy import STLVerifier, STLResult, ReasoningStep, ReasoningTrace

verifier = STLVerifier(decay_fn=decay, threshold=0.3)

step1 = ReasoningStep(step_index=0, content="retrieve labs", facts_used=[lab_fact])
step2 = ReasoningStep(step_index=1, content="assess risk",   facts_used=[lab_fact, dx_fact])
trace = ReasoningTrace(steps=[step1, step2], query_time=datetime.now())

result = verifier.verify(trace)

print(f"ρ = {result.robustness:.4f}")           # scalar robustness
print(f"Satisfied: {result.satisfied}")          # ρ ≥ 0
print(f"Step validity: {result.step_validity}")  # per-step signals
print(f"C_out ≤ {result.output_confidence_bound:.4f}")  # Theorem 1 bound
print(f"Weakest fact: {result.weakest_fact.content}")

# Re-acquisition trigger
if not result.satisfied:
    print("Insufficient temporal context — data re-acquisition required.")
```

**Theorem 1 (Weakest-Link Bound):** For any reasoning chain using facts `{e_1, ..., e_k}`, the maximum reliable output confidence is bounded by `min_i V(e_i, T_q)`. This follows from the min-aggregation principle of possibilistic logic.

---

### CorpusStats — temporal health analysis

Statistical summary of a fact corpus at a given query time. Useful for understanding your data before any RAG run.

```python
from chronofy import CorpusStats, ExponentialDecay

decay = ExponentialDecay(beta={"lab_result": 2.0, "vital_sign": 5.0})
stats = CorpusStats(facts=my_facts, query_time=datetime.now(), decay_fn=decay)

# Age distribution (in days)
print(stats.age_stats.mean)    # mean age
print(stats.age_stats.p75)     # 75th percentile age
print(stats.temporal_span_days)

# Validity distribution
print(stats.validity_stats.min)
print(stats.effective_density)          # mean validity — single-number health score

# Staleness
print(stats.staleness_rate(threshold=0.1))   # fraction of facts below 10% validity

# Silent periods
gaps = stats.coverage_gaps(min_gap_days=7)   # gaps with no facts for ≥ 7 days
for start, end in gaps:
    print(f"  Gap: {start.date()} → {end.date()}")

# Full summary dict (JSON-serialisable)
summary = stats.summary()
```

---

### BetaEstimator — fit β from data

When you do not know the latent process dynamics analytically, fit β empirically from labeled observations `(age_in_days, still_valid: bool)`.

```python
from chronofy import BetaEstimator, MLEBernoulli, MomentMatching, EnsembleMethod

# Default: MLE under Bernoulli model P(valid | age, β) = exp(-β · age)
estimator = BetaEstimator()

result = estimator.fit(
    ages=[1.0, 5.0, 10.0, 30.0, 60.0],
    valid=[True,  True,  True,  False, False],
    fact_type="lab_result",
)
print(f"β̂ = {result.beta:.4f}")
print(f"Half-life = {result.half_life:.1f} days")
print(f"Log-likelihood = {result.log_likelihood:.4f}")

# Fit per fact_type from a labelled corpus
results = estimator.fit_corpus(
    facts=my_facts,
    labels=my_labels,      # parallel list of bools
    query_time=datetime.now(),
)
for fact_type, r in results.items():
    print(f"{fact_type:20s}  β={r.beta:.4f}  t½={r.half_life:.1f}d")

# Convert directly to a ready-to-use ExponentialDecay
decay = estimator.to_decay(results, default_beta=0.5)
score = decay.compute(my_fact, datetime.now())
```

#### Swapping the estimation backend

```python
# Method of moments — closed-form, faster, slightly less efficient
estimator = BetaEstimator(method=MomentMatching())

# Ensemble — weighted combination of any methods
estimator = BetaEstimator(
    method=EnsembleMethod(
        methods=[MLEBernoulli(), MomentMatching()],
        weights=[0.7, 0.3],
    )
)
```

---

## Custom Extensions

Chronofy is designed to be extended. Both the decay backend and the estimation backend are swappable via abstract base classes.

### Custom Decay Function

Subclass `DecayFunction` and implement two methods:

```python
import math
from datetime import datetime
from chronofy import DecayFunction, TemporalFact, validate_decay_function

class SeasonalDecay(DecayFunction):
    """Decay that accounts for seasonal recurrence patterns.

    Information that repeats on a yearly cycle (e.g. seasonal disease
    prevalence, annual financial reports) becomes relevant again after
    ~365 days rather than continuing to decay monotonically.
    """

    def __init__(self, base_rate: float = 0.5, period_days: float = 365.0):
        self.base_rate = base_rate
        self.period_days = period_days

    def compute(self, fact: TemporalFact, query_time: datetime) -> float:
        age_days = fact.age_at(query_time)
        # Monotone decay modulated by seasonal cosine
        monotone = math.exp(-self.base_rate * age_days)
        seasonal = 0.5 * (1 + math.cos(2 * math.pi * age_days / self.period_days))
        return fact.source_quality * monotone * seasonal

    def compute_batch(self, facts: list[TemporalFact], query_time: datetime) -> list[float]:
        return [self.compute(f, query_time) for f in facts]

# Validate before use — catches bugs immediately with clear error messages
decay = validate_decay_function(SeasonalDecay(base_rate=0.01, period_days=365))

# Use anywhere a DecayFunction is accepted
from chronofy import EpistemicFilter, STLVerifier, CorpusStats
ep_filter  = EpistemicFilter(decay_fn=decay, threshold=0.05)
verifier   = STLVerifier(decay_fn=decay, threshold=0.1)
stats      = CorpusStats(facts=my_facts, query_time=datetime.now(), decay_fn=decay)
```

### Custom Estimation Method

Subclass `EstimationMethod` and implement one method:

```python
import math
from chronofy import EstimationMethod, BetaEstimator, validate_estimation_method

class MedianHalfLifeEstimator(EstimationMethod):
    """Estimates β by finding the age at which ~50% of facts become invalid.

    A non-parametric alternative: sort facts by age, find the median
    age among invalid ones, then set β = ln(2) / median_invalid_age.
    Robust to outliers and requires no optimisation.
    """

    def fit(self, ages: list[float], valid: list[bool]) -> float:
        invalid_ages = sorted(a for a, v in zip(ages, valid) if not v)
        if not invalid_ages:
            return 0.0   # all valid → effectively no decay
        median_invalid_age = invalid_ages[len(invalid_ages) // 2]
        if median_invalid_age <= 0:
            return 50.0  # invalid at age 0 → very fast decay
        return math.log(2) / median_invalid_age

# Validate before use
method = validate_estimation_method(MedianHalfLifeEstimator())

# Plug into BetaEstimator
estimator = BetaEstimator(method=method)
result = estimator.fit(ages=[1.0, 5.0, 10.0, 30.0], valid=[True, True, False, False])

# Or combine with other methods in an ensemble
from chronofy import EnsembleMethod, MLEBernoulli
ensemble = EnsembleMethod(
    methods=[MLEBernoulli(), method],
    weights=[0.6, 0.4],
)
estimator = BetaEstimator(method=ensemble)
```

### Plugin Validation

`validate_decay_function` and `validate_estimation_method` run behavioural smoke-tests and raise `PluginValidationError` with a clear message if anything is wrong:

```python
from chronofy import validate_decay_function, validate_estimation_method, PluginValidationError

try:
    decay = validate_decay_function(MyCustomDecay())
except PluginValidationError as e:
    print(f"Fix your decay function: {e}")

# What gets checked for decay functions:
#   ✓ Is an instance of DecayFunction
#   ✓ compute() is callable and returns a float
#   ✓ All returned scores are finite (no NaN/inf)
#   ✓ All returned scores are in [0.0, 1.0]
#   ✓ compute_batch() returns a list of the same length as the input

# What gets checked for estimation methods:
#   ✓ Is an instance of EstimationMethod
#   ✓ fit() is callable and returns a float
#   ✓ Returned β is finite (no NaN/inf)
#   ✓ Returned β ≥ 0 (non-negative by definition)
```

---

## Decision-Theoretic Grounding

The decay coefficient β is not an arbitrary hyperparameter. Under Gaussian latent dynamics (Ornstein-Uhlenbeck process `dθ = -κ(θ-μ)dt + σdW`), the information content of a measurement at age `Δt` decays as:

```
I(t_e → t) ∝ exp(-2κ · Δt)
```

Therefore the **optimal decay coefficient is β = 2κ** — twice the mean-reversion rate of the underlying latent process. This means:

- `β_vital_sign ≈ 5.0` → κ ≈ 2.5 day⁻¹ (physiology reverts in hours)
- `β_lab_result ≈ 2.0` → κ ≈ 1.0 day⁻¹ (labs shift in days)
- `β_demographic = 0.0` → κ = 0 (blood type is invariant)

**Temporal Invariance Guarantee:** When β = 0, `exp(-β·Δt) = 1` for all ages. Chronofy correctly preserves the full value of stable facts regardless of how old they are.

See *Chronofy: A Temporal-Logical Decay Architecture for Information Validity in Time-Aware RAG* (IEEE IRI 2026) for the full derivation.

---

## Full Pipeline

The `ChronofyPipeline` composes all three layers:

```python
from chronofy import ChronofyPipeline, ExponentialDecay

decay = ExponentialDecay(beta={"vital_sign": 5.0, "demographic": 0.0})

pipeline = ChronofyPipeline(
    decay_fn=decay,
    filter_threshold=0.1,     # τ: epistemic filter threshold
    stl_threshold=0.3,        # γ: STL validity threshold
)

result = pipeline.run(
    facts=my_facts,
    query_time=datetime.now(),
    reasoning_steps=my_steps,
)

print(result.valid_facts)
print(result.stl_result.satisfied)
print(result.stl_result.output_confidence_bound)
```

---

## API Reference

### Models

| Class | Description |
|---|---|
| `TemporalFact` | Evidence tuple `(c, t_e, q, m)` with `.age_at(query_time)` |
| `ReasoningStep` | One step in a CoT trace with associated facts |
| `ReasoningTrace` | Full chain-of-thought trace over a query time |

### Decay Backends

| Class | Parameterisation | Best for |
|---|---|---|
| `ExponentialDecay` | `beta` per fact type | General default; OU-optimal |
| `HalfLifeDecay` | `half_life` in days | Intuitive configuration |
| `LinearDecay` | `rate` (validity/day) | Hard compliance deadlines |
| `PowerLawDecay` | `exponent` | Citations, web freshness |
| `WeibullDecay` | `scale`, `shape` | Survival analysis domains |
| `DecayFunction` | ABC | Custom implementations |

### Retrieval

| Class | Description |
|---|---|
| `EpistemicFilter` | τ-threshold gate; `.filter()`, `.partition()`, `.needs_reacquisition()` |

### Verification

| Class | Description |
|---|---|
| `STLVerifier` | STL robustness over knowledge validity; `.verify()`, `.robustness()` |
| `STLResult` | Robustness score, step validity, weakest fact, confidence bound |

### Analysis

| Class | Description |
|---|---|
| `CorpusStats` | Age/validity distribution, staleness rate, coverage gaps, effective density |
| `AgeStats` | Frozen dataclass: min/max/mean/median/std/p25/p75 of ages |
| `ValidityStats` | Frozen dataclass: min/max/mean/median/std/p25/p75 of validity scores |
| `BetaEstimator` | Fits β from `(age, valid)` observations; `.fit()`, `.fit_corpus()`, `.to_decay()` |
| `BetaEstimateResult` | β, half-life, log-likelihood, n_obs, n_valid, converged |

### Estimation Methods

| Class | Description |
|---|---|
| `MLEBernoulli` | MLE under `P(valid\|age,β) = exp(-β·age)`; golden-section search |
| `MomentMatching` | Closed-form: `β = -log(p_valid) / mean_age` |
| `EnsembleMethod` | Normalised weighted average of any EstimationMethod instances |
| `EstimationMethod` | ABC — subclass to add any estimation technique |

### Plugin Utilities

| Symbol | Description |
|---|---|
| `validate_decay_function(fn)` | Behavioural smoke-test; returns `fn` on success |
| `validate_estimation_method(m)` | Behavioural smoke-test; returns `m` on success |
| `PluginValidationError` | Raised with a descriptive message on validation failure |

---

## Citation

```bibtex
@inproceedings{syed2026chronofy,
  title     = {Chronofy: A Temporal-Logical Decay Architecture for Information
               Validity in Time-Aware Retrieval-Augmented Generation},
  author    = {Syed, Muntaser},
  booktitle = {Proceedings of the IEEE International Conference on
               Information Reuse and Integration (IRI)},
  year      = {2026},
}
```

---

## License

MIT
