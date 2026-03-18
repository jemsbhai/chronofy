# Chronofy

**Temporal validity framework for time-aware AI systems.**

[![npm version](https://img.shields.io/npm/v/chronofy.svg)](https://www.npmjs.com/package/chronofy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

JavaScript port of the [Chronofy Python package](https://pypi.org/project/chronofy/). Implements the **Temporal-Logical Decay Architecture (TLDA)** — a framework that embeds temporal validity into the representation, retrieval, and reasoning layers of AI systems.

Also available as a [Python package](https://pypi.org/project/chronofy/).

---

## The Problem

RAG systems treat all retrieved facts as equally valid regardless of age. A clinical lab reading from yesterday and one from six months ago receive identical weight if semantically similar. This causes **temporal hallucination** — plausible but factually obsolete outputs.

---

## Installation

```bash
npm install chronofy
```

---

## Quick Start

```js
const {
  TemporalFact,
  ExponentialDecay,
  EpistemicFilter,
  STLVerifier,
  ReasoningStep,
  ReasoningTrace,
} = require('chronofy');

const now = new Date();
const daysAgo = (n) => new Date(now.getTime() - n * 86400000);

// 1. Define timestamped facts
const facts = [
  new TemporalFact({
    content: 'Serum potassium: 4.1 mEq/L',
    timestamp: daysAgo(1),
    factType: 'vital_sign',
    sourceQuality: 0.95,
  }),
  new TemporalFact({
    content: 'Serum potassium: 3.2 mEq/L',
    timestamp: daysAgo(180),
    factType: 'vital_sign',
    sourceQuality: 0.95,
  }),
  new TemporalFact({
    content: 'Blood type: O+',
    timestamp: daysAgo(3650),
    factType: 'demographic',
    sourceQuality: 1.0,
  }),
];

// 2. Configure decay — β = 2κ (Proposition 1)
const decay = new ExponentialDecay({
  beta: {
    vital_sign:        5.0,   // volatile — decays in hours/days
    lab_result:        2.0,
    medication:        1.0,
    chronic_condition: 0.01,
    demographic:       0.0,   // invariant — never decays
  },
});

// 3. Filter to temporally valid facts only
const filter = new EpistemicFilter({ decayFn: decay, threshold: 0.1 });
const validFacts = filter.filter(facts, now);
// → [yesterday's reading, blood type]  (6-month reading filtered out)

// 4. Verify reasoning chain with STL
const step = new ReasoningStep({ stepIndex: 0, content: 'assess risk', factsUsed: validFacts });
const trace = new ReasoningTrace({ steps: [step], queryTime: now });

const verifier = new STLVerifier({ decayFn: decay, threshold: 0.05 });
const result = verifier.verify(trace);

console.log(`STL satisfied:    ${result.satisfied}`);
console.log(`Robustness:       ${result.robustness.toFixed(4)}`);
console.log(`Confidence bound: ${result.outputConfidenceBound.toFixed(4)}`);
```

---

## Core Concepts

### TemporalFact

Every piece of evidence is a tuple `(content, timestamp, sourceQuality, factType)`:

```js
const fact = new TemporalFact({
  content:     'Patient serum potassium: 4.1 mEq/L',
  timestamp:   new Date('2024-06-01T08:30:00Z'),  // observation time
  factType:    'lab_result',
  sourceQuality: 0.95,                             // reliability weight q ∈ (0, 1]
  source:      'EPIC/LAB',
  metadata:    { units: 'mEq/L' },
});

const ageDays = fact.ageAt(new Date());  // age in days at any reference time
```

### Decay Functions

All decay functions return a validity score `V(e, T_q) ∈ [0, 1]`.

```js
const { ExponentialDecay, HalfLifeDecay, LinearDecay, PowerLawDecay, WeibullDecay } = require('chronofy');

// ExponentialDecay — Bayesian-optimal under Ornstein-Uhlenbeck dynamics
// β = 2κ where κ is the latent process mean-reversion rate (Proposition 1)
const decay = new ExponentialDecay({ beta: { vital_sign: 5.0, demographic: 0.0 } });

// Construct directly from mean-reversion rates
const decay2 = ExponentialDecay.fromMeanReversionRate({ vital_sign: 2.5, lab_result: 1.0 });

// Half-life parameterised (same as exponential, different interface)
const decay3 = new HalfLifeDecay({ halfLife: { vital_sign: 3.0, lab_result: 14.0 } });

// Linear — hard expiry at 1/rate days
const decay4 = new LinearDecay({ rate: { regulatory: 0.1 } });

// Power-law — heavy-tailed; matches citation/web patterns
const decay5 = new PowerLawDecay({ exponent: { news: 1.5 } });

// Weibull — shape > 1: accelerating decay; shape < 1: decelerating
const decay6 = new WeibullDecay({ scale: { lab_result: 7.0 }, shape: { lab_result: 1.5 } });

// Compute a single score
const score = decay.compute(fact, new Date());

// Compute batch
const scores = decay.computeBatch(facts, new Date());
```

**Temporal Invariance Guarantee:** When `β = 0`, validity is always `1.0` regardless of age — stable facts like blood type are never penalised.

### EpistemicFilter

Structurally excludes stale evidence from the LLM context window:

```js
const { EpistemicFilter } = require('chronofy');

const filter = new EpistemicFilter({ decayFn: decay, threshold: 0.1 });

const fresh             = filter.filter(facts, now);
const [fresh2, stale]   = filter.partition(facts, now);
const needsRefresh      = filter.needsReacquisition(facts, now);
```

### STLVerifier

Applies Signal Temporal Logic robustness to **knowledge validity** — not to LLM output confidence. Catches cases where the LLM is confidently wrong due to stale retrieval.

**STL formula:** `φ_valid = G[0,n]( v(sᵢ) ≥ γ )`
**Robustness:** `ρ = minᵢ ( v(sᵢ) − γ )` — positive means satisfied.
**Theorem 1 (Weakest-Link Bound):** `C_out ≤ minᵢ V(eᵢ, T_q)`

```js
const { STLVerifier, ReasoningStep, ReasoningTrace } = require('chronofy');

const verifier = new STLVerifier({ decayFn: decay, threshold: 0.3 });
const result   = verifier.verify(trace);

result.satisfied              // boolean
result.robustness             // scalar ρ
result.stepValidity           // per-step signals
result.outputConfidenceBound  // Theorem 1 bound
result.weakestFact            // the single least-fresh fact
```

### CorpusStats

Temporal health statistics for a fact corpus:

```js
const { CorpusStats } = require('chronofy');

const stats = new CorpusStats({ facts, queryTime: new Date(), decayFn: decay });

stats.factCount                     // number of facts
stats.effectiveDensity              // mean validity (0–1 health score)
stats.temporalSpanDays              // range from oldest to newest fact
stats.ageStats                      // { min, max, mean, median, std, p25, p75 }
stats.validityStats                 // same for validity scores
stats.stalenessRate(0.1)            // fraction of facts below 10% validity
stats.coverageGaps(7)               // gaps with no facts for ≥ 7 days
stats.summary()                     // all of the above as a plain object
```

### BetaEstimator

Fit β empirically from labelled observations `(ageInDays, stillValid)`:

```js
const { BetaEstimator, MLEBernoulli, MomentMatching, EnsembleMethod } = require('chronofy');

// Default: MLE under P(valid | age, β) = exp(-β · age)
const estimator = new BetaEstimator();

const result = estimator.fit({
  ages:  [1, 5, 10, 30, 60],
  valid: [true, true, true, false, false],
  factType: 'lab_result',
});
console.log(`β̂ = ${result.beta.toFixed(4)}, half-life = ${result.halfLife.toFixed(1)} days`);

// Fit per fact type from a labelled corpus
const results = estimator.fitCorpus({ facts, labels, queryTime: new Date() });

// Convert directly to ExponentialDecay
const fittedDecay = estimator.toDecay(results);

// Swap the estimation backend
const mmEst      = new BetaEstimator({ method: new MomentMatching() });
const ensembleEst = new BetaEstimator({
  method: new EnsembleMethod({
    methods: [new MLEBernoulli(), new MomentMatching()],
    weights: [0.7, 0.3],
  }),
});
```

### TemporalScorer

Combines semantic similarity with temporal validity for retrieval ranking:

```js
const {
  TemporalScorer, MultiplicativeScoring, HarmonicScoring,
  WeightedBlendScoring, PowerScoring,
} = require('chronofy');

const scorer = new TemporalScorer({
  decayFn:  decay,
  strategy: new HarmonicScoring(),         // or Multiplicative, WeightedBlend, Power
});

// Score a single fact
const sf = scorer.scoreFact(fact, 0.85, new Date());
sf.similarity     // 0.85
sf.validity       // from decay function
sf.combinedScore  // from strategy

// Score and rank a batch
const ranked = scorer.rank(facts, similarities, new Date(), /* topK= */ 5);
```

| Strategy | Formula | Best for |
|---|---|---|
| `MultiplicativeScoring` | `sim × val` | Default; aggressively penalises either weakness |
| `HarmonicScoring` | `2·sim·val / (sim+val)` | Balanced; satisfies `min ≤ H ≤ max` |
| `WeightedBlendScoring(α)` | `α·sim + (1-α)·val` | Explicit trade-off control |
| `PowerScoring(α)` | `sim^α · val^(1-α)` | Geometric blend; α=0.5 is geometric mean |

### DecayComparison

Evaluate multiple decay backends on labelled data to select the best fit:

```js
const { DecayComparison, ExponentialDecay, LinearDecay } = require('chronofy');

const dc = new DecayComparison({
  decayFns: {
    exponential: new ExponentialDecay({ beta: { lab: 2.0 } }),
    linear:      new LinearDecay({ rate: { lab: 0.1 } }),
  },
  facts,
  labels,     // parallel boolean array
  queryTime:  new Date(),
});

const results = dc.compare();   // sorted by AUROC descending
for (const r of results) {
  console.log(`${r.name}: AUC=${r.auc_roc.toFixed(3)} Brier=${r.brier_score.toFixed(3)}`);
}

const best = dc.best('auc_roc');     // or 'brier_score', 'log_loss'
const summary = dc.summary();
```

---

## Custom Extensions

### Custom Decay Function

```js
const { DecayFunction, validateDecayFunction } = require('chronofy');

class SeasonalDecay extends DecayFunction {
  constructor({ baseRate = 0.01, periodDays = 365 } = {}) {
    super();
    this.baseRate = baseRate;
    this.periodDays = periodDays;
  }

  compute(fact, queryTime) {
    const age = fact.ageAt(queryTime);
    const monotone = Math.exp(-this.baseRate * age);
    const seasonal = 0.5 * (1 + Math.cos(2 * Math.PI * age / this.periodDays));
    return fact.sourceQuality * monotone * seasonal;
  }

  computeBatch(facts, queryTime) {
    return facts.map(f => this.compute(f, queryTime));
  }
}

// Validate before use — catches bugs immediately
const decay = validateDecayFunction(new SeasonalDecay());
```

### Custom Estimation Method

```js
const { EstimationMethod, BetaEstimator, validateEstimationMethod } = require('chronofy');

class MedianHalfLifeEstimator extends EstimationMethod {
  fit(ages, valid) {
    const invalidAges = ages.filter((_, i) => !valid[i]).sort((a, b) => a - b);
    if (!invalidAges.length) return 0;
    const median = invalidAges[Math.floor(invalidAges.length / 2)];
    return median > 0 ? Math.LN2 / median : 50;
  }
}

const method    = validateEstimationMethod(new MedianHalfLifeEstimator());
const estimator = new BetaEstimator({ method });
```

### Custom Scoring Strategy

```js
const { ScoringStrategy, TemporalScorer, validateScoringStrategy } = require('chronofy');

class MinScoring extends ScoringStrategy {
  score(similarity, validity) {
    return Math.min(similarity, validity);   // conservative lower bound
  }
}

const strategy = validateScoringStrategy(new MinScoring());
const scorer   = new TemporalScorer({ decayFn: decay, strategy });
```

---

## Decision-Theoretic Grounding

The decay coefficient β is not an arbitrary hyperparameter. Under Gaussian latent dynamics (Ornstein-Uhlenbeck process), the information-theoretically optimal decay rate is **β = 2κ**, where κ is the mean-reversion rate of the underlying process.

```js
// If you know the domain dynamics, construct from κ directly
const decay = ExponentialDecay.fromMeanReversionRate({
  vital_sign: 2.5,   // β = 5.0
  lab_result: 1.0,   // β = 2.0
  demographic: 0.0,  // β = 0.0 (invariant)
});
```

---

## API Reference

| Class | Description |
|---|---|
| `TemporalFact` | Evidence tuple with `.ageAt(queryTime)` |
| `ReasoningStep` | One CoT step with associated facts |
| `ReasoningTrace` | Full reasoning chain over a query time |
| `ExponentialDecay` | β-parameterised; OU-optimal |
| `HalfLifeDecay` | Half-life parameterised |
| `LinearDecay` | Hard expiry at 1/rate |
| `PowerLawDecay` | Heavy-tailed decay |
| `WeibullDecay` | Generalised exponential |
| `DecayFunction` | Base class for custom backends |
| `EpistemicFilter` | τ-threshold gate; `filter`, `partition`, `needsReacquisition` |
| `STLVerifier` | STL robustness over knowledge validity |
| `STLResult` | Frozen: robustness, stepValidity, weakestFact, confidenceBound |
| `CorpusStats` | Temporal health: age/validity stats, staleness, gaps |
| `BetaEstimator` | Fit β from data; `fit`, `fitCorpus`, `toDecay` |
| `MLEBernoulli` | MLE under Bernoulli decay model |
| `MomentMatching` | Closed-form method-of-moments estimator |
| `EnsembleMethod` | Weighted combination of estimators |
| `TemporalScorer` | Rank facts by sim + validity; `scoreFact`, `scoreFacts`, `rank` |
| `MultiplicativeScoring` | `sim × val` |
| `HarmonicScoring` | `2·sim·val / (sim+val)` |
| `WeightedBlendScoring` | `α·sim + (1-α)·val` |
| `PowerScoring` | `sim^α · val^(1-α)` |
| `ScoringStrategy` | Base class for custom strategies |
| `DecayComparison` | Compare decay backends: AUROC, Brier, log loss, Spearman ρ |
| `validateDecayFunction` | Smoke-test custom decay; returns fn on success |
| `validateEstimationMethod` | Smoke-test custom estimator |
| `validateScoringStrategy` | Smoke-test custom scoring strategy |
| `PluginValidationError` | Thrown by validators with descriptive message |

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
