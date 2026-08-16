# Chronofy–RuView integration roadmap

> Status: public research/engineering proposal
>
> Last external-source verification: **2026-08-15**
>
> Roadmap owner: **Unassigned**
>
> Upstream relationship: no RuView endorsement, commitment, or integration agreement is known

This document records a possible integration between Chronofy's general
temporal-validity primitives and RuView's WiFi sensing pipeline. It is a roadmap,
not an implementation report. All eight phases below are **proposed—not
started**. It makes no clinical-performance, safety, regulatory-compliance, or
publication claim.

## Status vocabulary

- **Shipped** means code and tests exist in this repository.
- **Proposed—not started** means no implementation evidence exists here.
- **Validation required** means a parameter or design needs empirical evidence.
- **External coordination required** means work would touch an independently
  maintained repository or interface.
- **Non-goal** means this roadmap intentionally does not promise that outcome.

## Evidence baseline

### Shipped in Chronofy

The current Python source (version `0.1.9`, which is an unreleased repository
target) provides general-purpose:

- exponential, half-life, linear, power-law, and Weibull decay functions;
- `TemporalFact`, decay-weighted filtering/scoring, temporal graph retrieval,
  and re-acquisition predicates;
- `STLVerifier` over timestamped facts used by a `ReasoningTrace`;
- temporal embedding/loss components; and
- optional Subjective Logic opinions, decay, trust, fusion, conflict, scoring,
  graph, filtering, and pipeline helpers through `chronofy[sl]`.

These capabilities are visible in the tracked [`chronofy/`](../chronofy/)
package and its tests. They are not sensor-specific. In particular:

- there is no Rust crate or Rust source in this repository;
- there is no `SignalDecayProfile`, sensor `ValidityWindow`, RuView gate
  adapter, CBOR annotation bridge, or ESP32 integration;
- Chronofy's current exponential contract is `source_quality × exp(-beta ×
  age)`, not the proposed three-factor `quality × confidence × decay`; and
- the current STL verifier evaluates reasoning-trace facts, not a sampled
  real-time sensor signal with gap semantics.

The root pipeline also describes Layer 1 embedding integration as future work.
Existence of an embedding component is therefore not evidence of a shipped
end-to-end embedded pipeline.

### Verified RuView snapshot

Concrete RuView statements in this roadmap were checked against the official
repository at commit
[`1d50518a70254660006911f37aede77fed142d43`](https://github.com/ruvnet/RuView/commit/1d50518a70254660006911f37aede77fed142d43)
(commit timestamp 2026-08-15T21:08:02Z). The links are commit-pinned evidence,
not promises that RuView's current or future API will remain compatible.

As of **2026-08-15**:

- RuView's `GateDecision` is not the unit enum assumed by the old draft. Its
  variants are `Accept { noise_multiplier }`, `PredictOnly`, `Reject`, and
  `Recalibrate { stale_frames }`; its configurable policy includes accept,
  reject, stale-frame, noise, and adaptive settings. See the official
  [`coherence_gate.rs`](https://github.com/ruvnet/RuView/blob/1d50518a70254660006911f37aede77fed142d43/v2/crates/wifi-densepose-signal/src/ruvsense/coherence_gate.rs#L18-L100).
- The coherence implementation uses per-subcarrier z-scores with
  inverse-variance weighting, so temporal freshness would answer a different
  question rather than replace the spatial/statistical gate. See
  [`coherence.rs`](https://github.com/ruvnet/RuView/blob/1d50518a70254660006911f37aede77fed142d43/v2/crates/wifi-densepose-signal/src/ruvsense/coherence.rs#L218-L268).
- Firmware constants define a 256-frame phase history and 1,200 calibration
  frames (documented there as about 60 seconds at 20 Hz). At nominal 20 Hz,
  256 frames span 12.8 seconds—not 30 seconds. The code does not support the old
  claim that a 30-second-old sample remains equally weighted in that buffer.
  See official
  [`edge_processing.h`](https://github.com/ruvnet/RuView/blob/1d50518a70254660006911f37aede77fed142d43/firmware/esp32-csi-node/main/edge_processing.h#L34-L66).
- RuView already has longitudinal Welford statistics and explicitly describes
  their output as non-diagnostic. See
  [`longitudinal.rs`](https://github.com/ruvnet/RuView/blob/1d50518a70254660006911f37aede77fed142d43/v2/crates/wifi-densepose-signal/src/ruvsense/longitudinal.rs#L1-L19).
- RuView also has an LTL safety-guard module. Any Chronofy temporal-logic work
  must first decide whether it complements, replaces, or is unrelated to that
  module. See
  [`tmp_temporal_logic_guard.rs`](https://github.com/ruvnet/RuView/blob/1d50518a70254660006911f37aede77fed142d43/v2/crates/wifi-densepose-wasm-edge/src/tmp_temporal_logic_guard.rs#L1-L52).

No external repository was cloned or modified for this audit.

## Intended engineering outcome

The narrow goal is to investigate an explicit, testable notion of per-reading
freshness and provenance that could *augment* RuView's existing quality and
coherence handling. A defensible design would expose both:

1. a **measurement disposition** such as Accept, PredictOnly, or Reject; and
2. an independent **maintenance action** such as None, Reacquire, or
   Recalibrate.

Separating those dimensions avoids the old draft's unsafe mapping of every
low-confidence or expired input directly to recalibration. Recalibration,
re-acquisition, and rejection have different causes and operational effects.

The design must preserve both projects' native concepts. Chronofy should not
claim that RuView lacks temporal handling, and a proposed temporal score should
not erase RuView's coherence gate, finite buffers, calibration, or longitudinal
state.

## Proposed technical shape

Everything in this section is **proposed—not started**.

### 1. Cross-language decay core

A small Rust core could implement validated scalar/batch decay and half-life
math using a versioned contract shared with Python. Before code exists, the
contract must settle:

- whether confidence is folded into Chronofy's existing `source_quality` or is
  a distinct factor;
- accepted time units and monotonic-clock representation;
- NaN, infinity, negative-age, clock-reset, wraparound, and out-of-order input
  behavior;
- `f32` versus `f64` tolerances; and
- allocation, `std`/`no_std`, `libm`, serde, and WASM target requirements.

Declaring a Cargo feature named `no_std` would not by itself make an
implementation `no_std`, and WASM does not imply `no_std`. Those are explicit
target decisions.

Golden vectors must include beta zero, extreme beta/age products, future
timestamps, source-quality endpoints, batch parity, and half-life conversion.

### 2. Signal profile schema

A versioned data schema could describe a signal class, time unit, beta,
fresh/degraded/expired boundaries, minimum quality, interpolation policy, and
provenance for its calibration. Python and Rust must consume the same fields;
the prior draft's Python-only `interpolation` field must not silently disappear
in Rust.

All numerical profiles are currently **unvalidated hypotheses**. In particular,
exponential half-life is:

```text
half_life = ln(2) / beta
```

For `beta = 10 s^-1`, this is `0.0693147 s`, approximately **69.3 ms**, not
50 ms. That value is an arithmetic example only; it is not an endorsed CSI
profile. A hard expiry, decay threshold crossing, signal acquisition window,
and alert retention period are separate concepts and need separate fields.

### 3. Gate composition

Spatial coherence and temporal freshness should be composed through an explicit
truth table with reason codes, hysteresis/debounce, and missing-input behavior.
An ordinal `minimum(spatial, temporal)` is insufficient because RuView variants
carry payloads and recalibration is not merely a lower rank than rejection.

An adapter must compile against a pinned RuView revision and map payload fields
deliberately. Phase 4 acceptance would require tests for every cross-product of
spatial and temporal states, including sustained staleness and recovery.

### 4. Time and sampled-signal semantics

Sensor age needs a clock/provenance model covering device boot-relative time,
host time, synchronization uncertainty, network latency, device reboot,
counter wrap, packet reordering, and gaps. A wall-clock `datetime` port is not
enough.

Likewise, a sampled formula such as `G[0,w](score >= gamma)` is not automatically
equivalent to Chronofy's current weakest-link reasoning-trace verifier. Window
endpoints, sampling rate, missing samples, insufficient acquisition duration,
and gap tolerance need formal definitions and property tests.

### 5. Subjective Logic/annotation bridge

The wire format and dependency are unresolved. A bridge must not assume a
specific CBOR-LD-ex type name, byte width, or `value / 255` quantization without
checking the selected library/schema. It must also choose, with evidence,
whether a projected probability, raw belief, uncertainty penalty, or some other
mapping supplies decay quality/confidence. Multiplying belief-derived quality by
`1 - uncertainty` can double-penalize uncertainty.

The first deliverable here is a versioned schema and golden encoded examples,
not an informal conversion function.

### 6. Composite scores and event lifecycles

Weakest-link (AND) aggregation may suit a conjunction but not a fall-alert
lifecycle or a redundant sensor set. Proposed composition must declare AND, OR,
quorum, missing-signal, and acknowledgement/TTL semantics. A fall event should
not automatically be modeled as an ordinary continuously decaying measurement.

## Eight-phase roadmap

No phase has an assigned owner or completion evidence. Effort estimates were
removed because the prerequisites and external review path are unresolved.

| Phase | Proposed deliverable | Status / owner | Prerequisites | Evidence required to complete |
|---|---|---|---|---|
| 1 | Rust decay/half-life core and sensor-window semantics | **Proposed—not started** / Unassigned | Frozen numeric/time contract; target matrix | Cross-language golden vectors, property tests, documented benchmark hardware/command |
| 2 | Versioned signal-profile schema and Python/Rust readers | **Proposed—not started** / Unassigned | Phase 1 contract; calibration plan | Schema compatibility tests; no default profile without labelled validation data |
| 3 | Sensor/vital-sign acquisition and freshness windows | **Proposed—not started** / Unassigned | Phase 2; requirements owner; lawful dataset protocol | Empirical calibration report, uncertainty bounds, failure/gap tests; no clinical label |
| 4 | RuView gate adapter | **Proposed—not started** / Unassigned | Phase 1; pinned RuView API; disposition/maintenance truth table | Compile/tests against the pinned snapshot; every state/payload/recovery path covered |
| 5 | Subjective Logic annotation/wire bridge | **Proposed—not started** / Unassigned | Selected library/schema and uncertainty mapping | Versioned schema, round-trip vectors, quantization/error analysis |
| 6 | RuView pipeline integration | **Proposed—not started** / Unassigned | Phases 1 and 4; upstream maintainer agreement | External review, integration tests, rollback/feature flag; **external coordination required** |
| 7 | Profile composition and composite reasoning | **Proposed—not started** / Unassigned | Phase 2; declared AND/OR/quorum/event semantics | Deterministic truth-table/property tests and missing-signal behavior |
| 8 | ESP32-to-gate end-to-end validation | **Proposed—not started** / Unassigned | Phases 4–7; lawful traces/hardware protocol | First a deterministic synthetic trace; separately a reproducible real-hardware evaluation with provenance |

Phase 8's synthetic test and real-world evaluation are distinct. Passing a
synthetic trace must never be reported as sensor, safety, or clinical
validation.

## Roadmap constraints already decided

- Temporal freshness augments rather than replaces RuView coherence.
- `beta = 10 s^-1` has a 69.3 ms half-life.
- All profile/window/threshold numbers require calibration evidence.
- External source links are pinned audit evidence, not stable-API guarantees.
- This repository will not vendor, clone, or modify RuView as part of roadmap
  documentation.
- Measurement disposition and maintenance action are separate outputs.
- Raw/restricted clinical or CSI data, credentials, and machine artifacts do
  not belong in version control.
- No phase is shipped until its acceptance evidence is tracked and reviewable.

## Unresolved decisions and prerequisites

1. Crate ownership, repository location, license, and dependency direction.
2. RuView maintainer/reviewer and upstream contribution policy.
3. Exact clock, timestamp, latency, synchronization, and provenance model.
4. Compatibility with RuView's existing LTL guard and coherence pipeline.
5. Confidence/Subjective Logic mapping and avoidance of double counting.
6. Gate truth table, reason codes, hysteresis, debounce, recovery, and missing
   data behavior.
7. Calibration datasets, lawful access, ground truth, metrics, uncertainty, and
   tolerances for every proposed signal profile.
8. Cross-language schema versioning and backward compatibility.
9. CPU/edge/WASM targets, numeric precision, allocation, and `no_std` policy.
10. Realistic throughput/latency targets tied to node/frame counts and named
    hardware; the old 54,000-scores/second number had no recorded basis.
11. Re-acquisition protocol. The inspected code does not establish a host-to-
    ESP32 request path that Chronofy can simply trigger.
12. Whether alerts use freshness decay, acknowledgement TTLs, state machines,
    or another event-lifecycle model.

## Clinical, regulatory, standards, and interoperability boundary

The old plan incorrectly used standards as support for proposed freshness
windows. The official scopes do not establish those values:

- [ISO 80601-2-61:2026](https://www.iso.org/standard/84595.html) concerns basic
  safety and essential performance of pulse oximeter equipment. It does not
  validate a WiFi breathing-rate freshness window.
- [IEC 60601-2-27:2011](https://webstore.iec.ch/en/publication/2638) concerns
  electrocardiographic monitoring equipment. It does not validate a freshness
  window for a WiFi-derived heart-rate estimate.

Accordingly, this roadmap specifies no clinical validity windows and makes no
claim of conformity to either standard. Engineering evidence, intended use,
risk management, clinical evaluation, and regulatory review are separate work.
The FDA's official
[Clinical Decision Support Software guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software)
also makes clear that regulatory analysis depends on the software function and
intended use; adding temporal metadata cannot confer compliance.

If a CBOR-based format is selected, the implementation must cite the exact
format/version. [RFC 8949](https://www.rfc-editor.org/rfc/rfc8949) defines CBOR
and [RFC 8610](https://www.rfc-editor.org/rfc/rfc8610) defines CDDL. Those
standards do not, by themselves, define a Subjective Logic profile or prove
interoperability with a particular external annotation library.

## Acceptance and reproducibility policy

Evidence for any completed phase must live at stable tracked paths and include:

- source version and, for external interfaces, a pinned upstream revision;
- commands, toolchain/target/hardware, seeds, configs, and numeric tolerances;
- synthetic, non-sensitive golden fixtures and cross-language vectors;
- lawful dataset acquisition/provenance/checksum instructions where real data
  are necessary;
- failure, missing-data, clock, recovery, and adverse-condition tests;
- benchmark methodology rather than an unsupported throughput target; and
- explicit separation of engineering validation from clinical/regulatory
  claims.

Research artifacts follow the repository's [release policy](../RELEASING.md)
and [experiment reproducibility guide](../experiments/README.md).

## Non-goals

- Diagnosis, treatment recommendations, medical-device performance, clinical
  safety/effectiveness, or regulatory compliance.
- Claiming RuView endorsement, a committed upstream roadmap, or API stability.
- Replacing RuView's coherence system or presenting existing RuView temporal
  handling as absent.
- Redistributing raw/restricted clinical or CSI data.
- Treating an API demonstration or synthetic fixture as real-world validation.
- Guaranteeing publication venue, novelty, acceptance, schedule, or effort.
