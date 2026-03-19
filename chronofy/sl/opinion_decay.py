"""Subjective Logic opinion-based temporal decay for Chronofy.

Integrates Jøsang's opinion aging operator (Subjective Logic §10.4) with
Chronofy's DecayFunction protocol, providing a richer temporal validity
model that distinguishes between evidential direction and epistemic uncertainty.

Core insight:
    Standard Chronofy decay produces a scalar V ∈ [0, 1] that conflates
    "weak evidence" with "stale evidence." Jøsang's aging operator
    separates these: decay migrates mass from belief/disbelief into
    uncertainty, preserving the b/d ratio. A 6-month-old potassium
    reading of 3.2 still tells you the value was LOW — you just don't
    know if it's still low.

Mathematical model (Jøsang §10.4):
    Given decay factor λ ∈ [0, 1]:
        b' = λ · b          (belief shrinks)
        d' = λ · d          (disbelief shrinks proportionally)
        u' = 1 - b' - d'    (uncertainty absorbs the loss)

    Invariants:
        b'/d' = b/d          (evidence direction preserved)
        b' + d' + u' = 1     (valid opinion)
        P(ω') → a as λ → 0  (reverts to base rate prior)

Backward compatibility:
    compute() returns P(ω') = b' + a·u' — the projected probability.
    This is a scalar in [0, 1], satisfying DecayFunction's contract.
    SL-aware consumers call compute_opinion() for the full Opinion.

Requires: pip install chronofy[sl]  (depends on jsonld-ex)

References:
    Jøsang, A. (2016). Subjective Logic, §10.4 (Opinion Aging).
    Jøsang, A. (2016). Subjective Logic, §3.2 (Evidence-to-Opinion Mapping).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.confidence_decay import (
    decay_opinion,
    exponential_decay as _jex_exponential_decay,
)

from chronofy.decay.base import DecayFunction
from chronofy.models import TemporalFact


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OpinionConfig:
    """Per-fact-type configuration for opinion-based decay.

    Bundles half-life, base rate, and base uncertainty for a single
    fact type, enabling fine-grained domain configuration.

    Args:
        half_life: Days for belief/disbelief to halve.
        base_rate: Prior probability a ∈ [0, 1] (default 0.5 = uninformative).
        base_uncertainty: Minimum uncertainty in initial opinion construction
            when using "confidence" mode. Represents irreducible epistemic
            uncertainty even for perfectly fresh evidence.
    """

    half_life: float
    base_rate: float = 0.5
    base_uncertainty: float = 0.0


# Default configs parallel to ExponentialDecay's DEFAULT_BETA
DEFAULT_CONFIGS: dict[str, OpinionConfig] = {
    "vital_sign": OpinionConfig(half_life=0.14, base_rate=0.5),       # ~3.4 hours
    "lab_result": OpinionConfig(half_life=0.35, base_rate=0.5),       # ~8.4 hours
    "medication": OpinionConfig(half_life=0.69, base_rate=0.5),       # ~16.6 hours
    "diagnosis": OpinionConfig(half_life=1.39, base_rate=0.5),        # ~1.4 days
    "chronic_condition": OpinionConfig(half_life=69.3, base_rate=0.5),  # ~69 days
    "demographic": OpinionConfig(half_life=36500.0, base_rate=0.9),   # ~100 years
    "general": OpinionConfig(half_life=1.39, base_rate=0.5),
}

_VALID_CONSTRUCTIONS = {"confidence", "evidence"}


# ═══════════════════════════════════════════════════════════════════
# OpinionDecayFunction
# ═══════════════════════════════════════════════════════════════════


class OpinionDecayFunction(DecayFunction):
    """DecayFunction that tracks full Subjective Logic opinions.

    Constructs an initial opinion from each TemporalFact's source_quality,
    then applies Jøsang's aging operator to decay it by elapsed time.

    ``compute()`` returns the projected probability P(ω') for backward
    compatibility with all existing Chronofy components (EpistemicFilter,
    TemporalScorer, STLVerifier, TemporalRuleGraph).

    ``compute_opinion()`` returns the full Opinion for SL-aware consumers
    that need the richer (b, d, u, a) representation.

    Construction modes:
        "confidence" — Maps source_quality to Opinion via from_confidence().
            Treats source_quality as the projected probability.
            base_uncertainty controls irreducible prior uncertainty.
            Best for: scalar quality scores, ML model outputs.

        "evidence" — Maps source_quality to Opinion via from_evidence().
            Treats source_quality as an evidence ratio (higher = more
            positive evidence). Always has nonzero uncertainty.
            Best for: evidence-counted sources, clinical observations.

    Args:
        construction: "confidence" or "evidence" (default: "confidence").
        configs: Per-fact-type OpinionConfig dict. Overrides individual params.
        half_lives: Per-fact-type half-lives in days (convenience shorthand).
        default_half_life: Fallback half-life for unknown fact types.
        base_rates: Per-fact-type base rates (convenience shorthand).
        default_base_rate: Fallback base rate.
        base_uncertainty: Irreducible uncertainty for "confidence" mode.
        evidence_prior_weight: Prior weight W for "evidence" mode (Jøsang §3.2).
    """

    def __init__(
        self,
        construction: str = "confidence",
        configs: dict[str, OpinionConfig] | None = None,
        half_lives: dict[str, float] | None = None,
        default_half_life: float = 1.39,
        base_rates: dict[str, float] | None = None,
        default_base_rate: float = 0.5,
        base_uncertainty: float = 0.0,
        evidence_prior_weight: float = 2.0,
    ) -> None:
        if construction not in _VALID_CONSTRUCTIONS:
            raise ValueError(
                f"construction must be one of {_VALID_CONSTRUCTIONS}, "
                f"got {construction!r}"
            )

        self._construction = construction
        self._default_half_life = default_half_life
        self._default_base_rate = default_base_rate
        self._base_uncertainty = base_uncertainty
        self._evidence_prior_weight = evidence_prior_weight

        # Build unified config map: explicit configs > convenience params > defaults
        # Update DEFAULT_CONFIGS "general" entry to respect caller's defaults
        self._configs: dict[str, OpinionConfig] = dict(DEFAULT_CONFIGS)
        if default_base_rate != 0.5 or base_uncertainty != 0.0:
            general = self._configs["general"]
            self._configs["general"] = OpinionConfig(
                half_life=general.half_life,
                base_rate=default_base_rate,
                base_uncertainty=base_uncertainty,
            )

        if configs is not None:
            self._configs.update(configs)

        # Convenience params override individual fields
        if half_lives is not None:
            for ft, hl in half_lives.items():
                existing = self._configs.get(ft)
                if existing is not None:
                    self._configs[ft] = OpinionConfig(
                        half_life=hl,
                        base_rate=existing.base_rate,
                        base_uncertainty=existing.base_uncertainty,
                    )
                else:
                    self._configs[ft] = OpinionConfig(
                        half_life=hl,
                        base_rate=default_base_rate,
                        base_uncertainty=base_uncertainty,
                    )

        if base_rates is not None:
            for ft, br in base_rates.items():
                existing = self._configs.get(ft)
                if existing is not None:
                    self._configs[ft] = OpinionConfig(
                        half_life=existing.half_life,
                        base_rate=br,
                        base_uncertainty=existing.base_uncertainty,
                    )
                else:
                    self._configs[ft] = OpinionConfig(
                        half_life=default_half_life,
                        base_rate=br,
                        base_uncertainty=base_uncertainty,
                    )

    def _get_config(self, fact_type: str) -> OpinionConfig:
        """Resolve config for a fact type, falling back to defaults."""
        if fact_type in self._configs:
            return self._configs[fact_type]
        return OpinionConfig(
            half_life=self._default_half_life,
            base_rate=self._default_base_rate,
            base_uncertainty=self._base_uncertainty,
        )

    def _build_initial_opinion(
        self, fact: TemporalFact, config: OpinionConfig
    ) -> Opinion:
        """Construct the pre-decay opinion from a TemporalFact.

        Two modes:
        - "confidence": source_quality → from_confidence(q, uncertainty, base_rate)
        - "evidence": source_quality → from_evidence(positive, negative, W, base_rate)
        """
        q = fact.source_quality
        base_rate = config.base_rate
        u = max(config.base_uncertainty, self._base_uncertainty)

        if self._construction == "confidence":
            return Opinion.from_confidence(
                confidence=q,
                uncertainty=u,
                base_rate=base_rate,
            )
        else:
            # "evidence" mode: map quality to evidence counts
            # Higher quality → more positive evidence relative to negative
            # Scale so q=1.0 gives strong evidence, q=0.5 gives balanced
            W = self._evidence_prior_weight
            # Total evidence scales with quality — more quality = more observations
            total_evidence = q * 10.0  # 10 observations at max quality
            positive = total_evidence * q
            negative = total_evidence * (1.0 - q)
            return Opinion.from_evidence(
                positive=positive,
                negative=negative,
                prior_weight=W,
                base_rate=base_rate,
            )

    def compute_opinion(
        self, fact: TemporalFact, query_time: datetime
    ) -> Opinion:
        """Compute the full SL opinion for a fact at query time.

        1. Build initial opinion from source_quality and fact_type config.
        2. Compute elapsed time in days.
        3. Apply Jøsang's aging operator via jsonld-ex's decay_opinion().

        Returns:
            Opinion with temporal decay applied. Uncertainty increases
            with age while the b/d ratio is preserved.
        """
        config = self._get_config(fact.fact_type)
        initial = self._build_initial_opinion(fact, config)

        # Elapsed time in days
        elapsed_days = max(
            (query_time - fact.timestamp).total_seconds() / 86400.0,
            0.0,
        )

        if elapsed_days == 0.0:
            return initial

        # Apply Jøsang aging operator
        # half_life is in days, elapsed is in days — consistent units
        return decay_opinion(
            opinion=initial,
            elapsed=elapsed_days,
            half_life=config.half_life,
            decay_fn=_jex_exponential_decay,
        )

    def compute(self, fact: TemporalFact, query_time: datetime) -> float:
        """Compute scalar temporal validity — projected probability P(ω').

        This satisfies DecayFunction's contract: returns a float in [0, 1].
        The projected probability P(ω') = b' + a·u' collapses the full
        opinion to a scalar, distributing uncertainty mass according to
        the base rate.
        """
        return self.compute_opinion(fact, query_time).projected_probability()

    def compute_batch(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> list[float]:
        """Batch computation of scalar validity scores."""
        return [self.compute(f, query_time) for f in facts]

    def compute_opinion_batch(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> list[Opinion]:
        """Batch computation of full SL opinions."""
        return [self.compute_opinion(f, query_time) for f in facts]

    def get_beta(self, fact_type: str) -> float | None:
        """Return an approximate β equivalent for compatibility.

        Converts half-life to β via β = ln(2) / half_life, matching
        ExponentialDecay's parameterization.
        """
        import math

        config = self._get_config(fact_type)
        if config.half_life <= 0:
            return None
        return math.log(2) / config.half_life

    def __repr__(self) -> str:
        custom = {
            k: f"hl={v.half_life:.1f}d, a={v.base_rate}"
            for k, v in self._configs.items()
            if k not in DEFAULT_CONFIGS or v != DEFAULT_CONFIGS.get(k)
        }
        return (
            f"OpinionDecayFunction(mode={self._construction!r}, "
            f"custom={custom or 'defaults'})"
        )
