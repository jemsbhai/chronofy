"""Trust-discounted temporal evidence via Subjective Logic.

Models source reliability as a full Opinion rather than a scalar multiplier,
absorbing ASEV Axioms 3-4:

  Axiom 3 (Reliability Monotonicity): higher-quality measurements have
    weakly greater ASEV at the same age.
  Axiom 4 (Blackwell Monotonicity): if channel A Blackwell-dominates B,
    then A has at least as much ASEV.

Two independent decay channels:
    ω_Ax = decay(ω_AB, age_of_trust) ⊗ decay(ω_Bx, age_of_evidence)
             ↑ trust decays                ↑ evidence decays

Pipeline: TemporalFact → temporal decay (OpinionDecayFunction) →
    trust decay (Jøsang aging on trust opinion) → trust discount (§14.3)
    → final Opinion.

Trust discount (⊗):
    Given trust opinion ω_AB about source B, and B's evidence opinion ω_Bx:
        b_Ax = b_AB · b_Bx
        d_Ax = b_AB · d_Bx
        u_Ax = d_AB + u_AB + b_AB · u_Bx

    Full trust (b_AB=1):  A adopts B's opinion unchanged.
    Zero trust (b_AB=0):  result is vacuous (total uncertainty).
    Partial trust:        dilutes evidence toward uncertainty.

Trust aging (Jøsang §10.4 applied to trust):
    A lab certification from 5 years ago is less trustworthy than one
    from yesterday. Trust opinions decay by migrating mass into uncertainty,
    using a configurable decay function (exponential, linear, step, etc.).

Design notes (SOLID):
    - TrustEntry bundles opinion + temporal metadata (SRP).
    - TrustProfile stores TrustEntries, applies trust decay on lookup (SRP).
    - TrustWeightedDecay composes evidence decay + trust decay + discount (SRP).
    - Configurable trust_decay_fn via dependency injection (DIP/OCP).
    - Reuses jsonld-ex's trust_discount and decay_opinion directly.

Requires: pip install chronofy[sl]

References:
    Jøsang (2016) §14.3 (Trust Discount), §14.5 (Trust Networks),
    §10.4 (Opinion Aging).
    Blackwell (1951, 1953): Informativeness ranking of experiments.
    ASEV Evaluation, Axioms 3-4.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from jsonld_ex.confidence_algebra import Opinion, trust_discount
from jsonld_ex.confidence_decay import (
    DecayFunction as JexDecayFunction,
    decay_opinion,
    exponential_decay as _jex_exponential_decay,
)

from chronofy.models import TemporalFact
from chronofy.sl.opinion_decay import OpinionDecayFunction


# Vacuous opinion: total ignorance — the default for unknown sources.
_VACUOUS = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)


# ═══════════════════════════════════════════════════════════════════
# TrustEntry
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TrustEntry:
    """A trust opinion bundled with optional temporal metadata.

    When ``timestamp`` and ``half_life`` are provided, the trust opinion
    itself decays over time — a lab certified 5 years ago is less
    trustworthy than one certified yesterday.

    Args:
        opinion: The trust Opinion at the time it was established.
        timestamp: When the trust assessment was made (None = static/eternal).
        half_life: Days for the trust opinion's belief/disbelief to halve
            (None = no trust decay, even if timestamp is set).
    """

    opinion: Opinion
    timestamp: datetime | None = None
    half_life: float | None = None

    @property
    def is_temporal(self) -> bool:
        """Whether this entry has temporal decay information."""
        return self.timestamp is not None and self.half_life is not None


# ═══════════════════════════════════════════════════════════════════
# TrustProfile
# ═══════════════════════════════════════════════════════════════════


class TrustProfile:
    """Maps source identifiers to trust Opinions with optional temporal decay.

    A trust Opinion ω_AB = (b, d, u, a) represents agent A's trust in
    source B. High belief means high trust; high uncertainty means A
    has little evidence about B's reliability.

    When trust entries carry temporal metadata (timestamp + half_life),
    the trust opinion decays on lookup via ``get_trust(source, query_time)``.
    This models the reality that certifications, calibrations, and
    reputation assessments lose validity over time.

    Unknown sources receive the ``default_trust`` (vacuous by default),
    reflecting total ignorance about an unencountered source.

    Args:
        default_trust: Opinion assigned to sources not in the profile.
            Defaults to vacuous (u=1) — maximum epistemic caution.
        trust_decay_fn: Decay function for aging trust opinions.
            Signature: ``(elapsed_days, half_life_days) -> factor in [0, 1]``.
            Defaults to exponential decay. Can use any function from
            ``jsonld_ex.confidence_decay`` (linear_decay, step_decay, etc.).
    """

    def __init__(
        self,
        default_trust: Opinion | None = None,
        trust_decay_fn: JexDecayFunction | None = None,
    ) -> None:
        self._entries: dict[str, TrustEntry] = {}
        self._default = default_trust if default_trust is not None else _VACUOUS
        self._trust_decay_fn = trust_decay_fn or _jex_exponential_decay

    def set_trust(
        self,
        source_id: str,
        trust: Opinion,
        timestamp: datetime | None = None,
        half_life: float | None = None,
    ) -> None:
        """Set the trust opinion for a source.

        Args:
            source_id: Source identifier.
            trust: Trust opinion at the time of assessment.
            timestamp: When the trust was established (None = static).
            half_life: Days for trust belief/disbelief to halve (None = no decay).
        """
        self._entries[source_id] = TrustEntry(
            opinion=trust,
            timestamp=timestamp,
            half_life=half_life,
        )

    def get_trust(
        self,
        source_id: str | None,
        query_time: datetime | None = None,
    ) -> Opinion:
        """Get the trust opinion for a source, optionally decayed by age.

        If the entry has temporal metadata and ``query_time`` is provided,
        the trust opinion is decayed by the elapsed time since it was
        established. Otherwise, the raw opinion is returned.

        Args:
            source_id: Source to look up. None returns default.
            query_time: Reference time for trust decay. None skips decay.

        Returns:
            Trust Opinion (possibly decayed).
        """
        if source_id is None:
            return self._default

        entry = self._entries.get(source_id)
        if entry is None:
            return self._default

        # Static entry or no query_time: return raw opinion
        if not entry.is_temporal or query_time is None:
            return entry.opinion

        # Temporal entry: decay trust by age
        elapsed_days = max(
            (query_time - entry.timestamp).total_seconds() / 86400.0,
            0.0,
        )

        if elapsed_days == 0.0:
            return entry.opinion

        return decay_opinion(
            opinion=entry.opinion,
            elapsed=elapsed_days,
            half_life=entry.half_life,
            decay_fn=self._trust_decay_fn,
        )

    @property
    def sources(self) -> set[str]:
        """All explicitly registered source IDs."""
        return set(self._entries.keys())

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    @classmethod
    def from_scalars(
        cls,
        trust_scores: dict[str, float],
        uncertainty: float = 0.0,
        default_trust: Opinion | None = None,
        trust_decay_fn: JexDecayFunction | None = None,
        timestamps: dict[str, datetime] | None = None,
        half_lives: dict[str, float] | None = None,
    ) -> TrustProfile:
        """Convenience: build from scalar trust scores with optional temporal info.

        Each scalar t ∈ [0, 1] is mapped to an Opinion via
        ``Opinion.from_confidence(t, uncertainty=uncertainty)``.

        Args:
            trust_scores: ``{source_id: trust_score}``.
            uncertainty: Base uncertainty for each trust opinion.
            default_trust: Fallback for unknown sources.
            trust_decay_fn: Custom decay function for trust aging.
            timestamps: ``{source_id: datetime}`` — when trust was established.
            half_lives: ``{source_id: float}`` — trust half-life in days.
        """
        profile = cls(default_trust=default_trust, trust_decay_fn=trust_decay_fn)
        timestamps = timestamps or {}
        half_lives = half_lives or {}

        for source_id, score in trust_scores.items():
            profile.set_trust(
                source_id,
                Opinion.from_confidence(score, uncertainty=uncertainty),
                timestamp=timestamps.get(source_id),
                half_life=half_lives.get(source_id),
            )
        return profile

    def __repr__(self) -> str:
        sources = ", ".join(
            f"{k}: b={v.opinion.belief:.2f}"
            + (f" (t={v.is_temporal})" if v.is_temporal else "")
            for k, v in sorted(self._entries.items())
        )
        return f"TrustProfile({sources})"


# ═══════════════════════════════════════════════════════════════════
# TrustWeightedDecay
# ═══════════════════════════════════════════════════════════════════


class TrustWeightedDecay:
    """Temporal decay + trust decay + trust discount pipeline.

    Composes OpinionDecayFunction (evidence aging) with TrustProfile
    (source reliability, itself subject to aging) via Jøsang's trust
    discount operator.

    Full pipeline for each fact:
        1. Decay evidence by age → ω_Bx
        2. Decay trust by age → ω_AB (via TrustProfile with query_time)
        3. Trust discount: ω_Ax = ω_AB ⊗ ω_Bx

    This gives two independent decay channels:
        ω_Ax = decay(ω_AB, age_of_trust) ⊗ decay(ω_Bx, age_of_evidence)

    This is NOT a DecayFunction subclass — it is a higher-level
    component that produces Opinions. For consumers needing a scalar,
    use ``compute()`` which returns ``P(ω_Ax)``.

    Args:
        decay_fn: OpinionDecayFunction for evidence aging.
        trust_profile: Mapping from source IDs to trust Opinions
            (with optional temporal decay on the trust itself).

    Example::

        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        tp = TrustProfile.from_scalars(
            {"lab": 0.95}, timestamps={"lab": last_calibration},
            half_lives={"lab": 365.0},
        )
        twd = TrustWeightedDecay(decay_fn=odf, trust_profile=tp)
        opinion = twd.compute_opinion(fact, query_time)
    """

    def __init__(
        self,
        decay_fn: OpinionDecayFunction,
        trust_profile: TrustProfile,
    ) -> None:
        self._decay_fn = decay_fn
        self._trust_profile = trust_profile

    @property
    def trust_profile(self) -> TrustProfile:
        return self._trust_profile

    def compute_opinion(
        self, fact: TemporalFact, query_time: datetime
    ) -> Opinion:
        """Compute trust-discounted, temporally-decayed Opinion.

        1. Evidence decay: fact → ω_Bx (aged by evidence Δt)
        2. Trust decay: source → ω_AB (aged by trust Δt via TrustProfile)
        3. Trust discount: ω_Ax = ω_AB ⊗ ω_Bx
        """
        evidence_opinion = self._decay_fn.compute_opinion(fact, query_time)
        trust_opinion = self._trust_profile.get_trust(
            fact.source, query_time=query_time
        )
        return trust_discount(trust_opinion, evidence_opinion)

    def compute(self, fact: TemporalFact, query_time: datetime) -> float:
        """Scalar projected probability of the trust-discounted opinion."""
        return self.compute_opinion(fact, query_time).projected_probability()

    def compute_batch(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> list[float]:
        """Batch scalar computation."""
        return [self.compute(f, query_time) for f in facts]

    def compute_opinion_batch(
        self, facts: list[TemporalFact], query_time: datetime
    ) -> list[Opinion]:
        """Batch opinion computation."""
        return [self.compute_opinion(f, query_time) for f in facts]

    def __repr__(self) -> str:
        return (
            f"TrustWeightedDecay("
            f"decay_fn={self._decay_fn!r}, "
            f"trust_profile={self._trust_profile!r})"
        )
