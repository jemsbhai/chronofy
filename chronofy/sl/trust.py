"""Trust-discounted temporal evidence via Subjective Logic.

Models source reliability as a full Opinion rather than a scalar multiplier,
absorbing ASEV Axioms 3-4:

  Axiom 3 (Reliability Monotonicity): higher-quality measurements have
    weakly greater ASEV at the same age.
  Axiom 4 (Blackwell Monotonicity): if channel A Blackwell-dominates B,
    then A has at least as much ASEV.

Pipeline: TemporalFact → temporal decay (OpinionDecayFunction) → trust
discount (Jøsang §14.3) → final Opinion.

Trust discount (⊗):
    Given trust opinion ω_AB about source B, and B's evidence opinion ω_Bx:
        b_Ax = b_AB · b_Bx
        d_Ax = b_AB · d_Bx
        u_Ax = d_AB + u_AB + b_AB · u_Bx

    Full trust (b_AB=1):  A adopts B's opinion unchanged.
    Zero trust (b_AB=0):  result is vacuous (total uncertainty).
    Partial trust:        dilutes evidence toward uncertainty.

Design notes (SOLID):
    - TrustProfile is a pure data container mapping source IDs → trust Opinions.
    - TrustWeightedDecay composes OpinionDecayFunction + TrustProfile via
      dependency injection. It does NOT subclass DecayFunction — it is a
      higher-level component that produces Opinions, not raw scalars.
    - Reuses jsonld-ex's trust_discount operator directly.

Requires: pip install chronofy[sl]

References:
    Jøsang (2016) §14.3 (Trust Discount), §14.5 (Trust Networks).
    Blackwell (1951, 1953): Informativeness ranking of experiments.
    ASEV Evaluation, Axioms 3-4.
"""

from __future__ import annotations

from datetime import datetime

from jsonld_ex.confidence_algebra import Opinion, trust_discount

from chronofy.models import TemporalFact
from chronofy.sl.opinion_decay import OpinionDecayFunction


# Vacuous opinion: total ignorance — the default for unknown sources.
_VACUOUS = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)


class TrustProfile:
    """Maps source identifiers to trust Opinions.

    A trust Opinion ω_AB = (b, d, u, a) represents agent A's trust in
    source B. High belief means high trust; high uncertainty means A
    has little evidence about B's reliability.

    Unknown sources receive the ``default_trust`` (vacuous by default),
    reflecting total ignorance about an unencountered source.

    Args:
        default_trust: Opinion assigned to sources not in the profile.
            Defaults to vacuous (u=1) — maximum epistemic caution.

    Example::

        tp = TrustProfile.from_scalars(
            {"hospital_lab": 0.95, "patient_report": 0.6},
            uncertainty=0.05,
        )
        tp.get_trust("hospital_lab")  # Opinion(b=0.9025, d=0.0475, u=0.05)
    """

    def __init__(self, default_trust: Opinion | None = None) -> None:
        self._trusts: dict[str, Opinion] = {}
        self._default = default_trust if default_trust is not None else _VACUOUS

    def set_trust(self, source_id: str, trust: Opinion) -> None:
        """Set the trust opinion for a source."""
        self._trusts[source_id] = trust

    def get_trust(self, source_id: str | None) -> Opinion:
        """Get the trust opinion for a source, falling back to default."""
        if source_id is None:
            return self._default
        return self._trusts.get(source_id, self._default)

    @property
    def sources(self) -> set[str]:
        """All explicitly registered source IDs."""
        return set(self._trusts.keys())

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._trusts

    def __len__(self) -> int:
        return len(self._trusts)

    @classmethod
    def from_scalars(
        cls,
        trust_scores: dict[str, float],
        uncertainty: float = 0.0,
        default_trust: Opinion | None = None,
    ) -> TrustProfile:
        """Convenience: build from scalar trust scores.

        Each scalar t ∈ [0, 1] is mapped to an Opinion via
        ``Opinion.from_confidence(t, uncertainty=uncertainty)``.

        Args:
            trust_scores: ``{source_id: trust_score}``.
            uncertainty: Base uncertainty for each trust opinion.
            default_trust: Fallback for unknown sources.
        """
        profile = cls(default_trust=default_trust)
        for source_id, score in trust_scores.items():
            profile.set_trust(
                source_id,
                Opinion.from_confidence(score, uncertainty=uncertainty),
            )
        return profile

    def __repr__(self) -> str:
        sources = ", ".join(
            f"{k}: b={v.belief:.2f}" for k, v in sorted(self._trusts.items())
        )
        return f"TrustProfile({sources})"


class TrustWeightedDecay:
    """Temporal decay + trust discount pipeline.

    Composes OpinionDecayFunction (temporal aging) with TrustProfile
    (source reliability) via Jøsang's trust discount operator.

    Pipeline for each fact:
        1. Decay by age → evidence Opinion ω_Bx
        2. Look up trust Opinion ω_AB for the fact's source
        3. Apply trust discount: ω_Ax = ω_AB ⊗ ω_Bx

    This is NOT a DecayFunction subclass — it is a higher-level
    component that produces Opinions. For consumers needing a scalar,
    use ``compute()`` which returns ``P(ω_Ax)``.

    Args:
        decay_fn: OpinionDecayFunction for temporal aging.
        trust_profile: Mapping from source IDs to trust Opinions.

    Example::

        odf = OpinionDecayFunction(half_lives={"vital_sign": 1.0})
        tp = TrustProfile.from_scalars({"lab": 0.95, "self_report": 0.5})
        twd = TrustWeightedDecay(decay_fn=odf, trust_profile=tp)

        fact = TemporalFact(content="K+=4.1", ..., source="lab")
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

        1. Temporal decay: fact → ω_Bx (evidence opinion aged by Δt)
        2. Trust lookup: source → ω_AB (trust in this source)
        3. Trust discount: ω_Ax = ω_AB ⊗ ω_Bx
        """
        evidence_opinion = self._decay_fn.compute_opinion(fact, query_time)
        trust_opinion = self._trust_profile.get_trust(fact.source)
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
