'use strict';

/** Abstract base for scoring strategies. */
class ScoringStrategy {
  /**
   * @param {number} similarity Semantic similarity ∈ [0, 1]
   * @param {number} validity   Temporal validity ∈ [0, 1]
   * @returns {number} Combined score ∈ [0, 1]
   */
  score(similarity, validity) {
    throw new Error(`${this.constructor.name} must implement score()`);
  }
}

/** sim × val — default; aggressive penalty for either dimension. */
class MultiplicativeScoring extends ScoringStrategy {
  score(similarity, validity) { return similarity * validity; }
}

/** 2·sim·val / (sim+val) — balanced; satisfies min ≤ H ≤ max. */
class HarmonicScoring extends ScoringStrategy {
  score(similarity, validity) {
    const d = similarity + validity;
    return d === 0 ? 0 : (2 * similarity * validity) / d;
  }
}

/** α·sim + (1-α)·val — explicit trade-off control. */
class WeightedBlendScoring extends ScoringStrategy {
  /** @param {number} alpha ∈ [0, 1] */
  constructor(alpha) {
    super();
    if (alpha < 0 || alpha > 1) throw new Error('alpha must be in [0, 1]');
    this._alpha = alpha;
  }
  get alpha() { return this._alpha; }
  score(similarity, validity) {
    return this._alpha * similarity + (1 - this._alpha) * validity;
  }
  toString() { return `WeightedBlendScoring(alpha=${this._alpha})`; }
}

/** sim^α · val^(1-α) — geometric blend; α=0.5 is geometric mean. */
class PowerScoring extends ScoringStrategy {
  /** @param {number} alpha ∈ [0, 1] */
  constructor(alpha) {
    super();
    if (alpha < 0 || alpha > 1) throw new Error('alpha must be in [0, 1]');
    this._alpha = alpha;
  }
  get alpha() { return this._alpha; }
  score(similarity, validity) {
    if (similarity <= 0 || validity <= 0) return 0;
    return Math.pow(similarity, this._alpha) * Math.pow(validity, 1 - this._alpha);
  }
  toString() { return `PowerScoring(alpha=${this._alpha})`; }
}

class ScoredFact {
  constructor({ fact, similarity, validity, combinedScore }) {
    this.fact = fact;
    this.similarity = similarity;
    this.validity = validity;
    this.combinedScore = combinedScore;
    Object.freeze(this);
  }
}

/**
 * Scores and ranks TemporalFacts by combining semantic similarity and
 * temporal validity via a pluggable ScoringStrategy.
 */
class TemporalScorer {
  /**
   * @param {object} opts
   * @param {import('../decay/base').DecayFunction} opts.decayFn
   * @param {ScoringStrategy} [opts.strategy] Defaults to MultiplicativeScoring
   */
  constructor({ decayFn, strategy = null }) {
    if (!decayFn) throw new Error('decayFn is required');
    this._decayFn = decayFn;
    this._strategy = strategy || new MultiplicativeScoring();
  }

  get strategy() { return this._strategy; }
  get decayFn()  { return this._decayFn; }

  /**
   * @param {import('../models').TemporalFact} fact
   * @param {number} similarity
   * @param {Date}   queryTime
   * @returns {ScoredFact}
   */
  scoreFact(fact, similarity, queryTime) {
    const validity = this._decayFn.compute(fact, queryTime);
    return new ScoredFact({
      fact, similarity, validity,
      combinedScore: this._strategy.score(similarity, validity),
    });
  }

  /**
   * @param {import('../models').TemporalFact[]} facts
   * @param {number[]} similarities
   * @param {Date}     queryTime
   * @returns {ScoredFact[]}
   */
  scoreFacts(facts, similarities, queryTime) {
    if (facts.length !== similarities.length)
      throw new Error('facts and similarities must have equal length');
    if (!facts.length) return [];
    const validities = this._decayFn.computeBatch(facts, queryTime);
    return facts.map((fact, i) => new ScoredFact({
      fact,
      similarity: similarities[i],
      validity: validities[i],
      combinedScore: this._strategy.score(similarities[i], validities[i]),
    }));
  }

  /**
   * Score and rank facts by combined score, descending.
   * @param {import('../models').TemporalFact[]} facts
   * @param {number[]} similarities
   * @param {Date}     queryTime
   * @param {number|null} [topK]
   * @returns {ScoredFact[]}
   */
  rank(facts, similarities, queryTime, topK = null) {
    if (topK !== null && topK <= 0) throw new Error('topK must be a positive integer');
    const scored = this.scoreFacts(facts, similarities, queryTime);
    if (!scored.length) return [];
    // Stable sort descending
    scored.sort((a, b) => b.combinedScore - a.combinedScore);
    return topK !== null ? scored.slice(0, topK) : scored;
  }

  toString() {
    return `TemporalScorer(strategy=${this._strategy})`;
  }
}

module.exports = {
  ScoringStrategy, MultiplicativeScoring, HarmonicScoring,
  WeightedBlendScoring, PowerScoring, ScoredFact, TemporalScorer,
};
