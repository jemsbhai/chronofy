'use strict';

/**
 * EpistemicFilter — threshold gate for temporally valid facts.
 *
 * Facts below validity threshold τ are structurally excluded from the
 * context window delivered to the LLM.
 *
 * Property (Retrieval Completeness): Under threshold τ, the retrieved
 * context excludes all facts with validity < τ, guaranteeing the LLM
 * never receives evidence below the minimum validity floor.
 */
class EpistemicFilter {
  /**
   * @param {object} opts
   * @param {import('../decay/base').DecayFunction} opts.decayFn
   * @param {number} [opts.threshold] Minimum validity τ ∈ [0, 1]
   */
  constructor({ decayFn, threshold = 0.1 }) {
    if (!decayFn) throw new Error('decayFn is required');
    if (threshold < 0 || threshold > 1)
      throw new Error('threshold must be in [0, 1]');
    this._decayFn = decayFn;
    this._threshold = threshold;
  }

  get threshold() { return this._threshold; }

  /**
   * Return only facts with validity >= threshold.
   * @param {import('../models').TemporalFact[]} facts
   * @param {Date} queryTime
   * @returns {import('../models').TemporalFact[]}
   */
  filter(facts, queryTime) {
    return facts.filter(f => this._decayFn.compute(f, queryTime) >= this._threshold);
  }

  /**
   * Partition facts into [fresh, stale].
   * @param {import('../models').TemporalFact[]} facts
   * @param {Date} queryTime
   * @returns {[import('../models').TemporalFact[], import('../models').TemporalFact[]]}
   */
  partition(facts, queryTime) {
    const fresh = [], stale = [];
    for (const f of facts) {
      if (this._decayFn.compute(f, queryTime) >= this._threshold) fresh.push(f);
      else stale.push(f);
    }
    return [fresh, stale];
  }

  /**
   * Returns true if any fact type in the corpus has all facts below threshold.
   * Signals that fresh data acquisition is needed.
   * @param {import('../models').TemporalFact[]} facts
   * @param {Date} queryTime
   * @returns {boolean}
   */
  needsReacquisition(facts, queryTime) {
    if (facts.length === 0) return false;
    return facts.every(f => this._decayFn.compute(f, queryTime) < this._threshold);
  }
}

module.exports = { EpistemicFilter };
