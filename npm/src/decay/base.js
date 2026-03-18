'use strict';

/**
 * Abstract base class for decay functions.
 * Subclass and implement compute() and computeBatch().
 */
class DecayFunction {
  /**
   * Compute validity score for a single fact.
   * @param {import('../models').TemporalFact} fact
   * @param {Date} queryTime
   * @returns {number} Validity score in [0, 1]
   */
  compute(fact, queryTime) {
    throw new Error(`${this.constructor.name} must implement compute()`);
  }

  /**
   * Compute validity scores for a batch of facts.
   * @param {import('../models').TemporalFact[]} facts
   * @param {Date} queryTime
   * @returns {number[]}
   */
  computeBatch(facts, queryTime) {
    return facts.map(f => this.compute(f, queryTime));
  }

  /**
   * Return the decay coefficient for a fact type, if applicable.
   * @param {string} factType
   * @returns {number|null}
   */
  getBeta(factType) {
    return null;
  }
}

module.exports = { DecayFunction };
