'use strict';

/**
 * Signal Temporal Logic verification for knowledge temporal validity.
 *
 * THIS IS THE HEADLINE NOVELTY OF CHRONOFY.
 *
 * Applies STL robustness to the temporal validity of retrieved knowledge —
 * not to LLM output confidence. Catches cases where the LLM is confidently
 * wrong because it retrieved plausible but temporally stale facts.
 *
 * STL specification: φ_valid = G[0,n]( v(s_i) ≥ γ )
 * Robustness:        ρ = min_i ( v(s_i) − γ )
 *
 * Theorem 1 (Weakest-Link Bound):
 *   C_out ≤ min_i V(e_i, T_q)
 *   Follows from min-aggregation in possibilistic logic.
 */

class STLResult {
  constructor({ robustness, satisfied, stepValidity, weakestStepIndex,
                weakestFact, outputConfidenceBound }) {
    this.robustness = robustness;
    this.satisfied = satisfied;
    this.stepValidity = stepValidity;
    this.weakestStepIndex = weakestStepIndex;
    this.weakestFact = weakestFact;
    this.outputConfidenceBound = outputConfidenceBound;
    Object.freeze(this);
  }
}

class STLVerifier {
  /**
   * @param {object} opts
   * @param {import('../decay/base').DecayFunction} opts.decayFn
   * @param {number} [opts.threshold] Minimum validity γ ∈ [0, 1]
   */
  constructor({ decayFn, threshold = 0.5 }) {
    if (!decayFn) throw new Error('decayFn is required');
    if (threshold < 0 || threshold > 1)
      throw new Error('threshold must be in [0, 1]');
    this._decayFn = decayFn;
    this._threshold = threshold;
  }

  get threshold() { return this._threshold; }

  /**
   * Compute v(s_i) = min over facts used at step i of V(e, T_q).
   * Returns 1.0 (vacuous truth) if the step uses no facts.
   * @param {import('../models').ReasoningStep} step
   * @param {Date} queryTime
   * @returns {number}
   */
  stepValidity(step, queryTime) {
    if (!step.factsUsed || step.factsUsed.length === 0) return 1.0;
    const scores = this._decayFn.computeBatch(step.factsUsed, queryTime);
    return Math.min(...scores);
  }

  /**
   * Verify a complete reasoning trace against φ_valid = G[0,n](v(s_i) ≥ γ).
   * @param {import('../models').ReasoningTrace} trace
   * @returns {STLResult}
   */
  verify(trace) {
    if (!trace.steps || trace.steps.length === 0) {
      return new STLResult({
        robustness: 0.0,
        satisfied: true,
        stepValidity: [],
        weakestStepIndex: null,
        weakestFact: null,
        outputConfidenceBound: 1.0,
      });
    }

    // Per-step validity signals
    const stepVals = trace.steps.map(s => this.stepValidity(s, trace.queryTime));

    // Robustness ρ = min_i (v(s_i) − γ)
    const margins = stepVals.map(v => v - this._threshold);
    const rho = Math.min(...margins);
    const weakestIdx = margins.indexOf(Math.min(...margins));

    // Weakest-link bound: global min over ALL facts
    let globalMin = Infinity;
    let globalWeakestFact = null;
    for (const step of trace.steps) {
      for (const fact of (step.factsUsed || [])) {
        const score = this._decayFn.compute(fact, trace.queryTime);
        if (score < globalMin) {
          globalMin = score;
          globalWeakestFact = fact;
        }
      }
    }
    if (globalMin === Infinity) globalMin = 1.0;

    return new STLResult({
      robustness: rho,
      satisfied: rho >= 0,
      stepValidity: stepVals,
      weakestStepIndex: weakestIdx,
      weakestFact: globalWeakestFact,
      outputConfidenceBound: globalMin,
    });
  }

  /**
   * Convenience: return just the scalar robustness score ρ.
   * @param {import('../models').ReasoningTrace} trace
   * @returns {number}
   */
  robustness(trace) {
    return this.verify(trace).robustness;
  }

  toString() {
    return `STLVerifier(threshold=${this._threshold})`;
  }
}

module.exports = { STLVerifier, STLResult };
