'use strict';

const { DecayFunction } = require('./base');

/**
 * Default β values grounded in clinical domain knowledge (β = 2κ).
 */
const DEFAULT_BETA = {
  vital_sign:        5.0,
  lab_result:        2.0,
  medication:        1.0,
  diagnosis:         0.5,
  chronic_condition: 0.01,
  demographic:       0.0,
  general:           0.5,
};

/**
 * Exponential temporal decay: V(e, T_q) = q_e · exp(-β_j · Δt)
 *
 * Decision-theoretic grounding (Proposition 1):
 *   Under an Ornstein-Uhlenbeck latent process with mean-reversion rate κ,
 *   the optimal decay coefficient is β = 2κ.
 */
class ExponentialDecay extends DecayFunction {
  /**
   * @param {object} [opts]
   * @param {Object.<string,number>} [opts.beta]        fact type → β
   * @param {number}                 [opts.defaultBeta] fallback β
   * @param {string}                 [opts.timeUnit]    'days'|'hours'|'seconds'
   */
  constructor({ beta = {}, defaultBeta = 0.5, timeUnit = 'days' } = {}) {
    super();
    this._beta = Object.assign({}, DEFAULT_BETA, beta);
    this._defaultBeta = defaultBeta;
    const divisors = { seconds: 1000, hours: 3600000, days: 86400000 };
    if (!divisors[timeUnit]) throw new Error(`Unknown timeUnit: ${timeUnit}`);
    this._timeDivisor = divisors[timeUnit];
  }

  _getBetaForType(factType) {
    return factType in this._beta ? this._beta[factType] : this._defaultBeta;
  }

  _ageInUnits(fact, queryTime) {
    const deltaMs = queryTime.getTime() - fact.timestamp.getTime();
    return Math.max(deltaMs / this._timeDivisor, 0);
  }

  compute(fact, queryTime) {
    const beta = this._getBetaForType(fact.factType);
    const age = this._ageInUnits(fact, queryTime);
    if (beta === 0) return fact.sourceQuality;
    return fact.sourceQuality * Math.exp(-beta * age);
  }

  getBeta(factType) {
    return this._getBetaForType(factType);
  }

  /**
   * Half-life in the configured time unit: ln(2) / β.
   * Returns null for invariant fact types (β = 0).
   * @param {string} factType
   * @returns {number|null}
   */
  halfLife(factType) {
    const beta = this._getBetaForType(factType);
    if (beta <= 0) return null;
    return Math.LN2 / beta;
  }

  /**
   * Construct from mean-reversion rates κ, using β = 2κ (Proposition 1).
   * @param {Object.<string,number>} kappa
   * @param {object} [opts]
   * @returns {ExponentialDecay}
   */
  static fromMeanReversionRate(kappa, opts = {}) {
    const beta = {};
    for (const [k, v] of Object.entries(kappa)) beta[k] = 2 * v;
    return new ExponentialDecay({ ...opts, beta });
  }

  toString() {
    const entries = Object.entries(this._beta)
      .filter(([, v]) => v > 0)
      .map(([k, v]) => `${k}=${v.toFixed(2)}`)
      .join(', ');
    return `ExponentialDecay(${entries})`;
  }
}

module.exports = { ExponentialDecay, DEFAULT_BETA };
