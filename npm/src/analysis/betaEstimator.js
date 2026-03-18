'use strict';

const { ExponentialDecay } = require('../decay/exponential');

// Golden-section search to maximise a unimodal function on [lo, hi]
function _goldenSectionMax(f, lo, hi, tol = 1e-8) {
  const phi = (Math.sqrt(5) - 1) / 2;
  let a = lo, b = hi;
  let c = b - phi * (b - a), d = a + phi * (b - a);
  let fc = f(c), fd = f(d);
  while (Math.abs(b - a) > tol) {
    if (fc < fd) { a = c; c = d; fc = fd; d = a + phi * (b - a); fd = f(d); }
    else         { b = d; d = c; fd = fc; c = b - phi * (b - a); fc = f(c); }
  }
  const xMax = (a + b) / 2;
  return [xMax, f(xMax)];
}

// Bernoulli log-likelihood
function _logLikelihood(beta, ages, valid) {
  let ll = 0;
  for (let i = 0; i < ages.length; i++) {
    const p = Math.exp(-beta * ages[i]);
    ll += valid[i] ? Math.log(Math.max(p, 1e-300))
                   : Math.log(Math.max(1 - p, 1e-300));
  }
  return ll;
}

const MAX_BETA = 50;

/** Abstract base for estimation methods. */
class EstimationMethod {
  /**
   * @param {number[]} ages
   * @param {boolean[]} valid
   * @returns {number} beta >= 0
   */
  fit(ages, valid) {
    throw new Error(`${this.constructor.name} must implement fit()`);
  }
}

/** MLE under P(valid | age, β) = exp(-β · age). */
class MLEBernoulli extends EstimationMethod {
  fit(ages, valid) {
    const nValid = valid.filter(Boolean).length;
    const nInvalid = valid.length - nValid;
    if (nInvalid === 0) return 0;
    if (nValid === 0) return MAX_BETA;
    const [beta] = _goldenSectionMax(b => _logLikelihood(b, ages, valid), 0, MAX_BETA);
    return Math.max(0, beta);
  }
}

/** Method of moments: β = -log(p_valid) / mean_age */
class MomentMatching extends EstimationMethod {
  fit(ages, valid) {
    const n = ages.length;
    const pValid = valid.filter(Boolean).length / n;
    if (pValid >= 1) return 0;
    const meanAge = ages.reduce((a, b) => a + b, 0) / n;
    if (pValid <= 0 || meanAge <= 0) return MAX_BETA;
    return Math.min(Math.max(-Math.log(pValid) / meanAge, 0), MAX_BETA);
  }
}

/** Weighted average of multiple EstimationMethod backends. */
class EnsembleMethod extends EstimationMethod {
  /**
   * @param {object} opts
   * @param {EstimationMethod[]} opts.methods
   * @param {number[]} [opts.weights]
   */
  constructor({ methods, weights = null }) {
    super();
    if (!methods || methods.length === 0)
      throw new Error('EnsembleMethod requires at least one method');
    if (weights !== null) {
      if (weights.length !== methods.length)
        throw new Error('weights length must match methods length');
      if (weights.some(w => w <= 0))
        throw new Error('All weights must be strictly positive');
      const total = weights.reduce((a, b) => a + b, 0);
      this._weights = weights.map(w => w / total);
    } else {
      this._weights = methods.map(() => 1 / methods.length);
    }
    this._methods = methods;
  }

  fit(ages, valid) {
    const betas = this._methods.map(m => m.fit(ages, valid));
    return betas.reduce((s, b, i) => s + this._weights[i] * b, 0);
  }
}

class BetaEstimateResult {
  constructor({ beta, halfLife, logLikelihood, nObs, nValid, converged, factType }) {
    this.beta = beta;
    this.halfLife = halfLife;
    this.logLikelihood = logLikelihood;
    this.nObs = nObs;
    this.nValid = nValid;
    this.converged = converged;
    this.factType = factType;
    Object.freeze(this);
  }

  toDict() {
    return { beta: this.beta, halfLife: this.halfLife,
             logLikelihood: this.logLikelihood, nObs: this.nObs,
             nValid: this.nValid, converged: this.converged,
             factType: this.factType };
  }
}

/**
 * Fits decay coefficients β from empirical (age, valid) observations.
 * Accepts any EstimationMethod backend, defaulting to MLEBernoulli.
 */
class BetaEstimator {
  /** @param {object} [opts]
   *  @param {EstimationMethod} [opts.method]
   */
  constructor({ method = null } = {}) {
    this._method = method || new MLEBernoulli();
  }

  /**
   * @param {object} opts
   * @param {number[]}  opts.ages
   * @param {boolean[]} opts.valid
   * @param {string}    [opts.factType]
   * @returns {BetaEstimateResult}
   */
  fit({ ages, valid, factType = 'general' }) {
    if (!ages || ages.length === 0) throw new Error('ages must not be empty');
    if (ages.length !== valid.length)
      throw new Error('ages and valid must have equal length');
    const beta = Math.max(0, this._method.fit(ages, valid));
    const halfLife = beta > 0 ? Math.LN2 / beta : Infinity;
    const ll = _logLikelihood(beta, ages, valid);
    return new BetaEstimateResult({
      beta, halfLife, logLikelihood: ll,
      nObs: ages.length, nValid: valid.filter(Boolean).length,
      converged: true, factType,
    });
  }

  /**
   * Fit β per factType from a labelled TemporalFact corpus.
   * @param {object} opts
   * @param {import('../models').TemporalFact[]} opts.facts
   * @param {boolean[]} opts.labels
   * @param {Date}      opts.queryTime
   * @returns {Object.<string, BetaEstimateResult>}
   */
  fitCorpus({ facts, labels, queryTime }) {
    if (!facts || facts.length === 0) throw new Error('facts must not be empty');
    if (facts.length !== labels.length)
      throw new Error('facts and labels must have equal length');
    const groups = {};
    for (let i = 0; i < facts.length; i++) {
      const ft = facts[i].factType;
      if (!groups[ft]) groups[ft] = { ages: [], valid: [] };
      groups[ft].ages.push(facts[i].ageAt(queryTime));
      groups[ft].valid.push(labels[i]);
    }
    const results = {};
    for (const [ft, { ages, valid }] of Object.entries(groups)) {
      results[ft] = this.fit({ ages, valid, factType: ft });
    }
    return results;
  }

  /**
   * Convert fit results into a ready-to-use ExponentialDecay.
   * @param {Object.<string, BetaEstimateResult>} results
   * @param {number} [defaultBeta]
   * @returns {ExponentialDecay}
   */
  toDecay(results, defaultBeta = 0.5) {
    const beta = {};
    for (const [ft, r] of Object.entries(results)) beta[ft] = r.beta;
    return new ExponentialDecay({ beta, defaultBeta });
  }
}

module.exports = {
  EstimationMethod, MLEBernoulli, MomentMatching, EnsembleMethod,
  BetaEstimateResult, BetaEstimator,
};
