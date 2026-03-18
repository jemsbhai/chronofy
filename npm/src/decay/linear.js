'use strict';

const { DecayFunction } = require('./base');

/** Linear decay: V = q · max(0, 1 − rate · Δt). Hard expiry at 1/rate. */
class LinearDecay extends DecayFunction {
  constructor({ rate = {}, defaultRate = 0.1, timeUnit = 'days' } = {}) {
    super();
    this._rate = rate;
    this._defaultRate = defaultRate;
    const divisors = { seconds: 1000, hours: 3600000, days: 86400000 };
    if (!divisors[timeUnit]) throw new Error(`Unknown timeUnit: ${timeUnit}`);
    this._timeDivisor = divisors[timeUnit];
  }

  _getRate(factType) {
    return factType in this._rate ? this._rate[factType] : this._defaultRate;
  }

  _ageInUnits(fact, queryTime) {
    return Math.max((queryTime.getTime() - fact.timestamp.getTime()) / this._timeDivisor, 0);
  }

  compute(fact, queryTime) {
    const alpha = this._getRate(fact.factType);
    const age = this._ageInUnits(fact, queryTime);
    return fact.sourceQuality * Math.max(0, 1 - alpha * age);
  }

  /** Age at which validity reaches zero (with q=1). */
  expiryTime(factType) {
    const alpha = this._getRate(factType);
    return alpha > 0 ? 1 / alpha : Infinity;
  }

  toString() {
    return `LinearDecay(defaultRate=${this._defaultRate})`;
  }
}

module.exports = { LinearDecay };
