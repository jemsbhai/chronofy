'use strict';

const { DecayFunction } = require('./base');

/** Half-life parameterised exponential: β = ln(2) / halfLife */
class HalfLifeDecay extends DecayFunction {
  /**
   * @param {object} [opts]
   * @param {Object.<string,number>} [opts.halfLife]        fact type → half-life in days
   * @param {number}                 [opts.defaultHalfLife]
   * @param {string}                 [opts.timeUnit]
   */
  constructor({ halfLife = {}, defaultHalfLife = 1.0, timeUnit = 'days' } = {}) {
    super();
    this._halfLife = halfLife;
    this._defaultHalfLife = defaultHalfLife;
    const divisors = { seconds: 1000, hours: 3600000, days: 86400000 };
    if (!divisors[timeUnit]) throw new Error(`Unknown timeUnit: ${timeUnit}`);
    this._timeDivisor = divisors[timeUnit];
  }

  _getHalfLife(factType) {
    return factType in this._halfLife ? this._halfLife[factType] : this._defaultHalfLife;
  }

  _ageInUnits(fact, queryTime) {
    return Math.max((queryTime.getTime() - fact.timestamp.getTime()) / this._timeDivisor, 0);
  }

  compute(fact, queryTime) {
    const hl = this._getHalfLife(fact.factType);
    const age = this._ageInUnits(fact, queryTime);
    if (hl <= 0) return fact.sourceQuality;
    const beta = Math.LN2 / hl;
    return fact.sourceQuality * Math.exp(-beta * age);
  }

  getBeta(factType) {
    const hl = this._getHalfLife(factType);
    return hl > 0 ? Math.LN2 / hl : 0;
  }

  toString() {
    return `HalfLifeDecay(default=${this._defaultHalfLife})`;
  }
}

module.exports = { HalfLifeDecay };
