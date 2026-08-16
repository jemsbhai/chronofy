'use strict';

const { DecayFunction } = require('./base');
const {
  validateParameter,
  validateParameterMap,
  validateTimeUnit,
} = require('./validation');

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
    this._halfLife = validateParameterMap(halfLife, 'halfLife', { positive: true });
    this._defaultHalfLife = validateParameter(
      defaultHalfLife, 'defaultHalfLife', { positive: true }
    );
    this._timeDivisor = validateTimeUnit(timeUnit);
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
    return fact.sourceQuality * Math.pow(0.5, age / hl);
  }

  getBeta(factType) {
    const hl = this._getHalfLife(factType);
    return Math.LN2 / hl;
  }

  toString() {
    return `HalfLifeDecay(default=${this._defaultHalfLife})`;
  }
}

module.exports = { HalfLifeDecay };
