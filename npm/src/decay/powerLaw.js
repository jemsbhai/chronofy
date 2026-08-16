'use strict';

const { DecayFunction } = require('./base');
const {
  validateParameter,
  validateParameterMap,
  validateTimeUnit,
} = require('./validation');

/** Power-law decay: V = q · 1 / (1 + age)^exponent. Heavy-tailed. */
class PowerLawDecay extends DecayFunction {
  constructor({ exponent = {}, defaultExponent = 1.0, timeUnit = 'days' } = {}) {
    super();
    this._exponent = validateParameterMap(exponent, 'exponent', { positive: false });
    this._defaultExponent = validateParameter(
      defaultExponent, 'defaultExponent', { positive: false }
    );
    this._timeDivisor = validateTimeUnit(timeUnit);
  }

  _getExponent(factType) {
    return factType in this._exponent ? this._exponent[factType] : this._defaultExponent;
  }

  _ageInUnits(fact, queryTime) {
    return Math.max((queryTime.getTime() - fact.timestamp.getTime()) / this._timeDivisor, 0);
  }

  compute(fact, queryTime) {
    const exp = this._getExponent(fact.factType);
    const age = this._ageInUnits(fact, queryTime);
    return fact.sourceQuality / Math.pow(1 + age, exp);
  }

  toString() {
    return `PowerLawDecay(defaultExponent=${this._defaultExponent})`;
  }
}

module.exports = { PowerLawDecay };
