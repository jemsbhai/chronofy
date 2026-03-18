'use strict';

const { DecayFunction } = require('./base');

/**
 * Weibull decay: V = q · exp(-(age/scale)^shape)
 * shape > 1: accelerating decay. shape < 1: decelerating decay.
 */
class WeibullDecay extends DecayFunction {
  constructor({ scale = {}, shape = {}, defaultScale = 7.0,
                defaultShape = 1.0, timeUnit = 'days' } = {}) {
    super();
    this._scale = scale;
    this._shape = shape;
    this._defaultScale = defaultScale;
    this._defaultShape = defaultShape;
    const divisors = { seconds: 1000, hours: 3600000, days: 86400000 };
    if (!divisors[timeUnit]) throw new Error(`Unknown timeUnit: ${timeUnit}`);
    this._timeDivisor = divisors[timeUnit];
  }

  _getScale(factType) {
    return factType in this._scale ? this._scale[factType] : this._defaultScale;
  }

  _getShape(factType) {
    return factType in this._shape ? this._shape[factType] : this._defaultShape;
  }

  _ageInUnits(fact, queryTime) {
    return Math.max((queryTime.getTime() - fact.timestamp.getTime()) / this._timeDivisor, 0);
  }

  compute(fact, queryTime) {
    const scale = this._getScale(fact.factType);
    const shape = this._getShape(fact.factType);
    const age = this._ageInUnits(fact, queryTime);
    if (scale <= 0) return fact.sourceQuality;
    return fact.sourceQuality * Math.exp(-Math.pow(age / scale, shape));
  }

  toString() {
    return `WeibullDecay(defaultScale=${this._defaultScale}, defaultShape=${this._defaultShape})`;
  }
}

module.exports = { WeibullDecay };
