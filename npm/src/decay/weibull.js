'use strict';

const { DecayFunction } = require('./base');
const {
  validateParameter,
  validateParameterMap,
  validateTimeUnit,
} = require('./validation');

/**
 * Weibull decay: V = q · exp(-(age/scale)^shape)
 * shape > 1: accelerating decay. shape < 1: decelerating decay.
 */
class WeibullDecay extends DecayFunction {
  constructor({ scale = {}, shape = {}, defaultScale = 7.0,
                defaultShape = 1.0, timeUnit = 'days' } = {}) {
    super();
    this._scale = validateParameterMap(scale, 'scale', { positive: true });
    this._shape = validateParameterMap(shape, 'shape', { positive: true });
    this._defaultScale = validateParameter(defaultScale, 'defaultScale', { positive: true });
    this._defaultShape = validateParameter(defaultShape, 'defaultShape', { positive: true });
    this._timeDivisor = validateTimeUnit(timeUnit);
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
    return fact.sourceQuality * Math.exp(-Math.pow(age / scale, shape));
  }

  toString() {
    return `WeibullDecay(defaultScale=${this._defaultScale}, defaultShape=${this._defaultShape})`;
  }
}

module.exports = { WeibullDecay };
