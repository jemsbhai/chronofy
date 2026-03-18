'use strict';

const { DecayFunction } = require('./base');
const { ExponentialDecay } = require('./exponential');
const { HalfLifeDecay } = require('./halfLife');
const { LinearDecay } = require('./linear');
const { PowerLawDecay } = require('./powerLaw');
const { WeibullDecay } = require('./weibull');

module.exports = {
  DecayFunction,
  ExponentialDecay,
  HalfLifeDecay,
  LinearDecay,
  PowerLawDecay,
  WeibullDecay,
};
