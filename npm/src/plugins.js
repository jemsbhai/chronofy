'use strict';

const { DecayFunction } = require('./decay/base');
const { EstimationMethod } = require('./analysis/betaEstimator');
const { ScoringStrategy } = require('./scoring/temporalScorer');

class PluginValidationError extends Error {
  constructor(message) { super(message); this.name = 'PluginValidationError'; }
}

const PROBE_FACTS_RAW = [
  { ageDays: 0, q: 1.0 }, { ageDays: 1, q: 1.0 },
  { ageDays: 30, q: 0.8 }, { ageDays: 365, q: 0.5 },
];
const PROBE_QUERY = new Date('2024-01-01T12:00:00Z');

function _makeProbeFactObj(ageDays, q) {
  // Minimal duck-type TemporalFact for probing
  const ts = new Date(PROBE_QUERY.getTime() - ageDays * 86400000);
  return { content: `probe`, timestamp: ts, factType: 'general', sourceQuality: q,
           ageAt: (qt) => Math.max((qt.getTime() - ts.getTime()) / 86400000, 0) };
}

const PROBE_FACTS = PROBE_FACTS_RAW.map(({ ageDays, q }) => _makeProbeFactObj(ageDays, q));
const PROBE_AGES  = [1.0, 5.0, 10.0, 30.0];
const PROBE_VALID = [true, true, false, false];
const PROBE_PAIRS = [
  [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.8, 0.3], [0.3, 0.8], [0.6, 0.9],
];

function validateDecayFunction(fn) {
  const cls = fn && fn.constructor ? fn.constructor.name : String(fn);
  if (!(fn instanceof DecayFunction))
    throw new PluginValidationError(
      `'${cls}' is not an instance of DecayFunction. ` +
      `Custom decay functions must extend chronofy/src/decay/base.DecayFunction.`
    );
  if (typeof fn.compute !== 'function')
    throw new PluginValidationError(`${cls}.compute() is missing or not a function`);

  for (const probe of PROBE_FACTS) {
    let score;
    try { score = fn.compute(probe, PROBE_QUERY); }
    catch (e) { throw new PluginValidationError(`${cls}.compute() threw: ${e.message}`); }
    if (typeof score !== 'number')
      throw new PluginValidationError(
        `${cls}.compute() must return a numeric value, got ${typeof score}`);
    if (!isFinite(score))
      throw new PluginValidationError(`${cls}.compute() returned non-finite (NaN/Inf)`);
    if (score < 0)
      throw new PluginValidationError(`${cls}.compute() returned ${score} (negative)`);
    if (score > 1)
      throw new PluginValidationError(`${cls}.compute() returned ${score} (above 1.0)`);
  }

  if (typeof fn.computeBatch !== 'function')
    throw new PluginValidationError(`${cls}.computeBatch() is missing or not a function`);

  const batch = fn.computeBatch(PROBE_FACTS, PROBE_QUERY);
  if (!Array.isArray(batch) || batch.length !== PROBE_FACTS.length)
    throw new PluginValidationError(
      `${cls}.computeBatch() must return an array of the same length as input`);

  return fn;
}

function validateEstimationMethod(method) {
  const cls = method && method.constructor ? method.constructor.name : String(method);
  if (!(method instanceof EstimationMethod))
    throw new PluginValidationError(
      `'${cls}' is not an instance of EstimationMethod.`);
  if (typeof method.fit !== 'function')
    throw new PluginValidationError(`${cls}.fit() is missing or not a function`);

  let beta;
  try { beta = method.fit(PROBE_AGES, PROBE_VALID); }
  catch (e) { throw new PluginValidationError(`${cls}.fit() threw: ${e.message}`); }

  if (typeof beta !== 'number')
    throw new PluginValidationError(
      `${cls}.fit() must return a numeric float, got ${typeof beta}`);
  if (!isFinite(beta))
    throw new PluginValidationError(`${cls}.fit() returned non-finite (NaN/Inf)`);
  if (beta < 0)
    throw new PluginValidationError(`${cls}.fit() returned β=${beta} (negative)`);

  return method;
}

function validateScoringStrategy(strategy) {
  const cls = strategy && strategy.constructor ? strategy.constructor.name : String(strategy);
  if (!(strategy instanceof ScoringStrategy))
    throw new PluginValidationError(
      `'${cls}' is not an instance of ScoringStrategy.`);
  if (typeof strategy.score !== 'function')
    throw new PluginValidationError(`${cls}.score() is missing or not a function`);

  for (const [sim, val] of PROBE_PAIRS) {
    let result;
    try { result = strategy.score(sim, val); }
    catch (e) {
      throw new PluginValidationError(
        `${cls}.score() threw for (${sim}, ${val}): ${e.message}`);
    }
    if (typeof result !== 'number')
      throw new PluginValidationError(
        `${cls}.score() must return a numeric float, got ${typeof result}`);
    if (!isFinite(result))
      throw new PluginValidationError(
        `${cls}.score() returned non-finite (NaN/Inf) for (${sim}, ${val})`);
    if (result < 0)
      throw new PluginValidationError(
        `${cls}.score() returned ${result} (negative) for (${sim}, ${val})`);
    if (result > 1)
      throw new PluginValidationError(
        `${cls}.score() returned ${result} (above 1.0) for (${sim}, ${val})`);
  }
  return strategy;
}

module.exports = {
  PluginValidationError,
  validateDecayFunction,
  validateEstimationMethod,
  validateScoringStrategy,
};
