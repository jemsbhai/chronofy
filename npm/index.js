'use strict';

/**
 * Chronofy — Temporal validity framework for time-aware AI systems.
 * JavaScript port of chronofy v0.1.6 (Python).
 *
 * @see https://github.com/jemsbhai/chronofy
 */

const { TemporalFact, ReasoningStep, ReasoningTrace } = require('./src/models');
const { DecayFunction } = require('./src/decay/base');
const { ExponentialDecay } = require('./src/decay/exponential');
const { HalfLifeDecay }    = require('./src/decay/halfLife');
const { LinearDecay }      = require('./src/decay/linear');
const { PowerLawDecay }    = require('./src/decay/powerLaw');
const { WeibullDecay }     = require('./src/decay/weibull');
const { EpistemicFilter }  = require('./src/retrieval/filter');
const { STLVerifier, STLResult } = require('./src/verification/stl');
const { CorpusStats, AgeStats, ValidityStats } = require('./src/analysis/corpusStats');
const {
  EstimationMethod, MLEBernoulli, MomentMatching, EnsembleMethod,
  BetaEstimateResult, BetaEstimator,
} = require('./src/analysis/betaEstimator');
const {
  ScoringStrategy, MultiplicativeScoring, HarmonicScoring,
  WeightedBlendScoring, PowerScoring, ScoredFact, TemporalScorer,
} = require('./src/scoring/temporalScorer');
const { DecayComparison, DecayComparisonResult } = require('./src/comparison/decayComparison');
const {
  PluginValidationError,
  validateDecayFunction,
  validateEstimationMethod,
  validateScoringStrategy,
} = require('./src/plugins');

module.exports = {
  // Models
  TemporalFact, ReasoningStep, ReasoningTrace,
  // Decay
  DecayFunction, ExponentialDecay, HalfLifeDecay,
  LinearDecay, PowerLawDecay, WeibullDecay,
  // Retrieval
  EpistemicFilter,
  // Verification
  STLVerifier, STLResult,
  // Analysis
  CorpusStats, AgeStats, ValidityStats,
  EstimationMethod, MLEBernoulli, MomentMatching, EnsembleMethod,
  BetaEstimateResult, BetaEstimator,
  // Scoring
  ScoringStrategy, MultiplicativeScoring, HarmonicScoring,
  WeightedBlendScoring, PowerScoring, ScoredFact, TemporalScorer,
  // Comparison
  DecayComparison, DecayComparisonResult,
  // Plugins
  PluginValidationError,
  validateDecayFunction, validateEstimationMethod, validateScoringStrategy,
};
