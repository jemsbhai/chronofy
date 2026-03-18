'use strict';

const {
  TemporalFact, ReasoningStep, ReasoningTrace,
  ExponentialDecay, HalfLifeDecay, LinearDecay, PowerLawDecay, WeibullDecay,
  EpistemicFilter,
  STLVerifier, STLResult,
  CorpusStats,
  BetaEstimator, MLEBernoulli, MomentMatching, EnsembleMethod,
  MultiplicativeScoring, HarmonicScoring, WeightedBlendScoring, PowerScoring,
  ScoredFact, TemporalScorer,
  DecayComparison,
  PluginValidationError, validateDecayFunction, validateEstimationMethod,
  validateScoringStrategy,
  DecayFunction, EstimationMethod, ScoringStrategy,
} = require('../index');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const QUERY = new Date('2024-06-01T12:00:00Z');
const daysAgo = (n) => new Date(QUERY.getTime() - n * 86400000);

function fact(ageDays, factType = 'general', q = 1.0) {
  return new TemporalFact({
    content: `fact-${ageDays}d`, timestamp: daysAgo(ageDays),
    factType, sourceQuality: q,
  });
}

// ---------------------------------------------------------------------------
// 1. TemporalFact
// ---------------------------------------------------------------------------
describe('TemporalFact', () => {
  test('ageAt returns correct days', () => {
    const f = fact(10);
    expect(f.ageAt(QUERY)).toBeCloseTo(10, 5);
  });
  test('ageAt is never negative', () => {
    const f = fact(0);
    const future = new Date(QUERY.getTime() - 1000);
    expect(f.ageAt(future)).toBeGreaterThanOrEqual(0);
  });
  test('throws on missing content', () => {
    expect(() => new TemporalFact({ content: '', timestamp: QUERY })).toThrow();
  });
  test('throws on invalid sourceQuality', () => {
    expect(() => new TemporalFact({
      content: 'x', timestamp: QUERY, sourceQuality: 1.5
    })).toThrow();
  });
  test('defaults factType to general', () => {
    expect(new TemporalFact({ content: 'x', timestamp: QUERY }).factType).toBe('general');
  });
});

// ---------------------------------------------------------------------------
// 2. ReasoningTrace.allFacts
// ---------------------------------------------------------------------------
describe('ReasoningTrace', () => {
  test('allFacts deduplicates by content', () => {
    const f = fact(1);
    const s1 = new ReasoningStep({ stepIndex: 0, content: 'step1', factsUsed: [f] });
    const s2 = new ReasoningStep({ stepIndex: 1, content: 'step2', factsUsed: [f] });
    const trace = new ReasoningTrace({ steps: [s1, s2], queryTime: QUERY });
    expect(trace.allFacts.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// 3. ExponentialDecay
// ---------------------------------------------------------------------------
describe('ExponentialDecay', () => {
  const decay = new ExponentialDecay({ beta: { general: 1.0 }, defaultBeta: 1.0 });

  test('age=0 returns sourceQuality', () => {
    expect(decay.compute(fact(0), QUERY)).toBeCloseTo(1.0, 9);
  });
  test('age=1 returns exp(-1)', () => {
    expect(decay.compute(fact(1), QUERY)).toBeCloseTo(Math.exp(-1), 6);
  });
  test('beta=0 returns sourceQuality regardless of age', () => {
    const d = new ExponentialDecay({ beta: { general: 0 }, defaultBeta: 0 });
    expect(d.compute(fact(1000), QUERY)).toBeCloseTo(1.0, 9);
  });
  test('halfLife returns ln2/beta', () => {
    expect(decay.halfLife('general')).toBeCloseTo(Math.LN2, 9);
  });
  test('halfLife returns null for beta=0', () => {
    const d = new ExponentialDecay({ beta: { general: 0 }, defaultBeta: 0 });
    expect(d.halfLife('general')).toBeNull();
  });
  test('fromMeanReversionRate sets beta=2kappa', () => {
    const d = ExponentialDecay.fromMeanReversionRate({ vital_sign: 2.5 });
    expect(d.getBeta('vital_sign')).toBeCloseTo(5.0, 9);
  });
  test('sourceQuality multiplied in', () => {
    const f = new TemporalFact({ content: 'x', timestamp: daysAgo(0), sourceQuality: 0.5 });
    expect(decay.compute(f, QUERY)).toBeCloseTo(0.5, 9);
  });
  test('computeBatch returns same length', () => {
    const facts = [fact(1), fact(5), fact(10)];
    expect(decay.computeBatch(facts, QUERY)).toHaveLength(3);
  });
});

// ---------------------------------------------------------------------------
// 4. LinearDecay
// ---------------------------------------------------------------------------
describe('LinearDecay', () => {
  const decay = new LinearDecay({ rate: { general: 0.1 }, defaultRate: 0.1 });

  test('age=0 returns 1.0', () => {
    expect(decay.compute(fact(0), QUERY)).toBeCloseTo(1.0);
  });
  test('reaches 0 at expiry', () => {
    expect(decay.compute(fact(10), QUERY)).toBeCloseTo(0.0, 5);
  });
  test('expiryTime returns 1/rate', () => {
    expect(decay.expiryTime('general')).toBeCloseTo(10, 9);
  });
});

// ---------------------------------------------------------------------------
// 5. WeibullDecay
// ---------------------------------------------------------------------------
describe('WeibullDecay', () => {
  test('shape=1 matches exponential', () => {
    const scale = 7.0;
    const w = new WeibullDecay({ scale: { general: scale }, shape: { general: 1.0 } });
    const e = new ExponentialDecay({ beta: { general: 1 / scale }, defaultBeta: 1 / scale });
    expect(w.compute(fact(3), QUERY)).toBeCloseTo(e.compute(fact(3), QUERY), 5);
  });
});

// ---------------------------------------------------------------------------
// 6. EpistemicFilter
// ---------------------------------------------------------------------------
describe('EpistemicFilter', () => {
  const decay = new ExponentialDecay({ beta: { general: 1.0 }, defaultBeta: 1.0 });
  const ep = new EpistemicFilter({ decayFn: decay, threshold: 0.1 });

  test('filter removes stale facts', () => {
    const facts = [fact(0), fact(100)];
    expect(ep.filter(facts, QUERY)).toHaveLength(1);
  });
  test('partition returns [fresh, stale]', () => {
    const [fresh, stale] = ep.partition([fact(0), fact(100)], QUERY);
    expect(fresh).toHaveLength(1);
    expect(stale).toHaveLength(1);
  });
  test('needsReacquisition true when all stale', () => {
    expect(ep.needsReacquisition([fact(100)], QUERY)).toBe(true);
  });
  test('needsReacquisition false when some fresh', () => {
    expect(ep.needsReacquisition([fact(0), fact(100)], QUERY)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// 7. STLVerifier
// ---------------------------------------------------------------------------
describe('STLVerifier', () => {
  const decay = new ExponentialDecay({ beta: { general: 1.0 }, defaultBeta: 1.0 });
  const verifier = new STLVerifier({ decayFn: decay, threshold: 0.5 });

  test('empty trace is satisfied', () => {
    const trace = new ReasoningTrace({ steps: [], queryTime: QUERY });
    const result = verifier.verify(trace);
    expect(result.satisfied).toBe(true);
  });
  test('step with no facts has validity 1.0', () => {
    const step = new ReasoningStep({ stepIndex: 0, content: 'x', factsUsed: [] });
    expect(verifier.stepValidity(step, QUERY)).toBeCloseTo(1.0);
  });
  test('fresh fact satisfies constraint', () => {
    const step = new ReasoningStep({ stepIndex: 0, content: 'x', factsUsed: [fact(0)] });
    const trace = new ReasoningTrace({ steps: [step], queryTime: QUERY });
    const result = verifier.verify(trace);
    expect(result.satisfied).toBe(true);
    expect(result.robustness).toBeGreaterThan(0);
  });
  test('stale fact violates constraint', () => {
    const step = new ReasoningStep({ stepIndex: 0, content: 'x', factsUsed: [fact(100)] });
    const trace = new ReasoningTrace({ steps: [step], queryTime: QUERY });
    const result = verifier.verify(trace);
    expect(result.satisfied).toBe(false);
    expect(result.robustness).toBeLessThan(0);
  });
  test('weakest-link bound is min validity', () => {
    const step = new ReasoningStep({
      stepIndex: 0, content: 'x', factsUsed: [fact(0), fact(100)]
    });
    const trace = new ReasoningTrace({ steps: [step], queryTime: QUERY });
    const result = verifier.verify(trace);
    const minV = decay.compute(fact(100), QUERY);
    expect(result.outputConfidenceBound).toBeCloseTo(minV, 6);
  });
  test('STLResult is frozen', () => {
    const trace = new ReasoningTrace({ steps: [], queryTime: QUERY });
    const result = verifier.verify(trace);
    expect(() => { result.satisfied = false; }).toThrow();
  });
});

// ---------------------------------------------------------------------------
// 8. CorpusStats
// ---------------------------------------------------------------------------
describe('CorpusStats', () => {
  const decay = new ExponentialDecay({ beta: { general: 0 }, defaultBeta: 0 });

  test('empty corpus returns 0 factCount', () => {
    const s = new CorpusStats({ facts: [], queryTime: QUERY, decayFn: decay });
    expect(s.factCount).toBe(0);
    expect(s.effectiveDensity).toBe(0);
    expect(s.ageStats).toBeNull();
  });
  test('correct mean age', () => {
    const facts = [fact(0), fact(2), fact(4), fact(6), fact(8)];
    const s = new CorpusStats({ facts, queryTime: QUERY, decayFn: decay });
    expect(s.ageStats.mean).toBeCloseTo(4.0, 5);
  });
  test('temporal span is max-min age', () => {
    const facts = [fact(0), fact(10)];
    const s = new CorpusStats({ facts, queryTime: QUERY, decayFn: decay });
    expect(s.temporalSpanDays).toBeCloseTo(10, 5);
  });
  test('stalenessRate negative threshold throws', () => {
    const s = new CorpusStats({ facts: [fact(1)], queryTime: QUERY, decayFn: decay });
    expect(() => s.stalenessRate(-0.1)).toThrow();
  });
  test('coverageGaps finds gap', () => {
    const facts = [fact(0), fact(10)];
    const s = new CorpusStats({ facts, queryTime: QUERY, decayFn: decay });
    const gaps = s.coverageGaps(5);
    expect(gaps).toHaveLength(1);
  });
  test('coverageGaps zero minGapDays throws', () => {
    const s = new CorpusStats({ facts: [fact(1)], queryTime: QUERY, decayFn: decay });
    expect(() => s.coverageGaps(0)).toThrow();
  });
  test('summary has required keys', () => {
    const s = new CorpusStats({ facts: [fact(1)], queryTime: QUERY, decayFn: decay });
    const keys = Object.keys(s.summary());
    expect(keys).toEqual(expect.arrayContaining(
      ['factCount', 'temporalSpanDays', 'effectiveDensity', 'ageStats', 'validityStats']
    ));
  });
});

// ---------------------------------------------------------------------------
// 9. BetaEstimator
// ---------------------------------------------------------------------------
describe('BetaEstimator', () => {
  const est = new BetaEstimator();

  test('throws on empty ages', () => {
    expect(() => est.fit({ ages: [], valid: [] })).toThrow();
  });
  test('throws on length mismatch', () => {
    expect(() => est.fit({ ages: [1, 2], valid: [true] })).toThrow();
  });
  test('all valid returns low beta', () => {
    const r = est.fit({ ages: [1, 10, 100], valid: [true, true, true] });
    expect(r.beta).toBeLessThan(0.1);
  });
  test('all invalid returns high beta', () => {
    const r = est.fit({ ages: [1, 1, 1], valid: [false, false, false] });
    expect(r.beta).toBeGreaterThan(1.0);
  });
  test('half life consistent with beta', () => {
    const r = est.fit({ ages: [1, 5, 10], valid: [true, true, false] });
    if (r.beta > 0)
      expect(r.halfLife).toBeCloseTo(Math.LN2 / r.beta, 6);
    else
      expect(r.halfLife).toBe(Infinity);
  });
  test('nObs and nValid correct', () => {
    const r = est.fit({ ages: [1, 2, 3], valid: [true, true, false] });
    expect(r.nObs).toBe(3);
    expect(r.nValid).toBe(2);
  });
  test('MomentMatching gives non-negative beta', () => {
    const e = new BetaEstimator({ method: new MomentMatching() });
    expect(e.fit({ ages: [1, 5, 10], valid: [true, false, false] }).beta).toBeGreaterThanOrEqual(0);
  });
  test('EnsembleMethod works', () => {
    const e = new BetaEstimator({
      method: new EnsembleMethod({ methods: [new MLEBernoulli(), new MomentMatching()] })
    });
    const r = e.fit({ ages: [1, 5, 10], valid: [true, true, false] });
    expect(r.beta).toBeGreaterThanOrEqual(0);
  });
  test('fitCorpus groups by factType', () => {
    const facts = [
      fact(1, 'fast'), fact(2, 'fast'), fact(50, 'slow'), fact(100, 'slow'),
    ];
    const labels = [true, false, true, false];
    const results = est.fitCorpus({ facts, labels, queryTime: QUERY });
    expect(Object.keys(results).sort()).toEqual(['fast', 'slow']);
    expect(results.fast.beta).toBeGreaterThan(results.slow.beta);
  });
  test('toDecay returns ExponentialDecay', () => {
    const results = { lab: est.fit({ ages: [1, 10], valid: [true, false], factType: 'lab' }) };
    const decay = est.toDecay(results);
    expect(decay).toBeInstanceOf(ExponentialDecay);
    expect(decay.getBeta('lab')).toBeCloseTo(results.lab.beta, 9);
  });
});

// ---------------------------------------------------------------------------
// 10. Scoring strategies
// ---------------------------------------------------------------------------
describe('ScoringStrategies', () => {
  test('MultiplicativeScoring: 0.8 * 0.5 = 0.4', () => {
    expect(new MultiplicativeScoring().score(0.8, 0.5)).toBeCloseTo(0.4);
  });
  test('HarmonicScoring: known value', () => {
    const expected = 2 * 0.8 * 0.5 / (0.8 + 0.5);
    expect(new HarmonicScoring().score(0.8, 0.5)).toBeCloseTo(expected);
  });
  test('HarmonicScoring: symmetric', () => {
    const h = new HarmonicScoring();
    expect(h.score(0.3, 0.7)).toBeCloseTo(h.score(0.7, 0.3));
  });
  test('HarmonicScoring: zero returns 0', () => {
    expect(new HarmonicScoring().score(0, 0.9)).toBe(0);
  });
  test('WeightedBlend alpha=1 is pure sim', () => {
    expect(new WeightedBlendScoring(1.0).score(0.7, 0.2)).toBeCloseTo(0.7);
  });
  test('WeightedBlend alpha=0 is pure validity', () => {
    expect(new WeightedBlendScoring(0.0).score(0.7, 0.2)).toBeCloseTo(0.2);
  });
  test('WeightedBlend alpha=0.5 is mean', () => {
    expect(new WeightedBlendScoring(0.5).score(0.6, 0.4)).toBeCloseTo(0.5);
  });
  test('WeightedBlend throws on bad alpha', () => {
    expect(() => new WeightedBlendScoring(1.5)).toThrow();
  });
  test('PowerScoring alpha=0.5 is geometric mean', () => {
    expect(new PowerScoring(0.5).score(0.64, 1.0)).toBeCloseTo(0.8);
  });
  test('PowerScoring zero returns 0', () => {
    expect(new PowerScoring(0.5).score(0, 0.9)).toBe(0);
  });
  test('PowerScoring throws on bad alpha', () => {
    expect(() => new PowerScoring(-0.1)).toThrow();
  });
});

// ---------------------------------------------------------------------------
// 11. TemporalScorer
// ---------------------------------------------------------------------------
describe('TemporalScorer', () => {
  const decay = new ExponentialDecay({ beta: { general: 1.0 }, defaultBeta: 1.0 });
  const scorer = new TemporalScorer({ decayFn: decay });

  test('scoreFact returns ScoredFact', () => {
    const sf = scorer.scoreFact(fact(1), 0.8, QUERY);
    expect(sf).toBeInstanceOf(ScoredFact);
  });
  test('scoreFact validity matches decay', () => {
    const f = fact(1);
    const sf = scorer.scoreFact(f, 0.8, QUERY);
    expect(sf.validity).toBeCloseTo(decay.compute(f, QUERY), 9);
  });
  test('scoreFact combined = sim * validity (multiplicative default)', () => {
    const f = fact(1);
    const sf = scorer.scoreFact(f, 0.8, QUERY);
    expect(sf.combinedScore).toBeCloseTo(0.8 * decay.compute(f, QUERY), 9);
  });
  test('scoreFacts throws on length mismatch', () => {
    expect(() => scorer.scoreFacts([fact(1)], [0.8, 0.9], QUERY)).toThrow();
  });
  test('scoreFacts returns empty for empty input', () => {
    expect(scorer.scoreFacts([], [], QUERY)).toHaveLength(0);
  });
  test('rank sorts descending', () => {
    const facts = [fact(1), fact(5), fact(30)];
    const sims = [0.9, 0.7, 0.5];
    const ranked = scorer.rank(facts, sims, QUERY);
    const scores = ranked.map(r => r.combinedScore);
    expect(scores).toEqual([...scores].sort((a, b) => b - a));
  });
  test('rank topK limits output', () => {
    const facts = [fact(1), fact(5), fact(30)];
    expect(scorer.rank(facts, [0.9, 0.7, 0.5], QUERY, 2)).toHaveLength(2);
  });
  test('rank topK=0 throws', () => {
    expect(() => scorer.rank([fact(1)], [0.8], QUERY, 0)).toThrow();
  });
  test('ScoredFact is frozen', () => {
    const sf = scorer.scoreFact(fact(1), 0.8, QUERY);
    expect(() => { sf.combinedScore = 0.99; }).toThrow();
  });
});

// ---------------------------------------------------------------------------
// 12. DecayComparison
// ---------------------------------------------------------------------------
describe('DecayComparison', () => {
  const facts = [fact(0), fact(0.5), fact(50), fact(200)];
  const labels = [true, true, false, false];
  const decayFns = {
    fast: new ExponentialDecay({ beta: { general: 1.0 }, defaultBeta: 1.0 }),
    none: new ExponentialDecay({ beta: { general: 0.0 }, defaultBeta: 0.0 }),
  };

  test('throws on empty decayFns', () => {
    expect(() => new DecayComparison({ decayFns: {}, facts, labels, queryTime: QUERY })).toThrow();
  });
  test('throws on empty facts', () => {
    expect(() => new DecayComparison({ decayFns, facts: [], labels: [], queryTime: QUERY })).toThrow();
  });
  test('throws on length mismatch', () => {
    expect(() => new DecayComparison({ decayFns, facts, labels: [true], queryTime: QUERY })).toThrow();
  });
  test('compare returns one result per decay fn', () => {
    const dc = new DecayComparison({ decayFns, facts, labels, queryTime: QUERY });
    expect(dc.compare()).toHaveLength(2);
  });
  test('compare sorted by auc_roc descending', () => {
    const dc = new DecayComparison({ decayFns, facts, labels, queryTime: QUERY });
    const aucs = dc.compare().map(r => r.auc_roc);
    expect(aucs).toEqual([...aucs].sort((a, b) => b - a));
  });
  test('fast decay has higher auc than no decay', () => {
    const dc = new DecayComparison({ decayFns, facts, labels, queryTime: QUERY });
    const fast = dc.compare().find(r => r.name === 'fast');
    const none = dc.compare().find(r => r.name === 'none');
    expect(fast.auc_roc).toBeGreaterThan(none.auc_roc);
  });
  test('no decay gives auc~0.5', () => {
    const dc = new DecayComparison({ decayFns, facts, labels, queryTime: QUERY });
    const none = dc.compare().find(r => r.name === 'none');
    expect(none.auc_roc).toBeCloseTo(0.5, 1);
  });
  test('best auc_roc returns highest', () => {
    const dc = new DecayComparison({ decayFns, facts, labels, queryTime: QUERY });
    const best = dc.best('auc_roc');
    const maxAuc = Math.max(...dc.compare().map(r => r.auc_roc));
    expect(best.auc_roc).toBeCloseTo(maxAuc, 9);
  });
  test('best unknown metric throws', () => {
    const dc = new DecayComparison({ decayFns, facts, labels, queryTime: QUERY });
    expect(() => dc.best('made_up')).toThrow();
  });
  test('summary has required keys', () => {
    const dc = new DecayComparison({ decayFns, facts, labels, queryTime: QUERY });
    const s = dc.summary();
    expect(Object.keys(s)).toEqual(expect.arrayContaining(
      ['nFacts', 'nValid', 'nDecayFns', 'bestByAuc', 'bestByBrier', 'results']
    ));
  });
  test('DecayComparisonResult is frozen', () => {
    const dc = new DecayComparison({ decayFns, facts, labels, queryTime: QUERY });
    const r = dc.compare()[0];
    expect(() => { r.auc_roc = 0.99; }).toThrow();
  });
});

// ---------------------------------------------------------------------------
// 13. Plugin validation
// ---------------------------------------------------------------------------
describe('Plugin validation', () => {
  class GoodDecay extends DecayFunction {
    compute(f, qt) { return f.sourceQuality * Math.exp(-0.1 * f.ageAt(qt)); }
    computeBatch(fs, qt) { return fs.map(f => this.compute(f, qt)); }
  }
  class GoodEstimator extends EstimationMethod {
    fit(ages, valid) { return 0.5; }
  }
  class GoodScorer extends ScoringStrategy {
    score(sim, val) { return sim * val; }
  }

  test('valid decay passes', () => {
    expect(validateDecayFunction(new GoodDecay())).toBeInstanceOf(GoodDecay);
  });
  test('valid estimator passes', () => {
    expect(validateEstimationMethod(new GoodEstimator())).toBeInstanceOf(GoodEstimator);
  });
  test('valid scorer passes', () => {
    expect(validateScoringStrategy(new GoodScorer())).toBeInstanceOf(GoodScorer);
  });
  test('built-in ExponentialDecay passes', () => {
    expect(() => validateDecayFunction(new ExponentialDecay())).not.toThrow();
  });
  test('built-in MLEBernoulli passes', () => {
    expect(() => validateEstimationMethod(new MLEBernoulli())).not.toThrow();
  });
  test('built-in MultiplicativeScoring passes', () => {
    expect(() => validateScoringStrategy(new MultiplicativeScoring())).not.toThrow();
  });
  test('decay returning >1 throws PluginValidationError', () => {
    class Bad extends DecayFunction {
      compute() { return 1.5; }
      computeBatch(fs) { return fs.map(() => 1.5); }
    }
    expect(() => validateDecayFunction(new Bad())).toThrow(PluginValidationError);
  });
  test('decay returning NaN throws', () => {
    class Bad extends DecayFunction {
      compute() { return NaN; }
      computeBatch(fs) { return fs.map(() => NaN); }
    }
    expect(() => validateDecayFunction(new Bad())).toThrow(PluginValidationError);
  });
  test('estimator returning negative beta throws', () => {
    class Bad extends EstimationMethod { fit() { return -1; } }
    expect(() => validateEstimationMethod(new Bad())).toThrow(PluginValidationError);
  });
  test('scorer returning >1 throws', () => {
    class Bad extends ScoringStrategy { score() { return 1.5; } }
    expect(() => validateScoringStrategy(new Bad())).toThrow(PluginValidationError);
  });
  test('non-DecayFunction throws', () => {
    expect(() => validateDecayFunction('not a decay')).toThrow(PluginValidationError);
  });
  test('validators return object unchanged on success', () => {
    const d = new GoodDecay();
    expect(validateDecayFunction(d)).toBe(d);
  });
});
