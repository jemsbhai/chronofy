'use strict';

const {
  TemporalFact,
  ExponentialDecay,
  HalfLifeDecay,
  LinearDecay,
  PowerLawDecay,
  WeibullDecay,
  PowerScoring,
} = require('../index');

const QUERY_TIME = new Date('2026-03-15T00:00:00Z');

function fact(daysAgo, sourceQuality = 0.4) {
  return new TemporalFact({
    content: 'configuration probe',
    timestamp: new Date(QUERY_TIME.getTime() - daysAgo * 86400000),
    factType: 'probe',
    sourceQuality,
  });
}

const nonnegativeConfigs = [
  ['beta', value => new ExponentialDecay({ beta: { probe: value } })],
  ['defaultBeta', value => new ExponentialDecay({ defaultBeta: value })],
  ['rate', value => new LinearDecay({ rate: { probe: value } })],
  ['defaultRate', value => new LinearDecay({ defaultRate: value })],
  ['exponent', value => new PowerLawDecay({ exponent: { probe: value } })],
  ['defaultExponent', value => new PowerLawDecay({ defaultExponent: value })],
];

const positiveConfigs = [
  ['halfLife', value => new HalfLifeDecay({ halfLife: { probe: value } })],
  ['defaultHalfLife', value => new HalfLifeDecay({ defaultHalfLife: value })],
  ['scale', value => new WeibullDecay({ scale: { probe: value } })],
  ['defaultScale', value => new WeibullDecay({ defaultScale: value })],
  ['shape', value => new WeibullDecay({ shape: { probe: value } })],
  ['defaultShape', value => new WeibullDecay({ defaultShape: value })],
];

describe('decay configuration validation', () => {
  test.each(nonnegativeConfigs)('%s rejects invalid values', (_name, factory) => {
    for (const value of [-0.1, NaN, Infinity, -Infinity, true, '0.1', null]) {
      expect(() => factory(value)).toThrow(/must be a finite nonnegative number/);
    }
  });

  test.each(positiveConfigs)('%s rejects invalid values', (_name, factory) => {
    for (const value of [0, -0.1, NaN, Infinity, -Infinity, true, '1', null]) {
      expect(() => factory(value)).toThrow(/must be a finite positive number/);
    }
  });

  test.each([
    ExponentialDecay,
    HalfLifeDecay,
    LinearDecay,
    PowerLawDecay,
    WeibullDecay,
  ])('%p reports the shared timeUnit error', DecayClass => {
    expect(() => new DecayClass({ timeUnit: 'weeks' })).toThrow(
      "timeUnit must be one of 'seconds', 'hours', or 'days', got \"weeks\"."
    );
  });

  test.each([
    new ExponentialDecay({ beta: { probe: Number.MAX_VALUE } }),
    new HalfLifeDecay({ halfLife: { probe: Number.MIN_VALUE } }),
    new LinearDecay({ rate: { probe: Number.MAX_VALUE } }),
    new PowerLawDecay({ exponent: { probe: Number.MAX_VALUE } }),
    new WeibullDecay({ scale: { probe: 1e-300 }, shape: { probe: Number.MAX_VALUE } }),
  ])('%p keeps valid extreme configuration outputs bounded', decay => {
    for (const daysAgo of [-1, 0, 1, 1000]) {
      const score = decay.compute(fact(daysAgo), QUERY_TIME);
      expect(Number.isFinite(score)).toBe(true);
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    }
  });

  test('zero is valid for beta, rate, and exponent', () => {
    const decays = [
      new ExponentialDecay({ beta: { probe: 0 } }),
      new LinearDecay({ rate: { probe: 0 } }),
      new PowerLawDecay({ exponent: { probe: 0 } }),
    ];
    for (const decay of decays) {
      expect(decay.compute(fact(1000), QUERY_TIME)).toBeCloseTo(0.4);
    }
  });

  test.each([-1, NaN, Infinity, true, '0.1'])(
    'mean-reversion rate %p is rejected', value => {
      expect(() => ExponentialDecay.fromMeanReversionRate({ probe: value })).toThrow(
        /kappa for fact type 'probe'/
      );
    }
  );
});

describe('PowerScoring endpoint semantics', () => {
  test.each([NaN, Infinity, -Infinity, true, '0.5'])(
    'rejects invalid alpha %p', alpha => {
      expect(() => new PowerScoring(alpha)).toThrow(/finite number in \[0, 1\]/);
    }
  );

  test.each([
    [1, 0, 0, 0],
    [1, 0, 0.8, 0],
    [1, 0.7, 0, 0.7],
    [1, 0.7, 0.8, 0.7],
    [0, 0, 0, 0],
    [0, 0, 0.8, 0.8],
    [0, 0.7, 0, 0],
    [0, 0.7, 0.8, 0.8],
    [0.5, 0, 0.8, 0],
    [0.5, 0.7, 0, 0],
  ])(
    'alpha=%s score(%s, %s) is %s',
    (alpha, similarity, validity, expected) => {
      expect(new PowerScoring(alpha).score(similarity, validity)).toBeCloseTo(expected);
    }
  );
});
