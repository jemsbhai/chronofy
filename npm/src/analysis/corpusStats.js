'use strict';

/** Pure-JS descriptive stat helpers */
function _mean(xs) { return xs.reduce((a, b) => a + b, 0) / xs.length; }

function _std(xs) {
  if (xs.length <= 1) return 0;
  const mu = _mean(xs);
  return Math.sqrt(xs.reduce((s, x) => s + (x - mu) ** 2, 0) / xs.length);
}

function _percentile(xs, p) {
  const sorted = [...xs].sort((a, b) => a - b);
  const n = sorted.length;
  if (n === 1) return sorted[0];
  const idx = (p / 100) * (n - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] * (1 - (idx - lo)) + sorted[hi] * (idx - lo);
}

class AgeStats {
  constructor({ min, max, mean, median, std, p25, p75 }) {
    Object.assign(this, { min, max, mean, median, std, p25, p75 });
    Object.freeze(this);
  }
  toDict() {
    return { min: this.min, max: this.max, mean: this.mean,
             median: this.median, std: this.std, p25: this.p25, p75: this.p75 };
  }
}

class ValidityStats {
  constructor({ min, max, mean, median, std, p25, p75 }) {
    Object.assign(this, { min, max, mean, median, std, p25, p75 });
    Object.freeze(this);
  }
  toDict() {
    return { min: this.min, max: this.max, mean: this.mean,
             median: this.median, std: this.std, p25: this.p25, p75: this.p75 };
  }
}

/**
 * Temporal health statistics for a collection of TemporalFacts.
 *
 * Answers: "What is the temporal health of this fact corpus at query time?"
 */
class CorpusStats {
  /**
   * @param {object} opts
   * @param {import('../models').TemporalFact[]} opts.facts
   * @param {Date}   opts.queryTime
   * @param {import('../decay/base').DecayFunction} opts.decayFn
   */
  constructor({ facts, queryTime, decayFn }) {
    this._facts = facts;
    this._queryTime = queryTime;
    this._decayFn = decayFn;
    this._ages = null;
    this._validities = null;
    this._ageStats = undefined;
    this._validityStats = undefined;
  }

  get factCount() { return this._facts.length; }

  _getAges() {
    if (!this._ages)
      this._ages = this._facts.map(f => f.ageAt(this._queryTime));
    return this._ages;
  }

  _getValidities() {
    if (!this._validities)
      this._validities = this._decayFn.computeBatch(this._facts, this._queryTime);
    return this._validities;
  }

  get temporalSpanDays() {
    const ages = this._getAges();
    if (ages.length < 2) return 0;
    return Math.max(...ages) - Math.min(...ages);
  }

  get effectiveDensity() {
    const vals = this._getValidities();
    if (!vals.length) return 0;
    return _mean(vals);
  }

  get ageStats() {
    if (this._ageStats === undefined) {
      const ages = this._getAges();
      if (!ages.length) { this._ageStats = null; return null; }
      this._ageStats = new AgeStats({
        min: Math.min(...ages), max: Math.max(...ages),
        mean: _mean(ages), median: _percentile(ages, 50),
        std: _std(ages), p25: _percentile(ages, 25), p75: _percentile(ages, 75),
      });
    }
    return this._ageStats;
  }

  get validityStats() {
    if (this._validityStats === undefined) {
      const vals = this._getValidities();
      if (!vals.length) { this._validityStats = null; return null; }
      this._validityStats = new ValidityStats({
        min: Math.min(...vals), max: Math.max(...vals),
        mean: _mean(vals), median: _percentile(vals, 50),
        std: _std(vals), p25: _percentile(vals, 25), p75: _percentile(vals, 75),
      });
    }
    return this._validityStats;
  }

  /**
   * Fraction of facts with validity strictly below threshold.
   * @param {number} threshold Must be >= 0
   * @returns {number}
   */
  stalenessRate(threshold) {
    if (threshold < 0) throw new Error('threshold must be >= 0');
    const vals = this._getValidities();
    if (!vals.length) return 0;
    return vals.filter(v => v < threshold).length / vals.length;
  }

  /**
   * Temporal silent periods between consecutive fact timestamps.
   * @param {number} minGapDays Must be > 0
   * @returns {Array<[Date, Date]>}
   */
  coverageGaps(minGapDays) {
    if (minGapDays <= 0) throw new Error('minGapDays must be > 0');
    if (this._facts.length < 2) return [];
    const timestamps = [...new Set(this._facts.map(f => f.timestamp.getTime()))]
      .sort((a, b) => a - b)
      .map(t => new Date(t));
    if (timestamps.length < 2) return [];
    const minMs = minGapDays * 86400000;
    const gaps = [];
    for (let i = 0; i < timestamps.length - 1; i++) {
      const gap = timestamps[i + 1].getTime() - timestamps[i].getTime();
      if (gap >= minMs) gaps.push([timestamps[i], timestamps[i + 1]]);
    }
    return gaps;
  }

  summary() {
    return {
      factCount: this.factCount,
      temporalSpanDays: this.temporalSpanDays,
      effectiveDensity: this.effectiveDensity,
      ageStats: this.ageStats ? this.ageStats.toDict() : null,
      validityStats: this.validityStats ? this.validityStats.toDict() : null,
    };
  }
}

module.exports = { CorpusStats, AgeStats, ValidityStats };
