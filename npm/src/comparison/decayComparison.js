'use strict';

const LOG_CLIP = 1e-9;

const METRICS = {
  auc_roc:             false,  // higher is better
  brier_score:         true,   // lower is better
  log_loss:            true,   // lower is better
  ranking_correlation: false,  // higher is better
};

// --- Pure-JS metric implementations ---

function _auroc(scores, labels) {
  const pos = scores.filter((_, i) => labels[i] === 1);
  const neg = scores.filter((_, i) => labels[i] === 0);
  if (!pos.length || !neg.length) return 0.5;
  let concordant = 0;
  for (const p of pos) for (const n of neg) {
    if (p > n) concordant += 1;
    else if (p === n) concordant += 0.5;
  }
  return concordant / (pos.length * neg.length);
}

function _brierScore(scores, labels) {
  if (!scores.length) return 0;
  return scores.reduce((s, sc, i) => s + (sc - labels[i]) ** 2, 0) / scores.length;
}

function _logLoss(scores, labels) {
  if (!scores.length) return 0;
  let total = 0;
  for (let i = 0; i < scores.length; i++) {
    const p = Math.min(Math.max(scores[i], LOG_CLIP), 1 - LOG_CLIP);
    total -= labels[i] === 1 ? Math.log(p) : Math.log(1 - p);
  }
  return total / scores.length;
}

function _rank(xs) {
  const n = xs.length;
  const indexed = [...Array(n).keys()].sort((a, b) => xs[a] - xs[b]);
  const ranks = new Array(n).fill(0);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j < n - 1 && xs[indexed[j + 1]] === xs[indexed[j]]) j++;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[indexed[k]] = avgRank;
    i = j + 1;
  }
  return ranks;
}

function _pearson(xs, ys) {
  const n = xs.length;
  if (n < 2) return 0;
  const mx = xs.reduce((a, b) => a + b) / n;
  const my = ys.reduce((a, b) => a + b) / n;
  const cov = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0);
  const vx = xs.reduce((s, x) => s + (x - mx) ** 2, 0);
  const vy = ys.reduce((s, y) => s + (y - my) ** 2, 0);
  const denom = Math.sqrt(vx * vy);
  return denom === 0 ? 0 : cov / denom;
}

function _spearman(scores, labels) {
  if (scores.length < 2) return 0;
  return _pearson(_rank(scores), _rank(labels.map(Number)));
}

// --- Result ---

class DecayComparisonResult {
  constructor({ name, decayFn, auc_roc, brier_score, log_loss, ranking_correlation }) {
    this.name = name;
    this.decayFn = decayFn;
    this.auc_roc = auc_roc;
    this.brier_score = brier_score;
    this.log_loss = log_loss;
    this.ranking_correlation = ranking_correlation;
    Object.freeze(this);
  }

  toDict() {
    return { name: this.name, auc_roc: this.auc_roc,
             brier_score: this.brier_score, log_loss: this.log_loss,
             ranking_correlation: this.ranking_correlation };
  }
}

// --- Orchestrator ---

class DecayComparison {
  /**
   * @param {object} opts
   * @param {Object.<string, import('../decay/base').DecayFunction>} opts.decayFns
   * @param {import('../models').TemporalFact[]} opts.facts
   * @param {boolean[]} opts.labels
   * @param {Date}      opts.queryTime
   */
  constructor({ decayFns, facts, labels, queryTime }) {
    if (!decayFns || Object.keys(decayFns).length === 0)
      throw new Error('decayFns must contain at least one DecayFunction');
    if (!facts || facts.length === 0)
      throw new Error('facts must not be empty');
    if (facts.length !== labels.length)
      throw new Error(`facts and labels length mismatch: ${facts.length} vs ${labels.length}`);
    this._decayFns = decayFns;
    this._facts = facts;
    this._labels = labels;
    this._queryTime = queryTime;
    this._cache = null;
  }

  compare() {
    if (!this._cache) this._cache = this._run();
    return [...this._cache];
  }

  best(metric = 'auc_roc') {
    if (!(metric in METRICS))
      throw new Error(`Unknown metric '${metric}'. Valid: ${Object.keys(METRICS).join(', ')}`);
    const results = this.compare();
    return METRICS[metric]
      ? results.reduce((b, r) => r[metric] < b[metric] ? r : b)
      : results.reduce((b, r) => r[metric] > b[metric] ? r : b);
  }

  summary() {
    const results = this.compare();
    return {
      nFacts: this._facts.length,
      nValid: this._labels.filter(Boolean).length,
      nDecayFns: Object.keys(this._decayFns).length,
      queryTime: this._queryTime.toISOString(),
      bestByAuc: this.best('auc_roc').name,
      bestByBrier: this.best('brier_score').name,
      results: results.map(r => r.toDict()),
    };
  }

  _run() {
    const intLabels = this._labels.map(l => l ? 1 : 0);
    const results = [];
    for (const [name, decayFn] of Object.entries(this._decayFns)) {
      const scores = decayFn.computeBatch(this._facts, this._queryTime);
      results.push(new DecayComparisonResult({
        name, decayFn,
        auc_roc: _auroc(scores, intLabels),
        brier_score: _brierScore(scores, intLabels),
        log_loss: _logLoss(scores, intLabels),
        ranking_correlation: _spearman(scores, intLabels),
      }));
    }
    return results.sort((a, b) => b.auc_roc - a.auc_roc);
  }
}

module.exports = { DecayComparison, DecayComparisonResult };
