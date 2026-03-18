'use strict';

/**
 * Core data models for Chronofy.
 *
 * Implements the evidence tuple representation (c, t_e, q, m):
 *   c  — content
 *   t_e — observation timestamp (Date)
 *   q  — source reliability weight in (0, 1]
 *   m  — metadata (factType, source, etc.)
 */

class TemporalFact {
  /**
   * @param {object} opts
   * @param {string} opts.content
   * @param {Date}   opts.timestamp         Observation timestamp
   * @param {string} [opts.factType]        Category for decay rate lookup
   * @param {number} [opts.sourceQuality]   Reliability weight q ∈ (0, 1]
   * @param {Date}   [opts.publicationTimestamp]
   * @param {string} [opts.source]
   * @param {object} [opts.metadata]
   */
  constructor({ content, timestamp, factType = 'general', sourceQuality = 1.0,
                publicationTimestamp = null, source = null, metadata = {} }) {
    if (!content) throw new Error('content is required');
    if (!(timestamp instanceof Date)) throw new Error('timestamp must be a Date');
    if (sourceQuality <= 0 || sourceQuality > 1)
      throw new Error('sourceQuality must be in (0, 1]');

    this.content = content;
    this.timestamp = timestamp;
    this.factType = factType;
    this.sourceQuality = sourceQuality;
    this.publicationTimestamp = publicationTimestamp;
    this.source = source;
    this.metadata = metadata;
  }

  /**
   * Wall-clock age in days at queryTime. Always >= 0.
   * @param {Date} queryTime
   * @returns {number}
   */
  ageAt(queryTime) {
    const deltaMs = queryTime.getTime() - this.timestamp.getTime();
    return Math.max(deltaMs / 86400000, 0);
  }
}

class ReasoningStep {
  /**
   * @param {object} opts
   * @param {number}        opts.stepIndex
   * @param {string}        opts.content
   * @param {TemporalFact[]} [opts.factsUsed]
   */
  constructor({ stepIndex, content, factsUsed = [] }) {
    this.stepIndex = stepIndex;
    this.content = content;
    this.factsUsed = factsUsed;
  }
}

class ReasoningTrace {
  /**
   * @param {object} opts
   * @param {ReasoningStep[]} [opts.steps]
   * @param {Date}            [opts.queryTime]
   */
  constructor({ steps = [], queryTime = new Date() }) {
    this.steps = steps;
    this.queryTime = queryTime;
  }

  /** All unique facts used across the entire trace. */
  get allFacts() {
    const seen = new Set();
    const unique = [];
    for (const step of this.steps) {
      for (const fact of step.factsUsed) {
        if (!seen.has(fact.content)) {
          seen.add(fact.content);
          unique.push(fact);
        }
      }
    }
    return unique;
  }
}

module.exports = { TemporalFact, ReasoningStep, ReasoningTrace };
