'use strict';

const TIME_DIVISORS = Object.freeze({
  seconds: 1000,
  hours: 3600000,
  days: 86400000,
});

function validateTimeUnit(timeUnit) {
  if (!Object.prototype.hasOwnProperty.call(TIME_DIVISORS, timeUnit)) {
    throw new Error(
      `timeUnit must be one of 'seconds', 'hours', or 'days', got ${JSON.stringify(timeUnit)}.`
    );
  }
  return TIME_DIVISORS[timeUnit];
}

function validateParameter(value, name, { positive }) {
  const requirement = positive ? 'positive' : 'nonnegative';
  if (typeof value !== 'number' || !Number.isFinite(value)
      || (positive ? value <= 0 : value < 0)) {
    throw new Error(
      `${name} must be a finite ${requirement} number, got ${String(value)}.`
    );
  }
  return value;
}

function validateParameterMap(values, name, { positive }) {
  const requirement = positive ? 'positive' : 'nonnegative';
  if (values === null || typeof values !== 'object' || Array.isArray(values)) {
    throw new Error(
      `${name} must be an object mapping fact types to finite ${requirement} numbers.`
    );
  }

  return Object.fromEntries(Object.entries(values).map(([factType, value]) => [
    factType,
    validateParameter(value, `${name} for fact type '${factType}'`, { positive }),
  ]));
}

module.exports = { validateTimeUnit, validateParameter, validateParameterMap };
