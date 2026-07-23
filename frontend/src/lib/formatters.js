import { FEATURE_ORDER, getFeatureMetadata } from './featureMetadata'

const SCENARIO_LABELS = {
  combined_dataset: 'Combined dataset',
  combined_test: 'Combined held-out test set',
  phiusiil_main: 'PhiUSIIL main dataset',
  legitphish: 'LegitPhish dataset',
  phishstorm: 'PhishStorm dataset',
}

export function humanizeIdentifier(value) {
  return String(value || '')
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export function formatScenario(value) {
  return SCENARIO_LABELS[value] || humanizeIdentifier(value)
}

export function formatPercentage(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return 'Not available'
  }

  return `${(Number(value) * 100).toFixed(2)}%`
}

export function formatPredictionLabel(value) {
  return humanizeIdentifier(value)
}

export function formatFeatureValue(featureKey, value) {
  const metadata = getFeatureMetadata(featureKey)
  if (metadata.isBoolean) {
    return Number(value) === 1 ? 'Yes' : 'No'
  }

  return Number(value).toLocaleString()
}

export function formatLabelMapping(labelMapping = {}) {
  return Object.entries(labelMapping).map(([key, value]) => ({
    key,
    value: formatPredictionLabel(value),
  }))
}

export function getFeatureEntries(featureValues = {}) {
  const orderedKeys = FEATURE_ORDER.filter((key) => key in featureValues)
  const remainingKeys = Object.keys(featureValues).filter(
    (key) => !orderedKeys.includes(key),
  )

  return [...orderedKeys, ...remainingKeys].map((key) => ({
    key,
    value: featureValues[key],
    ...getFeatureMetadata(key),
  }))
}
