const MODEL_DESCRIPTIONS = [
  {
    pattern: /xgboost/i,
    shortLabel: 'Ensemble (Gradient Boosting)',
    description:
      'Gradient boosting ensemble. It is useful here because it can capture non-linear URL patterns strongly across mixed datasets.',
  },
  {
    pattern: /random forest/i,
    shortLabel: 'Ensemble (Random Forest)',
    description:
      'Ensemble of many decision trees. It is useful because it usually generalizes better than one single tree.',
  },
  {
    pattern: /decision tree/i,
    shortLabel: 'Tree-based baseline',
    description:
      'Single tree-based model. It is easy to interpret, so it works well as a simple baseline.',
  },
  {
    pattern: /svm/i,
    shortLabel: 'Margin-based classifier',
    description:
      'Margin-based classifier. It is a strong classical baseline for structured numeric URL features.',
  },
  {
    pattern: /logistic regression/i,
    shortLabel: 'Linear probabilistic baseline',
    description:
      'Linear probabilistic baseline. It is fast, explainable, and good for checking whether the URL-only features already separate the classes.',
  },
]

export function getModelDescription(modelName) {
  const matched = MODEL_DESCRIPTIONS.find(({ pattern }) => pattern.test(modelName || ''))
  return (
    matched?.description ||
    'This trained model is loaded from the dissertation experiment outputs and used to score the URL.'
  )
}

export function getModelTypeSummary(modelName) {
  const matched = MODEL_DESCRIPTIONS.find(({ pattern }) => pattern.test(modelName || ''))
  return matched?.shortLabel || 'Trained classifier'
}
