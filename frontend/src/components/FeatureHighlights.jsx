import { formatFeatureValue, getFeatureEntries } from '../lib/formatters'

function FeatureHighlights({ featureValues = {} }) {
  const entries = getFeatureEntries(featureValues)
  const featuredEntries = entries.slice(0, 8)

  if (!entries.length) {
    return null
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold text-slate-700">Feature Summary</h3>
      </div>

      <div className="flex flex-wrap gap-2">
        {featuredEntries.map((entry) => (
          <span
            key={entry.key}
            title={entry.description}
            className="rounded-full border border-teal-200 bg-teal-50 px-3 py-1.5 text-xs font-medium text-teal-700"
          >
            {entry.label}
          </span>
        ))}
        {entries.length > featuredEntries.length && (
          <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-600">
            + {entries.length - featuredEntries.length} more
          </span>
        )}
      </div>

      <details className="rounded-lg border border-slate-200 bg-slate-50">
        <summary className="cursor-pointer list-none px-4 py-3 text-sm font-medium text-slate-700">
          View all extracted features
        </summary>
        <div className="border-t border-slate-200 px-4 py-4">
          <dl className="grid gap-3 sm:grid-cols-2">
            {entries.map((entry) => (
              <div
                key={entry.key}
                className="rounded-lg border border-slate-200 bg-white px-3 py-3"
                title={entry.description}
              >
                <dt className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                  {entry.label}
                </dt>
                <dd className="mt-1 text-sm font-medium text-slate-900">
                  {formatFeatureValue(entry.key, entry.value)}
                </dd>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  {entry.description}
                </p>
              </div>
            ))}
          </dl>
        </div>
      </details>
    </div>
  )
}

export default FeatureHighlights
