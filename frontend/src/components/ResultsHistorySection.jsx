import { Clock3, History, ShieldAlert, ShieldCheck } from 'lucide-react'

import { formatPercentage, formatPredictionLabel, formatScenario } from '../lib/formatters'

function formatTimestamp(value) {
  if (!value) {
    return 'Just now'
  }

  return new Date(value).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  })
}

function ResultsHistorySection({ items = [] }) {
  return (
    <section
      id="results-history"
      className="rounded-xl border border-slate-200 bg-white shadow-sm"
    >
      <div className="flex items-center justify-between gap-4 px-6 py-5">
        <div>
          <h2 className="text-2xl font-semibold text-slate-900">Recent Results</h2>
          <p className="mt-2 text-sm text-slate-500">
            Your latest URL analyses stay here for quick comparison during this session.
          </p>
        </div>
        <div className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600">
          <History className="h-4 w-4 text-teal-600" />
          {items.length} recorded
        </div>
      </div>

      <div className="border-t border-slate-200 px-6 py-5">
        {!items.length && (
          <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm text-slate-500">
            No URLs have been analyzed yet in this session.
          </div>
        )}

        {!!items.length && (
          <div className="space-y-3">
            {items.map((item) => {
              const isPhishing = String(item.predictedClass).toLowerCase().includes('phish')
              const Icon = isPhishing ? ShieldAlert : ShieldCheck
              const tone = isPhishing
                ? 'border-amber-200 bg-amber-50 text-amber-800'
                : 'border-emerald-200 bg-emerald-50 text-emerald-800'

              return (
                <article
                  key={item.id}
                  className="grid gap-4 rounded-xl border border-slate-200 px-4 py-4 lg:grid-cols-[1.5fr_auto_auto_auto]"
                >
                  <div className="min-w-0">
                    <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
                      URL
                    </p>
                    <p className="mt-1 break-all text-sm font-medium text-slate-800">
                      {item.url}
                    </p>
                  </div>

                  <div className={`inline-flex items-center gap-2 self-start rounded-full border px-3 py-2 text-sm font-semibold ${tone}`}>
                    <Icon className="h-4 w-4" />
                    {formatPredictionLabel(item.predictedClass)}
                  </div>

                  <div className="self-start text-sm text-slate-600">
                    <div className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
                      Model
                    </div>
                    <div className="mt-1 font-semibold text-slate-800">
                      {item.modelName}
                    </div>
                    <div className="text-xs text-slate-500">
                      {formatScenario(item.trainingScenario)}
                    </div>
                  </div>

                  <div className="self-start text-sm text-slate-600">
                    <div className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
                      Confidence
                    </div>
                    <div className="mt-1 font-semibold text-slate-800">
                      {formatPercentage(item.confidence)}
                    </div>
                  </div>

                  <div className="inline-flex items-center gap-2 self-start text-sm text-slate-500">
                    <Clock3 className="h-4 w-4" />
                    {formatTimestamp(item.createdAt)}
                  </div>
                </article>
              )
            })}
          </div>
        )}
      </div>
    </section>
  )
}

export default ResultsHistorySection
