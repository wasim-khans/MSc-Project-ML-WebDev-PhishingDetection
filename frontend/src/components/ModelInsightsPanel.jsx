import { BrainCircuit, FileCode2, Files, Tags } from 'lucide-react'

import { formatLabelMapping, formatScenario } from '../lib/formatters'
import { getModelDescription } from '../lib/modelMetadata'

function DetailTile({ icon: Icon, label, value }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/5 px-4 py-4">
      <div className="flex items-center gap-2 text-slate-400">
        <Icon className="h-4 w-4" />
        <p className="text-xs font-semibold uppercase tracking-[0.16em]">{label}</p>
      </div>
      <p className="mt-3 text-sm font-medium leading-6 text-white">{value}</p>
    </div>
  )
}

function ModelInsightsPanel({ error, loading, modelInfo }) {
  const labelMapping = formatLabelMapping(modelInfo?.label_mapping)

  return (
    <section className="rounded-xl border border-white/10 bg-slate-950/70 p-6 shadow-xl shadow-black/20">
      <div className="space-y-2">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-300">
          3. Backend model context
        </p>
        <h2 className="text-2xl font-semibold text-white">Active backend selection</h2>
        <p className="text-sm leading-6 text-slate-300">
          The backend loads its recommendation from the dissertation experiment outputs,
          so this page always reflects the currently selected trained model.
        </p>
      </div>

      <div className="mt-6 space-y-4">
        {loading && (
          <div className="rounded-lg border border-white/10 bg-white/5 px-4 py-4 text-sm text-slate-300">
            Loading model metadata from the backend...
          </div>
        )}

        {!loading && error && (
          <div className="rounded-lg border border-amber-400/20 bg-amber-400/10 px-4 py-4 text-sm leading-6 text-amber-100">
            {error}
          </div>
        )}

        {!loading && modelInfo && (
          <>
            <div className="rounded-xl border border-teal-400/20 bg-teal-400/10 px-5 py-5">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-black/15 p-2 text-teal-100">
                  <BrainCircuit className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-teal-100/80">
                    Current model
                  </p>
                  <h3 className="mt-1 text-2xl font-semibold text-white">
                    {modelInfo.model_name}
                  </h3>
                </div>
              </div>
              <p className="mt-4 text-sm leading-6 text-slate-100/90">
                {getModelDescription(modelInfo.model_name)}
              </p>
            </div>

            <div className="grid gap-3 md:grid-cols-2">
              <DetailTile
                icon={Files}
                label="Training scenario"
                value={formatScenario(modelInfo.training_scenario)}
              />
              <DetailTile
                icon={Tags}
                label="Feature set"
                value={`${modelInfo.feature_columns.length} URL-only input features`}
              />
              <DetailTile
                icon={FileCode2}
                label="Model file"
                value={modelInfo.model_file}
              />
              <DetailTile
                icon={Tags}
                label="Label mapping"
                value={labelMapping.map((item) => `${item.key} = ${item.value}`).join(', ')}
              />
            </div>
          </>
        )}
      </div>
    </section>
  )
}

export default ModelInsightsPanel
