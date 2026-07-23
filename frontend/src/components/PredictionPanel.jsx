import {
  AlertTriangle,
  CalendarClock,
  Cpu,
  Database,
  LoaderCircle,
  ShieldAlert,
  ShieldCheck,
  Tags,
} from 'lucide-react'

import FeatureHighlights from './FeatureHighlights.jsx'
import {
  formatLabelMapping,
  formatPercentage,
  formatPredictionLabel,
  formatScenario,
} from '../lib/formatters'
import { getModelTypeSummary } from '../lib/modelMetadata'

function DetailRow({ icon: Icon, label, value }) {
  return (
    <div className="flex items-center gap-4 border border-slate-200 bg-white px-4 py-4">
      <Icon className="h-5 w-5 text-slate-500" />
      <div className="grid flex-1 gap-1 text-sm md:grid-cols-[140px_1fr]">
        <span className="text-slate-500">{label}</span>
        <span className="font-medium text-slate-800">{value}</span>
      </div>
    </div>
  )
}

function EmptyState({ modelInfo }) {
  return (
    <div className="rounded-xl border border-teal-200 bg-teal-50 px-5 py-5 text-slate-800">
      <div className="flex flex-col gap-5 md:flex-row md:items-center md:justify-between">
        <div className="flex items-start gap-4">
          <div className="rounded-full bg-teal-700 p-4 text-white">
            <ShieldCheck className="h-6 w-6" />
          </div>
          <div>
            <h3 className="text-4xl font-semibold text-teal-800">Ready to Analyze</h3>
            <p className="mt-2 max-w-xl text-sm leading-6 text-slate-600">
              Submit a URL from the left panel to generate a prediction result and
              confidence score.
            </p>
          </div>
        </div>

        <div className="min-w-[220px] space-y-2">
          <div className="flex items-center justify-between gap-4 text-sm font-medium text-slate-600">
            <span>Confidence</span>
            <span>Awaiting input</span>
          </div>
          <div className="h-2 rounded-full bg-slate-200">
            <div className="h-2 w-0 rounded-full bg-teal-700" />
          </div>
        </div>
      </div>

      <div className="mt-5 grid gap-px overflow-hidden rounded-xl border border-slate-200 bg-slate-200 md:grid-cols-2">
        <DetailRow
          icon={Cpu}
          label="Active Model"
          value={modelInfo?.model_name || 'Loading model metadata'}
        />
        <DetailRow
          icon={Database}
          label="Training Scenario"
          value={
            modelInfo?.training_scenario
              ? formatScenario(modelInfo.training_scenario)
              : 'Checking dataset configuration'
          }
        />
        <DetailRow
          icon={Cpu}
          label="Model Type"
          value={getModelTypeSummary(modelInfo?.model_name)}
        />
        <DetailRow
          icon={CalendarClock}
          label="Feature Count"
          value={
            modelInfo?.feature_columns?.length
              ? `${modelInfo.feature_columns.length} URL-only features`
              : 'Loading feature list'
          }
        />
      </div>

      {!!modelInfo?.feature_columns?.length && (
        <div className="mt-5">
          <FeatureHighlights
            featureValues={Object.fromEntries(
              modelInfo.feature_columns.map((featureKey) => [featureKey, 0]),
            )}
          />
        </div>
      )}
    </div>
  )
}

function PredictionPanel({ error, loading, modelInfo, prediction }) {
  const predictedClass = prediction?.predicted_class?.toLowerCase() || ''
  const isPhishing = predictedClass.includes('phish')
  const Icon = isPhishing ? ShieldAlert : ShieldCheck
  const tone = isPhishing
    ? 'border-amber-200 bg-amber-50 text-amber-800'
    : 'border-emerald-200 bg-emerald-50 text-emerald-800'
  const progressWidth =
    prediction?.confidence === null || prediction?.confidence === undefined
      ? '0%'
      : `${Math.max(0, Math.min(100, Number(prediction.confidence) * 100))}%`
  const labelMapping = formatLabelMapping(modelInfo?.label_mapping)

  return (
    <section className="rounded-xl border border-slate-200 bg-white px-6 py-5 shadow-sm">
      <div className="space-y-2">
        <h2 className="text-2xl font-semibold text-slate-900">2. Prediction Result</h2>
      </div>

      <div className="mt-6 space-y-5" aria-live="polite">
        {loading && (
          <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-5 text-sm text-slate-700">
            <LoaderCircle className="h-5 w-5 animate-spin text-teal-600" />
            <span>The backend is extracting features and scoring the URL.</span>
          </div>
        )}

        {!loading && error && (
          <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-4 text-sm text-amber-800">
            <div className="flex items-center gap-2 font-semibold">
              <AlertTriangle className="h-4 w-4" />
              Prediction request failed
            </div>
            <p className="mt-2 leading-6">{error}</p>
          </div>
        )}

        {!loading && !error && !prediction && <EmptyState modelInfo={modelInfo} />}

        {!loading && prediction && (
          <div className="space-y-5">
            <div className={`rounded-xl border px-5 py-5 ${tone}`}>
              <div className="flex flex-col gap-5 md:flex-row md:items-center md:justify-between">
                <div className="flex items-start gap-4">
                  <div className="rounded-full bg-white p-4 shadow-sm">
                    <Icon className="h-7 w-7" />
                  </div>
                  <div>
                    <h3 className="text-4xl font-semibold">
                      {formatPredictionLabel(prediction.predicted_class)}
                    </h3>
                    <p className="mt-2 max-w-xl text-sm leading-6 text-current/85">
                      This URL is predicted to be{' '}
                      {formatPredictionLabel(prediction.predicted_class).toLowerCase()}.
                    </p>
                  </div>
                </div>

                <div className="min-w-[220px] space-y-2">
                  <div className="text-sm text-slate-600">Confidence</div>
                  <div className="text-4xl font-semibold text-current">
                    {formatPercentage(prediction.confidence)}
                  </div>
                  <div className="h-2 rounded-full bg-slate-200">
                    <div
                      className="h-2 rounded-full bg-current transition-[width]"
                      style={{ width: progressWidth }}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="grid gap-px overflow-hidden rounded-xl border border-slate-200 bg-slate-200 md:grid-cols-2">
              <DetailRow
                icon={Cpu}
                label="Active Model"
                value={prediction.model_name}
              />
              <DetailRow
                icon={Database}
                label="Training Scenario"
                value={formatScenario(prediction.training_scenario)}
              />
              <DetailRow
                icon={Cpu}
                label="Model Type"
                value={getModelTypeSummary(prediction.model_name)}
              />
              <DetailRow
                icon={Tags}
                label="Label Mapping"
                value={labelMapping.map((item) => `${item.key} = ${item.value}`).join(', ')}
              />
            </div>

            <FeatureHighlights featureValues={prediction.feature_values} />

            <div className="flex items-start gap-2 text-sm text-slate-500">
              <AlertTriangle className="mt-0.5 h-4 w-4 text-slate-400" />
              <span>
                Results are generated by the selected model using engineered features
                from the URL.
              </span>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}

export default PredictionPanel
