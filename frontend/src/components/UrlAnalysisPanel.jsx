import { LoaderCircle, Search, Server, ShieldAlert, ShieldCheck } from 'lucide-react'

import { formatScenario } from '../lib/formatters'
import { SAMPLE_URLS } from '../lib/sampleUrls'

function StatusRow({ loading, error, isReady }) {
  let icon = <Server className="h-4 w-4" />
  let label = 'Checking backend connection...'
  let tone = 'text-slate-500'

  if (error) {
    icon = <ShieldAlert className="h-4 w-4" />
    label = error
    tone = 'text-amber-700'
  } else if (isReady) {
    icon = <ShieldCheck className="h-4 w-4" />
    label = 'Backend connected'
    tone = 'text-emerald-700'
  } else if (loading) {
    icon = <LoaderCircle className="h-4 w-4 animate-spin" />
  }

  return (
    <div className={`flex items-center gap-2 text-sm ${tone}`}>
      {icon}
      <span>{label}</span>
    </div>
  )
}

function UrlAnalysisPanel({
  backendError,
  backendLoading,
  backendReady,
  loading,
  modelOptions = [],
  onSampleSelect,
  onModelChange,
  onSubmit,
  onUrlChange,
  selectedModelId,
  url,
}) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white px-6 py-5 shadow-sm">
      <div className="space-y-2">
        <h2 className="text-2xl font-semibold text-slate-900">1. Analyze a URL</h2>
      </div>

      <form className="mt-6 space-y-5" onSubmit={onSubmit}>
        <div className="space-y-2">
          <label
            className="block text-sm font-medium text-slate-700"
            htmlFor="model-select"
          >
            Active Model
          </label>
          <select
            id="model-select"
            value={selectedModelId}
            onChange={(event) => onModelChange(event.target.value)}
            disabled={!modelOptions.length || backendLoading || loading}
            className="w-full rounded-lg border border-slate-300 bg-white px-4 py-3 text-sm text-slate-800 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {!modelOptions.length && (
              <option value="">Loading trained models...</option>
            )}
            {modelOptions.map((model) => (
              <option key={model.model_id} value={model.model_id}>
                {model.model_name} trained on {formatScenario(model.training_scenario)}
                {model.is_recommended ? ' - recommended' : ''}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label
            className="block text-sm font-medium text-slate-700"
            htmlFor="url-input"
          >
            URL
          </label>
          <textarea
            id="url-input"
            rows="5"
            value={url}
            onChange={(event) => onUrlChange(event.target.value)}
            placeholder="https://example.com/login"
            className="w-full resize-none rounded-lg border border-slate-300 bg-white px-4 py-3 text-sm text-slate-800 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
          />
          <p className="text-sm text-slate-500">
            Enter a full URL including http:// or https://
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          {SAMPLE_URLS.map((sample) => (
            <button
              key={sample.label}
              type="button"
              onClick={() => onSampleSelect(sample.url)}
              className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600 transition hover:border-teal-200 hover:bg-teal-50 hover:text-teal-700"
            >
              {sample.label}
            </button>
          ))}
        </div>

        <button
          type="submit"
          disabled={loading || !url.trim()}
          className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-teal-700 px-4 py-3 text-base font-semibold text-white transition hover:bg-teal-800 disabled:cursor-not-allowed disabled:opacity-55"
        >
          {loading ? (
            <>
              <LoaderCircle className="h-4 w-4 animate-spin" />
              Analyzing URL...
            </>
          ) : (
            <>
              <Search className="h-4 w-4" />
              Analyze URL
            </>
          )}
        </button>
      </form>

      <div className="mt-6 border-t border-slate-200 pt-4">
        <p className="mb-3 text-sm font-semibold text-slate-800">Backend Status</p>
        <StatusRow
          loading={backendLoading}
          error={backendError}
          isReady={backendReady}
        />
        {backendReady && (
          <p className="mt-2 text-sm text-slate-500">
            Model service is online and ready.
          </p>
        )}
      </div>
    </section>
  )
}

export default UrlAnalysisPanel
