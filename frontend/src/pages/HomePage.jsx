import { useMemo, useState } from 'react'

import DisclaimerSection from '../components/DisclaimerSection.jsx'
import HowItWorksSection from '../components/HowItWorksSection.jsx'
import PageHeader from '../components/PageHeader.jsx'
import PredictionPanel from '../components/PredictionPanel.jsx'
import ResultsHistorySection from '../components/ResultsHistorySection.jsx'
import UrlAnalysisPanel from '../components/UrlAnalysisPanel.jsx'
import { useModelInfo } from '../hooks/useModelInfo'
import { usePredictionHistory } from '../hooks/usePredictionHistory'
import { useUrlPrediction } from '../hooks/useUrlPrediction'

function HomePage() {
  const [url, setUrl] = useState('')
  const {
    error: backendError,
    loading: backendLoading,
    modelInfo,
    modelOptions,
    selectModel,
    selectedModelId,
  } = useModelInfo()
  const { history, recordPrediction } = usePredictionHistory()
  const {
    error: predictionError,
    loading: predictionLoading,
    prediction,
    submitUrl,
  } = useUrlPrediction()

  const isBackendReady = useMemo(
    () => Boolean(modelInfo) && !backendError,
    [backendError, modelInfo],
  )

  const handleSubmit = async (event) => {
    event.preventDefault()

    if (!url.trim()) {
      return
    }

    try {
      const nextPrediction = await submitUrl(url.trim(), selectedModelId)
      recordPrediction(nextPrediction)
    } catch (error) {
      return error
    }
  }

  return (
    <main
      id="top"
      className="min-h-screen bg-transparent px-4 py-5 text-slate-900 sm:px-6 lg:px-8"
    >
      <div className="mx-auto flex w-full max-w-[1500px] flex-col gap-6">
        <PageHeader modelInfo={modelInfo} />

        <section id="analyzer" className="grid gap-6 xl:grid-cols-[0.95fr_1.35fr]">
          <UrlAnalysisPanel
            backendError={backendError}
            backendLoading={backendLoading}
            backendReady={isBackendReady}
            loading={predictionLoading}
            modelOptions={modelOptions}
            onSampleSelect={setUrl}
            onModelChange={selectModel}
            onSubmit={handleSubmit}
            onUrlChange={setUrl}
            selectedModelId={selectedModelId}
            url={url}
          />

          <div className="grid gap-6">
            <PredictionPanel
              error={predictionError}
              loading={predictionLoading}
              modelInfo={modelInfo}
              prediction={prediction}
            />
          </div>
        </section>

        <ResultsHistorySection items={history} />
        <HowItWorksSection />
        <DisclaimerSection />

        <footer className="flex flex-col gap-2 px-1 pb-4 text-sm text-slate-500 sm:flex-row sm:items-center sm:justify-between">
          <span>© 2026 Phishing URL Detection – Dissertation Prototype. All rights reserved.</span>
          <span>Built for academic research.</span>
        </footer>
      </div>
    </main>
  )
}

export default HomePage
