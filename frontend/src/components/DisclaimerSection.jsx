import { AlertTriangle } from 'lucide-react'

function DisclaimerSection() {
  return (
    <section id="about" className="rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="px-6 py-5">
        <h2 className="text-2xl font-semibold text-slate-900">
          4. Research Disclaimer & Limitations
        </h2>
      </div>

      <div className="grid gap-6 border-t border-slate-200 px-6 py-5 md:grid-cols-[48px_1fr_1fr]">
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-amber-50 text-amber-600">
          <AlertTriangle className="h-5 w-5" />
        </div>

        <ul className="space-y-3 text-sm leading-7 text-slate-600">
          <li>This system is a dissertation prototype developed for academic research purposes only.</li>
          <li>Predictions should not be used as the sole basis for security decisions.</li>
          <li>The model is trained on publicly available datasets and may not generalize to unseen or future phishing techniques.</li>
        </ul>

        <ul className="space-y-3 text-sm leading-7 text-slate-600">
          <li>Feature engineering and model performance may be biased by dataset quality and class balance.</li>
          <li>Results can change when the selected model or training data changes.</li>
          <li>Always verify suspicious URLs using multiple trusted security tools.</li>
        </ul>
      </div>
    </section>
  )
}

export default DisclaimerSection
