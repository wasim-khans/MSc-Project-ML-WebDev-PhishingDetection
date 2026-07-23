import { Cpu, Link2, ShieldCheck } from 'lucide-react'

function StepCard({ description, icon: Icon, number, title }) {
  return (
    <div className="flex gap-4 px-5 py-5">
      <div className="flex items-start gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-teal-700 text-sm font-semibold text-white">
          {number}
        </div>
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-teal-50 text-teal-700">
          <Icon className="h-7 w-7" />
        </div>
      </div>
      <div>
        <h3 className="text-xl font-semibold text-slate-900">{title}</h3>
        <p className="mt-2 text-sm leading-7 text-slate-600">{description}</p>
      </div>
    </div>
  )
}

function HowItWorksSection() {
  return (
    <section
      id="how-it-works"
      className="rounded-xl border border-slate-200 bg-white shadow-sm"
    >
      <div className="px-6 py-5">
        <h2 className="text-2xl font-semibold text-slate-900">
          3. How the Result is Produced
        </h2>
      </div>

      <div className="grid divide-y divide-slate-200 md:grid-cols-3 md:divide-x md:divide-y-0">
        <StepCard
          number="1"
          icon={Link2}
          title="Feature Extraction"
          description="The URL is parsed and transformed into numerical features such as length, HTTPS usage, subdomains, and suspicious lexical patterns."
        />
        <StepCard
          number="2"
          icon={Cpu}
          title="Model Prediction"
          description="The active backend model analyzes the extracted URL-only features and predicts whether the URL looks legitimate or phishing-related."
        />
        <StepCard
          number="3"
          icon={ShieldCheck}
          title="Decision & Confidence"
          description="The prediction is returned as a class label with a confidence score so you can see both the model decision and its certainty."
        />
      </div>
    </section>
  )
}

export default HowItWorksSection
