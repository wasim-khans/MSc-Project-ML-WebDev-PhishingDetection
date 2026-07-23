import { BookOpenText, CircleHelp, TriangleAlert } from 'lucide-react'

function NoteBlock({ children, icon: Icon, title }) {
  return (
    <details className="rounded-xl border border-white/10 bg-slate-950/70 px-5 py-4" open>
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 text-left">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-white/5 p-2 text-teal-300">
            <Icon className="h-5 w-5" />
          </div>
          <h3 className="text-lg font-semibold text-white">{title}</h3>
        </div>
        <span className="text-xs uppercase tracking-[0.16em] text-slate-400">Toggle</span>
      </summary>
      <div className="pt-4 text-sm leading-7 text-slate-300">{children}</div>
    </details>
  )
}

function ResearchNotesSection({ modelInfo }) {
  return (
    <section className="space-y-4">
      <div className="space-y-2">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-300">
          4. Research notes
        </p>
        <h2 className="text-2xl font-semibold text-white">How to read this prototype</h2>
      </div>

      <div className="grid gap-4">
        <NoteBlock icon={BookOpenText} title="What the frontend is doing">
          <ul className="list-disc space-y-2 pl-5">
            <li>The page sends one URL to the FastAPI backend.</li>
            <li>The backend extracts the same URL-only features used in your machine-learning experiments.</li>
            <li>
              The active model is currently{' '}
              <strong className="text-white">
                {modelInfo?.model_name || 'being loaded'}
              </strong>
              , using the training scenario{' '}
              <strong className="text-white">
                {modelInfo?.training_scenario || 'being loaded'}
              </strong>
              .
            </li>
          </ul>
        </NoteBlock>

        <NoteBlock icon={CircleHelp} title="What the output means">
          <ul className="list-disc space-y-2 pl-5">
            <li>
              <strong className="text-white">Predicted class</strong> is the model's final label for the submitted URL.
            </li>
            <li>
              <strong className="text-white">Confidence</strong> is the model's numeric certainty for the chosen class.
            </li>
            <li>
              <strong className="text-white">Feature values</strong> are the engineered lexical URL measurements that the model sees as inputs.
            </li>
          </ul>
        </NoteBlock>

        <NoteBlock icon={TriangleAlert} title="Important limitations">
          <ul className="list-disc space-y-2 pl-5">
            <li>This is a dissertation prototype and should not be the sole basis for real security decisions.</li>
            <li>Performance can change when the training dataset changes, even when the feature set stays the same.</li>
            <li>Confidence does not guarantee correctness; it only reflects how sure the current model is about its own output.</li>
          </ul>
        </NoteBlock>
      </div>
    </section>
  )
}

export default ResearchNotesSection
