import { BookOpenText, CircleHelp, Info, ShieldCheck } from 'lucide-react'

function NavLink({ href, icon: Icon, label, tone = 'muted' }) {
  const className =
    tone === 'active'
      ? 'inline-flex items-center gap-2 border-b-2 border-teal-600 pb-3 font-semibold text-teal-700'
      : 'inline-flex items-center gap-2 transition hover:text-teal-700'

  return (
    <a className={className} href={href}>
      <Icon className="h-4 w-4" />
      {label}
    </a>
  )
}

function PageHeader() {
  return (
    <header className="rounded-xl border border-slate-200 bg-white/92 shadow-sm">
      <div className="flex flex-col gap-4 px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-center gap-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-xl border border-teal-200 bg-teal-50 text-teal-700">
            <ShieldCheck className="h-8 w-8" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold tracking-normal text-slate-900">
              Phishing URL Detection
            </h1>
            <p className="text-sm text-slate-500">Dissertation Prototype</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 lg:gap-6">
          <div className="flex items-center gap-5 text-sm text-slate-500">
            <NavLink href="#analyzer" icon={BookOpenText} label="Analyzer" tone="active" />
            <NavLink href="#results-history" icon={Info} label="Results History" />
            <NavLink href="#about" icon={CircleHelp} label="About" />
          </div>

          <div className="ml-auto flex items-center gap-3">
            <a
              href="#how-it-works"
              className="inline-flex items-center gap-2 rounded-xl border border-teal-200 bg-white px-4 py-2 text-sm font-semibold text-teal-700 shadow-sm transition hover:bg-teal-50"
            >
              <BookOpenText className="h-4 w-4" />
              Research Mode
            </a>
            <a
              href="#about"
              aria-label="Open about and limitations"
              className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-slate-200 text-slate-500 transition hover:border-teal-200 hover:text-teal-700"
            >
              <CircleHelp className="h-4 w-4" />
            </a>
          </div>
        </div>
      </div>
    </header>
  )
}

export default PageHeader
