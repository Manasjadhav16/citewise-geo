import { useEffect, useState } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import React from 'react';
import { apiFetch } from '../../lib/api';
import {
  CategoryChip,
  DataDisclaimer,
  ShareChart,
  StatTile,
  fmtNum,
  fmtPct,
} from '../../components/twoStage';

type ObservationSource = {
  id: string;
  url: string | null;
  domain: string;
  title: string | null;
  category: string;
  categorySource: string;
  llmCategory: string | null;
  heuristicCategory: string | null;
  stability: number;
  appearances: number;
  overSourceCap: boolean;
  crawlError: string | null;
  evidence: string | null;
  evidenceTokenEstimate: number | null;
  evidenceTokenBudget: number | null;
  hasRelevantEvidence: boolean | null;
  evidenceError: string | null;
};

type ObservationQuery = {
  id: string;
  queryIndex: number;
  query: string;
  successfulRuns: number;
  sourceSetDiversity: number | null;
  sourceSetStability: number | null;
  domainSetStability: number | null;
  mentionRate: number | null;
  citationRate: number | null;
  meanCitationPosition: number | null;
  sources: ObservationSource[];
  responses: { id: string; runIndex: number; ok: boolean; error: string | null }[];
};

type ObservationEvent = { id: string; step: string; status: string; payload: Record<string, any> | null; timestamp: string };

type Observation = {
  id: string;
  sandboxUrl: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  error: string | null;
  dataDisclaimer: string | null;
  config: Record<string, any> | null;
  context: { brand_name?: string; business_category?: string } | null;
  runsAttempted: number | null;
  runsSucceeded: number | null;
  uniqueSources: number | null;
  uniqueDomains: number | null;
  mentionRate: number | null;
  citationRate: number | null;
  meanCitationPosition: number | null;
  medianCitationPosition: number | null;
  meanSourceSetDiversity: number | null;
  meanSourceSetStability: number | null;
  categoryBreakdown: Record<string, number> | null;
  competitorGroupBreakdown: Record<string, number> | null;
  events: ObservationEvent[];
  queries: ObservationQuery[];
};

const STEP_LABELS: Record<string, string> = {
  context: 'Analysing the page',
  query_selection: 'Selecting queries',
  observation: 'Observing grounded answers',
  categorization: 'Categorising sources',
  crawling_sources: 'Crawling cited pages',
  evidence_compression: 'Extracting evidence',
  completed: 'Completed',
  failed: 'Failed',
};

function SourceRow({ source }: { source: ObservationSource }) {
  const [open, setOpen] = useState(false);
  const problem = source.crawlError || source.evidenceError;
  return (
    <div className="border-b border-gray-800 py-3">
      <div className="flex flex-wrap items-center gap-3">
        <div className="w-28 shrink-0">
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden" title={`Cited in ${source.appearances} run(s)`}>
            <div className="h-full bg-purple-500" style={{ width: `${source.stability * 100}%` }} />
          </div>
          <p className="text-xs text-gray-500 mt-1">{fmtPct(source.stability * 100, 0)} of runs</p>
        </div>
        <div className="flex-1 min-w-0">
          <a href={source.url || undefined} target="_blank" rel="noreferrer" className="text-blue-300 hover:text-blue-200 break-all text-sm">
            {source.title && source.title.toLowerCase() !== source.domain ? source.title : source.url || source.domain}
          </a>
          <p className="text-xs text-gray-500">{source.domain}</p>
        </div>
        <CategoryChip category={source.category} />
        {source.evidence ? (
          <button onClick={() => setOpen(!open)} className="text-xs text-purple-300 hover:text-purple-200">
            {open ? 'Hide evidence' : `Evidence (${source.evidenceTokenEstimate}/${source.evidenceTokenBudget} tokens)`}
          </button>
        ) : (
          <span className="text-xs text-gray-500" title={problem || undefined}>
            {source.overSourceCap ? 'Over source cap' : problem ? 'Not extracted' : source.hasRelevantEvidence === false ? 'No relevant evidence' : ''}
          </span>
        )}
      </div>
      {source.heuristicCategory && source.llmCategory && source.heuristicCategory !== source.llmCategory && (
        <p className="text-xs text-amber-400/80 mt-1 ml-32">
          Domain heuristics suggested {source.heuristicCategory}; the classifier chose {source.llmCategory}.
        </p>
      )}
      {open && source.evidence && (
        <pre className="mt-3 ml-32 whitespace-pre-wrap text-xs text-gray-300 bg-gray-900/70 rounded-lg p-4 border border-gray-800">
          {source.evidence}
        </pre>
      )}
    </div>
  );
}

const ObservationPage = () => {
  const router = useRouter();
  const { id } = router.query;
  const [observation, setObservation] = useState<Observation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    if (!id || typeof id !== 'string') return;
    let active = true;
    let timer: NodeJS.Timeout | null = null;

    const load = async () => {
      try {
        const data = await apiFetch<Observation>(`/api/observation/${id}`);
        if (!active) return;
        setObservation(data);
        if ((data.status === 'completed' || data.status === 'failed') && timer) clearInterval(timer);
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : String(err));
      }
    };
    load();
    timer = setInterval(load, 4000);
    return () => {
      active = false;
      if (timer) clearInterval(timer);
    };
  }, [id]);

  const startBeforeRun = async () => {
    if (!observation) return;
    setStarting(true);
    setError(null);
    try {
      const { run_id } = await apiFetch<{ run_id: string }>('/api/sandbox/run', {
        method: 'POST',
        body: JSON.stringify({ url: observation.sandboxUrl, observation_id: observation.id, run_tag: 'before' }),
      });
      router.push(`/${run_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStarting(false);
    }
  };

  const latest = observation?.events?.[0];
  const inProgress = observation && (observation.status === 'pending' || observation.status === 'running');

  return (
    <>
      <Head>
        <title>Real-World Observation - CiteLabs</title>
      </Head>
      <main className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white">
        <div className="container mx-auto px-4 py-8 max-w-7xl space-y-8">
          <header>
            <button onClick={() => router.push('/')} className="mb-4 text-gray-400 hover:text-white transition-colors">
              ← Back to Home
            </button>
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
              Stage 1: Real-World Observation
            </h1>
            {observation && (
              <p className="mt-2 text-gray-400 text-sm">
                {observation.sandboxUrl} · <span className="font-mono">{observation.id}</span>
                {observation.config?.runs_per_query && ` · ${observation.config.runs_per_query} runs per query`}
              </p>
            )}
          </header>

          <DataDisclaimer text={observation?.dataDisclaimer} />

          {error && <div className="bg-red-900/50 border border-red-700 text-red-200 px-4 py-3 rounded-lg">{error}</div>}

          {inProgress && (
            <section className="bg-gray-800/50 rounded-2xl border border-gray-700 p-8">
              <h2 className="text-2xl font-semibold mb-2">{latest ? STEP_LABELS[latest.step] || latest.step : 'Starting…'}</h2>
              {latest?.payload?.completed != null && (
                <div className="mt-4">
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div className="h-full bg-purple-500 transition-all" style={{ width: `${(latest.payload.completed / latest.payload.total) * 100}%` }} />
                  </div>
                  <p className="text-sm text-gray-400 mt-2">{latest.payload.completed} of {latest.payload.total} grounded answers observed</p>
                </div>
              )}
              <p className="text-sm text-gray-500 mt-4">Stage 1 makes many rate-limited model calls and can take a while. This page updates automatically.</p>
            </section>
          )}

          {observation?.status === 'failed' && (
            <section className="bg-red-900/20 rounded-2xl border border-red-700 p-8">
              <h2 className="text-2xl font-bold text-red-400 mb-2">Observation failed</h2>
              <p className="text-gray-300">{observation.error}</p>
            </section>
          )}

          {observation?.status === 'completed' && (
            <>
              <section className="bg-gray-800/50 rounded-2xl border border-gray-700 p-8 space-y-6">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <h2 className="text-3xl font-bold">Summary</h2>
                  <button
                    onClick={startBeforeRun}
                    disabled={starting}
                    className="px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 rounded-lg font-semibold disabled:opacity-50"
                  >
                    {starting ? 'Starting…' : 'Run controlled GEO score (before)'}
                  </button>
                </div>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <StatTile label="Mention rate" value={fmtPct(observation.mentionRate)} hint="Answers that name your brand" />
                  <StatTile label="Citation rate" value={fmtPct(observation.citationRate)} hint="Answers that cite your site as a source" />
                  <StatTile label="Mean / median position" value={`${fmtNum(observation.meanCitationPosition, 1)} / ${fmtNum(observation.medianCitationPosition, 1)}`} hint="Your rank among cited domains, when cited" />
                  <StatTile label="Source-set stability" value={fmtNum(observation.meanSourceSetStability)} hint="Mean pairwise Jaccard between runs (1 = identical sources every run)" />
                  <StatTile label="Unique sources" value={`${observation.uniqueSources ?? '—'}`} hint={`${observation.uniqueDomains ?? '—'} domains across all queries`} />
                  <StatTile label="Source-set diversity" value={fmtNum(observation.meanSourceSetDiversity, 1)} hint="Distinct sources per query across all runs" />
                  <StatTile label="Runs" value={`${observation.runsSucceeded ?? 0} / ${observation.runsAttempted ?? 0}`} hint="Successful / attempted observations" />
                  <StatTile label="Business category" value={<span className="text-base font-medium">{observation.context?.business_category || '—'}</span>} />
                </div>
                <div className="grid lg:grid-cols-2 gap-6">
                  <ShareChart title="Citation share by competitor group" breakdown={observation.competitorGroupBreakdown} />
                  <ShareChart title="AI visibility competition (share by category)" breakdown={observation.categoryBreakdown} />
                </div>
              </section>

              {observation.queries.map((q) => (
                <section key={q.id} className="bg-gray-800/50 rounded-2xl border border-gray-700 p-8">
                  <h2 className="text-xl font-semibold mb-4">
                    <span className="text-gray-500 mr-2">Q{q.queryIndex + 1}</span>
                    {q.query}
                  </h2>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6 text-sm">
                    <div><span className="text-gray-400">Runs:</span> {q.successfulRuns}/{q.responses.length}</div>
                    <div><span className="text-gray-400">Mentioned:</span> {q.successfulRuns > 0 ? fmtPct(q.mentionRate, 0) : 'no data'}</div>
                    <div><span className="text-gray-400">Cited:</span> {q.successfulRuns > 0 ? fmtPct(q.citationRate, 0) : 'no data'}</div>
                    <div><span className="text-gray-400">Diversity:</span> {q.sourceSetDiversity ?? '—'} sources</div>
                    <div><span className="text-gray-400">Stability:</span> {fmtNum(q.sourceSetStability)}</div>
                  </div>
                  {q.responses.some((r) => !r.ok) && (
                    <p className="text-xs text-amber-400 mb-4">
                      {q.responses.filter((r) => !r.ok).length} run(s) failed and are excluded from these metrics.
                    </p>
                  )}
                  <h3 className="text-sm uppercase tracking-wide text-gray-400 mb-2">Union source pool ({q.sources.length})</h3>
                  <div>
                    {q.sources.map((s) => (
                      <SourceRow key={s.id} source={s} />
                    ))}
                  </div>
                </section>
              ))}
            </>
          )}
        </div>
      </main>
    </>
  );
};

export default ObservationPage;
