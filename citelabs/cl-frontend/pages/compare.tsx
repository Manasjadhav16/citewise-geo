import { useEffect, useState } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { useRouter } from 'next/router';
import React from 'react';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { apiFetch } from '../lib/api';
import { ChangeBadge, DataDisclaimer, StatTile, categoryLabel, fmtDelta, fmtNum, fmtPct } from '../components/twoStage';

type Delta = { before: number | null; after: number | null; delta: number | null };
type Deltas = { geo_improvement: number | null; aeo_improvement: number | null; deltas: Record<string, Delta> };

type QuerySide = {
  appeared: boolean;
  cited: boolean;
  mentioned: boolean | null;
  citation_position: number | null;
  diagnosis_type: string | null;
} | null;

type Comparison = {
  before: { id: string; run_tag: string | null; created_at: string };
  after: { id: string; run_tag: string | null; baseline_run_id: string | null; created_at: string };
  warnings: string[];
  controlled: Deltas;
  per_query: {
    question: string;
    before: QuerySide;
    after: QuerySide;
    change: string;
    real_world: {
      successful_runs: number;
      mention_rate: number | null;
      citation_rate: number | null;
      mean_citation_position: number | null;
      source_set_stability: number | null;
    } | null;
  }[];
  real_world: (Deltas & { before_observation: string; after_observation: string; data_disclaimer: string | null }) | null;
  data_disclaimer: string | null;
};

// Rate-style metrics shown side by side (all 0-100)
const RATE_METRICS: [string, string][] = [
  ['geo_score', 'GEO score'],
  ['aeo_score', 'AEO score'],
  ['citation_rate', 'Citation rate (appeared)'],
  ['strict_citation_rate', 'Strict citation rate'],
  ['mention_rate', 'Mention rate'],
  ['coverage', 'Coverage'],
  ['competitor_dominance', 'Competitor dominance'],
];

const METRIC_LABELS: Record<string, string> = {
  ...Object.fromEntries(RATE_METRICS),
  mean_citation_position: 'Mean citation position',
  median_citation_position: 'Median citation position',
  source_set_diversity: 'Source-set diversity',
  source_set_stability: 'Source-set stability (Jaccard)',
  domain_set_stability: 'Domain-set stability (Jaccard)',
};

const LOWER_IS_BETTER = new Set(['mean_citation_position', 'median_citation_position', 'competitor_dominance']);

function metricLabel(key: string) {
  if (METRIC_LABELS[key]) return METRIC_LABELS[key];
  const [prefix, name] = key.split('.');
  if (prefix === 'group_share') return `Citation share (group): ${categoryLabel(name)}`;
  if (prefix === 'category_share') return `Citation share (category): ${categoryLabel(name)}`;
  return key;
}

// GEO/AEO are 0-100 scores, not percentages; positions, diversity and Jaccard are plain numbers
function isPercent(key: string) {
  return !['geo_score', 'aeo_score', 'source_set_diversity'].includes(key) && !key.includes('position') && !key.endsWith('stability');
}

const isShare = (key: string) => key.startsWith('group_share.') || key.startsWith('category_share.');
// Headline metrics first, then group shares, then category shares
const rowOrder = (key: string) => (key.startsWith('group_share.') ? 1 : key.startsWith('category_share.') ? 2 : 0);

function DeltaTable({ deltas }: { deltas: Record<string, Delta> }) {
  const rows = Object.entries(deltas)
    .filter(([, d]) => d.before != null || d.after != null)
    // A category share that is 0% in both runs says nothing
    .filter(([key, d]) => !isShare(key) || (d.before || 0) !== 0 || (d.after || 0) !== 0)
    .sort(([a], [b]) => rowOrder(a) - rowOrder(b));
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-gray-400 border-b border-gray-700">
            <th className="py-2 pr-4">Metric</th>
            <th className="py-2 pr-4">Before</th>
            <th className="py-2 pr-4">After</th>
            <th className="py-2">Change</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([key, d]) => {
            const fmt = isPercent(key) ? (v: number | null) => fmtPct(v) : (v: number | null) => fmtNum(v, key.endsWith('_score') ? 1 : 2);
            const change = fmtDelta(d.delta, isPercent(key) || key.endsWith('_score') ? ' pts' : '', LOWER_IS_BETTER.has(key));
            return (
              <tr key={key} className="border-b border-gray-800">
                <td className="py-2 pr-4 text-gray-300">{metricLabel(key)}</td>
                <td className="py-2 pr-4">{fmt(d.before)}</td>
                <td className="py-2 pr-4">{fmt(d.after)}</td>
                <td className={`py-2 font-medium ${change.className}`}>{change.text}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function sideText(side: QuerySide) {
  if (!side) return '—';
  if (side.cited) return `Cited #${side.citation_position ?? '?'}`;
  if (side.appeared) return 'Mentioned';
  return side.diagnosis_type === 'content_gap' ? 'Not cited (content gap)' : side.diagnosis_type === 'competitor_advantage' ? 'Not cited (competitor)' : 'Not cited';
}

const ComparePage = () => {
  const router = useRouter();
  const { before, after, before_observation, after_observation } = router.query;
  const [data, setData] = useState<Comparison | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!router.isReady) return;
    if (typeof before !== 'string' || typeof after !== 'string') {
      setError('Open this page with ?before=<run id>&after=<run id>');
      return;
    }
    const params = new URLSearchParams({ before, after });
    if (typeof before_observation === 'string' && typeof after_observation === 'string') {
      params.set('before_observation', before_observation);
      params.set('after_observation', after_observation);
    }
    apiFetch<Comparison>(`/api/sandbox/compare?${params}`)
      .then(setData)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [router.isReady, before, after, before_observation, after_observation]);

  const d = data?.controlled.deltas || {};
  const chartData = RATE_METRICS.filter(([key]) => d[key]).map(([key, label]) => ({
    name: label,
    Before: d[key].before ?? 0,
    After: d[key].after ?? 0,
  }));
  const geo = fmtDelta(data?.controlled.geo_improvement, ' pts');
  const aeo = fmtDelta(data?.controlled.aeo_improvement, ' pts');
  const position = fmtDelta(d.mean_citation_position?.delta, '', true);

  return (
    <>
      <Head>
        <title>Before / After - CiteLabs</title>
      </Head>
      <main className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white">
        <div className="container mx-auto px-4 py-8 max-w-7xl space-y-8">
          <header>
            <button onClick={() => router.push('/')} className="mb-4 text-gray-400 hover:text-white transition-colors">
              ← Back to Home
            </button>
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
              Before / After Comparison
            </h1>
            {data && (
              <p className="mt-2 text-gray-400 text-sm">
                <Link href={`/${data.before.id}`} className="text-blue-300 hover:underline">{data.before.id}</Link>
                {' → '}
                <Link href={`/${data.after.id}`} className="text-blue-300 hover:underline">{data.after.id}</Link>
              </p>
            )}
          </header>

          {error && <div className="bg-red-900/50 border border-red-700 text-red-200 px-4 py-3 rounded-lg">{error}</div>}

          {data && (
            <>
              {data.warnings.map((w) => (
                <div key={w} className="bg-yellow-900/30 border border-yellow-700 text-yellow-200 px-4 py-3 rounded-lg text-sm">{w}</div>
              ))}

              <section className="bg-gray-800/50 rounded-2xl border border-gray-700 p-8 space-y-6">
                <h2 className="text-3xl font-bold">Controlled GEO score (Stage 2)</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <StatTile label="GEO improvement" value={<span className={geo.className}>{geo.text}</span>} hint="GEO(after) − GEO(before)" />
                  <StatTile label="AEO improvement" value={<span className={aeo.className}>{aeo.text}</span>} hint="AEO(after) − AEO(before)" />
                  <StatTile label="Mean citation position" value={<span className={position.className}>{position.text}</span>} hint="Negative is better: cited earlier" />
                </div>
                <div className="bg-gray-900/50 rounded-xl p-6">
                  <div className="w-full h-[360px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} margin={{ bottom: 40 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="name" stroke="#9ca3af" angle={-20} textAnchor="end" interval={0} height={60} />
                        <YAxis stroke="#9ca3af" domain={[0, 100]} />
                        <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                        <Legend verticalAlign="top" height={32} />
                        <Bar dataKey="Before" fill="#6b7280" />
                        <Bar dataKey="After" fill="#8b5cf6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                <DeltaTable deltas={data.controlled.deltas} />
              </section>

              <section className="bg-gray-800/50 rounded-2xl border border-gray-700 p-8">
                <h2 className="text-3xl font-bold mb-2">Per question</h2>
                <p className="text-gray-400 text-sm mb-6">
                  Controlled results before and after, next to the real-world (Stage 1) metrics for the same query.
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-gray-400 border-b border-gray-700">
                        <th className="py-2 pr-4">Question</th>
                        <th className="py-2 pr-4">Before</th>
                        <th className="py-2 pr-4">After</th>
                        <th className="py-2 pr-4">Change</th>
                        <th className="py-2">Real world (Stage 1)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.per_query.map((q) => (
                        <tr key={q.question} className="border-b border-gray-800 align-top">
                          <td className="py-3 pr-4 text-gray-200 max-w-md">{q.question}</td>
                          <td className="py-3 pr-4 whitespace-nowrap">{sideText(q.before)}</td>
                          <td className="py-3 pr-4 whitespace-nowrap">{sideText(q.after)}</td>
                          <td className="py-3 pr-4"><ChangeBadge change={q.change} /></td>
                          <td className="py-3 text-gray-400 whitespace-nowrap">
                            {!q.real_world
                              ? '—'
                              : q.real_world.successful_runs === 0
                                ? <span className="text-amber-400/80">No data (no successful observations)</span>
                                : `cited ${fmtPct(q.real_world.citation_rate, 0)} · mentioned ${fmtPct(q.real_world.mention_rate, 0)} · stability ${fmtNum(q.real_world.source_set_stability)} · ${q.real_world.successful_runs} run(s)`}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              {data.real_world ? (
                <section className="bg-gray-800/50 rounded-2xl border border-gray-700 p-8 space-y-4">
                  <h2 className="text-3xl font-bold">Real-world validation (Stage 1)</h2>
                  <DataDisclaimer text={data.real_world.data_disclaimer} />
                  <p className="text-sm text-gray-400">
                    <Link href={`/observation/${data.real_world.before_observation}`} className="text-blue-300 hover:underline">{data.real_world.before_observation}</Link>
                    {' → '}
                    <Link href={`/observation/${data.real_world.after_observation}`} className="text-blue-300 hover:underline">{data.real_world.after_observation}</Link>
                  </p>
                  <DeltaTable deltas={data.real_world.deltas} />
                </section>
              ) : (
                <section className="bg-gray-800/30 rounded-2xl border border-gray-700 p-6 text-sm text-gray-400">
                  Real-world validation: once the changed page has been indexed, run a new Stage 1 observation and add
                  <code className="mx-1 text-gray-300">&amp;before_observation=…&amp;after_observation=…</code>
                  to this page's URL to compare the two.
                </section>
              )}

              {data.data_disclaimer && <DataDisclaimer text={data.data_disclaimer} />}
            </>
          )}
        </div>
      </main>
    </>
  );
};

export default ComparePage;
