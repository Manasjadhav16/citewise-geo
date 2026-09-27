import React from 'react';
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

/** Shared UI for the two-stage GEO views (Stage 1 observation, run metrics, before/after comparison). */

export const CATEGORY_LABELS: Record<string, string> = {
  target_company: 'Your site',
  direct_competitor: 'Direct competitor',
  indirect_competitor: 'Indirect competitor',
  government: 'Government',
  reference: 'Reference',
  media: 'Media',
  community: 'Community',
  industry_organization: 'Industry org',
  academic: 'Academic',
  other: 'Other',
};

export const GROUP_LABELS: Record<string, string> = {
  target: 'Your site',
  direct_business: 'Direct business competitors',
  ai_visibility: 'AI visibility competitors',
};

export const CATEGORY_COLORS: Record<string, string> = {
  target_company: '#10b981',
  target: '#10b981',
  direct_competitor: '#ef4444',
  direct_business: '#ef4444',
  indirect_competitor: '#f97316',
  government: '#3b82f6',
  reference: '#8b5cf6',
  media: '#ec4899',
  community: '#f59e0b',
  industry_organization: '#06b6d4',
  academic: '#6366f1',
  other: '#6b7280',
  ai_visibility: '#8b5cf6',
};

export const categoryLabel = (key: string) => CATEGORY_LABELS[key] || GROUP_LABELS[key] || key;

export function fmtPct(value: number | null | undefined, digits = 1): string {
  return value == null ? '—' : `${value.toFixed(digits)}%`;
}

export function fmtNum(value: number | null | undefined, digits = 2): string {
  return value == null ? '—' : Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

/** Signed change; for metrics where lower is better (citation position) pass lowerIsBetter. */
export function fmtDelta(delta: number | null | undefined, unit = '', lowerIsBetter = false) {
  if (delta == null) return { text: '—', className: 'text-gray-400' };
  const good = lowerIsBetter ? delta < 0 : delta > 0;
  const bad = lowerIsBetter ? delta > 0 : delta < 0;
  return {
    text: `${delta > 0 ? '+' : ''}${delta.toFixed(Math.abs(delta) < 10 ? 2 : 1)}${unit}`,
    className: good ? 'text-green-400' : bad ? 'text-red-400' : 'text-gray-300',
  };
}

export function DataDisclaimer({ text }: { text?: string | null }) {
  return (
    <div className="bg-amber-900/20 border border-amber-700/60 text-amber-200 rounded-xl px-4 py-3 text-sm">
      <strong>About this data:</strong>{' '}
      {text ||
        'Real-world observations come from Gemini with Google Search grounding: an approximate proxy for Google AI Overview citations, not real AI Overview data.'}
    </div>
  );
}

export function StatTile({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <div className="bg-gray-900/60 border border-gray-700 rounded-xl p-5" title={hint}>
      <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">{label}</h3>
      <p className="text-3xl font-bold text-white">{value}</p>
      {hint && <p className="text-xs text-gray-500 mt-2">{hint}</p>}
    </div>
  );
}

export function CategoryChip({ category }: { category: string }) {
  const color = CATEGORY_COLORS[category] || CATEGORY_COLORS.other;
  return (
    <span
      className="inline-block text-xs px-2 py-0.5 rounded-full border whitespace-nowrap"
      style={{ color, borderColor: `${color}80`, backgroundColor: `${color}1a` }}
    >
      {categoryLabel(category)}
    </span>
  );
}

/** Horizontal bars of % share per category, largest first; zero shares hidden. */
export function ShareChart({ breakdown, title }: { breakdown?: Record<string, number> | null; title: string }) {
  const rows = Object.entries(breakdown || {})
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([key, value]) => ({ key, name: categoryLabel(key), value: Number(value.toFixed(1)) }));
  return (
    <div className="bg-gray-900/50 rounded-xl p-6">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      {rows.length === 0 ? (
        <p className="text-gray-500 text-sm">No citations recorded.</p>
      ) : (
        <div className="w-full" style={{ height: Math.max(120, rows.length * 38) }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={rows} layout="vertical" margin={{ left: 20, right: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" domain={[0, 100]} unit="%" stroke="#9ca3af" />
              <YAxis type="category" dataKey="name" width={170} stroke="#9ca3af" />
              <Tooltip
                formatter={(v: number) => [`${v}%`, 'Share of citations']}
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
              />
              <Bar dataKey="value">
                {rows.map((row) => (
                  <Cell key={row.key} fill={CATEGORY_COLORS[row.key] || CATEGORY_COLORS.other} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

const CHANGE_STYLES: Record<string, { label: string; className: string }> = {
  gained_citation: { label: 'Gained citation', className: 'bg-green-500/20 text-green-300' },
  position_improved: { label: 'Position improved', className: 'bg-green-500/20 text-green-300' },
  gained_appearance: { label: 'Now mentioned', className: 'bg-emerald-500/20 text-emerald-300' },
  unchanged: { label: 'Unchanged', className: 'bg-gray-700 text-gray-300' },
  lost_appearance: { label: 'No longer mentioned', className: 'bg-orange-500/20 text-orange-300' },
  position_worsened: { label: 'Position worsened', className: 'bg-red-500/20 text-red-300' },
  lost_citation: { label: 'Lost citation', className: 'bg-red-500/20 text-red-300' },
  not_in_both_runs: { label: 'Only in one run', className: 'bg-gray-700 text-gray-400' },
};

export function ChangeBadge({ change }: { change: string }) {
  const style = CHANGE_STYLES[change] || CHANGE_STYLES.unchanged;
  return <span className={`text-xs px-3 py-1 rounded-full whitespace-nowrap ${style.className}`}>{style.label}</span>;
}
