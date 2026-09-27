/**
 * Two-stage GEO before/after comparison: flattening stored metrics into the
 * {name: number} maps the worker's compare_run_metrics diffs, and pairing up
 * per-question results. The subtraction itself lives in the worker
 * (geo_metrics.compare_run_metrics) so there is one tested implementation.
 */

export type MetricMap = Record<string, number | null>;

function flattenShares(prefix: string, breakdown: unknown): MetricMap {
  const out: MetricMap = {};
  if (breakdown && typeof breakdown === 'object') {
    for (const [key, value] of Object.entries(breakdown as Record<string, unknown>)) {
      out[`${prefix}.${key}`] = typeof value === 'number' ? value : null;
    }
  }
  return out;
}

/** Stage 2 (controlled) metrics of one run, from its sandbox_scores row. */
export function scoreMetrics(score: {
  geoScore: number;
  aeoScore: number;
  citationRate: number;
  coverage: number | null;
  competitorDominance: number | null;
  mentionRate: number | null;
  strictCitationRate: number | null;
  meanCitationPosition: number | null;
  medianCitationPosition: number | null;
  categoryBreakdown: unknown;
  competitorGroupBreakdown: unknown;
}): MetricMap {
  return {
    geo_score: score.geoScore,
    aeo_score: score.aeoScore,
    citation_rate: score.citationRate,
    coverage: score.coverage,
    competitor_dominance: score.competitorDominance,
    mention_rate: score.mentionRate,
    strict_citation_rate: score.strictCitationRate,
    mean_citation_position: score.meanCitationPosition,
    median_citation_position: score.medianCitationPosition,
    ...flattenShares('category_share', score.categoryBreakdown),
    ...flattenShares('group_share', score.competitorGroupBreakdown),
  };
}

/** Stage 1 (real-world proxy) metrics of one observation. */
export function observationMetrics(observation: {
  mentionRate: number | null;
  citationRate: number | null;
  meanCitationPosition: number | null;
  medianCitationPosition: number | null;
  meanSourceSetDiversity: number | null;
  meanSourceSetStability: number | null;
  meanDomainSetStability: number | null;
  categoryBreakdown: unknown;
  competitorGroupBreakdown: unknown;
}): MetricMap {
  return {
    mention_rate: observation.mentionRate,
    citation_rate: observation.citationRate,
    mean_citation_position: observation.meanCitationPosition,
    median_citation_position: observation.medianCitationPosition,
    source_set_diversity: observation.meanSourceSetDiversity,
    source_set_stability: observation.meanSourceSetStability,
    domain_set_stability: observation.meanDomainSetStability,
    ...flattenShares('category_share', observation.categoryBreakdown),
    ...flattenShares('group_share', observation.competitorGroupBreakdown),
  };
}

type RagRow = {
  question: string;
  didSandboxAppear: boolean;
  sandboxCitations: string[];
  sandboxCitationPosition: number | null;
  diagnosisType: string | null;
  metrics?: { isBrandMentioned: boolean } | null;
};

type Stage1QueryRow = {
  query: string;
  successfulRuns: number;
  mentionRate: number | null;
  citationRate: number | null;
  meanCitationPosition: number | null;
  sourceSetStability: number | null;
};

const key = (question: string) => question.trim().toLowerCase();

function side(row: RagRow | undefined) {
  if (!row) return null;
  return {
    appeared: row.didSandboxAppear,
    cited: row.sandboxCitations.length > 0,
    mentioned: row.metrics?.isBrandMentioned ?? null,
    citation_position: row.sandboxCitationPosition,
    diagnosis_type: row.diagnosisType,
  };
}

type Side = ReturnType<typeof side>;

/** What changed for one question between the runs (lower citation position is better). */
export function classifyChange(b: Side, a: Side): string {
  if (!b || !a) return 'not_in_both_runs';
  if (b.cited !== a.cited) return a.cited ? 'gained_citation' : 'lost_citation';
  if (b.cited && b.citation_position != null && a.citation_position != null && b.citation_position !== a.citation_position) {
    return a.citation_position < b.citation_position ? 'position_improved' : 'position_worsened';
  }
  if (b.appeared !== a.appeared) return a.appeared ? 'gained_appearance' : 'lost_appearance';
  return 'unchanged';
}

/**
 * One entry per question (matched by text, not list position), in the before
 * run's order, then any questions only the after run asked. When the runs were
 * seeded from Stage 1, the matching real-world (proxy) query metrics are attached.
 */
export function perQueryComparison(before: RagRow[], after: RagRow[], stage1Queries: Stage1QueryRow[] = []) {
  const afterByKey = new Map(after.map((r) => [key(r.question), r]));
  const stage1ByKey = new Map(stage1Queries.map((q) => [key(q.query), q]));
  const order = [...before.map((r) => r.question), ...after.map((r) => r.question)];
  const seen = new Set<string>();
  const beforeByKey = new Map(before.map((r) => [key(r.question), r]));

  return order.flatMap((question) => {
    const k = key(question);
    if (seen.has(k)) return [];
    seen.add(k);
    const b = side(beforeByKey.get(k));
    const a = side(afterByKey.get(k));
    const s1 = stage1ByKey.get(k);
    return [
      {
        question,
        before: b,
        after: a,
        change: classifyChange(b, a),
        real_world: s1
          ? {
              successful_runs: s1.successfulRuns,
              // No successful observations means no data, not 0% (older rows stored 0)
              mention_rate: s1.successfulRuns > 0 ? s1.mentionRate : null,
              citation_rate: s1.successfulRuns > 0 ? s1.citationRate : null,
              mean_citation_position: s1.meanCitationPosition,
              source_set_stability: s1.sourceSetStability,
            }
          : null,
      },
    ];
  });
}
