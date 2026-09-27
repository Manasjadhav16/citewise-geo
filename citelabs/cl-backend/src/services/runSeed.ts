import { PrismaClient } from '@prisma/client';

/**
 * Two-stage GEO: validates a Stage 2 run request and builds the seed the worker
 * receives (see async_job.run_full_job in cl-workers).
 *
 * - "before" (or untagged) run with observation_id: seeded from Stage 1. Its
 *   questions are Stage 1's queries and its source pool is Stage 1's union pool,
 *   merged by the worker with its own SERP/LLM discovery.
 * - "after" run: must name a completed baseline_run_id. It reuses the baseline's
 *   exact questions and source pool, and inherits its observation.
 * - no observation and no tag: an ordinary, unseeded run.
 */

export type RunTag = 'before' | 'after';

export type RunSeedRequest = {
  url: string;
  observation_id?: string;
  run_tag?: string;
  baseline_run_id?: string;
};

export type WorkerSeed =
  | { mode: 'stage1'; questions: { q: string; type: string }[]; stage1_sources: Record<string, unknown>[] }
  | { mode: 'baseline'; questions: { q: string; type: string }[]; source_pool: unknown[] };

export type ResolvedRunSeed = {
  seed: WorkerSeed | null;
  observationId: string | null;
  runTag: RunTag | null;
  baselineRunId: string | null;
};

export class RunSeedError extends Error {
  constructor(public readonly statusCode: number, message: string) {
    super(message);
  }
}

/** Compare URLs ignoring scheme, "www.", trailing slash and fragment. */
export function sameUrl(a: string, b: string): boolean {
  const norm = (u: string) => {
    try {
      const parsed = new URL(u);
      const host = parsed.hostname.toLowerCase().replace(/^www\./, '');
      return `${host}${parsed.pathname.replace(/\/+$/, '')}${parsed.search}`;
    } catch {
      return u;
    }
  };
  return norm(a) === norm(b);
}

export async function resolveRunSeed(prisma: PrismaClient, req: RunSeedRequest): Promise<ResolvedRunSeed> {
  const { url, observation_id, run_tag, baseline_run_id } = req;

  if (run_tag !== undefined && run_tag !== 'before' && run_tag !== 'after') {
    throw new RunSeedError(400, "run_tag must be 'before' or 'after'");
  }
  if (baseline_run_id && run_tag !== 'after') {
    throw new RunSeedError(400, "baseline_run_id is only valid with run_tag 'after'");
  }

  if (run_tag === 'after') {
    if (!baseline_run_id) {
      throw new RunSeedError(400, "run_tag 'after' requires baseline_run_id");
    }
    const baseline = await prisma.sandboxRun.findUnique({ where: { id: baseline_run_id } });
    if (!baseline) {
      throw new RunSeedError(404, `Baseline run ${baseline_run_id} not found`);
    }
    if (baseline.status !== 'completed') {
      throw new RunSeedError(409, `Baseline run is '${baseline.status}', not completed`);
    }
    if (!sameUrl(baseline.sandboxUrl, url)) {
      throw new RunSeedError(400, `An "after" run must use the baseline's URL (${baseline.sandboxUrl})`);
    }
    if (observation_id && baseline.observationId && observation_id !== baseline.observationId) {
      throw new RunSeedError(400, 'An "after" run inherits its baseline\'s observation; a different observation_id was given');
    }
    const questions = (Array.isArray(baseline.questionSet) ? baseline.questionSet : []) as { q: string; type: string }[];
    const pool = Array.isArray(baseline.sourcePool) ? baseline.sourcePool : [];
    if (questions.length === 0 || pool.length === 0) {
      throw new RunSeedError(409, 'Baseline run has no stored question set or source pool (it predates two-stage support)');
    }
    return {
      seed: { mode: 'baseline', questions, source_pool: pool },
      observationId: baseline.observationId,
      runTag: 'after',
      baselineRunId: baseline.id,
    };
  }

  if (observation_id) {
    const observation = await prisma.observationRun.findUnique({
      where: { id: observation_id },
      include: { queries: { orderBy: { queryIndex: 'asc' }, include: { sources: true } } },
    });
    if (!observation) {
      throw new RunSeedError(404, `Observation ${observation_id} not found`);
    }
    if (observation.status !== 'completed') {
      throw new RunSeedError(409, `Observation is '${observation.status}', not completed`);
    }
    if (!sameUrl(observation.sandboxUrl, url)) {
      throw new RunSeedError(400, `Observation was made for a different URL (${observation.sandboxUrl})`);
    }
    const questions = observation.queries.map((q) => ({ q: q.query, type: q.queryType || 'intent' }));
    const stage1Sources = observation.queries.flatMap((q) =>
      q.sources.map((s) => ({
        url: s.url,
        domain: s.domain,
        category: s.category,
        competitor_group: s.competitorGroup,
        category_source: s.categorySource,
        stability: s.stability,
      })),
    );
    if (questions.length === 0) {
      throw new RunSeedError(409, 'Observation has no queries');
    }
    return {
      seed: { mode: 'stage1', questions, stage1_sources: stage1Sources },
      observationId: observation.id,
      runTag: (run_tag as RunTag) ?? null,
      baselineRunId: null,
    };
  }

  return { seed: null, observationId: null, runTag: (run_tag as RunTag) ?? null, baselineRunId: null };
}
