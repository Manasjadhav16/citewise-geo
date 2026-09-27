import { FastifyInstance } from 'fastify';
import { PrismaClient } from '@prisma/client';
import { worker } from '../services/pythonClient';
import { observationMetrics, perQueryComparison, scoreMetrics } from '../services/compareRuns';
import { sameUrl } from '../services/runSeed';

const prisma = new PrismaClient();

/**
 * Two-stage GEO before/after comparison.
 *
 * GET /api/sandbox/compare?before=<run_id>&after=<run_id>
 *     [&before_observation=<obs_id>&after_observation=<obs_id>]
 *
 * - controlled: Stage 2 score deltas, incl. GEO Improvement = GEO(after) - GEO(before)
 * - per_query: each question's before/after result, with the Stage 1 (real-world
 *   proxy) metrics for the same query when the runs were seeded from Stage 1
 * - real_world: Stage 1 deltas, when two observations are given (e.g. one made
 *   before the website change and one made after it has been indexed)
 */
export default async function compareRoutes(app: FastifyInstance) {
  app.get('/api/sandbox/compare', async (request, reply) => {
    const { before, after, before_observation, after_observation } = request.query as Record<string, string | undefined>;
    if (!before || !after) {
      return reply.status(400).send({ error: 'before and after run ids are required' });
    }
    if (Boolean(before_observation) !== Boolean(after_observation)) {
      return reply.status(400).send({ error: 'before_observation and after_observation must be given together' });
    }

    const include = {
      scores: true,
      ragResults: { include: { metrics: true } },
    } as const;
    const [beforeRun, afterRun] = await Promise.all([
      prisma.sandboxRun.findUnique({ where: { id: before }, include }),
      prisma.sandboxRun.findUnique({ where: { id: after }, include }),
    ]);
    for (const [label, run] of [['before', beforeRun], ['after', afterRun]] as const) {
      if (!run) {
        return reply.status(404).send({ error: `${label} run not found` });
      }
      if (run.status !== 'completed' || !run.scores) {
        return reply.status(409).send({ error: `${label} run is '${run.status}' and has no final scores yet` });
      }
    }
    const b = beforeRun!;
    const a = afterRun!;

    const warnings: string[] = [];
    if (a.baselineRunId !== b.id) {
      warnings.push('The after run was not created with this run as its baseline, so its questions and source pool may differ.');
    }
    if (!sameUrl(a.sandboxUrl, b.sandboxUrl)) {
      warnings.push(`The runs analysed different URLs (${b.sandboxUrl} vs ${a.sandboxUrl}).`);
    }
    const questions = (run: typeof b) => new Set(run.ragResults.map((r) => r.question.trim().toLowerCase()));
    const bq = questions(b);
    const aq = questions(a);
    if (bq.size !== aq.size || [...bq].some((q) => !aq.has(q))) {
      warnings.push('The runs asked different questions; per-query changes cover only the questions both asked.');
    }

    let stage1Queries: Awaited<ReturnType<typeof prisma.observationQuery.findMany>> = [];
    let disclaimer: string | null = null;
    if (b.observationId) {
      stage1Queries = await prisma.observationQuery.findMany({ where: { observationId: b.observationId } });
      const obs = await prisma.observationRun.findUnique({ where: { id: b.observationId }, select: { dataDisclaimer: true } });
      disclaimer = obs?.dataDisclaimer ?? null;
    }

    try {
      const controlled = await worker.compare({ before: scoreMetrics(b.scores!), after: scoreMetrics(a.scores!) });

      let realWorld = null;
      if (before_observation && after_observation) {
        const [bo, ao] = await Promise.all([
          prisma.observationRun.findUnique({ where: { id: before_observation } }),
          prisma.observationRun.findUnique({ where: { id: after_observation } }),
        ]);
        if (!bo || !ao) {
          return reply.status(404).send({ error: 'observation not found' });
        }
        if (bo.status !== 'completed' || ao.status !== 'completed') {
          return reply.status(409).send({ error: 'both observations must be completed' });
        }
        if (!sameUrl(bo.sandboxUrl, ao.sandboxUrl)) {
          warnings.push('The two observations were made for different URLs.');
        }
        realWorld = {
          before_observation: bo.id,
          after_observation: ao.id,
          data_disclaimer: ao.dataDisclaimer ?? bo.dataDisclaimer,
          ...(await worker.compare({ before: observationMetrics(bo), after: observationMetrics(ao) })),
        };
      }

      return reply.status(200).send({
        before: { id: b.id, run_tag: b.runTag, seed_mode: b.seedMode, observation_id: b.observationId, created_at: b.createdAt },
        after: { id: a.id, run_tag: a.runTag, seed_mode: a.seedMode, baseline_run_id: a.baselineRunId, created_at: a.createdAt },
        warnings,
        controlled,
        per_query: perQueryComparison(b.ragResults, a.ragResults, stage1Queries),
        real_world: realWorld,
        data_disclaimer: disclaimer,
      });
    } catch (err) {
      app.log.error({ err, before, after }, 'Comparison failed');
      return reply.status(502).send({ error: 'Comparison failed', detail: err instanceof Error ? err.message : String(err) });
    }
  });
}
