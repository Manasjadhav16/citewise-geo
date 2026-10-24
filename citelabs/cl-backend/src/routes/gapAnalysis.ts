import { randomUUID } from 'crypto';
import { FastifyInstance } from 'fastify';
import { Prisma, PrismaClient } from '@prisma/client';
import { worker, WorkerRequestError } from '../services/pythonClient';

const prisma = new PrismaClient();

// The report carries quotes and claims for every query; well under this in practice
const RESULT_BODY_LIMIT = 10 * 1024 * 1024;
// Page text sent to the worker per page (the worker passes at most 40,000 characters to one call)
const MAX_CACHED_CHARS = 40000;

type ResultBody = { status?: string; report?: Record<string, any>; error?: string; observation_id?: string };

/** URL identity ignoring scheme, "www.", trailing slash and fragment (as runSeed.sameUrl). */
function pageKey(url: string): string {
  try {
    const parsed = new URL(url);
    return `${parsed.hostname.toLowerCase().replace(/^www\./, '')}${parsed.pathname.replace(/\/+$/, '')}${parsed.search}`;
  } catch {
    return url;
  }
}

/** Spellings a stored URL may have: with/without "www." and trailing slash. */
function urlVariants(url: string): string[] {
  const variants = new Set<string>([url]);
  try {
    const parsed = new URL(url);
    const host = parsed.hostname.replace(/^www\./, '');
    const path = parsed.pathname.replace(/\/+$/, '');
    for (const h of [host, `www.${host}`]) {
      for (const p of [path, `${path}/`]) {
        variants.add(`${parsed.protocol}//${h}${p || '/'}${parsed.search}`);
      }
    }
  } catch {
    /* keep the URL as given */
  }
  return [...variants];
}

/**
 * Worker payload for one observation: each query's union source pool (with how
 * often and how high each source was cited), and already-crawled page text from
 * Stage 2 runs (sandbox_competitors / sandbox_pages) so the worker only crawls
 * what is missing.
 */
async function buildPayload(observationId: string) {
  const observation = await prisma.observationRun.findUnique({
    where: { id: observationId },
    include: {
      queries: {
        orderBy: { queryIndex: 'asc' },
        include: { sources: true, responses: { include: { citations: true } } },
      },
    },
  });
  if (!observation) return null;

  const queries = observation.queries.map((q) => {
    const bestPosition = new Map<string, number>();
    for (const response of q.responses) {
      for (const c of response.citations) {
        if (!c.url) continue;
        const key = pageKey(c.url);
        bestPosition.set(key, Math.min(bestPosition.get(key) ?? Infinity, c.position));
      }
    }
    return {
      query_index: q.queryIndex,
      query: q.query,
      sources: q.sources.map((s) => ({
        url: s.url,
        domain: s.domain,
        title: s.title,
        category: s.category,
        competitor_group: s.competitorGroup,
        appearances: s.appearances,
        stability: s.stability,
        best_position: s.url ? bestPosition.get(pageKey(s.url)) ?? null : null,
      })),
    };
  });

  const candidates = new Map<string, string>(); // page key -> URL
  for (const q of queries) for (const s of q.sources) if (s.url) candidates.set(pageKey(s.url), s.url);
  candidates.set(pageKey(observation.sandboxUrl), observation.sandboxUrl);
  const variants = [...candidates.values()].flatMap(urlVariants);

  const [competitors, pages] = await Promise.all([
    prisma.sandboxCompetitor.findMany({
      where: { url: { in: variants }, fullText: { not: null } },
      select: { url: true, fullText: true },
      orderBy: { createdAt: 'desc' },
    }),
    prisma.sandboxPage.findMany({
      where: { url: { in: variants }, fullText: { not: null } },
      select: { url: true, fullText: true },
      orderBy: { createdAt: 'desc' },
    }),
  ]);
  const cachedTexts: Record<string, string> = {};
  const seen = new Set<string>();
  for (const row of [...pages, ...competitors]) {
    const key = pageKey(row.url);
    const text = (row.fullText || '').trim();
    if (seen.has(key) || text.length < 200) continue; // newest non-trivial text per page
    seen.add(key);
    cachedTexts[candidates.get(key) ?? row.url] = text.slice(0, MAX_CACHED_CHARS);
  }

  const context = (observation.context ?? {}) as Record<string, unknown>;
  return {
    observation,
    payload: {
      observation_id: observation.id,
      sandbox_url: observation.sandboxUrl,
      brand_name: typeof context.brand_name === 'string' ? context.brand_name : null,
      queries,
      cached_texts: cachedTexts,
    },
  };
}

function toApi(row: { id: string; observationId: string; status: string; report: Prisma.JsonValue | null; error: string | null; modelCalls: number | null; createdAt: Date; updatedAt: Date }) {
  return {
    id: row.id,
    observation_id: row.observationId,
    status: row.status,
    model_calls: row.modelCalls,
    error: row.error,
    created_at: row.createdAt,
    updated_at: row.updatedAt,
    report: row.report,
  };
}

export default async function gapAnalysisRoutes(app: FastifyInstance) {
  // POST /api/observation/:id/gap-analysis - start a report for a completed observation
  app.post('/api/observation/:id/gap-analysis', async (request, reply) => {
    const { id } = request.params as { id: string };
    const running = await prisma.gapReport.findFirst({ where: { observationId: id, status: 'running' } });
    if (running) {
      return reply.status(409).send({ error: 'A gap analysis is already running for this observation', report_id: running.id });
    }
    const built = await buildPayload(id);
    if (!built) return reply.status(404).send({ error: `Observation ${id} not found` });
    if (built.observation.status !== 'completed') {
      return reply.status(409).send({ error: `Observation is '${built.observation.status}', not completed` });
    }
    if (built.payload.queries.length === 0) return reply.status(409).send({ error: 'Observation has no queries' });

    const reportId = `gap_${Date.now()}_${randomUUID().slice(0, 8)}`;
    try {
      // The worker makes one tiny model call before accepting: on failure nothing is started or saved
      await worker.startGapAnalysis({ report_id: reportId, payload: built.payload });
    } catch (err) {
      if (err instanceof WorkerRequestError && (err.status === 429 || err.status === 503)) {
        return reply.status(err.status).send({ error: err.detail || 'Gap analysis not started', saved: false });
      }
      const message = err instanceof Error ? err.message : String(err);
      return reply.status(502).send({ error: 'Failed to start gap analysis', detail: message, saved: false });
    }
    // Upsert: the worker's result can only arrive minutes later, but never fail on ordering
    await prisma.gapReport.upsert({
      where: { id: reportId },
      create: { id: reportId, observationId: id, status: 'running' },
      update: {},
    });
    return reply.status(202).send({
      report_id: reportId,
      status: 'running',
      cached_pages: Object.keys(built.payload.cached_texts).length,
    });
  });

  // POST /api/gap-analysis/:id/result - the worker's report (or failure)
  app.post('/api/gap-analysis/:id/result', { bodyLimit: RESULT_BODY_LIMIT }, async (request, reply) => {
    const { id } = request.params as { id: string };
    const body = (request.body || {}) as ResultBody;
    const existing = await prisma.gapReport.findUnique({ where: { id } });
    const observationId = existing?.observationId ?? body.observation_id;
    if (!observationId) return reply.status(404).send({ error: `Gap report ${id} not found` });
    const status = body.status === 'completed' ? 'completed' : 'failed';
    const calls = body.report?.model_calls?.total;
    const data = {
      status,
      report: body.report ? (body.report as Prisma.InputJsonValue) : Prisma.DbNull,
      error: status === 'failed' ? body.error || 'unknown error' : null,
      modelCalls: typeof calls === 'number' ? calls : null,
    };
    await prisma.gapReport.upsert({ where: { id }, create: { id, observationId, ...data }, update: data });
    return reply.status(200).send({ message: 'Gap report stored' });
  });

  // GET /api/gap-analysis/:id - one report
  app.get('/api/gap-analysis/:id', async (request, reply) => {
    const { id } = request.params as { id: string };
    const row = await prisma.gapReport.findUnique({ where: { id } });
    if (!row) return reply.status(404).send({ error: `Gap report ${id} not found` });
    return reply.status(200).send(toApi(row));
  });

  // GET /api/observation/:id/gap-analysis - the observation's latest report
  app.get('/api/observation/:id/gap-analysis', async (request, reply) => {
    const { id } = request.params as { id: string };
    const row = await prisma.gapReport.findFirst({ where: { observationId: id }, orderBy: { createdAt: 'desc' } });
    if (!row) return reply.status(404).send({ error: 'No gap analysis for this observation yet' });
    return reply.status(200).send(toApi(row));
  });
}
