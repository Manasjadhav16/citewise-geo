import 'dotenv/config';
import Fastify, { FastifyInstance } from 'fastify';
import cors from '@fastify/cors';

import crawlRoutes from './routes/crawl';
import schemaRoutes from './routes/schema';
import simulationRoutes from './routes/simulation';
import sandboxRoutes from './routes/sandbox';
import stagingRoutes from './routes/staging';
import chatRoutes from './routes/chat';

export const createServer = (): FastifyInstance => {
  const app = Fastify({
    logger: true,
  });

  app.register(cors, {
    origin: true,
  });

  app.register(crawlRoutes);
  app.register(schemaRoutes);
  app.register(simulationRoutes);
  app.register(sandboxRoutes);
  app.register(stagingRoutes);
  app.register(chatRoutes);

  return app;
};

const start = async () => {
  const app = createServer();

  try {
    await app.listen({ port: 4000, host: '0.0.0.0' });
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
};

if (require.main === module) {
  void start();
}

