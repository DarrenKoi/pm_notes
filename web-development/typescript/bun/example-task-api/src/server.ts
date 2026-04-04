import { createApp } from "./app.ts";

const port = Number(Bun.env.PORT ?? 3000);
const app = createApp();

const server = Bun.serve({
  port,
  fetch: app.fetch,
});

console.log(`Task API running at http://localhost:${server.port}`);
