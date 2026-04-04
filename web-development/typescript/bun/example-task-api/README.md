# example-task-api

Small Bun + TypeScript example for the Bun tutorial.

## Run

```bash
bun install
bun run dev
```

Server URL: `http://localhost:3000`

## Test

```bash
bun test
```

## Endpoints

```bash
curl http://localhost:3000/health
curl http://localhost:3000/tasks
curl -X POST http://localhost:3000/tasks \
  -H "Content-Type: application/json" \
  -d "{\"title\":\"learn bun\"}"
```
