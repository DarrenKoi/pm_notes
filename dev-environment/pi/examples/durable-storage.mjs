// Offline Pi Durable 1.0.1 smoke: SQLite documents and a checkpointed task.
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { createModels } from "@earendil-works/pi-ai/models";
import {
  createRegistry, defineDoc, defineExtension, defineTask, Harness,
} from "@earendil-works/pi-durable";
import { openNodeSqliteStorage } from "@earendil-works/pi-durable/storage/sqlite/node";

const context = BACKGROUND_CONTEXT;
const databasePath = resolve(process.argv[2] ?? "./durable-smoke.sqlite");
const Counter = defineDoc({
  kind: "example.counter", version: 1, scope: "conversation",
  history: "latest", fork: "initial", initial: () => ({ count: 0 }),
});
const Finish = defineTask({
  name: "example.finish", version: 1,
  initial: () => ({ phase: "finish" }),
  phases: {
    finish: async (task, runtime, taskContext) => {
      await runtime.commit(
        () => ({ status: "terminal", outcome: { status: "completed", result: task.input.text } }),
        taskContext,
      );
    },
  },
  abort: async (_task, runtime, taskContext) => {
    await runtime.commit(() => ({ status: "terminal", outcome: { status: "aborted" } }), taskContext);
  },
});
const registry = createRegistry();
registry.install(defineExtension({ name: "example", tasks: [Finish] }));
const open = async () => Harness.open(
  await openNodeSqliteStorage(databasePath),
  { models: createModels(), registry }, context,
);

// Do not start scheduling: persist a document and a pending task, then close.
const first = await open();
const root = await first.root(context);
const before = await first.snapshot(Counter, root.id, context);
const expectedCount = (before?.count ?? 0) + 1;
const taskId = await root.commit(async (tx) => {
  (await tx.doc(Counter, root.id)).count = expectedCount;
  return tx.createTask(Finish, { text: "Recovered without an LLM" }, {
    ownership: { kind: "conversation" },
  });
}, context);
assert.equal((await first.getTask(taskId, context)).state.status, "pending");
const rootId = root.id;
await first.close(context);

// A new harness over the same file recovers the same root and saved state.
const second = await open();
try {
  const recoveredRoot = await second.root(context);
  assert.equal(recoveredRoot.id, rootId);
  assert.equal((await second.snapshot(Counter, rootId, context)).count, expectedCount);
  assert.equal((await second.getTask(taskId, context)).state.status, "pending");
  second.resume();
  const finished = await second.waitForTask(taskId, context);
  assert.equal(finished.state.outcome.status, "completed");
  assert.equal(finished.state.outcome.result, "Recovered without an LLM");
  console.log(JSON.stringify({ databasePath, rootId, count: expectedCount, taskId, outcome: finished.state.outcome }, null, 2));
} finally {
  await second.close(context);
}
