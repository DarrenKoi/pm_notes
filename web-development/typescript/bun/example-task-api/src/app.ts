export type Task = {
  id: number;
  title: string;
  done: boolean;
};

type TaskInput = {
  title?: unknown;
};

type App = {
  fetch(request: Request): Promise<Response>;
};

const seedTasks: Task[] = [
  { id: 1, title: "install bun", done: true },
  { id: 2, title: "replace npx with bunx", done: false },
];

export function createApp(initialTasks: Task[] = seedTasks): App {
  const tasks = initialTasks.map((task) => ({ ...task }));
  let nextId = Math.max(...tasks.map((task) => task.id), 0) + 1;

  return {
    async fetch(request: Request): Promise<Response> {
      const url = new URL(request.url);

      if (request.method === "GET" && url.pathname === "/health") {
        return Response.json({
          ok: true,
          runtime: "bun",
          tasks: tasks.length,
        });
      }

      if (request.method === "GET" && url.pathname === "/tasks") {
        return Response.json({
          items: tasks,
          count: tasks.length,
        });
      }

      if (request.method === "POST" && url.pathname === "/tasks") {
        const payload = await parseTaskInput(request);

        if (!payload.ok) {
          return Response.json({ error: payload.error }, { status: 400 });
        }

        const task: Task = {
          id: nextId,
          title: payload.title,
          done: false,
        };

        nextId += 1;
        tasks.push(task);

        return Response.json(task, {
          status: 201,
          headers: {
            Location: `/tasks/${task.id}`,
          },
        });
      }

      return Response.json({ error: "Not found" }, { status: 404 });
    },
  };
}

async function parseTaskInput(
  request: Request,
): Promise<
  | { ok: true; title: string }
  | { ok: false; error: string }
> {
  let payload: TaskInput;

  try {
    payload = (await request.json()) as TaskInput;
  } catch {
    return { ok: false, error: "Body must be valid JSON" };
  }

  const title =
    typeof payload.title === "string" ? payload.title.trim() : "";

  if (!title) {
    return { ok: false, error: "title is required" };
  }

  return { ok: true, title };
}
