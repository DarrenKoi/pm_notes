import { beforeEach, describe, expect, test } from "bun:test";

import { createApp } from "./app.ts";

describe("task api", () => {
  let app = createApp();

  beforeEach(() => {
    app = createApp();
  });

  test("GET /health returns runtime info", async () => {
    const response = await app.fetch(new Request("http://localhost/health"));
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body).toEqual({
      ok: true,
      runtime: "bun",
      tasks: 2,
    });
  });

  test("GET /tasks returns seed tasks", async () => {
    const response = await app.fetch(new Request("http://localhost/tasks"));
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.count).toBe(2);
    expect(body.items[0].title).toBe("install bun");
  });

  test("POST /tasks creates a new task", async () => {
    const response = await app.fetch(
      new Request("http://localhost/tasks", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ title: "write bun tutorial" }),
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(201);
    expect(response.headers.get("Location")).toBe("/tasks/3");
    expect(body).toMatchObject({
      id: 3,
      title: "write bun tutorial",
      done: false,
    });
  });

  test("POST /tasks validates title", async () => {
    const response = await app.fetch(
      new Request("http://localhost/tasks", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ title: "   " }),
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body).toEqual({
      error: "title is required",
    });
  });
});
