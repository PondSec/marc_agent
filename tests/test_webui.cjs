const test = require("node:test");
const assert = require("node:assert/strict");

const {
  buildActivityClusters,
  buildReferenceHeroView,
  buildRuntimeStatusItems,
  buildUiRoute,
  buildWorkspaceShellView,
  buildPhaseSteps,
  buildSessionOverview,
  buildValidationSnapshot,
  createRefreshController,
  parseUiRoute,
  renderRichText,
  sanitizeAssistantMessageContent,
  sessionBadgeText,
  sessionStatusTone,
  shouldStartRefresh,
  updateRefreshBackoff,
} = require("../webui/app.js");

function makeSession(overrides = {}) {
  return {
    id: "session-1",
    status: "running",
    current_phase: "editing",
    validation_status: "not_run",
    stop_requested: false,
    changed_files: [],
    validation_runs: [],
    blockers: [],
    diagnostics: [],
    tool_calls: [],
    report: null,
    final_response: null,
    last_error: null,
    ...overrides,
  };
}

test("buildActivityClusters fasst wiederholten Modell-Progress zu einem Eintrag zusammen", () => {
  const session = makeSession();
  const logs = [
    {
      timestamp: "2026-03-27T10:00:00.000Z",
      event: "content_generation_progress",
      payload: { type: "heartbeat", path: "src/ui/thread.js", model: "qwen-coder" },
    },
    {
      timestamp: "2026-03-27T10:00:05.000Z",
      event: "content_generation_progress",
      payload: { type: "chunk", path: "src/ui/thread.js", model: "qwen-coder" },
    },
    {
      timestamp: "2026-03-27T10:00:08.000Z",
      event: "content_generation_progress",
      payload: { type: "chunk", path: "src/ui/thread.js", model: "qwen-coder" },
    },
  ];

  const items = buildActivityClusters(session, logs);

  assert.equal(items.length, 1);
  assert.equal(items[0].count, 3);
  assert.match(items[0].text, /thread\.js|streamt|generiert/i);
});

test("buildValidationSnapshot hebt fehlgeschlagene Checks klar hervor", () => {
  const session = makeSession({
    status: "partial",
    current_phase: "blocked",
    validation_status: "failed",
    changed_files: [{ path: "webui/app.js", operation: "write" }],
    validation_runs: [
      {
        command: "npm test -- webui",
        verification_scope: "runtime",
        status: "failed",
        summary: "Thread rail rendering failed in smoke test.",
      },
    ],
  });

  const snapshot = buildValidationSnapshot(session);

  assert.equal(snapshot.tone, "danger");
  assert.equal(snapshot.statusLabel, "Fehler");
  assert.match(snapshot.summary, /smoke test|failed|fehl/i);
});

test("buildSessionOverview gibt fertige Threads als sauberen Erfolgszustand aus", () => {
  const session = makeSession({
    status: "completed",
    current_phase: "completed",
    validation_status: "passed",
    changed_files: [{ path: "webui/styles.css", operation: "write" }],
    report: { summary: "Chat, Worklog und Validierung wurden sauber getrennt." },
    validation_runs: [
      {
        command: "node --test tests/test_webui.cjs",
        verification_scope: "runtime",
        status: "passed",
        summary: "Frontend UI logic checks passed.",
      },
    ],
  });

  const overview = buildSessionOverview(session);

  assert.equal(overview.tone, "success");
  assert.match(overview.summary, /getrennt|Validierung/i);
});

test("buildPhaseSteps markiert blockierte Threads im Validierungsschritt", () => {
  const session = makeSession({
    status: "partial",
    current_phase: "blocked",
    validation_status: "failed",
    changed_files: [{ path: "webui/app.js", operation: "write" }],
    validation_runs: [
      {
        command: "pytest",
        verification_scope: "static",
        status: "failed",
        summary: "One assertion failed.",
      },
    ],
  });

  const steps = buildPhaseSteps(session);
  const validationStep = steps.find((step) => step.label === "Validierung");

  assert.ok(validationStep);
  assert.equal(validationStep.state, "blocked");
  assert.equal(sessionBadgeText(session), "Offen");
  assert.equal(sessionStatusTone(session), "warning");
});

test("sanitizeAssistantMessageContent blendet technische Nachsaetze aus der Hauptantwort aus", () => {
  const raw = [
    "Ich habe den Fehler eingegrenzt und den relevanten Bereich vorbereitet.",
    "Geaendert: webui/app.js.",
    "Validierung: fehlgeschlagen.",
    'internal:web_artifact:[{"path":"webui/app.js"}]',
  ].join("\n\n");

  const cleaned = sanitizeAssistantMessageContent(raw);

  assert.equal(cleaned, "Ich habe den Fehler eingegrenzt und den relevanten Bereich vorbereitet.");
});

test("renderRichText rendert grundlegende Markdown-Struktur fuer den Thread", () => {
  const html = renderRichText([
    "## Ergebnis",
    "",
    "- Punkt eins",
    "- `snake_game.html` aktualisiert",
    "",
    "```html",
    "<main>Hallo</main>",
    "```",
  ].join("\n"));

  assert.match(html, /<h2>Ergebnis<\/h2>/);
  assert.match(html, /<ul class="rich-list">/);
  assert.match(html, /<code>snake_game\.html<\/code>/);
  assert.match(html, /<pre><code>&lt;main&gt;Hallo&lt;\/main&gt;<\/code><\/pre>/);
});

test("shouldStartRefresh dedupliziert inflight und respektiert Mindestabstaende", () => {
  const controller = createRefreshController({
    baseIntervalMs: 8000,
    maxIntervalMs: 30000,
    minGapMs: 1200,
  });

  assert.equal(shouldStartRefresh(controller, { now: 1_000 }), true);
  controller.lastStartedAt = 1_500;
  assert.equal(shouldStartRefresh(controller, { now: 2_000 }), false);
  controller.inflight = Promise.resolve([]);
  assert.equal(shouldStartRefresh(controller, { now: 5_000 }), false);
  controller.inflight = null;
  controller.nextDueAt = 7_000;
  assert.equal(shouldStartRefresh(controller, { now: 6_500 }), false);
  assert.equal(shouldStartRefresh(controller, { now: 6_500, force: true }), true);
});

test("updateRefreshBackoff vergroessert ruhige Polling-Intervalle und setzt sie bei Aenderungen zurueck", () => {
  const controller = createRefreshController({
    baseIntervalMs: 4000,
    maxIntervalMs: 30000,
    minGapMs: 1200,
  });

  updateRefreshBackoff(controller, { changed: false, now: 10_000 });
  assert.equal(controller.currentIntervalMs, 6000);
  assert.equal(controller.nextDueAt, 16_000);

  updateRefreshBackoff(controller, { changed: false, now: 16_000 });
  assert.equal(controller.currentIntervalMs, 9000);
  assert.equal(controller.nextDueAt, 25_000);

  updateRefreshBackoff(controller, { changed: true, now: 25_000 });
  assert.equal(controller.currentIntervalMs, 4000);
  assert.equal(controller.nextDueAt, 29_000);
});

test("parseUiRoute erkennt Settings-Ansicht und Session aus der URL", () => {
  const route = parseUiRoute("http://localhost/?session=abc123&view=settings");

  assert.equal(route.page, "settings");
  assert.equal(route.sessionId, "abc123");
});

test("buildUiRoute baut Workspace- und Settings-URLs stabil auf", () => {
  assert.equal(buildUiRoute({}), "/");
  assert.equal(buildUiRoute({ sessionId: "abc123" }), "/?session=abc123");
  assert.equal(buildUiRoute({ page: "settings" }), "/?view=settings");
  assert.equal(buildUiRoute({ sessionId: "abc123", page: "settings" }), "/?session=abc123&view=settings");
});

test("buildRuntimeStatusItems leitet zentrale Laufzeitinfos konsistent aus dem Shell-State ab", () => {
  const items = buildRuntimeStatusItems({
    config: { model_name: "qwen3-coder:30b" },
    composer: {
      modelName: "",
      accessMode: "approval",
      executionProfile: "balanced",
    },
    workspaces: [{ id: "ws-1", name: "alpha", path: "/tmp/alpha" }],
    sessions: [makeSession({ id: "session-2", workspace_id: "ws-1" })],
    selectedWorkspaceId: "ws-1",
    activeSession: makeSession({
      workspace_id: "ws-1",
      validation_status: "passed",
      status: "completed",
      current_phase: "completed",
    }),
  });

  assert.deepEqual(
    items.map((item) => item.label),
    ["Projekt", "Modell", "Zugriff", "Profil", "Status", "Checks", "Laeufe"],
  );
  assert.equal(items[0].value, "alpha");
  assert.equal(items[1].value, "qwen3-coder:30b");
  assert.equal(items[5].value, "Bestanden");
});

test("buildWorkspaceShellView kapselt Toolbar-Entscheidungen zentral", () => {
  const view = buildWorkspaceShellView({
    config: { model_name: "qwen3-coder:30b" },
    composer: {
      modelName: "",
      accessMode: "approval",
      executionProfile: "balanced",
    },
    workspaces: [{ id: "ws-1", name: "alpha", path: "/tmp/alpha" }],
    sessions: [],
    selectedWorkspaceId: "ws-1",
    activeSession: makeSession({
      workspace_id: "ws-1",
      status: "running",
      current_phase: "editing",
      changed_files: [{ path: "webui/app.js", operation: "write" }],
    }),
  });

  assert.equal(view.workspace.name, "alpha");
  assert.equal(view.statusText, "Aktiv");
  assert.equal(view.canPreview, false);
  assert.equal(view.canCommit, false);
  assert.equal(view.canDeleteSession, false);
  assert.equal(view.canDownloadHandoff, false);
  assert.match(view.subtitle, /alpha|\/tmp\/alpha/i);
});

test("buildReferenceHeroView erzeugt im Startzustand Welcome-Feeds aus dem Workspace-Kontext", () => {
  const hero = buildReferenceHeroView({
    config: { model_name: "qwen3-coder:30b" },
    composer: {
      modelName: "",
      accessMode: "approval",
      executionProfile: "balanced",
    },
    workspaces: [{ id: "ws-1", name: "alpha", path: "/tmp/alpha" }],
    sessions: [
      makeSession({
        id: "session-2",
        workspace_id: "ws-1",
        status: "completed",
        current_phase: "completed",
        updated_at: "2026-03-27T10:00:00.000Z",
      }),
    ],
    selectedWorkspaceId: "ws-1",
    activeSession: null,
    logs: [],
  });

  assert.equal(hero.compact, false);
  assert.equal(hero.welcomeTitle, "Welcome to MARC A2");
  assert.equal(hero.feeds[0].title, "Workspaces");
  assert.equal(hero.feeds[1].title, "Sessions · alpha");
  assert.equal(hero.feeds[2].title, "Operator");
  assert.equal(hero.feeds[0].lines[0].action, "select-workspace");
  assert.equal(hero.feeds[1].lines[0].action, "open-session");
  assert.match(hero.feeds[0].lines[0].meta, /alpha|tmp/i);
});

test("buildReferenceHeroView wechselt bei aktiver Session in den kompakten Modus", () => {
  const hero = buildReferenceHeroView({
    config: { model_name: "qwen3-coder:30b" },
    composer: {
      modelName: "",
      accessMode: "approval",
      executionProfile: "balanced",
    },
    workspaces: [{ id: "ws-1", name: "alpha", path: "/tmp/alpha" }],
    sessions: [],
    selectedWorkspaceId: "ws-1",
    activeSession: makeSession({
      workspace_id: "ws-1",
      status: "running",
      current_phase: "editing",
      changed_files: [{ path: "webui/app.js", operation: "write" }],
    }),
    logs: [],
  });

  assert.equal(hero.compact, true);
  assert.equal(hero.statusText, "Aktiv");
  assert.equal(hero.feeds[1].title, "Sessions · alpha");
  assert.equal(hero.feeds[2].title, "Operator");
  assert.match(hero.locationLabel, /alpha|tmp/i);
});
