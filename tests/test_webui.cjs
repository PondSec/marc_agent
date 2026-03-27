const test = require("node:test");
const assert = require("node:assert/strict");

const {
  buildActivityClusters,
  buildPhaseSteps,
  buildSessionOverview,
  buildValidationSnapshot,
  renderRichText,
  sanitizeAssistantMessageContent,
  sessionBadgeText,
  sessionStatusTone,
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
