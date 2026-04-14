const test = require("node:test");
const assert = require("node:assert/strict");

const {
  buildActivityClusters,
  buildConversationWindow,
  buildReferenceHeroView,
  buildRuntimeStatusItems,
  buildThreadRunFeed,
  buildThreadPresentationView,
  buildUiRoute,
  buildWorkspaceShellView,
  workspaceModalTargetPathFrom,
  workspaceGitBranchSuggestions,
  buildPhaseSteps,
  buildSessionOverview,
  buildValidationSnapshot,
  createRefreshController,
  currentRunLogs,
  currentThoughtFrom,
  expandedWorkspaceBrowserPathsFor,
  findBlockingRunForSubmission,
  formatSessionElapsed,
  parseUiRoute,
  pickFirstWorkspaceBrowserFile,
  renderRichText,
  sanitizeAssistantMessageContent,
  sessionBadgeText,
  sessionStatusTone,
  shouldStartRefresh,
  submissionSessionId,
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

test("currentRunLogs ignoriert Logeintraege aus frueheren Runs derselben Session", () => {
  const logs = [
    {
      timestamp: "2026-04-02T10:00:00.000Z",
      event: "task_started",
      payload: { task: "alter Lauf" },
    },
    {
      timestamp: "2026-04-02T10:00:04.000Z",
      event: "tool_requested",
      payload: { tool_name: "read_file", thought_summary: "Alten Kontext lesen." },
    },
    {
      timestamp: "2026-04-02T10:05:00.000Z",
      event: "task_started",
      payload: { task: "neuer Lauf" },
    },
    {
      timestamp: "2026-04-02T10:05:03.000Z",
      event: "decision",
      payload: { thought_summary: "Ich analysiere jetzt die aktuelle UI-Stelle." },
    },
  ];

  const current = currentRunLogs(logs);

  assert.equal(current.length, 2);
  assert.equal(current[0].event, "task_started");
  assert.match(JSON.stringify(current), /neuer Lauf|aktuelle UI-Stelle/i);
  assert.doesNotMatch(JSON.stringify(current), /alter Lauf|Alten Kontext/i);
});

test("currentThoughtFrom priorisiert den letzten formulierten Agenten-Schritt vor Modell-Heartbeat-Rauschen", () => {
  const logs = [
    {
      timestamp: "2026-04-02T10:05:00.000Z",
      event: "task_started",
      payload: { task: "UI fixen" },
    },
    {
      timestamp: "2026-04-02T10:05:02.000Z",
      event: "decision",
      payload: { thought_summary: "Ich analysiere jetzt das bestehende Projekt und suche den eigentlichen UI-Fehler." },
    },
    {
      timestamp: "2026-04-02T10:05:03.000Z",
      event: "content_generation_progress",
      payload: { type: "status", stage: "request_started", path: "webui/app.js" },
    },
  ];

  const thought = currentThoughtFrom({
    activeSession: makeSession({ status: "running", current_phase: "exploring" }),
    logs,
  });

  assert.match(thought, /analysiere jetzt das bestehende projekt/i);
  assert.doesNotMatch(thought, /Modellstart/i);
});

test("buildConversationWindow laesst bei fertigen Threads nur die letzte Frage und Antwort offen", () => {
  const entries = [
    { type: "message", message: { role: "user", content: "alt 1" } },
    { type: "message", message: { role: "assistant", content: "alt 2" } },
    { type: "message", message: { role: "user", content: "letzte Frage" } },
    { type: "message", message: { role: "assistant", content: "letzte Antwort" } },
  ];

  const window = buildConversationWindow(entries, { running: false });

  assert.equal(window.hidden.length, 2);
  assert.equal(window.visible.length, 2);
  assert.equal(window.visible[0].message.content, "letzte Frage");
  assert.equal(window.visible[1].message.content, "letzte Antwort");
});

test("buildConversationWindow trennt letzte Nutzerfrage und letzte Antwort fuer den Arbeitsverlauf", () => {
  const entries = [
    { type: "message", message: { role: "user", content: "alt 1" } },
    { type: "message", message: { role: "assistant", content: "alt 2" } },
    { type: "message", message: { role: "user", content: "letzte Frage" } },
    { type: "message", message: { role: "assistant", content: "letzte Antwort" } },
  ];

  const window = buildConversationWindow(entries, { running: false });

  assert.equal(window.leading.length, 1);
  assert.equal(window.leading[0].message.content, "letzte Frage");
  assert.equal(window.trailing.length, 1);
  assert.equal(window.trailing[0].message.content, "letzte Antwort");
});

test("buildThreadRunFeed zeigt abgeschlossene Schritte schlicht und nur den letzten aktiven Schritt shiny-faehig", () => {
  const session = makeSession({
    status: "running",
    current_phase: "editing",
    updated_at: "2026-04-14T20:44:10.000Z",
  });
  const logs = [
    {
      timestamp: "2026-04-14T20:44:00.000Z",
      event: "task_started",
      payload: { task: "UI fixen" },
    },
    {
      timestamp: "2026-04-14T20:44:02.000Z",
      event: "decision",
      payload: { thought_summary: "Ich analysiere jetzt die Thread-Ansicht." },
    },
    {
      timestamp: "2026-04-14T20:44:03.000Z",
      event: "tool_result",
      payload: { tool_name: "search_in_files", success: true, message: "3 matches found" },
    },
    {
      timestamp: "2026-04-14T20:44:05.000Z",
      event: "decision",
      payload: { thought_summary: "Ich fasse jetzt die eigentlichen Dateien an." },
    },
    {
      timestamp: "2026-04-14T20:44:07.000Z",
      event: "content_generation_progress",
      payload: { type: "status", stage: "request_started", path: "webui/app.js" },
    },
  ];

  const feed = buildThreadRunFeed(session, logs);

  assert.equal(feed.history.length, 2);
  assert.match(feed.history[0].text, /analysiere jetzt die thread-ansicht/i);
  assert.match(feed.history[1].text, /3 Treffer gefunden/i);
  assert.ok(feed.active);
  assert.match(feed.active.text, /fasse jetzt die eigentlichen dateien an/i);
  assert.equal(feed.active.active, true);
});

test("pickFirstWorkspaceBrowserFile waehlt die erste sichtbare Datei aus verschachtelten Ordnern", () => {
  const tree = [
    {
      kind: "directory",
      name: "src",
      path: "src",
      children: [
        {
          kind: "directory",
          name: "components",
          path: "src/components",
          children: [{ kind: "file", name: "Panel.js", path: "src/components/Panel.js", children: [] }],
        },
      ],
    },
    { kind: "file", name: "README.md", path: "README.md", children: [] },
  ];

  const picked = pickFirstWorkspaceBrowserFile(tree);

  assert.equal(picked, "src/components/Panel.js");
});

test("expandedWorkspaceBrowserPathsFor markiert alle Ordner entlang des Dateipfads", () => {
  const expanded = expandedWorkspaceBrowserPathsFor("src/components/Panel.js");

  assert.deepEqual(expanded, {
    src: true,
    "src/components": true,
  });
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

test("submissionSessionId verwendet keine laufende Session fuer neue Auftraege", () => {
  assert.equal(
    submissionSessionId({
      activeSessionId: "session-1",
      activeSession: makeSession({ id: "session-1", status: "running" }),
    }),
    null,
  );

  assert.equal(
    submissionSessionId({
      activeSessionId: "session-1",
      activeSession: makeSession({ id: "session-1", status: "completed" }),
    }),
    "session-1",
  );
});

test("findBlockingRunForSubmission erkennt aktive und globale Run-Blocker", () => {
  const active = makeSession({ id: "active", status: "running" });
  assert.equal(
    findBlockingRunForSubmission({
      activeSession: active,
      sessions: [],
    })?.id,
    "active",
  );

  const queued = makeSession({ id: "queued", status: "queued" });
  assert.equal(
    findBlockingRunForSubmission({
      activeSession: makeSession({ id: "done", status: "completed" }),
      sessions: [makeSession({ id: "done", status: "completed" }), queued],
    })?.id,
    "queued",
  );

  assert.equal(
    findBlockingRunForSubmission({
      activeSession: makeSession({ id: "done", status: "completed" }),
      sessions: [makeSession({ id: "done", status: "completed" })],
    }),
    null,
  );
});

test("formatSessionElapsed formatiert kurze und laengere Laufzeiten stabil", () => {
  assert.equal(
    formatSessionElapsed({
      created_at: "2026-04-02T10:00:00.000Z",
      updated_at: "2026-04-02T10:01:09.000Z",
    }),
    "1m 9s",
  );

  assert.equal(
    formatSessionElapsed({
      created_at: "2026-04-02T10:00:00.000Z",
      updated_at: "2026-04-02T12:07:00.000Z",
    }),
    "2h 7m",
  );

  assert.equal(formatSessionElapsed({ created_at: "", updated_at: "" }), "");
});

test("buildThreadPresentationView leitet laufende Threads in eine kompakte Live-Ansicht ueber", () => {
  const session = makeSession({
    status: "running",
    current_phase: "editing",
    created_at: "2026-04-02T10:00:00.000Z",
    updated_at: "2026-04-02T10:00:42.000Z",
    changed_files: [{ path: "webui/app.js", operation: "write", diff: "@@ -1 +1 @@\n-old\n+new" }],
  });
  const logs = [
    {
      timestamp: "2026-04-02T10:00:12.000Z",
      event: "tool_call",
      payload: { tool: "rg", mode: "read", summary: "Suche nach Thread-View-Komponenten" },
    },
    {
      timestamp: "2026-04-02T10:00:24.000Z",
      event: "content_generation_progress",
      payload: { type: "chunk", path: "webui/app.js", model: "qwen-coder" },
    },
  ];

  const presentation = buildThreadPresentationView(session, logs);

  assert.equal(presentation.running, true);
  assert.equal(presentation.durationLabel, "42s");
  assert.equal(presentation.changes.length, 1);
  assert.ok(presentation.activity.length >= 1);
  assert.match(presentation.currentStep, /bearbeitet|edit|denke|webui/i);
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

test("workspaceModalTargetPathFrom verwendet lokale Repos direkt statt eines abgeleiteten Workspace-Pfads", () => {
  assert.equal(
    workspaceModalTargetPathFrom({
      workspaceMode: "create",
      workspaceSource: "git",
      workspaceName: "alpha-copy",
      workspaceGitInspection: {
        source_kind: "local_path",
        resolved_path: "/srv/repos/alpha",
      },
    }),
    "/srv/repos/alpha",
  );
});

test("workspaceGitBranchSuggestions fuehrt Branches dedupliziert und in stabiler Reihenfolge zusammen", () => {
  assert.deepEqual(
    workspaceGitBranchSuggestions({
      workspaceGitBranch: "feature/ui-shell",
      workspaceGitInspection: {
        current_branch: "feature/ui-shell",
        default_branch: "main",
        local_branches: ["feature/ui-shell", "main", "develop"],
        remote_branches: ["origin/main", "origin/develop", "origin/main"],
      },
      workspaceGitStatus: {
        current_branch: "feature/ui-shell",
        configured_branch: "develop",
        default_branch: "main",
        local_branches: ["develop", "release"],
        remote_branches: ["origin/release", "origin/main"],
      },
    }),
    ["feature/ui-shell", "main", "develop", "origin/main", "origin/develop", "release", "origin/release"],
  );
});

test("buildWorkspaceShellView zeigt Git-Metadaten und Sync-Aktionen fuer verbundene Projekte", () => {
  const view = buildWorkspaceShellView({
    config: { model_name: "qwen3-coder:30b" },
    composer: {
      modelName: "",
      accessMode: "approval",
      executionProfile: "balanced",
    },
    workspaces: [
      {
        id: "ws-1",
        name: "alpha",
        path: "/tmp/alpha",
        git_sync_source: "https://example.invalid/alpha.git",
        git_branch: "develop",
        git_remote_name: "origin",
        last_git_sync_at: "2026-04-11T08:00:00.000Z",
      },
    ],
    sessions: [],
    selectedWorkspaceId: "ws-1",
    activeSession: null,
  });

  const gitMeta = view.metaItems.find((item) => item.label === "Git");

  assert.ok(gitMeta);
  assert.equal(gitMeta.value, "develop");
  assert.equal(view.canSyncGit, true);
  assert.match(view.subtitle, /Git .*develop/i);
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
  assert.equal(hero.feeds[0].title, "Workspace");
  assert.equal(hero.feeds[2].title, "Recent activity");
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
  assert.equal(hero.feeds[1].title, "Current session");
  assert.match(hero.locationLabel, /alpha|tmp/i);
});
