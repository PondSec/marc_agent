const STORAGE_KEY = "marc_a1.workspace_ui.v3";
const CHAT_SCROLL_BOTTOM_THRESHOLD = 24;
const COMMIT_AND_PUSH_PROMPT =
  "Bitte pruefe den aktuellen Git-Status in diesem Workspace, erstelle einen kleinen sinnvollen Commit mit einer kurzen passenden Message und pushe den aktuellen Branch zu origin. Wenn es nichts zu committen gibt oder der Push scheitert, erklaere kurz den Grund im Chat.";
const APP_BRAND_NAME = "M.A.R.C A2";
const APP_BRAND_PLAIN = "MARC A2";
const REFERENCE_WELCOME_ART = [
  "                                                          ",
  "     *                                       █████▓▓░     ",
  "                                 *         ███▓░     ░░   ",
  "            ░░░░░░                        ███▓░           ",
  "    ░░░   ░░░░░░░░░░                      ███▓░           ",
  "   ░░░░░░░░░░░░░░░░░    *                ██▓░░      ▓   ",
  "                                             ░▓▓███▓▓░    ",
  " *                                 ░░░░                   ",
  "                                 ░░░░░░░░                 ",
  "                               ░░░░░░░░░░░░░░░░           ",
  "      █████████                                        * ",
  "      ██▄█████▄██                        *                ",
  "      █████████     *                                   ",
  ".......█ █   █ █..........................................",
];
let sessionRefreshHandle = null;
let modelRefreshHandle = null;
let authTickerHandle = null;
let scheduledSessionRefreshHandle = null;
let scheduledModelRefreshHandle = null;
let terminalPollHandle = null;

const refreshControllers = {
  sessions: createRefreshController({ baseIntervalMs: 8000, maxIntervalMs: 30000, minGapMs: 1200 }),
  models: createRefreshController({ baseIntervalMs: 4000, maxIntervalMs: 30000, minGapMs: 1800 }),
};

const state = {
  config: null,
  health: { ok: false, active_sessions: [] },
  models: {
    installed_models: [],
    recommended_models: [],
  },
  workspaces: [],
  sessions: [],
  logs: [],
  activeSessionId: null,
  activeSession: null,
  selectedWorkspaceId: null,
  stream: null,
  setup: {
    checked: false,
    required: false,
    submitting: false,
    seeded: false,
    step: 1,
    reason: "",
    envPath: "",
    hasEnvFile: false,
    passwordPolicy: null,
    installedModels: [],
    recommendedModels: [],
    form: {
      adminDisplayName: "",
      adminEmail: "",
      adminPassword: "",
      adminPasswordConfirm: "",
      initialWorkspaceName: "",
      initialWorkspacePath: "",
      ollamaHost: "",
      modelName: "",
      routerModelName: "",
      accessMode: "approval",
      authCookieSecure: false,
      publicBaseUrl: "",
    },
    error: "",
  },
  auth: {
    checked: false,
    authenticated: false,
    submitting: false,
    success: false,
    handlingExpiry: false,
    user: null,
    session: null,
    passwordPolicy: null,
    csrfHeaderName: "X-CSRF-Token",
    login: {
      email: "",
      password: "",
      totpCode: "",
    },
    error: "",
    rateLimitedUntil: null,
  },
  composer: {
    prompt: "",
    agentProfile: "a2",
    accessMode: "approval",
    modelName: "",
    executionProfile: "balanced",
    dryRun: false,
  },
  ui: {
    booting: true,
    page: "workspace",
    sessionLoading: false,
    workspaceModalOpen: false,
    workspaceMode: "create",
    editingWorkspaceId: null,
    workspaceName: "",
    workspacePath: "",
    terminal: {
      open: false,
      starting: false,
      sessionId: null,
      cursor: 0,
      output: "",
      input: "",
      cwd: "",
      shell: "",
      status: "idle",
      exitCode: null,
      error: "",
    },
    toast: null,
    toastTimer: null,
    diffViewer: {
      open: false,
      expanded: false,
      path: null,
    },
    chatScroll: {
      stickToBottom: true,
      distanceFromBottom: 0,
    },
  },
};

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => {
    initializeApp().catch((error) => {
      console.error(error);
      showToast(`Initialisierung fehlgeschlagen: ${error.message}`, "error");
    });
  });
}

async function initializeApp() {
  hydrateRouteFromLocation();
  bindEvents();
  renderApp();
  await boot();
}

function createRefreshController({ baseIntervalMs, maxIntervalMs, minGapMs }) {
  return {
    baseIntervalMs,
    currentIntervalMs: baseIntervalMs,
    maxIntervalMs,
    minGapMs,
    lastStartedAt: 0,
    lastCompletedAt: 0,
    nextDueAt: 0,
    inflight: null,
  };
}

function shouldStartRefresh(controller, { force = false, now = Date.now() } = {}) {
  if (!controller) {
    return true;
  }
  if (force) {
    return true;
  }
  if (controller.inflight) {
    return false;
  }
  if (controller.nextDueAt && now < controller.nextDueAt) {
    return false;
  }
  if (controller.lastStartedAt && now - controller.lastStartedAt < controller.minGapMs) {
    return false;
  }
  return true;
}

function updateRefreshBackoff(controller, { changed = false, errored = false, now = Date.now() } = {}) {
  if (!controller) {
    return;
  }
  if (changed) {
    controller.currentIntervalMs = controller.baseIntervalMs;
  } else if (errored) {
    controller.currentIntervalMs = Math.min(
      controller.maxIntervalMs,
      Math.max(controller.currentIntervalMs, controller.baseIntervalMs) * 2,
    );
  } else {
    controller.currentIntervalMs = Math.min(
      controller.maxIntervalMs,
      Math.round(Math.max(controller.currentIntervalMs, controller.baseIntervalMs) * 1.5),
    );
  }
  controller.lastCompletedAt = now;
  controller.nextDueAt = now + controller.currentIntervalMs;
}

function scheduleSessionRefresh({ delayMs = 1200, force = false, source = "stream" } = {}) {
  if (scheduledSessionRefreshHandle) {
    window.clearTimeout(scheduledSessionRefreshHandle);
    scheduledSessionRefreshHandle = null;
  }
  scheduledSessionRefreshHandle = window.setTimeout(() => {
    scheduledSessionRefreshHandle = null;
    refreshSessions({ force, source }).catch(() => {});
  }, delayMs);
}

function scheduleModelRefresh({ delayMs = 1800, force = false, source = "poll" } = {}) {
  if (scheduledModelRefreshHandle) {
    window.clearTimeout(scheduledModelRefreshHandle);
    scheduledModelRefreshHandle = null;
  }
  scheduledModelRefreshHandle = window.setTimeout(() => {
    scheduledModelRefreshHandle = null;
    refreshModelCatalog({ force, silent: true, source }).catch(() => {});
  }, delayMs);
}

function bindEvents() {
  document.addEventListener("click", handleClick);
  document.addEventListener("submit", handleSubmit);
  document.addEventListener("input", handleInput);
  document.addEventListener("change", handleChange);
  document.addEventListener("keydown", handleKeydown);
  document.addEventListener("scroll", handleScroll, true);
  window.addEventListener("popstate", handlePopState);
}

async function boot() {
  try {
    await refreshSetupState();
    if (state.setup.required) {
      clearApplicationState({ preserveAuthInputs: true, preserveRoute: true });
      return;
    }
    await refreshAuthState();
    if (!state.auth.authenticated) {
      clearApplicationState({ preserveAuthInputs: true, preserveRoute: true });
      return;
    }
    await loadAuthenticatedApplication();
  } finally {
    state.ui.booting = false;
    renderApp();
    ensureBackgroundPolling();
  }
}

async function refreshSetupState({ silent = false } = {}) {
  try {
    const payload = await fetchJSON("/api/setup/status");
    applySetupState(payload);
  } catch (error) {
    if (!silent) {
      showToast(`Setup-Status konnte nicht geladen werden: ${error.message}`, "error");
    }
    throw error;
  }
}

function applySetupState(payload) {
  state.setup.checked = true;
  state.setup.required = Boolean(payload?.required);
  state.setup.reason = String(payload?.reason || "");
  state.setup.envPath = String(payload?.env_path || "");
  state.setup.hasEnvFile = Boolean(payload?.has_env_file);
  state.setup.passwordPolicy = payload?.password_policy || null;
  state.setup.installedModels = Array.isArray(payload?.installed_ollama_models)
    ? payload.installed_ollama_models
    : [];
  state.setup.recommendedModels = Array.isArray(payload?.recommended_models)
    ? payload.recommended_models
    : [];

  if (!state.setup.seeded && payload?.defaults) {
    state.setup.form.adminDisplayName = "Administrator";
    state.setup.form.initialWorkspaceName = payload.defaults.initial_workspace_name || "";
    state.setup.form.initialWorkspacePath = payload.defaults.initial_workspace_path || "";
    state.setup.form.ollamaHost = payload.defaults.ollama_host || "";
    state.setup.form.modelName = payload.defaults.model_name || "";
    state.setup.form.routerModelName = payload.defaults.router_model_name || payload.defaults.model_name || "";
    state.setup.form.accessMode = payload.defaults.access_mode || "approval";
    state.setup.form.authCookieSecure = Boolean(payload.defaults.auth_cookie_secure);
    state.setup.form.publicBaseUrl = payload.defaults.public_base_url || "";
    state.setup.seeded = true;
  }
  if (!state.setup.required) {
    state.setup.error = "";
  }
}

async function loadAuthenticatedApplication() {
  state.config = await fetchJSON("/api/config");
  applyModelCatalog({
    installed_models: state.config?.installed_ollama_models || [],
    recommended_models: state.config?.recommended_models || [],
  });
  applyStoredPreferences();
  await Promise.all([refreshWorkspaces(), refreshSessions(), ensureRecommendedModels({ silent: true })]);

  if (state.activeSessionId) {
    await openSession(state.activeSessionId, { updateHistory: false });
  } else if (!state.selectedWorkspaceId && state.workspaces.length) {
    state.selectedWorkspaceId = state.workspaces[0].id;
  }
}

function ensureBackgroundPolling() {
  if (!sessionRefreshHandle) {
    sessionRefreshHandle = window.setInterval(() => {
      if (state.auth.authenticated) {
        refreshSessions({ source: "poll" }).catch(() => {});
      }
    }, 8000);
  }
  if (!modelRefreshHandle) {
    modelRefreshHandle = window.setInterval(() => {
      if (state.auth.authenticated) {
        refreshModelCatalog({ silent: true, source: "poll" }).catch(() => {});
      }
    }, 4000);
  }
  if (!authTickerHandle) {
    authTickerHandle = window.setInterval(() => {
      if (state.auth.rateLimitedUntil) {
        if (Date.now() >= state.auth.rateLimitedUntil) {
          state.auth.rateLimitedUntil = null;
          state.auth.error = "";
        }
        renderApp();
      }
    }, 1000);
  }
}

async function refreshAuthState({ silent = false } = {}) {
  try {
    const payload = await fetchJSON("/api/auth/session");
    applyAuthState(payload);
  } catch (error) {
    if (!silent) {
      showToast(`Anmeldestatus konnte nicht geladen werden: ${error.message}`, "error");
    }
    throw error;
  }
}

function applyAuthState(payload) {
  state.auth.checked = true;
  state.auth.authenticated = Boolean(payload?.authenticated);
  state.auth.user = payload?.user || null;
  state.auth.session = payload?.session || null;
  state.auth.passwordPolicy = payload?.password_policy || null;
  state.auth.csrfHeaderName = payload?.csrf_header_name || "X-CSRF-Token";
  if (state.auth.authenticated) {
    state.auth.error = "";
    state.auth.success = false;
    state.auth.rateLimitedUntil = null;
  }
  if (!state.auth.authenticated) {
    disconnectStream();
    state.activeSession = null;
    state.logs = [];
  }
}

function clearApplicationState({ preserveAuthInputs = false, preserveRoute = false } = {}) {
  disconnectStream();
  resetTerminalState();
  state.config = null;
  state.health = { ok: false, active_sessions: [] };
  state.models = { installed_models: [], recommended_models: [] };
  state.workspaces = [];
  state.sessions = [];
  state.logs = [];
  state.activeSessionId = preserveRoute ? state.activeSessionId : null;
  state.activeSession = null;
  state.selectedWorkspaceId = null;
  state.ui.page = preserveRoute ? state.ui.page : "workspace";
  state.ui.sessionLoading = false;
  state.ui.workspaceModalOpen = false;
  resetDiffViewer();
  resetChatScrollState();
  if (!preserveRoute) {
    syncHistory(null);
  }
  if (!preserveAuthInputs) {
    state.auth.login.password = "";
    state.auth.login.totpCode = "";
  }
}

function resetTerminalState() {
  if (terminalPollHandle) {
    window.clearTimeout(terminalPollHandle);
    terminalPollHandle = null;
  }
  state.ui.terminal = {
    open: false,
    starting: false,
    sessionId: null,
    cursor: 0,
    output: "",
    input: "",
    cwd: "",
    shell: "",
    status: "idle",
    exitCode: null,
    error: "",
  };
}

async function refreshHealth() {
  if (!state.auth.authenticated) {
    state.health = { ok: false, active_sessions: [] };
    return;
  }
  try {
    state.health = await fetchJSON("/api/health");
  } catch (error) {
    state.health = { ok: false, active_sessions: [] };
  }
}

async function refreshWorkspaces() {
  if (!state.auth.authenticated) {
    state.workspaces = [];
    return;
  }
  const previousSelectedWorkspaceId = state.selectedWorkspaceId;
  const nextWorkspaces = await fetchJSON("/api/workspaces");
  const changed = hasCollectionChanged(state.workspaces, nextWorkspaces);
  state.workspaces = nextWorkspaces;
  if (!state.selectedWorkspaceId && state.workspaces.length) {
    state.selectedWorkspaceId = state.workspaces[0].id;
  }
  if (
    state.selectedWorkspaceId &&
    !state.workspaces.some((workspace) => workspace.id === state.selectedWorkspaceId)
  ) {
    state.selectedWorkspaceId = state.workspaces[0]?.id || null;
  }
  if (changed || previousSelectedWorkspaceId !== state.selectedWorkspaceId) {
    renderApp();
  }
}

async function refreshSessions({ force = false, source = "manual" } = {}) {
  if (!state.auth.authenticated) {
    state.sessions = [];
    return;
  }
  const controller = refreshControllers.sessions;
  if (
    source === "poll" &&
    state.stream &&
    state.activeSession &&
    ["queued", "running"].includes(state.activeSession.status)
  ) {
    updateRefreshBackoff(controller, { changed: false });
    return state.sessions;
  }
  if (!shouldStartRefresh(controller, { force })) {
    return controller.inflight || state.sessions;
  }
  const previousActiveSessionId = state.activeSessionId;
  controller.lastStartedAt = Date.now();
  controller.inflight = fetchJSON("/api/sessions")
    .then((nextSessions) => {
      const changed = hasCollectionChanged(state.sessions, nextSessions);
      state.sessions = nextSessions;
      if (
        state.activeSessionId &&
        !state.sessions.some((session) => session.id === state.activeSessionId)
      ) {
        disconnectStream();
        state.activeSessionId = null;
        state.activeSession = null;
        state.logs = [];
      }
      if (changed || previousActiveSessionId !== state.activeSessionId) {
        renderApp();
      }
      updateRefreshBackoff(controller, { changed });
      return state.sessions;
    })
    .catch((error) => {
      updateRefreshBackoff(controller, { errored: true });
      throw error;
    })
    .finally(() => {
      controller.inflight = null;
    });
  return controller.inflight;
}

async function refreshModelCatalog({ silent = false, force = false, source = "manual" } = {}) {
  if (!state.auth.authenticated) {
    return;
  }
  const controller = refreshControllers.models;
  if (source === "poll" && !shouldStartRefresh(controller, { force })) {
    return controller.inflight || state.models;
  }
  try {
    controller.lastStartedAt = Date.now();
    controller.inflight = fetchJSON("/api/models")
      .then((catalog) => {
        const changed = applyModelCatalog(catalog);
        if (changed) {
          renderApp();
        }
        updateRefreshBackoff(controller, { changed });
        return state.models;
      })
      .finally(() => {
        controller.inflight = null;
      });
    await controller.inflight;
  } catch (error) {
    updateRefreshBackoff(controller, { errored: true });
    if (!silent) {
      showToast(`Modelle konnten nicht geladen werden: ${error.message}`, "error");
    }
  }
}

async function ensureRecommendedModels({ silent = false } = {}) {
  if (!state.auth.authenticated) {
    return;
  }
  try {
    const catalog = await fetchJSON("/api/models/ensure-recommended", {
      method: "POST",
    });
    const changed = applyModelCatalog(catalog);
    if (changed) {
      renderApp();
    }
  } catch (error) {
    if (!silent) {
      showToast(`Modelle konnten nicht geprueft werden: ${error.message}`, "error");
    }
  }
}

async function openSession(sessionId, { updateHistory = true } = {}) {
  if (!sessionId || !state.auth.authenticated) {
    return;
  }

  disconnectStream();
  resetChatScrollState();
  state.ui.sessionLoading = true;
  state.activeSessionId = sessionId;
  renderApp();

  try {
    const [session, logs] = await Promise.all([
      fetchJSON(`/api/sessions/${sessionId}`),
      fetchJSON(`/api/sessions/${sessionId}/logs`),
    ]);
    state.activeSession = session;
    syncDiffViewerState(session);
    state.logs = Array.isArray(logs) ? logs : [];
    state.activeSessionId = session.id;
    state.selectedWorkspaceId = session.workspace_id || state.selectedWorkspaceId;
    persistPreferences();
    if (updateHistory) {
      syncHistory(session.id);
    }
    if (["queued", "running"].includes(session.status)) {
      connectStream(session.id);
    }
  } finally {
    state.ui.sessionLoading = false;
    renderApp();
  }
}

function selectWorkspace(workspaceId) {
  state.selectedWorkspaceId = workspaceId;
  if (!state.activeSession || state.activeSession.workspace_id !== workspaceId) {
    clearActiveSession();
  }
  persistPreferences();
  renderApp();
}

function startNewChat(workspaceId) {
  state.selectedWorkspaceId = workspaceId || state.selectedWorkspaceId;
  clearActiveSession();
  state.composer.prompt = "";
  persistPreferences();
  renderApp();
  focusComposer();
}

function clearActiveSession() {
  state.activeSessionId = null;
  state.activeSession = null;
  state.logs = [];
  resetDiffViewer();
  resetChatScrollState();
  disconnectStream();
  syncHistory(null);
}

function resetDiffViewer() {
  state.ui.diffViewer = {
    open: false,
    expanded: false,
    path: null,
  };
}

function syncDiffViewerState(session) {
  const changes = Array.isArray(session?.changed_files) ? session.changed_files : [];
  if (!changes.length) {
    resetDiffViewer();
    return;
  }
  if (!state.ui.diffViewer.path || !changes.some((item) => item.path === state.ui.diffViewer.path)) {
    state.ui.diffViewer.path = changes[0].path;
  }
}

function openDiffFile(path) {
  const session = state.activeSession;
  const changes = Array.isArray(session?.changed_files) ? session.changed_files : [];
  if (!changes.length) {
    return;
  }
  state.ui.diffViewer.path = changes.some((item) => item.path === path) ? path : changes[0].path;
  state.ui.diffViewer.open = true;
  renderApp();
}

function toggleDiffPanel() {
  const session = state.activeSession;
  const changes = Array.isArray(session?.changed_files) ? session.changed_files : [];
  if (!changes.length) {
    return;
  }
  if (state.ui.diffViewer.open) {
    closeDiffPanel();
    return;
  }
  openDiffFile(state.ui.diffViewer.path || changes[0].path);
}

function toggleDiffExpanded() {
  const session = state.activeSession;
  const changes = Array.isArray(session?.changed_files) ? session.changed_files : [];
  if (!changes.length) {
    return;
  }
  if (!state.ui.diffViewer.open) {
    state.ui.diffViewer.open = true;
  }
  state.ui.diffViewer.expanded = !state.ui.diffViewer.expanded;
  if (!state.ui.diffViewer.path) {
    state.ui.diffViewer.path = changes[0].path;
  }
  renderApp();
}

function closeDiffPanel() {
  state.ui.diffViewer.open = false;
  state.ui.diffViewer.expanded = false;
  renderApp();
}

async function submitPrompt({ promptOverride = null, accessModeOverride = null } = {}) {
  const prompt = String(promptOverride ?? state.composer.prompt).trim();
  const workspaceId = state.activeSession?.workspace_id || state.selectedWorkspaceId;
  if (!prompt) {
    showToast("Bitte schreibe zuerst eine Nachricht.", "error");
    return;
  }
  if (isSessionRunning(state.activeSession)) {
    showToast("Warte bitte, bis der aktuelle Agent-Schritt abgeschlossen ist.", "error");
    return;
  }
  if (!workspaceId) {
    showToast("Lege zuerst ein Projekt an oder waehle eines aus.", "error");
    openWorkspaceModal("create");
    return;
  }

  const body = {
    prompt,
    session_id: state.activeSessionId,
    workspace_id: workspaceId,
    access_mode: accessModeOverride || state.composer.accessMode,
    dry_run: state.composer.dryRun,
    verbose: true,
    model_name: state.composer.modelName || state.config?.model_name || null,
    agent_profile: state.composer.agentProfile,
    execution_profile: state.composer.executionProfile,
  };

  try {
    const session = await fetchJSON("/api/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (promptOverride === null) {
      state.composer.prompt = "";
    }
    await refreshSessions();
    await openSession(session.id);
  } catch (error) {
    showToast(`Nachricht konnte nicht gesendet werden: ${error.message}`, "error");
  }
}

function requestCommitAndPush() {
  submitPrompt({
    promptOverride: COMMIT_AND_PUSH_PROMPT,
    accessModeOverride: "full",
  });
}

function triggerBrowserDownload(url) {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.rel = "noopener";
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

function downloadSessionHandoff(sessionId) {
  const session =
    (state.activeSession && state.activeSession.id === sessionId ? state.activeSession : null) ||
    state.sessions.find((item) => item.id === sessionId);
  if (!session) {
    showToast("Es ist noch kein Thread fuer den Handoff ausgewaehlt.", "error");
    return;
  }
  if (isSessionRunning(session)) {
    showToast("Warte bitte, bis der aktuelle Lauf abgeschlossen ist.", "error");
    return;
  }
  if (!Array.isArray(session.changed_files) || !session.changed_files.length) {
    showToast("Dieser Thread hat aktuell keine geaenderten Dateien fuer ein Handoff-Bundle.", "error");
    return;
  }
  triggerBrowserDownload(`/api/sessions/${session.id}/download`);
  showToast("Handoff-Bundle wird heruntergeladen.", "success");
}

function downloadWorkspaceExport(workspaceId) {
  const workspace = state.workspaces.find((item) => item.id === workspaceId) || selectedWorkspace();
  if (!workspace) {
    showToast("Waehle zuerst ein Projekt fuer den Export aus.", "error");
    return;
  }
  if (state.activeSession && isSessionRunning(state.activeSession)) {
    showToast("Warte bitte, bis der aktuelle Lauf abgeschlossen ist.", "error");
    return;
  }
  triggerBrowserDownload(`/api/workspaces/${workspace.id}/download`);
  showToast("Workspace-Export wird vorbereitet.", "success");
}

function openWorkspacePreview(workspaceId) {
  const workspace = state.workspaces.find((item) => item.id === workspaceId) || selectedWorkspace();
  if (!workspace) {
    showToast("Waehle zuerst ein Projekt fuer die Cloud-Vorschau aus.", "error");
    return;
  }
  if (state.activeSession && isSessionRunning(state.activeSession)) {
    showToast("Starte die Vorschau bitte erst nach dem aktuellen Lauf.", "error");
    return;
  }
  const previewUrl = `/preview/workspaces/${workspace.id}`;
  const opened = window.open(previewUrl, "_blank", "noopener,noreferrer");
  if (!opened) {
    window.location.href = previewUrl;
    return;
  }
  showToast("Cloud-Vorschau in neuem Tab geoeffnet.", "success");
}

async function deleteSession(sessionId) {
  const session = state.sessions.find((item) => item.id === sessionId);
  if (!session) {
    return;
  }
  if (isSessionRunning(session)) {
    showToast("Laufende Chats koennen erst nach dem Abschluss geloescht werden.", "error");
    return;
  }

  const label = session.title || session.last_message_preview || session.task || "Dieser Chat";
  const confirmed = window.confirm(`Chat "${shorten(label, 60)}" wirklich loeschen?`);
  if (!confirmed) {
    return;
  }

  try {
    await fetchJSON(`/api/sessions/${sessionId}`, { method: "DELETE" });
    if (state.activeSessionId === sessionId) {
      clearActiveSession();
    }
    await refreshSessions();
    renderApp();
    showToast("Chat geloescht.");
  } catch (error) {
    showToast(`Chat konnte nicht geloescht werden: ${error.message}`, "error");
  }
}

async function deleteWorkspace(workspaceId) {
  const workspace = state.workspaces.find((item) => item.id === workspaceId);
  if (!workspace) {
    return;
  }
  if (isWorkspaceBusy(workspaceId)) {
    showToast("Projekte mit laufenden Threads koennen noch nicht geloescht werden.", "error");
    return;
  }

  const sessionCount = state.sessions.filter((session) => session.workspace_id === workspaceId).length;
  const chatLabel = sessionCount === 1 ? "1 Chat" : `${sessionCount} Chats`;
  const confirmed = window.confirm(
    sessionCount
      ? `Projekt "${workspace.name}" und ${chatLabel} aus der Webapp loeschen?\n\nDer Projektordner auf der Platte bleibt erhalten.`
      : `Projekt "${workspace.name}" aus der Webapp loeschen?\n\nDer Projektordner auf der Platte bleibt erhalten.`,
  );
  if (!confirmed) {
    return;
  }

  try {
    await fetchJSON(`/api/workspaces/${workspaceId}`, { method: "DELETE" });
    if (state.activeSession?.workspace_id === workspaceId) {
      clearActiveSession();
    }
    if (state.selectedWorkspaceId === workspaceId) {
      state.selectedWorkspaceId = null;
    }
    persistPreferences();
    await Promise.all([refreshWorkspaces(), refreshSessions()]);
    renderApp();
    showToast("Projekt geloescht.");
  } catch (error) {
    showToast(`Projekt konnte nicht geloescht werden: ${error.message}`, "error");
  }
}

async function clearWorkspaceContents(workspaceId) {
  const workspace = state.workspaces.find((item) => item.id === workspaceId);
  if (!workspace) {
    return;
  }
  if (isWorkspaceBusy(workspaceId)) {
    showToast("Projekte mit laufenden Threads koennen nicht geleert werden.", "error");
    return;
  }

  const confirmed = window.confirm(
    `Den Inhalt von "${workspace.name}" jetzt wirklich auf der Platte loeschen?\n\nDer Projektordner bleibt bestehen, aber alle Dateien und Unterordner darin werden entfernt.`,
  );
  if (!confirmed) {
    return;
  }

  try {
    await fetchJSON(`/api/workspaces/${workspaceId}/clear`, { method: "POST" });
    await Promise.all([refreshWorkspaces(), refreshSessions()]);
    if (state.activeSession?.workspace_id === workspaceId) {
      clearActiveSession();
    }
    renderApp();
    showToast("Projektordner geleert.", "success");
  } catch (error) {
    showToast(`Projektordner konnte nicht geleert werden: ${error.message}`, "error");
  }
}

function scheduleTerminalPoll(delayMs = 700) {
  if (terminalPollHandle) {
    window.clearTimeout(terminalPollHandle);
  }
  if (!state.ui.terminal.open || !state.ui.terminal.sessionId) {
    terminalPollHandle = null;
    return;
  }
  terminalPollHandle = window.setTimeout(() => {
    terminalPollHandle = null;
    pollTerminalSession().catch((error) => {
      state.ui.terminal.error = error.message;
      renderApp();
    });
  }, delayMs);
}

function sanitizeTerminalOutput(value) {
  return String(value || "")
    .replace(/\u001b\][^\u0007]*(?:\u0007|\u001b\\)/g, "")
    .replace(/\u001b\[[0-9;?]*[ -/]*[@-~]/g, "")
    .replace(/\u001b[@-_]/g, "")
    .replace(/\r(?!\n)/g, "");
}

function terminalDisplayMarkup() {
  const terminal = state.ui.terminal;
  const text = terminal.output || (terminal.starting ? "Terminal wird gestartet ..." : "");
  return `${escapeHtml(text + terminal.input)}<span class="terminal-caret" aria-hidden="true"></span>`;
}

function focusTerminalCapture() {
  window.requestAnimationFrame(() => {
    const input = document.getElementById("terminalCaptureInput");
    if (input && typeof input.focus === "function") {
      try {
        input.focus({ preventScroll: true });
      } catch (error) {
        input.focus();
      }
    }
  });
}

function syncTerminalViewport() {
  const output = document.getElementById("terminalOutput");
  if (!output) {
    renderApp();
    focusTerminalCapture();
    return;
  }
  output.innerHTML = terminalDisplayMarkup();
  if (output.parentElement) {
    output.parentElement.scrollTop = output.parentElement.scrollHeight;
  }
}

async function openTerminalModal() {
  if (state.ui.terminal.open && state.ui.terminal.sessionId) {
    renderApp();
    focusTerminalCapture();
    scheduleTerminalPoll(100);
    return;
  }
  resetTerminalState();
  state.ui.terminal.open = true;
  state.ui.terminal.starting = true;
  renderApp();

  try {
    const workspaceId = activeWorkspaceId();
    const payload = await fetchJSON("/api/admin/terminal/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(workspaceId ? { workspace_id: workspaceId } : {}),
    });
    state.ui.terminal = {
      ...state.ui.terminal,
      open: true,
      starting: false,
      sessionId: payload.id,
      cursor: payload.cursor || 0,
      output: sanitizeTerminalOutput(payload.output || ""),
      cwd: payload.cwd || "",
      shell: payload.shell || "",
      status: payload.status || "running",
      exitCode: payload.exit_code ?? null,
      error: "",
    };
    renderApp();
    focusTerminalCapture();
    scheduleTerminalPoll(150);
  } catch (error) {
    resetTerminalState();
    showToast(`Terminal konnte nicht gestartet werden: ${error.message}`, "error");
    renderApp();
  }
}

async function closeTerminalModal() {
  const sessionId = state.ui.terminal.sessionId;
  resetTerminalState();
  renderApp();
  if (!sessionId) {
    return;
  }
  try {
    await fetchJSON(`/api/admin/terminal/sessions/${sessionId}`, { method: "DELETE" });
  } catch (error) {
    showToast(`Terminal konnte nicht sauber beendet werden: ${error.message}`, "error");
  }
}

async function pollTerminalSession() {
  const sessionId = state.ui.terminal.sessionId;
  if (!state.ui.terminal.open || !sessionId) {
    return;
  }
  const payload = await fetchJSON(`/api/admin/terminal/sessions/${sessionId}?cursor=${state.ui.terminal.cursor}`);
  state.ui.terminal.cursor = payload.cursor || 0;
  state.ui.terminal.cwd = payload.cwd || state.ui.terminal.cwd;
  state.ui.terminal.shell = payload.shell || state.ui.terminal.shell;
  state.ui.terminal.status = payload.status || state.ui.terminal.status;
  state.ui.terminal.exitCode = payload.exit_code ?? state.ui.terminal.exitCode;
  state.ui.terminal.error = "";
  if (payload.reset) {
    state.ui.terminal.output = sanitizeTerminalOutput(payload.output || "");
  } else if (payload.output) {
    state.ui.terminal.output += sanitizeTerminalOutput(payload.output);
  }
  syncTerminalViewport();
  if (payload.status === "running") {
    scheduleTerminalPoll(700);
  }
}

async function sendTerminalInput() {
  const sessionId = state.ui.terminal.sessionId;
  const input = state.ui.terminal.input;
  if (!sessionId || !input) {
    return;
  }
  const nextInput = input.endsWith("\n") ? input : `${input}\n`;
  state.ui.terminal.input = "";
  syncTerminalViewport();
  try {
    await fetchJSON(`/api/admin/terminal/sessions/${sessionId}/input`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: nextInput }),
    });
    focusTerminalCapture();
    scheduleTerminalPoll(50);
  } catch (error) {
    state.ui.terminal.input = input;
    showToast(`Terminal-Eingabe fehlgeschlagen: ${error.message}`, "error");
    syncTerminalViewport();
  }
}

async function interruptTerminalSession() {
  const sessionId = state.ui.terminal.sessionId;
  if (!sessionId) {
    return;
  }
  try {
    await fetchJSON(`/api/admin/terminal/sessions/${sessionId}/interrupt`, { method: "POST" });
    scheduleTerminalPoll(50);
  } catch (error) {
    showToast(`Ctrl+C konnte nicht gesendet werden: ${error.message}`, "error");
  }
}

async function stopSession() {
  if (!state.activeSessionId || !isSessionRunning(state.activeSession)) {
    return;
  }

  try {
    const session = await fetchJSON(`/api/sessions/${state.activeSessionId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stop_requested: true }),
    });
    state.activeSession = session;
    await refreshSessions();
    renderApp();
  } catch (error) {
    showToast(`Session konnte nicht gestoppt werden: ${error.message}`, "error");
  }
}

async function submitLogin() {
  if (loginRetryAfterSeconds() > 0) {
    renderApp();
    return;
  }

  state.auth.submitting = true;
  state.auth.success = false;
  state.auth.error = "";
  renderApp();

  try {
    const payload = await fetchJSON("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: state.auth.login.email.trim(),
        password: state.auth.login.password,
        totp_code: state.auth.login.totpCode.trim() || null,
      }),
    });
    applyAuthState(payload);
    state.auth.success = true;
    state.auth.login.password = "";
    state.auth.login.totpCode = "";
    state.ui.booting = true;
    renderApp();
    await loadAuthenticatedApplication();
    showToast("Sichere Anmeldung erfolgreich.", "success");
  } catch (error) {
    state.auth.success = false;
    state.auth.error = error.message || "Anmeldung fehlgeschlagen.";
    if (error.retryAfter > 0) {
      state.auth.rateLimitedUntil = Date.now() + error.retryAfter * 1000;
    }
  } finally {
    state.auth.submitting = false;
    state.ui.booting = false;
    renderApp();
  }
}

async function submitSetup() {
  state.setup.submitting = true;
  state.setup.error = "";
  renderApp();

  try {
    const payload = await fetchJSON("/api/setup/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        admin_display_name: state.setup.form.adminDisplayName.trim(),
        admin_email: state.setup.form.adminEmail.trim(),
        admin_password: state.setup.form.adminPassword,
        admin_password_confirm: state.setup.form.adminPasswordConfirm,
        initial_workspace_name: state.setup.form.initialWorkspaceName.trim(),
        initial_workspace_path: state.setup.form.initialWorkspacePath.trim(),
        ollama_host: state.setup.form.ollamaHost.trim(),
        model_name: state.setup.form.modelName.trim(),
        router_model_name: state.setup.form.routerModelName.trim() || null,
        access_mode: state.setup.form.accessMode,
        auth_cookie_secure: state.setup.form.authCookieSecure,
        public_base_url: state.setup.form.publicBaseUrl.trim() || null,
      }),
    });
    state.setup.required = false;
    state.setup.error = "";
    state.ui.booting = true;
    applyAuthState(payload?.auth || {});
    renderApp();
    await loadAuthenticatedApplication();
    if (payload?.workspace?.id) {
      state.selectedWorkspaceId = payload.workspace.id;
      persistPreferences();
    }
    showToast("Setup abgeschlossen. Die Konsole ist jetzt bereit.", "success");
  } catch (error) {
    state.setup.error = error.message || "Setup konnte nicht abgeschlossen werden.";
  } finally {
    state.setup.submitting = false;
    state.ui.booting = false;
    renderApp();
  }
}

async function logoutUser() {
  try {
    await fetchJSON("/api/auth/logout", { method: "POST" });
  } catch (error) {
    showToast(`Abmeldung konnte nicht abgeschlossen werden: ${error.message}`, "error");
  } finally {
    clearApplicationState({ preserveAuthInputs: true });
    state.auth.authenticated = false;
    state.auth.user = null;
    state.auth.session = null;
    state.auth.success = false;
    state.auth.error = "";
    state.auth.rateLimitedUntil = null;
    await refreshAuthState({ silent: true }).catch(() => {});
    renderApp();
  }
}

async function handleUnauthorizedResponse(detail = "Sitzung abgelaufen. Bitte erneut anmelden.") {
  if (state.auth.handlingExpiry) {
    return;
  }
  state.auth.handlingExpiry = true;
  clearApplicationState({ preserveAuthInputs: true, preserveRoute: true });
  state.auth.authenticated = false;
  state.auth.user = null;
  state.auth.session = null;
  state.auth.error = detail;
  try {
    await refreshAuthState({ silent: true });
  } catch (error) {
    state.auth.checked = true;
  } finally {
    state.auth.handlingExpiry = false;
    renderApp();
    showToast(detail, "error");
  }
}

function connectStream(sessionId) {
  disconnectStream();
  const stream = new EventSource(`/api/sessions/${sessionId}/events`);
  state.stream = stream;

  stream.addEventListener("session", (event) => {
    const session = JSON.parse(event.data);
    if (state.activeSessionId !== session.id) {
      return;
    }
    state.activeSession = session;
    syncDiffViewerState(session);
    renderApp();
    scheduleSessionRefresh({ delayMs: 1200, source: "stream" });
  });

  stream.addEventListener("log", (event) => {
    const record = JSON.parse(event.data);
    appendLogRecord(record);
    renderApp();
  });

  stream.addEventListener("done", () => {
    disconnectStream();
    refreshSessions({ force: true, source: "stream-done" }).catch(() => {});
    renderApp();
  });

  stream.onerror = () => {
    disconnectStream();
    scheduleSessionRefresh({ delayMs: 1500, source: "stream-error" });
    renderApp();
  };
}

function disconnectStream() {
  if (state.stream) {
    state.stream.close();
    state.stream = null;
  }
}

function handleScroll(event) {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  if (!target.classList.contains("chat-stage")) {
    return;
  }
  syncChatScrollState(target);
}

function handleClick(event) {
  const target = event.target.closest("[data-action]");
  if (!target) {
    return;
  }
  const { action } = target.dataset;

  if (action === "select-workspace") {
    selectWorkspace(target.dataset.workspaceId);
    return;
  }

  if (action === "open-session") {
    openSession(target.dataset.sessionId);
    return;
  }

  if (action === "new-chat") {
    startNewChat(target.dataset.workspaceId);
    return;
  }

  if (action === "commit-push") {
    requestCommitAndPush();
    return;
  }

  if (action === "download-session-handoff") {
    downloadSessionHandoff(target.dataset.sessionId);
    return;
  }

  if (action === "download-workspace-export") {
    downloadWorkspaceExport(target.dataset.workspaceId);
    return;
  }

  if (action === "open-workspace-preview") {
    openWorkspacePreview(target.dataset.workspaceId);
    return;
  }

  if (action === "logout") {
    logoutUser();
    return;
  }

  if (action === "submit-prompt") {
    submitPrompt();
    return;
  }

  if (action === "stop-session") {
    stopSession();
    return;
  }

  if (action === "open-workspace-modal") {
    openWorkspaceModal("create");
    return;
  }

  if (action === "open-settings-page") {
    openSettingsPage();
    return;
  }

  if (action === "edit-workspace") {
    openWorkspaceModal("edit", target.dataset.workspaceId);
    return;
  }

  if (action === "delete-workspace") {
    deleteWorkspace(target.dataset.workspaceId);
    return;
  }

  if (action === "clear-workspace-contents") {
    clearWorkspaceContents(target.dataset.workspaceId);
    return;
  }

  if (action === "delete-session") {
    deleteSession(target.dataset.sessionId);
    return;
  }

  if (action === "close-workspace-modal") {
    closeWorkspaceModal();
    return;
  }

  if (action === "open-terminal-modal") {
    openTerminalModal();
    return;
  }

  if (action === "focus-terminal-input") {
    focusTerminalCapture();
    return;
  }

  if (action === "close-terminal-modal") {
    closeTerminalModal();
    return;
  }

  if (action === "interrupt-terminal-session") {
    interruptTerminalSession();
    return;
  }

  if (action === "clear-terminal-output") {
    state.ui.terminal.output = "";
    renderApp();
    return;
  }

  if (action === "close-settings-page") {
    closeSettingsPage();
    return;
  }

  if (action === "ensure-models") {
    ensureRecommendedModels();
    return;
  }

  if (action === "save-workspace") {
    saveWorkspace();
    return;
  }

  if (action === "clear-active-session") {
    startNewChat(state.selectedWorkspaceId);
    return;
  }

  if (action === "toggle-diff-panel") {
    toggleDiffPanel();
    return;
  }

  if (action === "open-diff-file") {
    openDiffFile(target.dataset.path);
    return;
  }

  if (action === "toggle-diff-expanded") {
    toggleDiffExpanded();
    return;
  }

  if (action === "close-diff-panel") {
    closeDiffPanel();
    return;
  }

  if (action === "setup-prev") {
    state.setup.step = Math.max(1, state.setup.step - 1);
    renderApp();
    return;
  }

  if (action === "setup-next") {
    state.setup.step = Math.min(3, state.setup.step + 1);
    renderApp();
    return;
  }

  if (action === "setup-step") {
    state.setup.step = Math.min(3, Math.max(1, Number(target.dataset.step) || 1));
    renderApp();
  }
}

function handleSubmit(event) {
  if (event.target.id === "setupForm") {
    event.preventDefault();
    submitSetup();
    return;
  }
  if (event.target.id === "loginForm") {
    event.preventDefault();
    submitLogin();
  }
}

function handleInput(event) {
  if (event.target.id === "loginEmailInput") {
    state.auth.login.email = event.target.value;
    state.auth.error = "";
    return;
  }
  if (event.target.id === "loginPasswordInput") {
    state.auth.login.password = event.target.value;
    state.auth.error = "";
    return;
  }
  if (event.target.id === "loginTotpInput") {
    state.auth.login.totpCode = event.target.value;
    state.auth.error = "";
    return;
  }
  if (event.target.id === "setupAdminDisplayNameInput") {
    state.setup.form.adminDisplayName = event.target.value;
    state.setup.error = "";
    return;
  }
  if (event.target.id === "setupAdminEmailInput") {
    state.setup.form.adminEmail = event.target.value;
    state.setup.error = "";
    return;
  }
  if (event.target.id === "setupAdminPasswordInput") {
    state.setup.form.adminPassword = event.target.value;
    state.setup.error = "";
    return;
  }
  if (event.target.id === "setupAdminPasswordConfirmInput") {
    state.setup.form.adminPasswordConfirm = event.target.value;
    state.setup.error = "";
    return;
  }
  if (event.target.id === "setupWorkspaceNameInput") {
    state.setup.form.initialWorkspaceName = event.target.value;
    state.setup.error = "";
    return;
  }
  if (event.target.id === "setupWorkspacePathInput") {
    state.setup.form.initialWorkspacePath = event.target.value;
    state.setup.error = "";
    return;
  }
  if (event.target.id === "setupOllamaHostInput") {
    state.setup.form.ollamaHost = event.target.value;
    state.setup.error = "";
    return;
  }
  if (event.target.id === "setupPublicBaseUrlInput") {
    state.setup.form.publicBaseUrl = event.target.value;
    state.setup.error = "";
    return;
  }
  if (event.target.id === "composerInput") {
    state.composer.prompt = event.target.value;
    autosizeTextarea(event.target);
    return;
  }
  if (event.target.id === "workspaceNameInput") {
    state.ui.workspaceName = event.target.value;
    return;
  }
  if (event.target.id === "terminalCaptureInput") {
    if (event.target.value) {
      state.ui.terminal.input += event.target.value;
      event.target.value = "";
      syncTerminalViewport();
    }
  }
}

function handleChange(event) {
  if (event.target.id === "setupModelNameSelect") {
    state.setup.form.modelName = event.target.value;
    if (!state.setup.form.routerModelName) {
      state.setup.form.routerModelName = event.target.value;
    }
    state.setup.error = "";
  } else if (event.target.id === "setupRouterModelNameSelect") {
    state.setup.form.routerModelName = event.target.value;
    state.setup.error = "";
  } else if (event.target.id === "setupAccessModeSelect") {
    state.setup.form.accessMode = event.target.value;
    state.setup.error = "";
  } else if (event.target.id === "setupAuthCookieSecureToggle") {
    state.setup.form.authCookieSecure = event.target.checked;
    state.setup.error = "";
  } else if (event.target.id === "agentProfileSelect") {
    state.composer.agentProfile = event.target.value;
  } else if (event.target.id === "accessModeSelect") {
    state.composer.accessMode = event.target.value;
  } else if (event.target.id === "modelNameSelect") {
    state.composer.modelName = event.target.value;
  } else if (event.target.id === "executionProfileSelect") {
    state.composer.executionProfile = event.target.value;
  } else if (event.target.id === "dryRunToggle") {
    state.composer.dryRun = event.target.checked;
  } else {
    return;
  }
  persistPreferences();
}

function handleKeydown(event) {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    if (event.target.id === "composerInput") {
      event.preventDefault();
      submitPrompt();
      return;
    }
    if (event.target.id === "setupAdminPasswordConfirmInput" || event.target.id === "setupWorkspacePathInput") {
      event.preventDefault();
      if (state.setup.step < 3) {
        state.setup.step += 1;
        renderApp();
      } else {
        submitSetup();
      }
      return;
    }
    if (event.target.id === "loginTotpInput" || event.target.id === "loginPasswordInput") {
      event.preventDefault();
      submitLogin();
      return;
    }
  }
  if (event.key === "Escape") {
    if (state.ui.terminal.open) {
      closeTerminalModal();
      return;
    }
    if (state.ui.workspaceModalOpen) {
      closeWorkspaceModal();
      return;
    }
    if (state.ui.page === "settings") {
      closeSettingsPage();
      return;
    }
    if (!state.auth.authenticated) {
      state.auth.error = "";
      state.setup.error = "";
      renderApp();
    }
  }
  if (event.target.id === "terminalCaptureInput") {
    if (event.key === "Enter") {
      event.preventDefault();
      sendTerminalInput();
      return;
    }
    if (event.key === "Backspace") {
      event.preventDefault();
      state.ui.terminal.input = state.ui.terminal.input.slice(0, -1);
      syncTerminalViewport();
      return;
    }
    if (event.key === "Tab") {
      event.preventDefault();
      state.ui.terminal.input += "\t";
      syncTerminalViewport();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "c") {
      event.preventDefault();
      if (state.ui.terminal.input) {
        state.ui.terminal.input = "";
        syncTerminalViewport();
      } else {
        interruptTerminalSession();
      }
      return;
    }
  }
}

function handlePopState() {
  hydrateRouteFromLocation();
  if (!state.auth.authenticated) {
    renderApp();
    return;
  }
  if (state.activeSessionId) {
    openSession(state.activeSessionId, { updateHistory: false });
  } else {
    state.activeSession = null;
    state.logs = [];
    disconnectStream();
    renderApp();
  }
}

function normalizeWorkspaceFolderName(value) {
  const text = String(value || "")
    .trim()
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "");
  const slug = text
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^[-_.]+|[-_.]+$/g, "");
  return slug || "projekt";
}

function derivedWorkspacePath(name, { keepExisting = false } = {}) {
  if (keepExisting && state.ui.workspacePath) {
    return state.ui.workspacePath;
  }
  const workspaceRoot = String(state.config?.workspace_root || "").trim().replace(/\/+$/g, "");
  const folderName = normalizeWorkspaceFolderName(name);
  if (!workspaceRoot) {
    return folderName;
  }
  return `${workspaceRoot}/${folderName}`;
}

function openWorkspaceModal(mode, workspaceId = null) {
  state.ui.workspaceModalOpen = true;
  state.ui.workspaceMode = mode;
  state.ui.editingWorkspaceId = workspaceId;

  if (mode === "edit" && workspaceId) {
    const workspace = state.workspaces.find((item) => item.id === workspaceId);
    state.ui.workspaceName = workspace?.name || "";
    state.ui.workspacePath = workspace?.path || "";
  } else {
    state.ui.workspaceName = "";
    state.ui.workspacePath = "";
  }
  renderApp();
}

function openSettingsPage() {
  state.ui.page = "settings";
  syncHistory({ page: "settings" });
  renderApp();
  refreshModelCatalog({ silent: true });
}

function closeWorkspaceModal() {
  state.ui.workspaceModalOpen = false;
  state.ui.workspaceMode = "create";
  state.ui.editingWorkspaceId = null;
  renderApp();
}

function closeSettingsPage() {
  state.ui.page = "workspace";
  syncHistory({ page: "workspace", replace: true });
  renderApp();
}

async function saveWorkspace() {
  const name = state.ui.workspaceName.trim();
  const path = derivedWorkspacePath(name, { keepExisting: state.ui.workspaceMode === "edit" });
  if (!name || !path) {
    showToast("Bitte gib mindestens einen Ordnernamen an.", "error");
    return;
  }

  try {
    let workspace;
    if (state.ui.workspaceMode === "edit" && state.ui.editingWorkspaceId) {
      workspace = await fetchJSON(`/api/workspaces/${state.ui.editingWorkspaceId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, path }),
      });
    } else {
      workspace = await fetchJSON("/api/workspaces", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, path }),
      });
    }
    await refreshWorkspaces();
    state.selectedWorkspaceId = workspace.id;
    persistPreferences();
    closeWorkspaceModal();
    showToast("Projekt gespeichert.", "success");
  } catch (error) {
    showToast(`Projekt konnte nicht gespeichert werden: ${error.message}`, "error");
  }
}

function renderApp() {
  const root = document.getElementById("app");
  if (!root) {
    return;
  }
  if (!state.setup.checked || state.setup.required) {
    root.innerHTML = `
      ${renderSetupShell()}
      ${renderToast()}
    `;
    return;
  }
  if (!state.auth.checked || !state.auth.authenticated) {
    root.innerHTML = `
      ${renderAuthShell()}
      ${renderToast()}
    `;
    return;
  }
  const uiSnapshot = captureUiSnapshot();
  const settingsPage = state.ui.page === "settings";
  if (!settingsPage) {
    captureChatScrollState();
  }
  root.innerHTML = settingsPage
    ? `
        ${renderSettingsPage()}
        ${renderWorkspaceModal()}
        ${renderTerminalModal()}
        ${renderToast()}
      `
    : `
        <div class="shell">
          <aside class="sidebar">
            ${renderSidebar()}
          </aside>
          <main class="workspace-shell">
            ${renderTopBar()}
            ${renderChatContainer()}
            ${renderChatInput()}
          </main>
        </div>
        ${renderWorkspaceModal()}
        ${renderTerminalModal()}
        ${renderToast()}
      `;
  if (!settingsPage) {
    syncComposerControls();
  }
  restoreUiSnapshot(uiSnapshot);
  if (!settingsPage) {
    restoreChatScrollState();
    window.requestAnimationFrame(() => {
      restoreChatScrollState();
    });
  }
}

function renderSetupShell() {
  const loading = !state.setup.checked || state.ui.booting;
  const tone = state.setup.error ? "danger" : loading ? "running" : "warning";
  const step = Math.min(Math.max(Number(state.setup.step) || 1, 1), 3);
  const submitLabel = state.setup.submitting ? "Konfiguration wird eingerichtet..." : "Setup abschliessen";
  const copy = loading
    ? "Die Runtime prueft gerade, ob bereits eine betriebsbereite Konfiguration vorhanden ist."
    : "Der Assistent richtet die Web-Konsole, die lokale Runtime und den ersten Administrator in einem Durchgang ein. Die Laufzeitkonfiguration wird als .env gespeichert, dein Passwort nur gehasht in der Auth-Datenbank.";

  return `
    <div class="auth-shell">
      <div class="auth-shell-inner">
        <section class="auth-hero surface-panel">
          <div class="brand-panel auth-brand-panel">
            <div class="brand-mark" aria-hidden="true">${icon("spark")}</div>
            <div class="brand-copy">
              <p class="brand-title">${APP_BRAND_NAME}</p>
              <p class="brand-subtitle">Gefuehrter Setup-Assistent fuer die lokale Agent-Runtime</p>
            </div>
          </div>
          <div class="auth-copy">
            <p class="panel-kicker">Erststart</p>
            <h1>In wenigen Schritten startklar</h1>
            <p>${escapeHtml(copy)}</p>
          </div>
          <div class="auth-feature-list">
            ${renderAuthFeature("Keine Handarbeit", "Die .env, der erste Admin und das erste Projekt werden direkt aus der Web-Oberflaeche angelegt.")}
            ${renderAuthFeature("Sichere Defaults", "Der Assistent erzeugt automatisch einen starken Secret Key und speichert Passwoerter nicht im Klartext in der .env.")}
            ${renderAuthFeature("Ollama im Blick", "Host, Standardmodell und Router-Modell werden sauber vorbelegt und koennen spaeter weiter angepasst werden.")}
            ${renderAuthFeature("Im selben Stil", "Der Setup-Flow nutzt bewusst dieselbe Oberflaechenlogik wie der restliche Workspace statt einer separaten Fremdoberflaeche.")}
          </div>
        </section>
        <section class="auth-card surface-panel setup-card">
          <div class="auth-card-head">
            <div>
              <p class="panel-kicker">Setup</p>
              <h2>${escapeHtml(loading ? "Assistent wird vorbereitet" : "Basis konfigurieren")}</h2>
            </div>
            ${renderStatusBadge(loading ? "Wird vorbereitet" : `Schritt ${step} / 3`, tone, {
              compact: true,
              live: loading || state.setup.submitting,
            })}
          </div>
          <p class="auth-card-copy">
            ${escapeHtml(
              state.setup.reason
                ? setupReasonCopy(state.setup.reason)
                : "Der Assistent sammelt nur die Angaben, die fuer einen sicheren und direkt nutzbaren Start wirklich benoetigt werden.",
            )}
          </p>
          ${renderSetupStatusPanel(tone, loading)}
          ${renderSetupStepTabs(step)}
          <form id="setupForm" class="auth-form setup-form">
            ${renderSetupStep(step, loading)}
            <div class="auth-meta-row setup-meta-row">
              <span class="auth-meta-copy">
                .env-Ziel: ${escapeHtml(state.setup.envPath || ".env")}
              </span>
              <span class="auth-meta-copy">
                Mindestlaenge Passwort: ${escapeHtml(String(state.setup.passwordPolicy?.min_length || 14))} Zeichen
              </span>
              <span class="auth-meta-copy">
                ${escapeHtml(state.setup.hasEnvFile ? "Vorhandene .env wird gezielt ergaenzt." : "Eine neue .env wird automatisch erzeugt.")}
              </span>
            </div>
            <div class="setup-actions">
              <button
                class="button-secondary"
                type="button"
                data-action="setup-prev"
                ${loading || state.setup.submitting || step === 1 ? "disabled" : ""}
              >
                Zurueck
              </button>
              ${
                step < 3
                  ? `
                    <button
                      class="button-primary"
                      type="button"
                      data-action="setup-next"
                      ${loading || state.setup.submitting ? "disabled" : ""}
                    >
                      Weiter
                    </button>
                  `
                  : `
                    <button
                      class="button-primary"
                      type="submit"
                      ${loading || state.setup.submitting ? "disabled" : ""}
                    >
                      ${escapeHtml(submitLabel)}
                    </button>
                  `
              }
            </div>
          </form>
        </section>
      </div>
    </div>
  `;
}

function renderSetupStatusPanel(tone, loading) {
  const message = loading
    ? "Pruefe vorhandene Konfiguration, Sicherheitskontext und Modell-Status."
    : state.setup.error
      ? state.setup.error
      : "Nach dem Abschluss wird die Konfiguration gespeichert, der erste Admin erzeugt und der erste Projektordner direkt eingebunden.";
  const heading = loading
    ? "Initialisierung"
    : state.setup.error
      ? "Setup braucht Aufmerksamkeit"
      : "Bereit fuer die Einrichtung";

  return `
    <div class="auth-status tone-${escapeHtml(tone)}" aria-live="polite">
      <strong>${escapeHtml(heading)}</strong>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
}

function renderSetupStepTabs(activeStep) {
  return `
    <div class="setup-step-tabs">
      ${renderSetupStepTab(1, "Runtime", "Ollama, Modell und Zugriff", activeStep)}
      ${renderSetupStepTab(2, "Admin", "Sicherer Erstzugang", activeStep)}
      ${renderSetupStepTab(3, "Projekt", "Erstes verbundenes Projekt", activeStep)}
    </div>
  `;
}

function renderSetupStepTab(step, title, copy, activeStep) {
  const active = step === activeStep;
  return `
    <button
      class="setup-step-tab ${active ? "active" : ""}"
      type="button"
      data-action="setup-step"
      data-step="${step}"
      ${state.setup.submitting ? "disabled" : ""}
    >
      <span class="setup-step-index">${step}</span>
      <span class="setup-step-copy">
        <strong>${escapeHtml(title)}</strong>
        <small>${escapeHtml(copy)}</small>
      </span>
    </button>
  `;
}

function renderSetupStep(step, loading) {
  if (step === 1) {
    return `
      <div class="setup-section">
        <label class="auth-field" for="setupOllamaHostInput">
          <span>Ollama-Host</span>
          <input
            id="setupOllamaHostInput"
            type="text"
            value="${escapeAttribute(state.setup.form.ollamaHost)}"
            placeholder="http://127.0.0.1:11434"
            ${loading ? "disabled" : ""}
            required
          />
        </label>
        <label class="auth-field" for="setupModelNameSelect">
          <span>Standardmodell</span>
          <select id="setupModelNameSelect" ${loading ? "disabled" : ""}>
            ${renderSetupModelOptions(state.setup.form.modelName)}
          </select>
        </label>
        <label class="auth-field" for="setupRouterModelNameSelect">
          <span>Router-Modell</span>
          <select id="setupRouterModelNameSelect" ${loading ? "disabled" : ""}>
            ${renderSetupModelOptions(state.setup.form.routerModelName || state.setup.form.modelName)}
          </select>
        </label>
        <label class="auth-field" for="setupAccessModeSelect">
          <span>Standard-Zugriffsmodus</span>
          <select id="setupAccessModeSelect" ${loading ? "disabled" : ""}>
            <option value="safe"${state.setup.form.accessMode === "safe" ? " selected" : ""}>Nur Lesen</option>
            <option value="approval"${state.setup.form.accessMode === "approval" ? " selected" : ""}>Mit Freigabe</option>
            <option value="full"${state.setup.form.accessMode === "full" ? " selected" : ""}>Voller Zugriff</option>
          </select>
        </label>
        <label class="settings-toggle setup-toggle">
          <span>
            <strong>Sichere Cookies nur ueber HTTPS</strong>
            <small>Fuer lokales HTTP oft deaktiviert, fuer Reverse Proxy oder HTTPS empfohlen.</small>
          </span>
          <input id="setupAuthCookieSecureToggle" type="checkbox"${state.setup.form.authCookieSecure ? " checked" : ""}${loading ? " disabled" : ""} />
        </label>
      </div>
    `;
  }

  if (step === 2) {
    return `
      <div class="setup-section">
        <label class="auth-field" for="setupAdminDisplayNameInput">
          <span>Anzeigename</span>
          <input
            id="setupAdminDisplayNameInput"
            type="text"
            value="${escapeAttribute(state.setup.form.adminDisplayName)}"
            placeholder="Administrator"
            ${loading ? "disabled" : ""}
            required
          />
        </label>
        <label class="auth-field" for="setupAdminEmailInput">
          <span>E-Mail-Adresse</span>
          <input
            id="setupAdminEmailInput"
            type="email"
            inputmode="email"
            autocomplete="username"
            value="${escapeAttribute(state.setup.form.adminEmail)}"
            placeholder="operator@example.com"
            ${loading ? "disabled" : ""}
            required
          />
        </label>
        <label class="auth-field" for="setupAdminPasswordInput">
          <span>Passwort</span>
          <input
            id="setupAdminPasswordInput"
            type="password"
            autocomplete="new-password"
            value="${escapeAttribute(state.setup.form.adminPassword)}"
            placeholder="Ein starkes Passwort"
            ${loading ? "disabled" : ""}
            required
          />
        </label>
        <label class="auth-field" for="setupAdminPasswordConfirmInput">
          <span>Passwort wiederholen</span>
          <input
            id="setupAdminPasswordConfirmInput"
            type="password"
            autocomplete="new-password"
            value="${escapeAttribute(state.setup.form.adminPasswordConfirm)}"
            placeholder="Zur Kontrolle erneut eingeben"
            ${loading ? "disabled" : ""}
            required
          />
        </label>
        <div class="setup-summary">
          <strong>Wichtig zu wissen</strong>
          <p>Das Passwort wird nicht im Klartext in der .env abgelegt. Es wird nur serverseitig gehasht in der Auth-Datenbank gespeichert.</p>
        </div>
      </div>
    `;
  }

  return `
    <div class="setup-section">
      <label class="auth-field" for="setupWorkspaceNameInput">
        <span>Projektname</span>
        <input
          id="setupWorkspaceNameInput"
          type="text"
          value="${escapeAttribute(state.setup.form.initialWorkspaceName)}"
          placeholder="z. B. agent_ai"
          ${loading ? "disabled" : ""}
          required
        />
      </label>
      <label class="auth-field" for="setupWorkspacePathInput">
        <span>Projektordner</span>
        <input
          id="setupWorkspacePathInput"
          type="text"
          value="${escapeAttribute(state.setup.form.initialWorkspacePath)}"
          placeholder="/Users/.../projekt"
          ${loading ? "disabled" : ""}
          required
        />
      </label>
      <label class="auth-field" for="setupPublicBaseUrlInput">
        <span>Oeffentliche Basis-URL</span>
        <input
          id="setupPublicBaseUrlInput"
          type="text"
          value="${escapeAttribute(state.setup.form.publicBaseUrl)}"
          placeholder="Optional, z. B. https://agent.local"
          ${loading ? "disabled" : ""}
        />
      </label>
      <div class="setup-summary">
        <strong>Was beim Abschluss passiert</strong>
        <p>Die Runtime schreibt die .env, erzeugt den ersten Admin, legt den Projektordner an falls noetig und meldet dich direkt in der Konsole an.</p>
      </div>
    </div>
  `;
}

function renderSetupModelOptions(selectedValue) {
  return setupModelCandidates()
    .map((name) => `<option value="${escapeAttribute(name)}"${name === selectedValue ? " selected" : ""}>${escapeHtml(name)}</option>`)
    .join("");
}

function setupModelCandidates() {
  const values = [
    state.setup.form.modelName,
    state.setup.form.routerModelName,
    ...(state.setup.installedModels || []).map((item) => item?.name),
    ...(state.setup.recommendedModels || []).map((item) => item?.name),
  ];
  return Array.from(new Set(values.filter(Boolean)));
}

function setupReasonCopy(reason) {
  if (reason === "missing_runtime_env") {
    return "Die gespeicherte Laufzeitkonfiguration fehlt. Der Assistent legt die .env neu an und stellt den Administrator-Zugang wieder her.";
  }
  if (reason === "missing_auth_secret_key") {
    return "Es wurde noch kein dauerhafter Security-Secret-Key eingerichtet. Der Assistent erzeugt ihn fuer dich und speichert ihn in der .env.";
  }
  if (reason === "missing_initial_admin") {
    return "Es existiert noch kein Administrator fuer die Web-Konsole. Richte jetzt den ersten sicheren Zugang ein.";
  }
  return "Die Runtime braucht noch eine Ersteinrichtung, bevor die normale Konsole freigeschaltet werden kann.";
}

function renderAuthShell() {
  const loading = !state.auth.checked || state.ui.booting;
  const lockedSeconds = loginRetryAfterSeconds();
  const tone = lockedSeconds > 0 ? "warning" : state.auth.error ? "danger" : state.auth.success ? "success" : "muted";
  const heading = loading
    ? "Sicherheitskontext wird geladen"
    : lockedSeconds > 0
      ? "Anmeldung temporaer gedrosselt"
      : "Sicher anmelden";
  const copy = loading
    ? "Sitzung, CSRF-Schutz und Sicherheitsparameter werden vorbereitet."
    : lockedSeconds > 0
      ? `Zu viele Fehlversuche erkannt. Bitte in etwa ${lockedSeconds} Sekunden erneut versuchen.`
      : "Die Konsole ist erst nach erfolgreicher Anmeldung freigeschaltet. Sessions laufen serverseitig, Cookies bleiben HttpOnly und 2FA per TOTP wird unterstuetzt.";
  const submitLabel = state.auth.submitting
    ? "Anmeldung wird geprueft..."
    : lockedSeconds > 0
      ? "Kurz warten"
      : "Anmelden";

  return `
    <div class="auth-shell">
      <div class="auth-shell-inner">
        <section class="auth-hero surface-panel">
          <div class="brand-panel auth-brand-panel">
            <div class="brand-mark" aria-hidden="true">${icon("spark")}</div>
            <div class="brand-copy">
              <p class="brand-title">${APP_BRAND_NAME}</p>
              <p class="brand-subtitle">Gesicherte Operator-Konsole fuer lokale Agent-Laufzeiten</p>
            </div>
          </div>
          <div class="auth-copy">
            <p class="panel-kicker">Security</p>
            <h1>Produktionsreife Anmeldung fuer die Runtime</h1>
            <p>
              Authentifizierung, Session-Verwaltung und API-Zugriffe laufen zentral ueber den Server.
              Das Frontend speichert keinen Auth-State im Local Storage.
            </p>
          </div>
          <div class="auth-feature-list">
            ${renderAuthFeature("Argon2id", "Passwoerter werden serverseitig mit moderner, speicherhaerter Hashing-Strategie verarbeitet.")}
            ${renderAuthFeature("Session-Cookies", "HttpOnly, Secure und SameSite-geschuetzt mit serverseitiger Invalidierung und Idle-Timeout.")}
            ${renderAuthFeature("CSRF + Origin Checks", "Schreibende Requests brauchen denselben Ursprung und einen gueltigen CSRF-Token.")}
            ${renderAuthFeature("Abuse-Schutz", "IP- und konto-bezogene Drosselung mit progressivem Backoff gegen Brute Force und Credential Stuffing.")}
          </div>
        </section>
        <section class="auth-card surface-panel">
          <div class="auth-card-head">
            <div>
              <p class="panel-kicker">Login</p>
              <h2>${escapeHtml(heading)}</h2>
            </div>
            ${renderStatusBadge(lockedSeconds > 0 ? "Gedrosselt" : loading ? "Wird vorbereitet" : "Geschuetzt", tone, {
              compact: true,
              live: state.auth.submitting || loading,
            })}
          </div>
          <p class="auth-card-copy">${escapeHtml(copy)}</p>
          ${renderAuthStatusPanel(tone, loading, lockedSeconds)}
          <form id="loginForm" class="auth-form">
            <label class="auth-field" for="loginEmailInput">
              <span>E-Mail-Adresse</span>
              <input
                id="loginEmailInput"
                type="email"
                inputmode="email"
                autocomplete="username"
                value="${escapeAttribute(state.auth.login.email)}"
                placeholder="operator@example.com"
                ${loading ? "disabled" : ""}
                required
              />
            </label>
            <label class="auth-field" for="loginPasswordInput">
              <span>Passwort</span>
              <input
                id="loginPasswordInput"
                type="password"
                autocomplete="current-password"
                value="${escapeAttribute(state.auth.login.password)}"
                placeholder="Passwort eingeben"
                ${loading ? "disabled" : ""}
                required
              />
            </label>
            <label class="auth-field" for="loginTotpInput">
              <span>Einmalcode</span>
              <input
                id="loginTotpInput"
                type="text"
                inputmode="numeric"
                autocomplete="one-time-code"
                value="${escapeAttribute(state.auth.login.totpCode)}"
                placeholder="Nur falls 2FA aktiviert ist"
                ${loading ? "disabled" : ""}
              />
            </label>
            <div class="auth-meta-row">
              <span class="auth-meta-copy">Mindestens ${escapeHtml(String(state.auth.passwordPolicy?.min_length || 14))} Zeichen. Gleiche Fehlermeldung fuer unbekannte und falsche Zugangsdaten.</span>
              <span class="auth-meta-copy">2FA via TOTP wird unterstuetzt.</span>
            </div>
            <button
              class="button-primary auth-submit"
              type="submit"
              ${loading || state.auth.submitting || lockedSeconds > 0 ? "disabled" : ""}
            >
              ${escapeHtml(submitLabel)}
            </button>
          </form>
        </section>
      </div>
    </div>
  `;
}

function renderAuthFeature(title, copy) {
  return `
    <article class="auth-feature">
      <strong>${escapeHtml(title)}</strong>
      <p>${escapeHtml(copy)}</p>
    </article>
  `;
}

function renderAuthStatusPanel(tone, loading, lockedSeconds) {
  const message = loading
    ? "Initiale Sitzung wird aufgebaut."
    : lockedSeconds > 0
      ? `Neue Versuche sind noch ${lockedSeconds} Sekunden gesperrt.`
      : state.auth.error || (state.auth.success ? "Anmeldung erfolgreich. Konsole wird geladen." : "Die Anmeldung erfolgt ueber denselben Ursprung mit CSRF-Schutz.");

  return `
    <div class="auth-status tone-${escapeHtml(tone)}" aria-live="polite">
      <strong>${escapeHtml(loading ? "Initialisierung" : lockedSeconds > 0 ? "Schutz aktiv" : state.auth.error ? "Anmeldung fehlgeschlagen" : state.auth.success ? "Erfolg" : "Bereit")}</strong>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
}

function renderSidebar() {
  const workspace = selectedWorkspace();
  const activeRuns = state.sessions.filter((session) => isSessionRunning(session)).length;
  const totalThreads = state.sessions.length;
  const primaryAction = workspace
    ? `
        <button
          class="sidebar-primary-action"
          type="button"
          data-action="new-chat"
          data-workspace-id="${escapeHtml(workspace.id)}"
        >
          ${icon("compose")}
          <span>Neuer Thread</span>
        </button>
      `
    : `
        <button class="sidebar-primary-action" type="button" data-action="open-workspace-modal">
          ${icon("plus")}
          <span>Projekt anlegen</span>
        </button>
      `;

  return `
    <div class="sidebar-shell sidebar-shell-minimal">
      <div class="sidebar-header sidebar-header-minimal">
        <div class="sidebar-brandline">
          <strong>${escapeHtml(APP_BRAND_NAME)}</strong>
          <span>${escapeHtml(activeRuns ? `${activeRuns} aktiv` : workspace ? "Bereit" : "Kein Projekt")}</span>
        </div>
        ${primaryAction}
      </div>
      <div class="sidebar-scroll">
        <section class="sidebar-section sidebar-project-section">
          <div class="sidebar-section-head">
            <p class="sidebar-label">Threads</p>
            <span class="sidebar-count">${escapeHtml(String(totalThreads))}</span>
          </div>
          ${renderSidebarProjectList()}
        </section>
      </div>
      ${renderSidebarFooter(activeRuns)}
    </div>
  `;
}

function renderSidebarProjectList() {
  if (!state.workspaces.length) {
    return `
      <div class="sidebar-empty sidebar-empty-compact">
        Noch kein Projekt verbunden. Lege einen lokalen Ordner an, damit Threads und Agentenlaeufe hier erscheinen.
      </div>
    `;
  }

  return `
    <div class="project-nav-list">
      ${state.workspaces.map(renderSidebarProject).join("")}
    </div>
  `;
}

function renderSidebarProject(workspace) {
  const active = workspace.id === activeWorkspaceId();
  const sessions = sessionsForWorkspace(workspace.id);
  const activeRuns = sessions.filter((session) => isSessionRunning(session)).length;
  const disabled = isWorkspaceBusy(workspace.id);

  return `
    <section class="project-group ${active ? "active" : ""}">
      <div class="project-group-head">
        <button
          class="project-button"
          type="button"
          data-action="select-workspace"
          data-workspace-id="${escapeHtml(workspace.id)}"
        >
          <span class="project-button-icon" aria-hidden="true">${icon("folder")}</span>
          <span class="project-button-copy">
            <span class="project-button-name">${escapeHtml(workspace.name)}</span>
            <span class="project-button-path">${escapeHtml(shortenPath(workspace.path, 42))}</span>
          </span>
          <span class="project-button-count">${escapeHtml(String(sessions.length))}</span>
        </button>
        <div class="project-tools">
          <button
            class="workspace-action"
            type="button"
            data-action="edit-workspace"
            data-workspace-id="${escapeHtml(workspace.id)}"
            aria-label="Projekt bearbeiten"
          >
            ${icon("edit")}
          </button>
          <button
            class="workspace-action warning-button"
            type="button"
            data-action="clear-workspace-contents"
            data-workspace-id="${escapeHtml(workspace.id)}"
            aria-label="Projektordner leeren"
            title="${escapeAttribute(
              disabled
                ? "Projekt kann erst geleert werden, wenn keine Threads mehr laufen."
                : "Dateien und Unterordner auf der Platte loeschen, Projekt behalten",
            )}"
            ${disabled ? "disabled" : ""}
          >
            ${icon("broom")}
          </button>
          <button
            class="workspace-action danger-button"
            type="button"
            data-action="delete-workspace"
            data-workspace-id="${escapeHtml(workspace.id)}"
            aria-label="Projekt loeschen"
            title="${escapeAttribute(
              disabled
                ? "Projekt kann erst geloescht werden, wenn keine Threads mehr laufen."
                : "Projekt aus der Webapp loeschen",
            )}"
            ${disabled ? "disabled" : ""}
          >
            ${icon("trash")}
          </button>
        </div>
      </div>
      ${
        active
          ? `
            <div class="project-thread-list">
              <div class="project-group-meta">
                <span>${escapeHtml(activeRuns ? `${activeRuns} aktiv` : "Bereit")}</span>
                <span>${escapeHtml(countLabel(sessions.length, "1 Thread", `${sessions.length} Threads`))}</span>
              </div>
              ${sessions.length ? sessions.map(renderSidebarThreadItem).join("") : `<div class="sidebar-empty sidebar-empty-compact">In diesem Projekt gibt es noch keinen Thread.</div>`}
            </div>
          `
          : ""
      }
    </section>
  `;
}

function renderSidebarThreadItem(session) {
  const active = session.id === state.activeSessionId;
  const title = session.title || session.last_message_preview || session.task || "Neuer Thread";
  const preview = threadPreview(session);
  const badgeTone = sessionStatusTone(session);
  const badgeLabel = sessionBadgeText(session);

  return `
    <button
      class="thread-nav-item ${active ? "active" : ""}"
      type="button"
      data-action="open-session"
      data-session-id="${escapeHtml(session.id)}"
    >
      <span class="thread-nav-main">
        <span class="thread-nav-title">${escapeHtml(shorten(title, 30))}</span>
        <span class="thread-nav-status tone-${escapeHtml(badgeTone)}">${escapeHtml(badgeLabel)}</span>
      </span>
      <span class="thread-nav-preview">${escapeHtml(shorten(preview, 72))}</span>
      <span class="thread-nav-meta">
        <span>${escapeHtml(formatSessionTimestamp(session.updated_at))}</span>
        ${
          session.changed_files?.length
            ? `<span>${escapeHtml(countLabel(session.changed_files.length, "1 Datei", `${session.changed_files.length} Dateien`))}</span>`
            : ""
        }
      </span>
    </button>
  `;
}

function renderSidebarFooter(activeRuns) {
  if (!state.auth.user) {
    return `
      <div class="sidebar-footer sidebar-footer-minimal">
        <button class="button-ghost sidebar-footer-button" type="button" data-action="open-settings-page">
          Einstellungen
        </button>
      </div>
    `;
  }

  return `
    <div class="sidebar-footer sidebar-footer-minimal">
      <div class="sidebar-footer-copy">
        <strong>${escapeHtml(state.auth.user.display_name || state.auth.user.email)}</strong>
        <span>${escapeHtml(activeRuns ? `${activeRuns} aktive Laeufe` : "Bereit")}</span>
      </div>
      <div class="sidebar-footer-actions">
        <button class="button-ghost sidebar-footer-button" type="button" data-action="open-settings-page">
          Einstellungen
        </button>
        <button class="button-ghost sidebar-footer-button" type="button" data-action="logout">
          Abmelden
        </button>
      </div>
    </div>
  `;
}

function renderIdentityPanel() {
  if (!state.auth.user) {
    return "";
  }
  const lastLogin = state.auth.user.last_login_at
    ? `Letzte Anmeldung ${formatSessionTimestamp(state.auth.user.last_login_at)}`
    : "Erste bekannte Anmeldung";
  const idleExpires = state.auth.session?.idle_expires_at
    ? `Idle-Timeout ${formatTime(state.auth.session.idle_expires_at)}`
    : "Serverseitige Session aktiv";

  return `
    <section class="surface-panel security-panel">
      <div class="security-panel-head">
        <div>
          <p class="panel-kicker">Operator</p>
          <h3>${escapeHtml(state.auth.user.display_name || state.auth.user.email)}</h3>
        </div>
        ${renderStatusBadge(state.auth.user.mfa_enabled ? "2FA aktiv" : "2FA verfuegbar", state.auth.user.mfa_enabled ? "success" : "warning", {
          compact: true,
        })}
      </div>
      <p class="security-panel-copy">${escapeHtml(state.auth.user.email)}</p>
      <div class="security-panel-meta">
        ${renderMetaChip(lastLogin, "muted")}
        ${renderMetaChip(idleExpires, "muted")}
      </div>
      <button class="button-secondary security-logout" type="button" data-action="logout">
        Sitzung beenden
      </button>
    </section>
  `;
}

function renderWorkspaceList() {
  if (!state.workspaces.length) {
    return `
      <div class="sidebar-empty">
        Noch kein Projekt verbunden. Lege zuerst einen lokalen Ordner an, damit Threads, Aktivitaet und Validierung gemeinsam sichtbar werden.
      </div>
    `;
  }

  return `
    <div class="workspace-list">
      ${state.workspaces.map(renderWorkspaceItem).join("")}
    </div>
  `;
}

function renderWorkspaceItem(workspace) {
  const active = workspace.id === activeWorkspaceId();
  const workspaceSessions = sessionsForWorkspace(workspace.id);
  const count = workspaceSessions.length;
  const activeRuns = workspaceSessions.filter((session) => isSessionRunning(session)).length;
  const disabled = isWorkspaceBusy(workspace.id);

  return `
    <div class="workspace-item ${active ? "active" : ""}">
      <button
        class="workspace-button"
        type="button"
        data-action="select-workspace"
        data-workspace-id="${escapeHtml(workspace.id)}"
      >
        <span class="workspace-topline">
          <span class="workspace-name">${escapeHtml(workspace.name)}</span>
          <span class="workspace-badge">${count}</span>
        </span>
        <span class="workspace-path">${escapeHtml(shortenPath(workspace.path, 58))}</span>
        <span class="workspace-meta">
          ${escapeHtml(
            activeRuns
              ? `${activeRuns} ${countLabel(activeRuns, "aktiver Agent", "aktive Agenten")}`
              : count
                ? `${count} ${countLabel(count, "Thread", "Threads")} im Verlauf`
                : "Noch kein Verlauf",
          )}
        </span>
      </button>
      <div class="workspace-item-actions">
        <button
          class="workspace-action"
          type="button"
          data-action="edit-workspace"
          data-workspace-id="${escapeHtml(workspace.id)}"
          aria-label="Projekt bearbeiten"
        >
          ${icon("edit")}
        </button>
        <button
          class="workspace-action warning-button"
          type="button"
          data-action="clear-workspace-contents"
          data-workspace-id="${escapeHtml(workspace.id)}"
          aria-label="Projektordner leeren"
          title="${escapeAttribute(
            disabled
              ? "Projekt kann erst geleert werden, wenn keine Threads mehr laufen."
              : "Dateien und Unterordner auf der Platte loeschen, Projekt behalten",
          )}"
          ${disabled ? "disabled" : ""}
        >
          ${icon("broom")}
        </button>
        <button
          class="workspace-action danger-button"
          type="button"
          data-action="delete-workspace"
          data-workspace-id="${escapeHtml(workspace.id)}"
          aria-label="Projekt loeschen"
          title="${escapeAttribute(
            disabled
              ? "Projekt kann erst geloescht werden, wenn keine Threads mehr laufen."
              : "Projekt aus der Webapp loeschen",
          )}"
          ${disabled ? "disabled" : ""}
        >
          ${icon("trash")}
        </button>
      </div>
    </div>
  `;
}

function renderThreadList() {
  const workspace = selectedWorkspace();
  if (!workspace) {
    return `
      <div class="sidebar-empty">
        Waehle ein Projekt aus, damit hier die Threads erscheinen.
      </div>
    `;
  }

  const sessions = sessionsForWorkspace(workspace.id);
  if (!sessions.length) {
    return `
      <div class="sidebar-empty">
        In ${escapeHtml(workspace.name)} gibt es noch keinen Thread. Starte rechts einen neuen Lauf.
      </div>
    `;
  }

  return `
    <div class="thread-list">
      ${sessions.map(renderThreadItem).join("")}
    </div>
  `;
}

function renderThreadItem(session) {
  const active = session.id === state.activeSessionId;
  const title = session.title || session.last_message_preview || session.task || "Neuer Chat";
  const preview = threadPreview(session);
  const disabled = isSessionRunning(session);
  const badgeTone = sessionStatusTone(session);
  const badgeLabel = sessionBadgeText(session);

  return `
    <div class="thread-row ${active ? "active" : ""}">
      <button
        class="thread-item ${active ? "active" : ""}"
        type="button"
        data-action="open-session"
        data-session-id="${escapeHtml(session.id)}"
      >
        <span class="thread-meta-row">
          ${renderStatusBadge(badgeLabel, badgeTone, {
            compact: true,
            live: isSessionRunning(session),
          })}
          <span class="thread-timestamp">${escapeHtml(formatSessionTimestamp(session.updated_at))}</span>
        </span>
        <span class="thread-title">${escapeHtml(shorten(title, 48))}</span>
        <span class="thread-preview">${escapeHtml(shorten(preview, 112))}</span>
      </button>
      <button
        class="thread-action danger-button"
        type="button"
        data-action="delete-session"
        data-session-id="${escapeHtml(session.id)}"
        aria-label="Chat loeschen"
        title="${escapeAttribute(
          disabled
            ? "Chat kann erst geloescht werden, wenn der Lauf beendet ist."
            : "Chat loeschen",
        )}"
        ${disabled ? "disabled" : ""}
      >
        ${icon("trash")}
      </button>
    </div>
  `;
}

function renderAgentProfileOptions() {
  const profiles = getAgentProfiles();
  if (!profiles.length) {
    return `<option value="${escapeHtml(state.composer.agentProfile || "a2")}" selected>${escapeHtml(state.composer.agentProfile || "a2")}</option>`;
  }

  return profiles
    .map(
      (item) =>
        `<option value="${escapeHtml(item.id)}"${item.id === state.composer.agentProfile ? " selected" : ""}>${escapeHtml(item.label)}</option>`,
    )
    .join("");
}

function renderAccessModeOptions() {
  return getAccessModes()
    .map(
      (mode) =>
        `<option value="${escapeHtml(mode)}"${mode === state.composer.accessMode ? " selected" : ""}>${escapeHtml(labelForAccessMode(mode))}</option>`,
    )
    .join("");
}

function renderModelOptions() {
  const options = getModelOptions();
  if (!options.length) {
    return `<option value=""${state.composer.modelName ? "" : " selected"}>Standard</option>`;
  }

  return options
    .map(
      (item) =>
        `<option value="${escapeHtml(item)}"${item === state.composer.modelName ? " selected" : ""}>${escapeHtml(item)}</option>`,
    )
    .join("");
}

function renderExecutionProfileOptions() {
  const profiles = getExecutionProfiles();
  if (!profiles.length) {
    return `<option value="${escapeHtml(state.composer.executionProfile || "balanced")}" selected>${escapeHtml(state.composer.executionProfile || "balanced")}</option>`;
  }

  return profiles
    .map(
      (item) =>
        `<option value="${escapeHtml(item.id)}"${item.id === state.composer.executionProfile ? " selected" : ""}>${escapeHtml(item.label)}</option>`,
    )
    .join("");
}

function executionProfilesFromState(sourceState = state) {
  return sourceState.config?.execution_profiles || [];
}

function executionProfileLabelFromState(sourceState = state) {
  return (
    executionProfilesFromState(sourceState).find((item) => item.id === sourceState.composer.executionProfile)?.label ||
    sourceState.composer.executionProfile ||
    "Standardprofil"
  );
}

function buildRuntimeStatusItems(sourceState = state) {
  const session = sourceState.activeSession || null;
  const workspace = workspaceForSessionFrom(sourceState, session) || selectedWorkspaceFrom(sourceState);
  const validation = session ? buildValidationSnapshot(session) : null;
  const activeRuns = Array.isArray(sourceState.sessions)
    ? sourceState.sessions.filter((item) => isSessionRunning(item)).length
    : 0;

  const items = [
    {
      label: "Projekt",
      value: workspace?.name || "Kein Projekt",
      tone: workspace ? "muted" : "warning",
    },
    {
      label: "Modell",
      value: sourceState.composer.modelName || sourceState.config?.model_name || "Standard",
      tone: "muted",
    },
    {
      label: "Zugriff",
      value: labelForAccessMode(session?.access_mode || sourceState.composer.accessMode),
      tone: "muted",
    },
    {
      label: "Profil",
      value: executionProfileLabelFromState(sourceState),
      tone: "muted",
    },
    {
      label: "Status",
      value: session ? sessionBadgeText(session) : "Bereit",
      tone: session ? sessionStatusTone(session) : "muted",
    },
    {
      label: "Laeufe",
      value: activeRuns ? `${activeRuns} aktiv` : "Leerlauf",
      tone: activeRuns ? "running" : "muted",
    },
  ];

  if (validation) {
    items.splice(5, 0, {
      label: "Checks",
      value: validation.statusLabel,
      tone: validation.tone,
    });
  }

  return items;
}

function buildWorkspaceShellView(sourceState = state) {
  const session = sourceState.activeSession || null;
  const workspace = workspaceForSessionFrom(sourceState, session) || selectedWorkspaceFrom(sourceState);
  const statusTone = session ? sessionStatusTone(session) : "muted";
  const statusText = session ? sessionBadgeText(session) : "Bereit";
  const running = isSessionRunning(session);
  const title = session
    ? session.title || shorten(session.task, 84) || "Neuer Thread"
    : workspace?.name || "Projekt auswaehlen";
  const subtitleParts = [];

  if (workspace?.name && workspace.name !== title) {
    subtitleParts.push(workspace.name);
  }
  if (workspace?.path) {
    subtitleParts.push(shortenPath(workspace.path, 90));
  }
  if (session?.updated_at) {
    subtitleParts.push(`Aktualisiert ${formatSessionTimestamp(session.updated_at)}`);
  }

  return {
    workspace,
    session,
    running,
    title,
    statusTone,
    statusText,
    runtimeStatusItems: buildRuntimeStatusItems(sourceState),
    subtitle:
      subtitleParts.join(" | ") || "Links ein Projekt waehlen oder einen neuen Thread starten.",
    canPreview: Boolean(workspace) && !running,
    canCommit: Boolean(workspace) && !running,
    canDeleteSession: Boolean(session) && !running,
    canDownloadHandoff:
      Boolean(session) && !running && Array.isArray(session?.changed_files) && session.changed_files.length > 0,
    canDownloadWorkspace: Boolean(workspace) && !running,
  };
}

function buildReferenceHeroView(sourceState = state) {
  const shell = buildWorkspaceShellView(sourceState);
  const { workspace, session } = shell;
  const validation = session ? buildValidationSnapshot(session) : null;
  const workspaceSessions = workspace ? sessionsForWorkspaceFrom(sourceState, workspace.id) : [];
  const activeRuns = Array.isArray(sourceState.sessions)
    ? sourceState.sessions.filter((item) => isSessionRunning(item)).length
    : 0;
  const recentSessions = workspaceSessions.slice(0, 3).map((item) => ({
    text: shorten(item.title || item.last_message_preview || item.task || "Neuer Thread", 56),
    meta: `${sessionBadgeText(item)} · ${formatSessionTimestamp(item.updated_at)}`,
  }));
  const currentStep = currentThoughtFrom(sourceState);
  const welcomeCopy = session
    ? "Transcript, Tool-Aktivitaet und Laufzustand bleiben in einer einzigen konzentrierten REPL-Sicht gebuendelt."
    : workspace
      ? "Die Workspace-Shell orientiert sich jetzt direkt an der Referenz-REPL: erst Kontext und Status, dann der Verlauf, dann der Prompt."
      : "Verbinde ein Projekt und arbeite dann in derselben transcript-zentrierten Shell weiter, statt in einer losen Dashboard-Ansicht.";
  const feeds = [
    {
      title: "Workspace",
      lines: workspace
        ? [
            { text: workspace.name, meta: shortenPath(workspace.path, 72) },
            {
              text: countLabel(workspaceSessions.length, "1 Thread", `${workspaceSessions.length} Threads`),
              meta: activeRuns ? `${activeRuns} aktive Laeufe` : "Keine aktiven Laeufe",
            },
          ]
        : [{ text: "Noch kein Projekt verbunden", meta: "Lege links einen lokalen Workspace an." }],
      footer: workspace ? "Lokaler Kontext verbunden" : "Projekt anlegen, um zu starten",
    },
    {
      title: session ? "Current session" : "Start prompt",
      lines: session
        ? [
            { text: shell.title, meta: labelForPhase(session.current_phase) },
            { text: labelForAccessMode(session.access_mode), meta: sessionStatusTone(session) === "running" ? "Agent arbeitet" : shell.statusText },
            {
              text: countLabel(session.changed_files?.length || 0, "1 Datei geaendert", `${session.changed_files?.length || 0} Dateien geaendert`),
              meta: validation ? validation.statusLabel : "Noch keine Checks",
            },
          ]
        : [
            { text: workspace ? composerPlaceholder(workspace) : "Verbinde zuerst ein lokales Projekt", meta: composerHint(workspace) },
            { text: "Ctrl+Enter sendet direkt", meta: "Zugriff, Modell und Profil bleiben im Footer sichtbar" },
          ],
      footer: currentStep || "Bereit fuer den naechsten Auftrag",
    },
    {
      title: "Recent activity",
      lines: recentSessions.length
        ? recentSessions
        : [
            {
              text: workspace ? "Noch kein Verlauf in diesem Workspace" : "Noch keine Sessions",
              meta: workspace ? "Starte den ersten Thread, um hier Aktivitaet zu sehen." : "Nach dem ersten Lauf erscheint hier der Verlauf.",
            },
          ],
      footer: validation
        ? `Checks: ${validation.statusLabel}`
        : activeRuns
          ? `${activeRuns} aktive Laeufe`
          : "Leerlauf",
    },
  ];

  return {
    shell,
    compact: Boolean(session),
    brand: APP_BRAND_NAME,
    plainBrand: APP_BRAND_PLAIN,
    welcomeTitle: session ? shell.title : `Welcome to ${APP_BRAND_PLAIN}`,
    welcomeCopy,
    promptHint: composerHint(workspace),
    modelLabel: sourceState.composer.modelName || sourceState.config?.model_name || "Standard",
    locationLabel: workspace?.path ? shortenPath(workspace.path, 84) : "Kein lokaler Workspace verbunden",
    statusText: shell.statusText,
    statusTone: shell.statusTone,
    currentStep: currentStep || "Bereit",
    feeds,
    runtimeStatusItems: shell.runtimeStatusItems,
  };
}

function renderReferenceHero() {
  const hero = buildReferenceHeroView();
  if (hero.compact) {
    return `
      <section class="reference-hero reference-hero-compact">
        <div class="reference-hero-compact-head">
          <div class="reference-hero-compact-brand">
            <div class="reference-hero-mark" aria-hidden="true">${icon("spark")}</div>
            <div class="reference-hero-compact-copy">
              <strong>${escapeHtml(hero.brand)}</strong>
              <span>${escapeHtml(hero.modelLabel)} · ${escapeHtml(hero.locationLabel)}</span>
            </div>
          </div>
          <span class="reference-hero-state tone-${escapeHtml(hero.statusTone)}">${escapeHtml(hero.statusText)}</span>
        </div>
        <div class="reference-hero-compact-body">
          <div class="reference-hero-compact-titleblock">
            <span class="reference-kicker">Current session</span>
            <h2>${escapeHtml(hero.welcomeTitle)}</h2>
            <p>${escapeHtml(hero.welcomeCopy)}</p>
          </div>
          ${renderReferenceStatusLine(hero.runtimeStatusItems, hero.currentStep)}
        </div>
      </section>
    `;
  }

  return `
    <section class="reference-hero reference-hero-full">
      <div class="reference-hero-grid">
        <div class="reference-hero-primary">
          <div class="reference-hero-heading">
            <span class="reference-kicker">Welcome</span>
            <h2>${escapeHtml(hero.welcomeTitle)}</h2>
            <p>${escapeHtml(hero.welcomeCopy)}</p>
          </div>
          <pre class="reference-welcome-art" aria-hidden="true">${escapeHtml(REFERENCE_WELCOME_ART.join("\n"))}</pre>
          <div class="reference-hero-meta">
            <span class="reference-meta-pill">${escapeHtml(hero.brand)}</span>
            <span class="reference-meta-pill">${escapeHtml(hero.modelLabel)}</span>
            <span class="reference-meta-pill">${escapeHtml(hero.locationLabel)}</span>
          </div>
        </div>
        <div class="reference-feed-column">
          ${hero.feeds.map(renderReferenceFeedCard).join("")}
        </div>
      </div>
      ${renderReferenceStatusLine(hero.runtimeStatusItems, hero.currentStep)}
    </section>
  `;
}

function renderReferenceFeedCard(feed) {
  const lines = Array.isArray(feed?.lines) ? feed.lines : [];
  return `
    <section class="reference-feed-card">
      <div class="reference-feed-head">
        <h3>${escapeHtml(feed?.title || "Feed")}</h3>
      </div>
      <div class="reference-feed-body">
        ${
          lines.length
            ? lines.map(renderReferenceFeedLine).join("")
            : `<p class="reference-feed-empty">Noch keine Daten verfuegbar.</p>`
        }
      </div>
      ${feed?.footer ? `<div class="reference-feed-footer">${escapeHtml(feed.footer)}</div>` : ""}
    </section>
  `;
}

function renderReferenceFeedLine(line) {
  return `
    <div class="reference-feed-line">
      <strong>${escapeHtml(line?.text || "")}</strong>
      ${line?.meta ? `<span>${escapeHtml(line.meta)}</span>` : ""}
    </div>
  `;
}

function renderReferenceStatusLine(items, currentStep = "") {
  const segments = Array.isArray(items)
    ? items.map(
        (item) => `
          <span class="reference-status-segment tone-${escapeHtml(item.tone || "muted")}">
            <span class="reference-status-label">${escapeHtml(item.label)}</span>
            <strong>${escapeHtml(item.value)}</strong>
          </span>
        `,
      )
    : [];
  if (currentStep) {
    segments.push(
      `<span class="reference-status-segment tone-running"><span class="reference-status-label">Now</span><strong>${escapeHtml(
        currentStep,
      )}</strong></span>`,
    );
  }
  return `<div class="reference-status-line">${segments.join("")}</div>`;
}

function renderRuntimeStatusStrip(items) {
  if (!Array.isArray(items) || !items.length) {
    return "";
  }

  return `
    <div class="thread-status-strip">
      ${items.map(renderRuntimeStatusItem).join("")}
    </div>
  `;
}

function renderRuntimeStatusItem(item) {
  return `
    <div class="runtime-status-item tone-${escapeHtml(item.tone || "muted")}">
      <span class="runtime-status-label">${escapeHtml(item.label)}</span>
      <strong class="runtime-status-value">${escapeHtml(item.value)}</strong>
    </div>
  `;
}

function renderTopBar() {
  const shell = buildWorkspaceShellView();
  const { workspace, session } = shell;
  const title = shell.title || APP_BRAND_NAME;
  const subtitle = workspace
    ? [workspace.name, workspace.path ? shortenPath(workspace.path, 48) : "", session?.updated_at ? `Aktualisiert ${formatSessionTimestamp(session.updated_at)}` : ""]
        .filter(Boolean)
        .join(" · ")
    : shell.subtitle;

  return `
    <header class="thread-topbar">
      <div class="thread-topbar-inner">
        <div class="thread-topbar-copy">
          <h1 class="thread-topbar-title">${escapeHtml(title)}</h1>
          <p class="thread-topbar-subtitle">${escapeHtml(subtitle)}</p>
        </div>
        <div class="thread-toolbar">
          <span class="thread-toolbar-status tone-${escapeHtml(shell.statusTone)}">${escapeHtml(shell.statusText)}</span>
          <button
            class="icon-button thread-toolbar-icon"
            type="button"
            data-action="open-workspace-preview"
            data-workspace-id="${escapeHtml(workspace?.id || "")}"
            aria-label="Workspace in der Cloud starten"
            title="${escapeAttribute(
              !shell.canPreview
                ? "Die Vorschau ist erst verfuegbar, wenn kein Agent-Schritt mehr laeuft."
                : "Workspace direkt auf dem Agent-Server testen",
            )}"
            ${shell.canPreview ? "" : "disabled"}
          >
            ${icon("play")}
          </button>
          <button
            class="button-ghost thread-toolbar-button"
            type="button"
            data-action="download-session-handoff"
            data-session-id="${escapeHtml(session?.id || "")}"
            title="${escapeAttribute(
              !shell.canDownloadHandoff
                ? "Der Handoff ist verfuegbar, sobald ein abgeschlossener Thread Dateien geaendert hat."
                : "Nur die geaenderten Dateien, Report und Logs herunterladen",
            )}"
            ${shell.canDownloadHandoff ? "" : "disabled"}
          >
            Handoff
          </button>
          <button
            class="button-ghost thread-toolbar-button"
            type="button"
            data-action="${workspace ? "new-chat" : "open-workspace-modal"}"
            ${workspace ? `data-workspace-id="${escapeHtml(workspace.id)}"` : ""}
          >
            Neuer Thread
          </button>
          <button
            class="icon-button thread-toolbar-icon"
            type="button"
            data-action="download-workspace-export"
            data-workspace-id="${escapeHtml(workspace?.id || "")}"
            aria-label="Kompletten Workspace herunterladen"
            title="${escapeAttribute(
              !shell.canDownloadWorkspace
                ? "Der Workspace-Export ist erst verfuegbar, wenn kein Agent-Schritt mehr laeuft."
                : "Kompletten Workspace als Zip herunterladen",
            )}"
            ${shell.canDownloadWorkspace ? "" : "disabled"}
          >
            ${icon("download")}
          </button>
          <button
            class="icon-button thread-toolbar-icon"
            type="button"
            data-action="commit-push"
            aria-label="Commit und Push an den Agenten senden"
            title="${escapeAttribute(
              !shell.canCommit
                ? "Commit und Push ist erst verfuegbar, wenn kein Agent-Schritt mehr laeuft."
                : "Commit und Push anfordern",
            )}"
            ${shell.canCommit ? "" : "disabled"}
          >
            ${icon("git-push")}
          </button>
          <button
            class="icon-button thread-toolbar-icon"
            type="button"
            data-action="delete-session"
            data-session-id="${escapeHtml(session?.id || "")}"
            aria-label="Thread loeschen"
            title="${escapeAttribute(
              !shell.canDeleteSession
                ? "Thread kann erst geloescht werden, wenn kein Lauf aktiv ist."
                : "Thread loeschen",
            )}"
            ${shell.canDeleteSession ? "" : "disabled"}
          >
            ${icon("trash")}
          </button>
          <button
            class="icon-button thread-toolbar-icon"
            type="button"
            data-action="open-settings-page"
            aria-label="Einstellungen"
            title="Agent- und Laufzeitoptionen"
          >
            ${icon("sliders")}
          </button>
        </div>
      </div>
    </header>
  `;
}

function renderChatContainer() {
  return `
    <section class="chat-stage">
      <div class="chat-stage-inner">
        ${renderChatStateMessages()}
      </div>
    </section>
  `;
}

function renderChatStateMessages() {
  if (state.ui.booting) {
    return `${renderStageState(
      "Oberflaeche wird vorbereitet",
      "Projekte, Threads und Laufzeitstatus werden geladen.",
    )}`;
  }

  if (state.ui.sessionLoading) {
    return `${renderStageState(
      "Thread wird geladen",
      "Konversation, Arbeitsdetails und Validierung werden zusammengestellt.",
    )}`;
  }

  if (!state.activeSession) {
    return renderEmptyThreadState();
  }

  return renderThreadView(state.activeSession);
}

function renderStageState(title, copy, actions = "") {
  return `
    <section class="surface-panel empty-state-panel">
      <div class="empty-state-copy">
        <p class="panel-kicker">Status</p>
        <h2>${escapeHtml(title)}</h2>
        <p>${escapeHtml(copy)}</p>
      </div>
      ${actions ? `<div class="empty-state-actions">${actions}</div>` : ""}
    </section>
  `;
}

function renderEmptyThreadState() {
  const workspace = selectedWorkspace();
  if (!workspace) {
    return renderStageState(
      "Verbinde zuerst ein Projekt",
      "Lege einen lokalen Projektordner an. Danach erscheinen Threads, Laufstatus und Dateiaenderungen direkt hier.",
      `
        <button class="button-primary" type="button" data-action="open-workspace-modal">
          Projekt anlegen
        </button>
      `,
    );
  }

  const sessions = sessionsForWorkspace(workspace.id);
  return renderStageState(
    `${workspace.name} ist bereit`,
    "Starte einen neuen Thread. Status, Dateien und Validierung erscheinen anschliessend direkt im Verlauf.",
    `
      <div class="empty-state-facts">
        ${renderMetaChip(shortenPath(workspace.path, 60), "muted")}
        ${renderMetaChip(countLabel(sessions.length, "1 Thread", `${sessions.length} Threads`), "muted")}
      </div>
      <button
        class="button-primary"
        type="button"
        data-action="new-chat"
        data-workspace-id="${escapeHtml(workspace.id)}"
      >
        Neuen Thread starten
      </button>
    `,
  );
}

function renderThreadView(session) {
  const presentation = buildThreadPresentationView(session, state.logs);
  const diffFile = activeDiffFile(session);
  return `
    <div class="thread-shell ${diffFile ? "with-diff" : ""} ${diffFile && state.ui.diffViewer.expanded ? "diff-expanded" : ""}">
      <div class="thread-column">
        ${renderConversationPanel(session, presentation)}
        ${
          presentation.running
            ? renderThreadLivePanel(session, presentation)
            : renderThreadOutcomePanel(session, presentation, diffFile)
        }
      </div>
      ${diffFile ? renderThreadDiffViewer(diffFile) : ""}
    </div>
  `;
}

function renderThreadHero(session) {
  const overview = buildSessionOverview(session);
  const validation = buildValidationSnapshot(session);
  const metaChips = [
    renderMetaChip(labelForAccessMode(session.access_mode), "muted"),
    renderMetaChip(
      countLabel(session.tool_calls?.length || 0, "1 Schritt", `${session.tool_calls?.length || 0} Schritte`),
      "muted",
    ),
    renderMetaChip(
      countLabel(session.changed_files?.length || 0, "1 Datei", `${session.changed_files?.length || 0} Dateien`),
      session.changed_files?.length ? "success" : "muted",
    ),
    renderMetaChip(validation.statusLabel, validation.tone),
  ].join("");

  return `
    <section class="surface-panel thread-header-card tone-${escapeHtml(overview.tone)}">
      <div class="thread-header-top">
        <div class="thread-header-copy">
          <p class="panel-kicker">Thread</p>
          <h2 class="thread-heading">${escapeHtml(session.title || shorten(session.task, 96) || "Neuer Thread")}</h2>
          <p class="thread-header-summary">${escapeHtml(overview.summary)}</p>
        </div>
        <div class="thread-header-side">
          ${renderStatusBadge(sessionBadgeText(session), sessionStatusTone(session), {
            live: isSessionRunning(session),
          })}
          <span class="thread-updated">Aktualisiert ${escapeHtml(formatTime(session.updated_at))}</span>
        </div>
      </div>
      <div class="thread-header-facts">${metaChips}</div>
    </section>
  `;
}

function renderMessageBubble(message, options = {}) {
  const display = messageDisplayState(message, options.session);
  return `
    <div class="message-row ${escapeHtml(message.role)}">
      <article class="message-bubble ${escapeHtml(message.role)}">
        <div class="message-head">
          <span class="message-author">${escapeHtml(roleLabel(message.role))}</span>
          <span class="message-time">${escapeHtml(formatTime(message.created_at))}</span>
        </div>
        <div class="message-body rich-text">${renderRichText(display.content)}</div>
        ${display.note ? `<p class="message-note">${escapeHtml(display.note)}</p>` : ""}
      </article>
    </div>
  `;
}

function renderConversationPanel(session, presentation = buildThreadPresentationView(session, state.logs)) {
  const timeline = conversationTimeline(session);
  const workspace = workspaceForSession(session);
  const title = session.title || shorten(session.task, 96) || "Neuer Thread";
  const contextMeta = [workspace?.name || "", workspace?.path ? shortenPath(workspace.path, 84) : "", labelForAccessMode(session.access_mode)]
    .filter(Boolean)
    .join(" · ");
  return `
    <section class="thread-canvas">
      <div class="thread-session-head">
        <div class="thread-session-headline">
          <span class="thread-session-kicker">Thread</span>
          <h2 class="thread-session-title">${escapeHtml(title)}</h2>
          ${contextMeta ? `<p class="thread-session-subtitle">${escapeHtml(contextMeta)}</p>` : ""}
        </div>
        <div class="thread-session-meta">
          <span class="thread-session-status tone-${escapeHtml(sessionStatusTone(session))}">${escapeHtml(sessionBadgeText(session))}</span>
          ${
            presentation.durationLabel
              ? `<span class="thread-session-duration">${escapeHtml(presentation.durationLabel)}</span>`
              : ""
          }
        </div>
      </div>
      <div class="thread-feed thread-feed-minimal">
        ${timeline.length ? timeline.map((entry) => renderTimelineEntry(entry, session)).join("") : `<div class="inline-note">Noch keine Nachrichten in diesem Thread.</div>`}
      </div>
    </section>
  `;
}

function renderThreadLivePanel(session, presentation) {
  const headline = presentation.currentStep || phaseHeadline(session?.current_phase);
  return `
    <section class="thread-live-panel">
      <div class="thread-panel-head">
        <div>
          <span class="thread-panel-kicker">Denke nach</span>
          <h3>${escapeHtml(headline)}</h3>
        </div>
        <span class="thread-panel-meta">${escapeHtml(labelForPhase(session?.current_phase || "planning"))}</span>
      </div>
      ${
        presentation.activity.length
          ? `
            <div class="thread-live-list">
              ${presentation.activity.map(renderThreadLiveRow).join("")}
            </div>
          `
          : `<p class="thread-panel-copy">Der Agent arbeitet gerade, aber es liegt noch keine sichtbare Aktivitaet vor.</p>`
      }
    </section>
  `;
}

function renderThreadLiveRow(item) {
  return `
    <div class="thread-live-row tone-${escapeHtml(item.tone || "muted")}">
      <div class="thread-live-copy">
        <strong>${escapeHtml(item.text || "")}</strong>
        ${item.meta ? `<p>${escapeHtml(item.meta)}</p>` : ""}
      </div>
      <span class="thread-live-time">${escapeHtml(formatTime(item.timestamp))}</span>
    </div>
  `;
}

function renderThreadOutcomePanel(session, presentation, diffFile) {
  const changeLabel = countLabel(presentation.changes.length, "1 Datei", `${presentation.changes.length} Dateien`);
  return `
    <section class="thread-outcome-panel tone-${escapeHtml(presentation.overview.tone)}">
      <div class="thread-panel-head">
        <div>
          <span class="thread-panel-kicker">Zusammenfassung</span>
          <h3>${escapeHtml(presentation.overview.title)}</h3>
        </div>
        <div class="thread-panel-actions">
          ${
            presentation.durationLabel
              ? `<span class="thread-panel-meta">${escapeHtml(presentation.durationLabel)} lang gearbeitet</span>`
              : ""
          }
          ${
            presentation.changes.length
              ? `
                <button class="thread-panel-toggle" type="button" data-action="toggle-diff-panel" aria-label="Diff-Bereich umschalten">
                  ${state.ui.diffViewer.open ? "−" : "+"}
                </button>
              `
              : ""
          }
        </div>
      </div>
      <p class="thread-panel-copy">${escapeHtml(presentation.overview.summary)}</p>
      <div class="thread-outcome-stats">
        ${renderThreadOutcomeStat("Status", sessionBadgeText(session))}
        ${renderThreadOutcomeStat("Checks", presentation.validation.statusLabel)}
        ${renderThreadOutcomeStat("Dateien", changeLabel)}
        ${renderThreadOutcomeStat("Schritte", String(session.tool_calls?.length || 0))}
      </div>
      ${
        presentation.changes.length
          ? `
            <div class="thread-outcome-files">
              ${presentation.changes
                .slice(0, 8)
                .map((change) => renderThreadOutcomeFile(change, diffFile?.path === change.path))
                .join("")}
            </div>
          `
          : ""
      }
    </section>
  `;
}

function renderThreadOutcomeStat(label, value) {
  return `
    <div class="thread-outcome-stat">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </div>
  `;
}

function renderThreadOutcomeFile(change, active = false) {
  return `
    <button
      class="thread-outcome-file ${active ? "active" : ""}"
      type="button"
      data-action="open-diff-file"
      data-path="${escapeAttribute(change.path)}"
    >
      <span>${escapeHtml(labelForFileOperation(change.operation))}</span>
      <strong>${escapeHtml(shortenPath(change.path, 104))}</strong>
    </button>
  `;
}

function activeDiffFile(session) {
  if (!state.ui.diffViewer.open) {
    return null;
  }
  const changes = Array.isArray(session?.changed_files) ? session.changed_files : [];
  if (!changes.length) {
    return null;
  }
  return changes.find((item) => item.path === state.ui.diffViewer.path) || changes[0] || null;
}

function renderThreadDiffViewer(change) {
  return `
    <aside class="thread-diff-panel">
      <div class="thread-diff-head">
        <div class="thread-diff-copy">
          <span class="thread-panel-kicker">Aenderung</span>
          <h3>${escapeHtml(shortenPath(change.path, 96))}</h3>
        </div>
        <div class="thread-diff-actions">
          <button class="thread-panel-toggle" type="button" data-action="toggle-diff-expanded" aria-label="Diff-Bereich vergroessern oder verkleinern">
            ${state.ui.diffViewer.expanded ? "−" : "+"}
          </button>
          <button class="thread-panel-toggle" type="button" data-action="close-diff-panel" aria-label="Diff-Bereich schliessen">
            ×
          </button>
        </div>
      </div>
      <div class="thread-diff-scroll">
        ${renderThreadDiff(change)}
      </div>
    </aside>
  `;
}

function renderThreadDiff(change) {
  const diff = String(change?.diff || "").trim();
  if (!diff) {
    return `
      <div class="thread-diff-empty">
        <p>Fuer diese Datei liegt aktuell kein Patch vor.</p>
      </div>
    `;
  }
  return `
    <div class="thread-diff-code">
      ${diff.split("\n").map(renderThreadDiffLine).join("")}
    </div>
  `;
}

function renderThreadDiffLine(line) {
  const value = String(line || "");
  let tone = "context";
  if (value.startsWith("@@") || value.startsWith("---") || value.startsWith("+++")) {
    tone = "meta";
  } else if (value.startsWith("+")) {
    tone = "add";
  } else if (value.startsWith("-")) {
    tone = "remove";
  }
  return `
    <div class="thread-diff-line tone-${escapeHtml(tone)}">
      <code>${escapeHtml(value || " ")}</code>
    </div>
  `;
}

function renderTimelineEntry(entry, session) {
  if (!entry) {
    return "";
  }
  if (entry.type === "message") {
    return renderMessageBubble(entry.message, { session });
  }
  if (entry.type === "activity") {
    return renderActivityStreamItem(entry.record);
  }
  return "";
}

function renderWorklogPanel(session, logs) {
  if (!hasWorklogContent(session, logs)) {
    return "";
  }

  const activity = [...buildActivityClusters(session, logs)].reverse();
  const transcript = [
    ...activity.map((item) => ({
      author: "Agent",
      tone: item.tone || "muted",
      timestamp: item.timestamp || session.updated_at,
      title: item.text,
      content: item.meta || "",
    })),
    ...(isSessionRunning(session) ? [] : buildThreadTranscriptNotes(session)),
  ];

  return transcript.map(renderTranscriptNote).join("");
}

function renderRunSummaryCard(session) {
  const overview = buildSessionOverview(session);
  const highlights = buildRunHighlights(session);
  return `
    <section class="surface-panel rail-panel status-panel tone-${escapeHtml(overview.tone)}">
      <div class="panel-head">
        <div>
          <p class="panel-kicker">Agent</p>
          <h3>Aktueller Lauf</h3>
        </div>
        ${renderStatusBadge(sessionBadgeText(session), sessionStatusTone(session), {
          compact: true,
          live: isSessionRunning(session),
        })}
      </div>
      <div class="status-panel-body">
        <strong class="status-panel-title">${escapeHtml(overview.title)}</strong>
        <p class="status-panel-copy">${escapeHtml(overview.summary)}</p>
        <div class="status-stats">
          ${highlights.map(renderStatCell).join("")}
        </div>
        <div class="phase-track phase-track-compact">
          ${buildPhaseSteps(session)
            .map(
              (step, index) => `
                <div class="phase-step ${escapeHtml(step.state)}">
                  <span class="phase-step-index">${index + 1}</span>
                  <span class="phase-step-label">${escapeHtml(step.label)}</span>
                </div>
              `,
            )
            .join("")}
        </div>
      </div>
    </section>
  `;
}

function renderValidationCard(session) {
  const validation = buildValidationSnapshot(session);
  return `
    <details
      class="surface-panel rail-panel detail-panel tone-${escapeHtml(validation.tone)}"
      data-preserve-open
      id="validation-${escapeHtml(session.id)}"
      ${shouldOpenDetailPanel(validation.tone, session) ? "open" : ""}
    >
      <summary class="detail-summary">
        <div>
          <p class="panel-kicker">Validierung</p>
          <h3>${escapeHtml(validation.title)}</h3>
        </div>
        ${renderStatusBadge(validation.statusLabel, validation.tone, { compact: true })}
      </summary>
      <div class="detail-body">
        <p class="detail-copy">${escapeHtml(validation.summary)}</p>
        ${validation.latest?.command ? `<code class="detail-code">${escapeHtml(validation.latest.command)}</code>` : ""}
        ${
          validation.runs.length
            ? `
              <div class="validation-list">
                ${validation.runs.map(renderValidationRun).join("")}
              </div>
            `
            : ""
        }
      </div>
    </details>
  `;
}

function renderFilesCard(session) {
  const changes = Array.isArray(session.changed_files) ? session.changed_files : [];
  if (!changes.length) {
    return "";
  }

  return `
    <details
      class="surface-panel rail-panel detail-panel"
      data-preserve-open
      id="changes-${escapeHtml(session.id)}"
      ${changes.length <= 4 ? "open" : ""}
    >
      <summary class="detail-summary">
        <div>
          <p class="panel-kicker">Aenderungen</p>
          <h3>Dateien</h3>
        </div>
        ${renderStatusBadge(countLabel(changes.length, "1 Datei", `${changes.length} Dateien`), "muted", {
          compact: true,
        })}
      </summary>
      <div class="detail-body">
        <div class="file-change-list">
          ${changes.map(renderFileChangeRow).join("")}
        </div>
      </div>
    </details>
  `;
}

function renderIssuesCard(session) {
  const issues = buildIssueItems(session);
  if (!issues.length) {
    return "";
  }

  const tone = issues.some((item) => item.tone === "danger")
    ? "danger"
    : issues.some((item) => item.tone === "warning")
      ? "warning"
      : "muted";

  return `
    <details
      class="surface-panel rail-panel detail-panel tone-${escapeHtml(tone)}"
      data-preserve-open
      id="issues-${escapeHtml(session.id)}"
      ${tone !== "muted" ? "open" : ""}
    >
      <summary class="detail-summary">
        <div>
          <p class="panel-kicker">Hinweise</p>
          <h3>${escapeHtml(tone === "danger" ? "Blocker" : "Risiken")}</h3>
        </div>
        ${renderStatusBadge(countLabel(issues.length, "1 Eintrag", `${issues.length} Eintraege`), tone, {
          compact: true,
        })}
      </summary>
      <div class="detail-body">
        <div class="issue-list">
          ${issues.map(renderIssueRow).join("")}
        </div>
      </div>
    </details>
  `;
}

function renderActivityCard(session, logs) {
  const activity = buildActivityClusters(session, logs);
  if (!activity.length) {
    return "";
  }

  return `
    <details
      class="surface-panel rail-panel detail-panel"
      data-preserve-open
      id="activity-${escapeHtml(session.id)}"
      ${isSessionRunning(session) ? "open" : ""}
    >
      <summary class="detail-summary">
        <div>
          <p class="panel-kicker">Arbeitsdetails</p>
          <h3>Aktivitaet</h3>
        </div>
        ${renderStatusBadge(countLabel(activity.length, "1 Eintrag", `${activity.length} Eintraege`), "muted", {
          compact: true,
        })}
      </summary>
      <div class="detail-body">
        <p class="detail-copy">
          Werkzeuge, Reparaturen und Zwischenphasen erscheinen hier kompakt, ohne den Chatfluss zu stoeren.
        </p>
        <div class="activity-feed">
          ${activity.map(renderActivityItem).join("")}
        </div>
      </div>
    </details>
  `;
}

function renderRunningMessage(session = state.activeSession) {
  const stopping = Boolean(session?.stop_requested);
  const tone = stopping ? "warning" : "running";
  const headline = stopping ? "Stop-Anfrage liegt vor" : phaseHeadline(session?.current_phase);
  const copy = stopping
    ? "Der aktuelle Schritt wird sauber beendet. Danach bleibt der Thread fuer den naechsten Auftrag offen."
    : currentThought() || "Der Agent arbeitet gerade im Hintergrund.";

  return renderActivityStreamItem({
    text: headline,
    meta: copy,
    tone,
    timestamp: session?.updated_at || new Date().toISOString(),
    count: 1,
    live: true,
  });
}

function renderThreadSideRail(session, logs) {
  if (!session) {
    return "";
  }

  const overview = buildSessionOverview(session);
  const validation = buildValidationSnapshot(session);
  const changes = Array.isArray(session.changed_files) ? session.changed_files : [];
  const issues = buildIssueItems(session);
  const activity = buildActivityClusters(session, logs).slice(0, 6);

  return `
    <div class="thread-side-rail">
      <section class="thread-side-card tone-${escapeHtml(overview.tone)}">
        <div class="thread-side-card-head">
          <div>
            <p class="panel-kicker">Thread</p>
            <h3>${escapeHtml(sessionBadgeText(session))}</h3>
          </div>
          <span class="thread-side-card-status tone-${escapeHtml(sessionStatusTone(session))}">
            ${escapeHtml(labelForPhase(session.current_phase))}
          </span>
        </div>
        <p class="thread-side-card-copy">${escapeHtml(overview.summary)}</p>
        <div class="thread-side-stat-grid">
          <div class="thread-side-stat">
            <span>Schritte</span>
            <strong>${escapeHtml(String(session.tool_calls?.length || 0))}</strong>
          </div>
          <div class="thread-side-stat">
            <span>Dateien</span>
            <strong>${escapeHtml(String(changes.length))}</strong>
          </div>
          <div class="thread-side-stat">
            <span>Checks</span>
            <strong>${escapeHtml(validation.statusLabel)}</strong>
          </div>
        </div>
      </section>

      <section class="thread-side-card">
        <div class="thread-side-card-head">
          <div>
            <p class="panel-kicker">Validierung</p>
            <h3>${escapeHtml(validation.title)}</h3>
          </div>
          <span class="thread-side-card-status tone-${escapeHtml(validation.tone)}">
            ${escapeHtml(validation.statusLabel)}
          </span>
        </div>
        <p class="thread-side-card-copy">${escapeHtml(validation.summary)}</p>
        ${
          validation.runs.length
            ? `
              <div class="thread-side-list">
                ${validation.runs.slice(0, 3).map(renderRailValidationItem).join("")}
              </div>
            `
            : ""
        }
      </section>

      ${
        changes.length
          ? `
            <section class="thread-side-card">
              <div class="thread-side-card-head">
                <div>
                  <p class="panel-kicker">Aenderungen</p>
                  <h3>${escapeHtml(countLabel(changes.length, "1 Datei", `${changes.length} Dateien`))}</h3>
                </div>
              </div>
              <div class="thread-side-list">
                ${changes.slice(0, 8).map(renderRailFileItem).join("")}
              </div>
            </section>
          `
          : ""
      }

      ${
        issues.length
          ? `
            <section class="thread-side-card tone-${escapeHtml(
              issues.some((item) => item.tone === "danger") ? "danger" : "warning",
            )}">
              <div class="thread-side-card-head">
                <div>
                  <p class="panel-kicker">Hinweise</p>
                  <h3>${escapeHtml(countLabel(issues.length, "1 Punkt", `${issues.length} Punkte`))}</h3>
                </div>
              </div>
              <div class="thread-side-list">
                ${issues.slice(0, 5).map(renderRailIssueItem).join("")}
              </div>
            </section>
          `
          : ""
      }

      ${
        activity.length
          ? `
            <section class="thread-side-card">
              <div class="thread-side-card-head">
                <div>
                  <p class="panel-kicker">Aktivitaet</p>
                  <h3>Zuletzt</h3>
                </div>
              </div>
              <div class="thread-side-list">
                ${activity.map(renderRailActivityItem).join("")}
              </div>
            </section>
          `
          : ""
      }
    </div>
  `;
}

function renderChatInput() {
  const shell = buildWorkspaceShellView();
  const workspace = shell.workspace;
  const thought = currentThought();
  const running = shell.running;
  const modelInstallNotice = currentModelInstallNotice();
  const notice = thought
    ? { label: "Laufstatus", text: thought, tone: running ? "running" : "muted" }
    : modelInstallNotice
      ? { label: "Modelle", text: modelInstallNotice, tone: "muted" }
      : null;

  return `
    <footer class="chat-input-shell">
      <div class="chat-input-inner">
        <div class="chat-input-container composer-panel minimal-composer-panel">
          ${notice ? `<div class="composer-inline-status">${renderComposerNotice(notice)}</div>` : ""}
          <div class="chat-input-row minimal-chat-input-row">
            <textarea
              id="composerInput"
              class="chat-input minimal-chat-input"
              rows="3"
              placeholder="${escapeAttribute(composerPlaceholder(workspace))}"
            ></textarea>
            <button
              class="send-button minimal-send-button ${running ? "stop" : "send"}"
              type="button"
              data-action="${running ? "stop-session" : "submit-prompt"}"
              aria-label="${running ? "Stoppen" : "Senden"}"
            >
              ${icon(running ? "stop" : "arrow")}
            </button>
          </div>
          <div class="composer-meta-row minimal-composer-footer">
            ${renderComposerMetaItem(workspace?.name || "Kein Projekt")}
            ${renderComposerMetaItem(state.composer.modelName || state.config?.model_name || "Standardmodell")}
            ${renderComposerMetaItem(labelForAccessMode(state.activeSession?.access_mode || state.composer.accessMode))}
            ${renderComposerMetaItem(executionProfileLabelFromState())}
            <button class="button-ghost composer-options-button minimal-composer-options" type="button" data-action="open-settings-page">
              Optionen
            </button>
          </div>
        </div>
      </div>
    </footer>
  `;
}

function currentExecutionProfileLabel() {
  return executionProfileLabelFromState();
}

function renderComposerMetaItem(value) {
  return `<span class="composer-meta-item">${escapeHtml(value)}</span>`;
}

function renderComposerNotice(notice) {
  return `
    <div class="composer-notice tone-${escapeHtml(notice.tone || "muted")}">
      <span class="composer-notice-label">${escapeHtml(notice.label)}</span>
      <span class="composer-notice-text">${escapeHtml(notice.text)}</span>
    </div>
  `;
}

function renderModelInstallStrip(notice) {
  return renderComposerNotice({ label: "Modelle", text: notice, tone: "muted" });
}

function renderThoughtStrip(thought) {
  return `
    ${renderComposerNotice({ label: "Laufstatus", text: thought, tone: "running" })}
  `;
}

function renderSettingsPage() {
  const config = state.config || {};
  const currentWorkspace = workspaceForSession(state.activeSession) || selectedWorkspace();
  const installedModels = Array.isArray(state.models?.installed_models) ? state.models.installed_models : [];
  const canOpenTerminal = state.auth.user?.role === "admin";
  return `
    <main class="settings-page">
      <div class="settings-page-inner">
        <header class="settings-page-header">
          <div class="settings-page-headline">
            <button class="button-ghost settings-back-button" type="button" data-action="close-settings-page">
              <span>Zurueck zur Konsole</span>
            </button>
            <div class="settings-page-copy">
              <p class="panel-kicker">Einstellungen</p>
              <h1 class="settings-page-title">System und Laufzeit</h1>
              <p class="settings-page-subtitle">
                Alle aktuell verfuegbaren Einstellungen sind hier gebuendelt: Agent-Laufverhalten, Modelle,
                Projektverwaltung und die wichtigsten Runtime-Werte.
              </p>
            </div>
          </div>
          <div class="settings-page-actions">
            <button class="button-secondary" type="button" data-action="open-workspace-modal">
              Projekt anlegen
            </button>
            ${
              canOpenTerminal
                ? `
                  <button class="button-ghost settings-action-with-icon" type="button" data-action="open-terminal-modal">
                    ${icon("terminal")}
                    <span>Server-Terminal</span>
                  </button>
                `
                : ""
            }
            <button class="button-ghost" type="button" data-action="ensure-models">
              Modelle aktualisieren
            </button>
          </div>
        </header>
        <div class="settings-page-chips">
          ${renderMetaChip(currentWorkspace?.name || "Kein Projekt", "muted")}
          ${renderMetaChip(state.composer.modelName || config.model_name || "Standardmodell", "muted")}
          ${renderMetaChip(labelForAccessMode(state.composer.accessMode), "muted")}
          ${renderMetaChip(currentExecutionProfileLabel(), "muted")}
        </div>
        <div class="settings-layout">
          <div class="settings-stack">
            <section class="surface-panel settings-page-panel">
              <div class="panel-head">
                <div>
                  <p class="panel-kicker">Agent</p>
                  <h3>Ausfuehrung</h3>
                </div>
                ${renderStatusBadge(state.composer.dryRun ? "Trockenlauf" : "Aktiv", state.composer.dryRun ? "warning" : "muted", {
                  compact: true,
                })}
              </div>
              <p class="settings-panel-copy">
                Diese Werte gelten fuer neue Agentenlaeufe, Antworten und Folgeauftraege im Chat.
              </p>
              <div class="settings-form-grid">
                <label class="settings-field">
                  <span>Agent</span>
                  <select id="agentProfileSelect">
                    ${renderAgentProfileOptions()}
                  </select>
                </label>
                <label class="settings-field">
                  <span>Zugriffsmodus</span>
                  <select id="accessModeSelect">
                    ${renderAccessModeOptions()}
                  </select>
                </label>
                <label class="settings-field">
                  <span>Modell</span>
                  <select id="modelNameSelect">
                    ${renderModelOptions()}
                  </select>
                  <small>${escapeHtml(modelFieldHelperText())}</small>
                </label>
                <label class="settings-field">
                  <span>Ausfuehrungsprofil</span>
                  <select id="executionProfileSelect">
                    ${renderExecutionProfileOptions()}
                  </select>
                </label>
              </div>
              <label class="settings-toggle settings-toggle-row">
                <span>
                  <strong>Trockenlauf</strong>
                  <small>Werkzeuge nur simulieren, ohne Dateien zu veraendern</small>
                </span>
                <input id="dryRunToggle" type="checkbox"${state.composer.dryRun ? " checked" : ""} />
              </label>
            </section>

            <section class="surface-panel settings-page-panel">
              <div class="settings-panel-head">
                <div>
                  <p class="panel-kicker">Modelle</p>
                  <h3>Lokale Modellverwaltung</h3>
                </div>
                <button class="button-ghost model-panel-action" type="button" data-action="ensure-models">
                  Jetzt aktualisieren
                </button>
              </div>
              <p class="settings-panel-copy">
                Empfohlene Coding- und Router-Modelle werden hier sichtbar und koennen direkt angestossen werden.
              </p>
              ${
                installedModels.length
                  ? `
                    <div class="settings-chip-row">
                      ${installedModels.map((item) => renderMetaChip(item.name, "muted")).join("")}
                    </div>
                  `
                  : `<div class="inline-note">Noch keine lokalen Modelle gefunden.</div>`
              }
              <div class="model-list">
                ${renderRecommendedModels()}
              </div>
            </section>

            <section class="surface-panel settings-page-panel">
              <div class="settings-panel-head">
                <div>
                  <p class="panel-kicker">Projekte</p>
                  <h3>Projektverwaltung</h3>
                </div>
                <button class="button-secondary" type="button" data-action="open-workspace-modal">
                  Projekt anlegen
                </button>
              </div>
              <p class="settings-panel-copy">
                Hier verwaltest du alle verbundenen Projektordner, die in der Konsole links auftauchen.
              </p>
              ${renderWorkspaceList()}
            </section>
          </div>

          <div class="settings-stack">
            ${renderIdentityPanel()}
            <section class="surface-panel settings-page-panel">
              <div class="settings-panel-head">
                <div>
                  <p class="panel-kicker">Runtime</p>
                  <h3>Systemwerte</h3>
                </div>
                ${renderStatusBadge(config.model_name ? "Geladen" : "Ohne Config", config.model_name ? "success" : "warning", {
                  compact: true,
                })}
              </div>
              <p class="settings-panel-copy">
                Diese Werte kommen direkt aus der aktuellen Server-Konfiguration. Sie sind hier sichtbar, auch wenn
                nicht jeder Eintrag in der Web-Oberflaeche direkt bearbeitet werden kann.
              </p>
              <div class="settings-info-grid">
                ${renderSettingsInfoCard("Ollama Host", config.ollama_host || "-", "Endpoint fuer lokale Modelle")}
                ${renderSettingsInfoCard("Standardmodell", config.preferred_model_name || config.model_name || "-", "Aktueller Standard fuer neue Laeufe")}
                ${renderSettingsInfoCard("Router-Modell", config.router_preferred_model_name || "-", "Routing und schnellere Vorentscheidungen")}
                ${renderSettingsInfoCard("Default Zugriff", labelForAccessMode(config.access_mode || state.composer.accessMode), "Serverseitiger Standardmodus")}
                ${renderSettingsInfoCard("Netzwerk", formatBooleanSetting(config.allow_network), "Externe Netzwerkzugriffe")}
                ${renderSettingsInfoCard("Gefaehrliche Befehle", formatBooleanSetting(config.allow_dangerous_commands), "Shell mit erweitertem Risiko")}
                ${renderSettingsInfoCard("Warmup beim Start", formatBooleanSetting(config.warmup_models_on_startup), "Modelle beim Boot vorladen")}
                ${renderSettingsInfoCard("Pfad-Scope", config.path_scope || "-", "Systemweit oder projektbezogen")}
                ${renderSettingsInfoCard("Workspace Root", config.workspace_root || "-", "Basisverzeichnis der Runtime")}
                ${renderSettingsInfoCard("State Root", config.state_root || "-", "Sessions, Logs und Reports")}
                ${renderSettingsInfoCard("Cookie Secure", formatBooleanSetting(config.auth_cookie_secure), "HTTPS-Schutz fuer Auth-Cookies")}
                ${renderSettingsInfoCard("Public Base URL", config.public_base_url || "Nicht gesetzt", "Optional fuer Reverse Proxy / externe URL")}
              </div>
            </section>
            ${
              canOpenTerminal
                ? `
                  <section class="surface-panel settings-page-panel">
                    <div class="settings-panel-head">
                      <div>
                        <p class="panel-kicker">Server</p>
                        <h3>Remote-Terminal</h3>
                      </div>
                      <button class="button-secondary settings-action-with-icon" type="button" data-action="open-terminal-modal">
                        ${icon("terminal")}
                        <span>${state.ui.terminal.open ? "Terminal offen" : "Terminal starten"}</span>
                      </button>
                    </div>
                    <p class="settings-panel-copy">
                      Direkter Shell-Zugriff auf den Server fuer Wartung, Hotfixes und Agent-Debugging unterwegs.
                    </p>
                    <div class="settings-info-grid">
                      ${renderSettingsInfoCard("Startpfad", state.ui.terminal.cwd || currentWorkspace?.path || config.workspace_root || "-", "Neue Sessions starten im aktiven Projekt oder im Workspace-Root")}
                      ${renderSettingsInfoCard("Shell", state.ui.terminal.shell || "-", "Interaktive Shell des laufenden Server-Users")}
                      ${renderSettingsInfoCard("Status", state.ui.terminal.status === "running" ? "Verbunden" : state.ui.terminal.status === "exited" ? "Beendet" : "Bereit", "Nur fuer angemeldete Admins sichtbar")}
                      ${renderSettingsInfoCard("Rolle", state.auth.user?.role || "-", "Admin-only Zugriff")}
                    </div>
                  </section>
                `
                : ""
            }
          </div>
        </div>
      </div>
    </main>
  `;
}

function renderSettingsInfoCard(label, value, hint = "") {
  return `
    <article class="settings-info-card">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value || "-"))}</strong>
      ${hint ? `<small>${escapeHtml(hint)}</small>` : ""}
    </article>
  `;
}

function formatBooleanSetting(value) {
  if (typeof value !== "boolean") {
    return "Nicht gesetzt";
  }
  return value ? "Ja" : "Nein";
}

function renderWorkspaceModal() {
  if (!state.ui.workspaceModalOpen) {
    return "";
  }

  const edit = state.ui.workspaceMode === "edit";
  const derivedPath = derivedWorkspacePath(state.ui.workspaceName, { keepExisting: edit });
  return `
    <div class="modal-backdrop" data-action="close-workspace-modal"></div>
    <div class="modal-layer">
      <section class="modal-card">
        <header class="modal-head">
          <div>
            <p class="modal-kicker">Projekt</p>
            <h3>${edit ? "Projekt bearbeiten" : "Projekt anlegen"}</h3>
          </div>
          <button class="icon-button modal-close" type="button" data-action="close-workspace-modal" aria-label="Schliessen">
            <span class="modal-close-glyph" aria-hidden="true">X</span>
          </button>
        </header>
        <div class="modal-body">
          <label class="modal-field">
            <span>${edit ? "Projektname" : "Ordnername"}</span>
            <input id="workspaceNameInput" type="text" placeholder="z. B. agent_ai" value="${escapeAttribute(state.ui.workspaceName)}" />
            <small>${edit ? "Der bestehende Projektordner bleibt erhalten, nur der Anzeigename aendert sich." : "Der Ordner wird automatisch unter dem konfigurierten Workspace-Root angelegt."}</small>
          </label>
          <div class="modal-preview-card">
            <span>Projektpfad</span>
            <strong>${escapeHtml(derivedPath || "-")}</strong>
            <small>${edit ? "Dieser Pfad bleibt fuer das bestehende Projekt fixiert." : `Automatisch aus Workspace Root und dem sicheren Ordnernamen "${normalizeWorkspaceFolderName(state.ui.workspaceName)}" erzeugt.`}</small>
          </div>
          <p class="modal-note">Die WebUI nimmt nur noch den Ordnernamen entgegen. Den vollen Serverpfad setzt das System selbst.</p>
        </div>
        <footer class="modal-actions">
          <button class="button-secondary" type="button" data-action="close-workspace-modal">Abbrechen</button>
          <button class="button-primary" type="button" data-action="save-workspace">${edit ? "Speichern" : "Anlegen"}</button>
        </footer>
      </section>
    </div>
  `;
}

function renderTerminalModal() {
  if (!state.ui.terminal.open) {
    return "";
  }

  const terminal = state.ui.terminal;
  const statusLabel =
    terminal.status === "running"
      ? "Verbunden"
      : terminal.status === "exited"
        ? terminal.exitCode === 0
          ? "Beendet"
          : `Beendet (${terminal.exitCode})`
        : terminal.starting
          ? "Wird gestartet"
          : "Bereit";

  return `
    <div class="modal-backdrop" data-action="close-terminal-modal"></div>
    <div class="modal-layer modal-layer-wide">
      <section class="modal-card terminal-modal">
        <header class="modal-head">
          <div>
            <p class="modal-kicker">Server-Terminal</p>
            <h3>Direkter Shell-Zugriff</h3>
          </div>
          <div class="terminal-head-actions">
            ${renderStatusBadge(statusLabel, terminal.status === "running" ? "success" : terminal.error ? "danger" : "muted", {
              compact: true,
            })}
            <button class="icon-button modal-close" type="button" data-action="close-terminal-modal" aria-label="Schliessen">
              <span class="modal-close-glyph" aria-hidden="true">X</span>
            </button>
          </div>
        </header>
        <div class="modal-body terminal-modal-body">
          <div class="terminal-meta-row">
            <div class="modal-preview-card terminal-meta-card">
              <span>Aktiver Pfad</span>
              <strong>${escapeHtml(terminal.cwd || state.config?.workspace_root || "-")}</strong>
              <small>${escapeHtml(terminal.shell || "Shell wird gestartet...")}</small>
            </div>
            <div class="terminal-toolbar">
              <button class="button-ghost settings-action-with-icon" type="button" data-action="clear-terminal-output">
                ${icon("broom")}
                <span>Ansicht leeren</span>
              </button>
              <button class="button-ghost settings-action-with-icon" type="button" data-action="interrupt-terminal-session" ${terminal.sessionId ? "" : "disabled"}>
                ${icon("stop")}
                <span>Ctrl+C</span>
              </button>
            </div>
          </div>
          <div class="terminal-output-shell" data-action="focus-terminal-input">
            <pre class="terminal-output" id="terminalOutput">${terminalDisplayMarkup()}</pre>
            <textarea
              id="terminalCaptureInput"
              class="terminal-capture-input"
              spellcheck="false"
              autocapitalize="off"
              autocomplete="off"
              autocorrect="off"
              aria-label="Server-Terminal"
            ></textarea>
          </div>
          <p class="modal-note">Klicke ins Terminal und tippe direkt dort. Enter sendet an die Shell, Backspace loescht lokal, Ctrl+C sendet ein Interrupt.</p>
          ${terminal.error ? `<p class="modal-note terminal-error">${escapeHtml(terminal.error)}</p>` : ""}
        </div>
        <footer class="modal-actions">
          <button class="button-secondary" type="button" data-action="close-terminal-modal">Schliessen</button>
          <button class="button-primary settings-action-with-icon" type="button" data-action="focus-terminal-input" ${terminal.sessionId ? "" : "disabled"}>
            ${icon("terminal")}
            <span>Terminal fokussieren</span>
          </button>
        </footer>
      </section>
    </div>
  `;
}

function renderToast() {
  if (!state.ui.toast) {
    return "";
  }
  return `
    <div class="toast ${escapeHtml(state.ui.toast.tone)}">
      ${escapeHtml(state.ui.toast.message)}
    </div>
  `;
}

function renderValidationRun(run) {
  const tone = validationRunTone(run);
  const summary = [run.summary, run.excerpt].filter(Boolean).join(" · ");
  return `
    <article class="validation-item tone-${escapeHtml(tone)}">
      <div class="validation-item-head">
        ${renderStatusBadge(labelForValidationRunStatus(run.status), tone, { compact: true })}
        <span class="validation-scope">${escapeHtml(labelForVerificationScope(run.verification_scope))}</span>
      </div>
      <code class="detail-code">${escapeHtml(run.command || "Ohne Befehl")}</code>
      ${summary ? `<p class="validation-summary">${escapeHtml(shorten(summary, 180))}</p>` : ""}
    </article>
  `;
}

function renderFileChangeRow(change) {
  return `
    <div class="file-change-row">
      ${renderStatusBadge(labelForFileOperation(change.operation), operationTone(change.operation), {
        compact: true,
      })}
      <span class="file-change-path">${escapeHtml(shortenPath(change.path, 72))}</span>
    </div>
  `;
}

function renderIssueRow(issue) {
  return `
    <article class="issue-row tone-${escapeHtml(issue.tone)}">
      <div class="issue-row-head">
        ${renderStatusBadge(issue.label, issue.tone, { compact: true })}
        ${issue.meta ? `<span class="issue-row-meta">${escapeHtml(issue.meta)}</span>` : ""}
      </div>
      <p class="issue-row-copy">${escapeHtml(issue.text)}</p>
    </article>
  `;
}

function renderActivityItem(item) {
  return `
    <article class="activity-item tone-${escapeHtml(item.tone || "muted")}">
      <div class="activity-item-copy">
        <div class="activity-item-head">
          <strong>${escapeHtml(item.text)}</strong>
          ${
            item.count > 1
              ? `<span class="activity-count">${escapeHtml(countLabel(item.count, "1x", `${item.count}x`))}</span>`
              : ""
          }
        </div>
        ${item.meta ? `<p class="activity-item-meta">${escapeHtml(item.meta)}</p>` : ""}
      </div>
      <span class="activity-time">${escapeHtml(formatTime(item.timestamp))}</span>
    </article>
  `;
}

function renderPhaseTrack(session) {
  return `
    <div class="phase-track">
      ${buildPhaseSteps(session)
        .map(
          (step, index) => `
            <div class="phase-step ${escapeHtml(step.state)}">
              <span class="phase-step-index">${index + 1}</span>
              <span class="phase-step-label">${escapeHtml(step.label)}</span>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderStatCell(item) {
  return `
    <div class="stat-cell tone-${escapeHtml(item.tone || "muted")}">
      <span class="stat-cell-label">${escapeHtml(item.label)}</span>
      <strong class="stat-cell-value">${escapeHtml(item.value)}</strong>
    </div>
  `;
}

function renderStatusBadge(label, tone = "muted", options = {}) {
  const { live = false, compact = false } = options;
  return `
    <span class="status-badge ${escapeHtml(tone)} ${compact ? "compact" : ""}">
      ${live ? `<span class="status-badge-dot" aria-hidden="true"></span>` : ""}
      ${escapeHtml(label)}
    </span>
  `;
}

function renderMetaChip(label, tone = "muted") {
  return `<span class="meta-chip ${escapeHtml(tone)}">${escapeHtml(label)}</span>`;
}

function renderActivityStreamItem(item) {
  if (!item) {
    return "";
  }
  const meta = String(item.meta || "").trim();
  const tone = String(item.tone || "muted");
  return `
    <div class="activity-stream-item tone-${escapeHtml(tone)} ${item.live ? "live" : ""}">
      <div class="activity-stream-marker" aria-hidden="true">
        <span class="activity-stream-dot"></span>
      </div>
      <div class="activity-stream-body">
        <div class="activity-stream-head">
          <span class="activity-stream-kicker">Agent</span>
          <span class="activity-stream-time">${escapeHtml(formatTime(item.timestamp))}</span>
        </div>
        <p class="activity-stream-title">${escapeHtml(item.text || "")}</p>
        ${
          meta
            ? isLikelyCodeMeta(meta)
              ? `<code class="activity-stream-pill">${escapeHtml(meta)}</code>`
              : `<p class="activity-stream-copy">${escapeHtml(meta)}</p>`
            : ""
        }
      </div>
    </div>
  `;
}

function renderRailValidationItem(run) {
  const tone = validationRunTone(run);
  const detail = shorten([run.command, run.summary || run.excerpt].filter(Boolean).join(" - "), 120);
  return `
    <article class="thread-side-item tone-${escapeHtml(tone)}">
      <div class="thread-side-item-top">
        <span class="thread-side-item-label">${escapeHtml(labelForValidationRunStatus(run.status))}</span>
        <span class="thread-side-item-time">${escapeHtml(labelForVerificationScope(run.verification_scope))}</span>
      </div>
      <p class="thread-side-item-copy">${escapeHtml(detail || "Ohne Details")}</p>
    </article>
  `;
}

function renderRailFileItem(change) {
  return `
    <article class="thread-side-item">
      <div class="thread-side-item-top">
        <span class="thread-side-item-label">${escapeHtml(labelForFileOperation(change.operation))}</span>
      </div>
      <p class="thread-side-item-copy">${escapeHtml(shortenPath(change.path, 84))}</p>
    </article>
  `;
}

function renderRailIssueItem(issue) {
  return `
    <article class="thread-side-item tone-${escapeHtml(issue.tone)}">
      <div class="thread-side-item-top">
        <span class="thread-side-item-label">${escapeHtml(issue.label)}</span>
        ${issue.meta ? `<span class="thread-side-item-time">${escapeHtml(issue.meta)}</span>` : ""}
      </div>
      <p class="thread-side-item-copy">${escapeHtml(issue.text)}</p>
    </article>
  `;
}

function renderRailActivityItem(item) {
  return `
    <article class="thread-side-item tone-${escapeHtml(item.tone || "muted")}">
      <div class="thread-side-item-top">
        <span class="thread-side-item-label">${escapeHtml(item.text)}</span>
        <span class="thread-side-item-time">${escapeHtml(formatTime(item.timestamp))}</span>
      </div>
      ${item.meta ? `<p class="thread-side-item-copy">${escapeHtml(item.meta)}</p>` : ""}
    </article>
  `;
}

function isLikelyCodeMeta(value) {
  const text = String(value || "").trim();
  if (!text) {
    return false;
  }
  return /[\\/]|--|=>|\.|:\s|^npm\s|^python\s|^git\s|^rg\s|^sed\s|^cat\s|^Get-/.test(text);
}

function renderTranscriptNote(note) {
  const title = String(note?.title || "").trim();
  const content = String(note?.content || "").trim();
  return `
    <div class="message-row system transcript-note tone-${escapeHtml(note?.tone || "muted")}">
      <article class="message-bubble system transcript-note-bubble">
        <div class="message-head">
          <span class="message-author">${escapeHtml(note?.author || "System")}</span>
          <span class="message-time">${escapeHtml(formatTime(note?.timestamp))}</span>
        </div>
        ${title ? `<p class="transcript-note-title">${escapeHtml(title)}</p>` : ""}
        ${content ? `<div class="message-body rich-text transcript-note-body">${renderRichText(content)}</div>` : ""}
      </article>
    </div>
  `;
}

function buildThreadTranscriptNotes(session) {
  const notes = [];
  const validation = buildValidationSnapshot(session);
  const changes = Array.isArray(session?.changed_files) ? session.changed_files : [];
  const issues = buildIssueItems(session);
  const issueTone = issues.some((item) => item.tone === "danger")
    ? "danger"
    : issues.some((item) => item.tone === "warning")
      ? "warning"
      : "muted";

  if (shouldRenderValidationTranscript(session, validation)) {
    notes.push({
      author: "Validierung",
      tone: validation.tone,
      timestamp: validation.latest?.finished_at || validation.latest?.started_at || session?.updated_at,
      title: validation.title,
      content: buildValidationTranscriptContent(validation),
    });
  }

  if (changes.length) {
    notes.push({
      author: "Aenderungen",
      tone: "success",
      timestamp: session?.updated_at,
      title: countLabel(changes.length, "1 Datei geaendert", `${changes.length} Dateien geaendert`),
      content: buildChangedFilesTranscriptContent(changes),
    });
  }

  if (issues.length) {
    notes.push({
      author: "Hinweise",
      tone: issueTone,
      timestamp: session?.updated_at,
      title: issueTone === "danger" ? "Offene Probleme" : "Wichtige Hinweise",
      content: buildIssueTranscriptContent(issues),
    });
  }

  return notes;
}

function shouldRenderValidationTranscript(session, validation) {
  return Boolean(
    validation?.runs?.length ||
      session?.validation_status ||
      (Array.isArray(session?.changed_files) && session.changed_files.length),
  );
}

function buildValidationTranscriptContent(validation) {
  const parts = [];
  if (validation?.summary) {
    parts.push(validation.summary);
  }

  const runs = Array.isArray(validation?.runs) ? validation.runs.slice(0, 3) : [];
  if (runs.length) {
    parts.push(
      [
        "Checks:",
        ...runs.map((run) => {
          const detail = shorten(
            [run.command, run.summary || run.excerpt].filter(Boolean).join(" - ") || "Ohne Details",
            160,
          );
          return `- ${labelForValidationRunStatus(run.status)}: ${detail}`;
        }),
      ].join("\n"),
    );
  }

  return parts.join("\n\n");
}

function buildChangedFilesTranscriptContent(changes) {
  const visible = changes.slice(0, 8).map((change) => {
    return `- ${labelForFileOperation(change.operation)}: \`${shortenPath(change.path, 120)}\``;
  });

  if (changes.length > visible.length) {
    visible.push(`- ... ${changes.length - visible.length} weitere Dateien`);
  }

  return visible.join("\n");
}

function buildIssueTranscriptContent(issues) {
  return issues
    .slice(0, 5)
    .map((issue) => {
      const meta = issue.meta ? ` (${issue.meta})` : "";
      return `- ${issue.label}: ${issue.text}${meta}`;
    })
    .join("\n");
}

function hasWorklogContent(session, logs) {
  return Boolean(
    (Array.isArray(logs) && logs.length) ||
      session?.validation_runs?.length ||
      session?.changed_files?.length ||
      session?.blockers?.length ||
      session?.diagnostics?.length ||
      session?.tool_calls?.length,
  );
}

function syncComposerControls() {
  syncTextArea("composerInput", state.composer.prompt);
}

function captureUiSnapshot() {
  const activeElement = document.activeElement;
  const snapshot = {
    activeId: activeElement?.id || null,
    selectionStart: null,
    selectionEnd: null,
    openDetailsIds: Array.from(
      document.querySelectorAll("details[data-preserve-open][open]"),
    )
      .map((item) => item.id)
      .filter(Boolean),
  };

  if (activeElement && typeof activeElement.selectionStart === "number") {
    snapshot.selectionStart = activeElement.selectionStart;
    snapshot.selectionEnd = activeElement.selectionEnd;
  }

  return snapshot;
}

function restoreUiSnapshot(snapshot) {
  if (!snapshot?.activeId) {
    restoreDetailPanels(snapshot);
  } else {
    const target = document.getElementById(snapshot.activeId);
    if (target) {
      if (typeof target.focus === "function") {
        try {
          target.focus({ preventScroll: true });
        } catch (error) {
          target.focus();
        }
      }

      if (
        typeof snapshot.selectionStart === "number" &&
        typeof snapshot.selectionEnd === "number" &&
        typeof target.setSelectionRange === "function"
      ) {
        const nextLength = typeof target.value === "string" ? target.value.length : 0;
        const start = Math.min(snapshot.selectionStart, nextLength);
        const end = Math.min(snapshot.selectionEnd, nextLength);
        target.setSelectionRange(start, end);
      }
    }
  }

  restoreDetailPanels(snapshot);
}

function restoreDetailPanels(snapshot) {
  const openDetailsIds = Array.isArray(snapshot?.openDetailsIds) ? snapshot.openDetailsIds : [];
  for (const id of openDetailsIds) {
    const panel = document.getElementById(id);
    if (panel instanceof HTMLDetailsElement) {
      panel.open = true;
    }
  }
}

function syncTextArea(id, value) {
  const target = document.getElementById(id);
  if (!target) {
    return;
  }
  if (target.value !== value) {
    target.value = value;
  }
  autosizeTextarea(target);
}

function resetChatScrollState() {
  state.ui.chatScroll = {
    stickToBottom: true,
    distanceFromBottom: 0,
  };
}

function captureChatScrollState() {
  const container = document.querySelector(".chat-stage");
  if (!(container instanceof HTMLElement)) {
    return;
  }
  syncChatScrollState(container);
}

function restoreChatScrollState() {
  const container = document.querySelector(".chat-stage");
  if (!(container instanceof HTMLElement)) {
    return;
  }

  const chatScroll = state.ui.chatScroll || {
    stickToBottom: true,
    distanceFromBottom: 0,
  };
  if (chatScroll.stickToBottom) {
    container.scrollTop = container.scrollHeight;
    return;
  }

  const nextDistance = Math.max(0, Number(chatScroll.distanceFromBottom) || 0);
  container.scrollTop = Math.max(
    0,
    container.scrollHeight - container.clientHeight - nextDistance,
  );
}

function syncChatScrollState(container) {
  const distanceFromBottom = Math.max(
    0,
    container.scrollHeight - container.scrollTop - container.clientHeight,
  );
  state.ui.chatScroll = {
    stickToBottom: distanceFromBottom <= CHAT_SCROLL_BOTTOM_THRESHOLD,
    distanceFromBottom,
  };
}

function autosizeTextarea(textarea) {
  if (!textarea) {
    return;
  }
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 280)}px`;
}

function focusComposer() {
  const target = document.getElementById("composerInput");
  if (target) {
    target.focus();
  }
}

function threadPreview(session) {
  const candidate = String(session?.last_message_preview || "").trim();
  if (!candidate) {
    return session?.task || "Noch keine Vorschau.";
  }
  if (
    /^(geaendert|validierung|validation|blocker|repair blocker|status=|workflow_stage=)/i.test(candidate) ||
    /internal:/i.test(candidate)
  ) {
    return session?.task || candidate;
  }
  return candidate;
}

function messageDisplayState(message, session) {
  const raw = String(message?.content || "").trim();
  if (!raw) {
    return { content: raw, note: "" };
  }
  return { content: raw, note: "" };
}

function isLatestAssistantMessage(session, message) {
  const latestAssistant = [...conversationMessages(session)]
    .reverse()
    .find((item) => item.role === "assistant");
  if (!latestAssistant) {
    return false;
  }
  return latestAssistant.id === message.id;
}

function sanitizeAssistantMessageContent(text) {
  const value = String(text || "").trim();
  if (!value) {
    return "";
  }

  const telemetryPatterns = [
    /^(geaendert|changed|validierung|validation|blocker|repair blocker|letzter check|last check)/i,
    /^status=/i,
    /^workflow_stage=/i,
    /internal:/i,
  ];

  const paragraphs = value
    .split(/\n{2,}/)
    .map((item) => item.trim())
    .filter(Boolean);

  const cleanedParagraphs = paragraphs.filter(
    (paragraph) => !telemetryPatterns.some((pattern) => pattern.test(paragraph)),
  );

  if (paragraphs.length > 1 && cleanedParagraphs.length) {
    return cleanedParagraphs[0];
  }

  const cleanedLines = value
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line && !telemetryPatterns.some((pattern) => pattern.test(line)));

  return cleanedLines.join("\n") || value;
}

function conversationMessages(session) {
  if (Array.isArray(session.messages) && session.messages.length) {
    return session.messages;
  }
  const fallback = [];
  if (session.task) {
    fallback.push({ id: "fallback-user", role: "user", content: session.task, created_at: session.created_at });
  }
  if (session.final_response) {
    fallback.push({
      id: "fallback-assistant",
      role: "assistant",
      content: session.final_response,
      created_at: session.updated_at,
    });
  }
  return fallback;
}

function conversationTimeline(session) {
  const messageEntries = conversationMessages(session).map((message) => ({
    type: "message",
    timestamp: message.created_at,
    message,
  }));

  return [...messageEntries].sort((left, right) =>
    String(left.timestamp || "").localeCompare(String(right.timestamp || "")),
  );
}

function buildSessionOverview(session) {
  const validation = buildValidationSnapshot(session);
  const issueText = latestIssueText(session);

  if (session?.stop_requested && isSessionRunning(session)) {
    return {
      tone: "warning",
      title: "Beendet den aktuellen Schritt",
      summary:
        "Die Stop-Anfrage ist aktiv. Der Agent schliesst den laufenden Schritt sauber ab und bleibt danach fuer den naechsten Auftrag bereit.",
    };
  }

  if (isSessionRunning(session)) {
    return {
      tone: "running",
      title: phaseHeadline(session.current_phase),
      summary: currentThought() || "Der Agent arbeitet gerade im Hintergrund.",
    };
  }

  if (session?.status === "completed") {
    return {
      tone: "success",
      title: "Ergebnis bereit",
      summary: buildCompletionSummary(session, validation),
    };
  }

  if (session?.status === "partial") {
    return {
      tone: "warning",
      title: "Offene Punkte im Thread",
      summary:
        issueText ||
        validation.summary ||
        "Der Thread wurde mit offenen Punkten beendet und braucht noch einen gezielten Nachschritt.",
    };
  }

  if (session?.status === "failed") {
    return {
      tone: "danger",
      title: "Eingriff erforderlich",
      summary:
        issueText ||
        validation.summary ||
        session?.last_error ||
        "Der Lauf ist mit einem Fehler beendet worden.",
    };
  }

  return {
    tone: "muted",
    title: "Bereit fuer den naechsten Auftrag",
    summary: "Dieser Thread wartet auf eine neue Nachricht.",
  };
}

function buildCompletionSummary(session, validation) {
  const parts = [];
  if (session?.report?.summary) {
    parts.push(session.report.summary);
  } else if (session?.changed_files?.length) {
    parts.push(
      session.changed_files.length === 1
        ? "1 Datei wurde angepasst."
        : `${session.changed_files.length} Dateien wurden angepasst.`,
    );
  } else if (session?.final_response) {
    parts.push("Die Antwort wurde bereitgestellt.");
  }

  if (validation.tone === "success") {
    parts.push("Validierung bestaetigt.");
  } else if (validation.tone === "warning" || validation.tone === "danger") {
    parts.push(validation.summary);
  }

  return shorten(parts.filter(Boolean).join(" ") || "Der Thread ist abgeschlossen.", 220);
}

function buildRunHighlights(session) {
  const validation = buildValidationSnapshot(session);
  return [
    {
      label: "Phase",
      value: labelForPhase(session?.current_phase),
      tone: sessionStatusTone(session),
    },
    {
      label: "Werkzeuge",
      value: String(session?.tool_calls?.length || 0),
      tone: "muted",
    },
    {
      label: "Dateien",
      value: String(session?.changed_files?.length || 0),
      tone: session?.changed_files?.length ? "success" : "muted",
    },
    {
      label: "Checks",
      value: buildValidationSnapshot(session).statusLabel,
      tone: validation.tone,
    },
  ];
}

function buildThreadPresentationView(session, logs) {
  return {
    running: isSessionRunning(session),
    currentStep: currentThoughtFrom({ activeSession: session, logs }),
    durationLabel: formatSessionElapsed(session),
    overview: buildSessionOverview(session),
    validation: buildValidationSnapshot(session),
    changes: Array.isArray(session?.changed_files) ? session.changed_files : [],
    activity: buildActivityClusters(session, logs).slice(0, 6),
    highlights: buildRunHighlights(session),
  };
}

function buildValidationSnapshot(session) {
  const runs = Array.isArray(session?.validation_runs)
    ? [...session.validation_runs].slice(-4).reverse()
    : [];
  const latest = runs[0] || null;
  const changedCount = Array.isArray(session?.changed_files) ? session.changed_files.length : 0;

  if (isSessionRunning(session) && session?.current_phase === "verifying") {
    return {
      tone: "running",
      statusLabel: "Laeuft",
      title: "Validierung laeuft",
      summary: currentThought() || "Die letzten Aenderungen werden gerade geprueft.",
      latest,
      runs,
    };
  }

  if (session?.validation_status === "passed") {
    return {
      tone: "success",
      statusLabel: "Bestanden",
      title: "Validierung bestaetigt",
      summary:
        latest?.summary ||
        latest?.excerpt ||
        "Die bestaetigenden Checks sind erfolgreich durchgelaufen.",
      latest,
      runs,
    };
  }

  if (session?.validation_status === "failed") {
    return {
      tone: "danger",
      statusLabel: "Fehler",
      title: "Validierung fehlgeschlagen",
      summary:
        latest?.summary ||
        latest?.excerpt ||
        "Mindestens ein Check ist fehlgeschlagen.",
      latest,
      runs,
    };
  }

  if (session?.validation_status === "blocked") {
    return {
      tone: "warning",
      statusLabel: "Blockiert",
      title: "Validierung blockiert",
      summary:
        latest?.summary ||
        latest?.excerpt ||
        "Der naechste Check konnte nicht sicher ausgefuehrt werden.",
      latest,
      runs,
    };
  }

  if (changedCount > 0) {
    return {
      tone: "warning",
      statusLabel: "Ausstehend",
      title: "Validierung ausstehend",
      summary: "Es liegen Aenderungen vor, aber noch kein bestaetigender Check.",
      latest,
      runs,
    };
  }

  return {
    tone: "muted",
    statusLabel: "Nicht noetig",
    title: "Noch keine Validierung",
    summary: "In diesem Thread wurden bisher keine geaenderten Dateien bestaetigend geprueft.",
    latest,
    runs,
  };
}

function buildIssueItems(session) {
  const items = [];
  const blockers = Array.isArray(session?.blockers) ? [...session.blockers].slice(-3).reverse() : [];

  for (const blocker of blockers) {
    items.push({
      label: "Blocker",
      tone: "danger",
      text: blocker,
      meta: null,
    });
  }

  if (session?.last_error && !items.some((item) => item.text === session.last_error)) {
    items.unshift({
      label: "Fehler",
      tone: "danger",
      text: session.last_error,
      meta: null,
    });
  }

  const diagnostics = Array.isArray(session?.diagnostics)
    ? [...session.diagnostics].slice(-3).reverse()
    : [];
  for (const diagnostic of diagnostics) {
    const metaParts = [
      diagnostic.command ? shorten(diagnostic.command, 56) : "",
      diagnostic.file_hints?.[0] ? shortenPath(diagnostic.file_hints[0], 48) : "",
    ].filter(Boolean);

    items.push({
      label:
        diagnostic.severity === "warning"
          ? "Hinweis"
          : diagnostic.severity === "info"
            ? "Info"
            : "Diagnose",
      tone:
        diagnostic.severity === "warning"
          ? "warning"
          : diagnostic.severity === "info"
            ? "muted"
            : "danger",
      text: shorten(
        [diagnostic.summary, diagnostic.action_hints?.[0]].filter(Boolean).join(" · ") ||
          "Diagnostischer Hinweis",
        180,
      ),
      meta: metaParts.join(" · ") || null,
    });
  }

  return items.slice(0, 5);
}

function latestIssueText(session) {
  const issues = buildIssueItems(session);
  return issues[0]?.text || "";
}

function buildActivityClusters(session, logs) {
  const source = (Array.isArray(logs) ? logs : [])
    .map((record) => {
      const detail = describeLogRecord(record);
      if (!detail) {
        return null;
      }
      return {
        ...detail,
        timestamp: record.timestamp,
      };
    })
    .filter(Boolean);

  const items = source.length ? source : buildActivityFallback(session);
  const clusters = [];

  for (const item of items) {
    const previous = clusters[clusters.length - 1];
    if (
      previous &&
      item.groupKey &&
      previous.groupKey === item.groupKey &&
      previous.tone === item.tone
    ) {
      previous.count += 1;
      previous.timestamp = item.timestamp;
      previous.text = item.text;
      if (item.meta) {
        previous.meta = item.meta;
      }
      continue;
    }

    clusters.push({
      ...item,
      count: 1,
    });
  }

  return clusters.slice(-12).reverse();
}

function buildActivityFallback(session) {
  const toolCalls = Array.isArray(session?.tool_calls) ? session.tool_calls.slice(-8) : [];
  return toolCalls.map((call) => {
    const path = extractPathFromPayload({ tool_args: call.tool_args || {} });
    const command = extractCommandFromPayload({ tool_args: call.tool_args || {} });
    return {
      text: call.success
        ? describeToolActivity({
            tool_name: call.tool_name,
            tool_args: call.tool_args || {},
            thought_summary: call.thought_summary,
            expected_outcome: call.expected_outcome,
          }) || "Werkzeugschritt abgeschlossen"
        : `Schritt fehlgeschlagen: ${humanizeToolName(call.tool_name || "tool")}`,
      meta: path ? shortenPath(path, 64) : command ? shorten(command, 72) : null,
      tone: call.success ? "muted" : "danger",
      timestamp: call.timestamp,
      groupKey: null,
    };
  });
}

function buildPhaseSteps(session) {
  const steps = [
    { key: "planning", label: "Plan" },
    { key: "exploring", label: "Analyse" },
    { key: "editing", label: "Umsetzung" },
    { key: "verifying", label: "Validierung" },
    { key: "reporting", label: "Ergebnis" },
  ];
  const activeKey = phaseStepKey(session);
  const activeIndex = Math.max(
    0,
    steps.findIndex((step) => step.key === activeKey),
  );
  const blocked = !isSessionRunning(session) && (session?.status === "failed" || session?.status === "partial");

  return steps.map((step, index) => {
    if (session?.status === "completed") {
      return { ...step, state: "completed" };
    }
    if (index < activeIndex) {
      return { ...step, state: "completed" };
    }
    if (index === activeIndex) {
      return { ...step, state: blocked ? "blocked" : "active" };
    }
    return { ...step, state: "pending" };
  });
}

function phaseStepKey(session) {
  const phase = String(session?.current_phase || "").trim();
  if (phase === "planning") return "planning";
  if (phase === "exploring") return "exploring";
  if (phase === "editing") return "editing";
  if (phase === "verifying" || phase === "repairing") return "verifying";
  if (phase === "reporting" || phase === "completed") return "reporting";
  if (phase === "blocked") {
    if (session?.validation_status === "failed" || session?.validation_status === "blocked" || session?.validation_runs?.length) {
      return "verifying";
    }
    if (session?.changed_files?.length) {
      return "editing";
    }
    return "exploring";
  }
  return "planning";
}

function phaseHeadline(phase) {
  if (phase === "planning") return "Plant den naechsten Schritt";
  if (phase === "exploring") return "Sammelt Projektkontext";
  if (phase === "editing") return "Setzt die Aenderung um";
  if (phase === "verifying") return "Prueft die Aenderung";
  if (phase === "repairing") return "Repariert nach einem Check";
  if (phase === "reporting") return "Bereitet das Ergebnis auf";
  if (phase === "blocked") return "Braucht einen Eingriff";
  if (phase === "completed") return "Ergebnis bereit";
  return "Agent arbeitet";
}

function labelForPhase(phase) {
  if (phase === "planning") return "Plan";
  if (phase === "exploring") return "Analyse";
  if (phase === "editing") return "Umsetzung";
  if (phase === "verifying") return "Validierung";
  if (phase === "repairing") return "Reparatur";
  if (phase === "reporting") return "Ergebnis";
  if (phase === "blocked") return "Blockiert";
  if (phase === "completed") return "Abgeschlossen";
  return phase || "-";
}

function sessionBadgeText(session) {
  if (session?.stop_requested && isSessionRunning(session)) return "Stoppt";
  if (session?.status === "queued") return "Startet";
  if (session?.status === "running") return "Aktiv";
  if (session?.status === "completed") return "Fertig";
  if (session?.status === "partial") return session?.blockers?.length ? "Blockiert" : "Offen";
  if (session?.status === "failed") return "Fehler";
  return "Bereit";
}

function sessionStatusTone(session) {
  if (session?.stop_requested && isSessionRunning(session)) return "warning";
  if (session?.status === "queued" || session?.status === "running") return "running";
  if (session?.status === "completed") return "success";
  if (session?.status === "partial") {
    return session?.blockers?.length || session?.validation_status === "failed" || session?.validation_status === "blocked"
      ? "warning"
      : "muted";
  }
  if (session?.status === "failed") return "danger";
  return "muted";
}

function validationRunTone(run) {
  if (run?.status === "passed") return "success";
  if (run?.status === "failed" || run?.status === "timeout") return "danger";
  if (run?.status === "blocked") return "warning";
  return "muted";
}

function labelForValidationRunStatus(status) {
  if (status === "passed") return "Bestanden";
  if (status === "failed") return "Fehler";
  if (status === "blocked") return "Blockiert";
  if (status === "timeout") return "Timeout";
  return status || "-";
}

function labelForVerificationScope(scope) {
  if (scope === "runtime") return "Runtime";
  if (scope === "structural") return "Struktur";
  if (scope === "static") return "Static";
  if (scope === "syntax") return "Syntax";
  return scope || "Check";
}

function labelForFileOperation(operation) {
  if (operation === "create") return "Neu";
  if (operation === "write" || operation === "update" || operation === "modify") return "Update";
  if (operation === "delete") return "Entfernt";
  if (operation === "append") return "Anhang";
  if (operation === "patch" || operation === "replace") return "Patch";
  return humanizeToolName(operation || "Datei");
}

function operationTone(operation) {
  if (operation === "delete") return "warning";
  if (["create", "write", "update", "modify", "append", "patch", "replace"].includes(operation || "")) {
    return "success";
  }
  return "muted";
}

function shouldOpenDetailPanel(tone, session) {
  return tone === "danger" || tone === "warning" || (tone === "running" && isSessionRunning(session));
}

function countLabel(count, singular, plural) {
  return Number(count) === 1 ? singular : plural;
}

function describeLogRecord(record) {
  if (!record || !record.event) {
    return null;
  }

  const payload = record.payload || {};
  const toolName = payload.tool || payload.tool_name || null;
  const message = String(payload.message || payload.error || "").trim();
  const progressLog = describeAgentProgressLog(record.event, payload);
  if (progressLog) {
    return progressLog;
  }

  if (record.event === "tool_requested") {
    return describeToolRequestLog(toolName, payload);
  }

  if (record.event === "tool_result") {
    return describeToolResultLog(toolName, payload);
  }

  if (record.event === "tool_blocked") {
    return {
      text: `Tool blockiert: ${humanizeToolName(toolName || "tool")}`,
      meta: ((payload.reasons || []).join(" · ") || "").trim() || null,
      tone: "danger",
      groupKey: `tool-blocked:${toolName || "tool"}`,
    };
  }

  if (
    record.event === "tool_error" ||
    record.event === "tool_execution_error" ||
    record.event === "tool_validation_error"
  ) {
    return {
      text: `Werkzeugfehler in ${humanizeToolName(toolName || "tool")}`,
      meta: message || null,
      tone: "danger",
      groupKey: `tool-error:${toolName || "tool"}`,
    };
  }

  if (record.event === "task_stop_requested") {
    return {
      text: "Stop-Anfrage registriert",
      meta: "Der aktuelle Schritt wird noch sauber abgeschlossen.",
      tone: "warning",
      groupKey: "task-stop-requested",
    };
  }

  if (record.event === "task_stopped") {
    return {
      text: "Lauf angehalten",
      meta: "Der Thread wurde auf Wunsch gestoppt.",
      tone: "warning",
      groupKey: "task-stopped",
    };
  }

  if (record.event === "task_crashed") {
    return {
      text: "Laufzeitfehler",
      meta: message || "Unbekannter Fehler",
      tone: "danger",
      groupKey: "task-crashed",
    };
  }

  if (record.event === "task_finished") {
    return {
      text:
        payload.status === "completed"
          ? "Lauf abgeschlossen"
          : `Lauf beendet: ${labelForStatus(payload.status || "")}`,
      meta: String(payload.stop_reason || payload.workflow_stage || "").trim() || null,
      tone: payload.status === "completed" ? "success" : payload.status === "failed" ? "danger" : "warning",
      groupKey: "task-finished",
    };
  }

  return null;
}

function describeAgentProgressLog(event, payload) {
  const text = describeAgentProgressActivity(event, payload);
  if (!text) {
    return null;
  }

  return {
    text,
    meta: describeActivityMeta(event, payload),
    tone: event.includes("retry") ? "warning" : event.includes("finished") ? "success" : "muted",
    groupKey: describeActivityGroupKey(event, payload),
  };
}

function describeActivityMeta(event, payload) {
  const path = extractPathFromPayload(payload || {});
  const model = String(payload?.model || "").trim();

  if ((event === "path_generation_started" || event === "path_generation_finished") && path) {
    return shortenPath(path, 64);
  }
  if (event === "content_generation_started" && path) {
    return shortenPath(path, 64);
  }
  if (
    event === "content_generation_fallback_started" ||
    event === "content_generation_retry_started" ||
    event === "content_generation_recovery_started"
  ) {
    return describeContentGenerationRetry(path, payload);
  }
  if (event === "content_generation_progress" || event === "final_response_generation_progress") {
    return model ? shorten(model, 28) : null;
  }
  if (event === "router_retry_started") {
    return "Fallback auf kompakteres Routing";
  }
  return null;
}

function describeActivityGroupKey(event, payload) {
  const path = extractPathFromPayload(payload || {});
  if (event === "content_generation_progress") {
    return `progress:content:${path || "-"}`;
  }
  if (event === "final_response_generation_progress") {
    return "progress:final-response";
  }
  if (event === "router_retry_started") {
    return "router-retry";
  }
  return null;
}

function describeToolRequestLog(toolName, payload) {
  const args = payload.args || payload.tool_args || {};
  const path = extractPathFromPayload(payload);
  const command = extractCommandFromPayload(payload);
  const query = typeof args.query === "string" ? args.query.trim() : "";
  const focus = typeof args.focus === "string" ? args.focus.trim() : "";

  if (toolName === "inspect_workspace") {
    return {
      text: "Projektkontext wird gesammelt",
      meta: focus ? shorten(focus, 60) : null,
      tone: "muted",
      groupKey: focus ? `inspect:${focus}` : "inspect-workspace",
    };
  }

  if (toolName === "search_in_files") {
    return {
      text: "Projekt wird durchsucht",
      meta: query || null,
      tone: "muted",
      groupKey: query ? `search:${query}` : "search-project",
    };
  }

  if (toolName === "read_file") {
    return {
      text: "Datei wird gelesen",
      meta: path ? shortenPath(path, 64) : null,
      tone: "muted",
      groupKey: path ? `read:${path}` : "read-file",
    };
  }

  if (toolName === "list_files") {
    return {
      text: "Projektstruktur wird geprueft",
      meta: path && path !== "." ? shortenPath(path, 64) : null,
      tone: "muted",
      groupKey: path && path !== "." ? `list:${path}` : "list-files",
    };
  }

  if (["create_file", "write_file", "append_file", "replace_in_file", "patch_file", "delete_file"].includes(toolName || "")) {
    return {
      text: toolName === "delete_file" ? "Datei wird entfernt" : "Datei wird bearbeitet",
      meta: path ? shortenPath(path, 64) : null,
      tone: "running",
      groupKey: path ? `write:${path}` : `write:${toolName}`,
    };
  }

  if (toolName === "run_tests") {
    return {
      text: "Check wird ausgefuehrt",
      meta: command ? shorten(command, 76) : null,
      tone: "running",
      groupKey: command ? `validation:${command}` : "validation-run",
    };
  }

  if (toolName === "run_shell") {
    return {
      text: "Befehl wird ausgefuehrt",
      meta: command ? shorten(command, 76) : humanizeToolName(toolName),
      tone: "running",
      groupKey: command ? `command:${command}` : "shell-run",
    };
  }

  if (toolName === "git_status") {
    return {
      text: "Git-Status wird geprueft",
      meta: null,
      tone: "muted",
      groupKey: "git-status",
    };
  }

  if (toolName === "git_diff" || toolName === "show_diff") {
    return {
      text: "Diff wird geprueft",
      meta: null,
      tone: "muted",
      groupKey: "git-diff",
    };
  }

  if (toolName === "git_log") {
    return {
      text: "Commit-Verlauf wird gelesen",
      meta: null,
      tone: "muted",
      groupKey: "git-log",
    };
  }

  return {
    text: `${humanizeToolName(toolName || "Tool")} wird gestartet`,
    meta: command ? shorten(command, 76) : null,
    tone: command ? "running" : "muted",
    groupKey: null,
  };
}

function describeToolResultLog(toolName, payload) {
  const message = String(payload.message || "").trim();
  const success = Boolean(payload.success);
  const changedFiles = Number(payload.changed_files || 0);

  if (!success) {
    return {
      text: `Schritt fehlgeschlagen: ${humanizeToolName(toolName || "tool")}`,
      meta: message || null,
      tone: "danger",
      groupKey: null,
    };
  }

  if (toolName === "inspect_workspace" || toolName === "read_file" || toolName === "list_files" || toolName === "git_status") {
    return null;
  }

  if (toolName === "search_in_files") {
    const count = extractMatchCount(message);
    return {
      text: count === 0 ? "Keine Treffer gefunden" : `${count} Treffer gefunden`,
      meta: null,
      tone: "muted",
      groupKey: null,
    };
  }

  if (toolName === "run_tests") {
    return {
      text: "Check abgeschlossen",
      meta: message ? shorten(message, 120) : null,
      tone: "success",
      groupKey: null,
    };
  }

  if (toolName === "run_shell") {
    return {
      text: "Befehl abgeschlossen",
      meta: message ? shorten(message, 120) : null,
      tone: "success",
      groupKey: null,
    };
  }

  if (changedFiles > 0) {
    return {
      text: changedFiles === 1 ? "1 Datei aktualisiert" : `${changedFiles} Dateien aktualisiert`,
      meta: message && !/changed|updated|created|deleted/i.test(message) ? shorten(message, 120) : null,
      tone: "success",
      groupKey: null,
    };
  }

  if (message) {
    return {
      text: shorten(message, 96),
      meta: null,
      tone: "success",
      groupKey: null,
    };
  }

  return null;
}

function extractMatchCount(message) {
  const match = String(message || "").match(/(\d+)\s+match/i);
  if (!match) {
    return null;
  }
  return Number(match[1]);
}

function humanizeToolName(toolName) {
  const normalized = String(toolName || "").trim();
  if (!normalized) {
    return "Tool";
  }
  return normalized
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function extractCommandFromPayload(payload) {
  const args = payload.args || payload.tool_args || {};
  if (typeof args.command === "string" && args.command.trim()) {
    return args.command.trim();
  }
  if (typeof payload.command === "string" && payload.command.trim()) {
    return payload.command.trim();
  }
  return "";
}

function appendLogRecord(record) {
  if (!record) {
    return;
  }

  const signature = logSignature(record);
  if (state.logs.some((item) => logSignature(item) === signature)) {
    return;
  }

  state.logs = [...state.logs, record];
}

function selectedWorkspaceFrom(sourceState = state) {
  return sourceState.workspaces.find((workspace) => workspace.id === sourceState.selectedWorkspaceId) || null;
}

function selectedWorkspace() {
  return selectedWorkspaceFrom(state);
}

function activeWorkspaceIdFrom(sourceState = state) {
  return sourceState.activeSession?.workspace_id || sourceState.selectedWorkspaceId;
}

function activeWorkspaceId() {
  return activeWorkspaceIdFrom(state);
}

function workspaceForSessionFrom(sourceState = state, session) {
  if (!session) {
    return null;
  }
  return sourceState.workspaces.find((workspace) => workspace.id === session.workspace_id) || null;
}

function workspaceForSession(session) {
  return workspaceForSessionFrom(state, session);
}

function sessionsForWorkspaceFrom(sourceState = state, workspaceId) {
  return sourceState.sessions
    .filter((session) => session.workspace_id === workspaceId && !session.archived)
    .sort((left, right) => new Date(right.updated_at) - new Date(left.updated_at));
}

function sessionsForWorkspace(workspaceId) {
  return sessionsForWorkspaceFrom(state, workspaceId);
}

function isWorkspaceBusy(workspaceId) {
  return state.sessions.some(
    (session) => session.workspace_id === workspaceId && isSessionRunning(session),
  );
}

function getAgentProfiles() {
  return state.config?.agent_profiles || [];
}

function getExecutionProfiles() {
  return state.config?.execution_profiles || [];
}

function getAccessModes() {
  return state.config?.access_modes || ["safe", "approval", "full"];
}

function getModelOptions() {
  return Array.from(
    new Set([
      state.composer.modelName,
      ...(state.config?.model_candidates || []),
      ...((state.models?.installed_models || []).map((item) => item.name)),
      state.config?.model_name,
    ]),
  ).filter(Boolean);
}

function modelFieldHelperText() {
  const installing = activeRecommendedModelJobs();
  if (!installing.length) {
    return "Empfohlene Modelle werden automatisch installiert, falls sie fehlen.";
  }
  const primary = installing[0];
  return `${primary.label || primary.name} wird gerade vorbereitet.`;
}

function renderRecommendedModels() {
  const models = state.models?.recommended_models || [];
  if (!models.length) {
    return `
      <div class="model-empty">
        Noch keine Modellinformationen verfuegbar.
      </div>
    `;
  }

  return models.map(renderRecommendedModelItem).join("");
}

function renderRecommendedModelItem(model) {
  const progressLabel = formatModelProgress(model);
  return `
    <article class="model-item">
      <div class="model-item-main">
        <div class="model-item-head">
          <strong>${escapeHtml(model.label || model.name)}</strong>
          <span class="model-status ${escapeHtml(model.status || "missing")}">${escapeHtml(labelForModelStatus(model.status))}</span>
        </div>
        <p class="model-summary">${escapeHtml(model.summary || "")}</p>
        <div class="model-meta">
          <span>${escapeHtml(model.name)}</span>
          <span>${escapeHtml(progressLabel)}</span>
        </div>
        ${renderModelProgressBar(model)}
        ${model.error ? `<p class="model-error">${escapeHtml(model.error)}</p>` : ""}
      </div>
    </article>
  `;
}

function renderModelProgressBar(model) {
  if (!["queued", "pulling", "verifying"].includes(model.status || "")) {
    return "";
  }
  const width = Math.max(4, Math.round((Number(model.progress) || 0) * 100));
  return `
    <div class="model-progress" aria-hidden="true">
      <span style="width: ${width}%"></span>
    </div>
  `;
}

function composerPlaceholder(workspace) {
  if (!workspace) {
    return "Verbinde zuerst links ein lokales Projekt";
  }
  if (state.activeSession) {
    return `Beschreibe den naechsten Schritt fuer ${workspace.name}`;
  }
  return `Starte einen neuen Thread in ${workspace.name}`;
}

function composerHint(workspace) {
  if (!workspace) {
    return "Projekt auswaehlen";
  }
  if (!state.activeSession) {
    return "Neuer Thread";
  }
  if (state.activeSession.stop_requested) {
    return "Stop wird vorbereitet";
  }
  return "Antwort oder Korrektur senden";
}

function applyModelCatalog(catalog) {
  const nextCatalog = {
    installed_models: Array.isArray(catalog?.installed_models) ? catalog.installed_models : [],
    recommended_models: Array.isArray(catalog?.recommended_models) ? catalog.recommended_models : [],
  };
  const changed = JSON.stringify(state.models || {}) !== JSON.stringify(nextCatalog);
  state.models = nextCatalog;

  const installedNames = nextCatalog.installed_models
    .map((item) => item?.name)
    .filter(Boolean);

  if (state.config) {
    state.config.installed_ollama_models = nextCatalog.installed_models;
    state.config.recommended_models = nextCatalog.recommended_models;
    state.config.model_candidates = Array.from(
      new Set([
        state.config.preferred_model_name,
        state.config.model_name,
        ...installedNames,
      ]),
    ).filter(Boolean);
  }

  return changed;
}

function activeRecommendedModelJobs() {
  return (state.models?.recommended_models || []).filter((item) =>
    ["queued", "pulling", "verifying"].includes(item.status || ""),
  );
}

function currentModelInstallNotice() {
  const jobs = activeRecommendedModelJobs();
  if (!jobs.length) {
    return "";
  }

  const primary = jobs[0];
  const progress = formatModelProgress(primary);
  if (jobs.length === 1) {
    return `${primary.label || primary.name}: ${progress}`;
  }
  return `${primary.label || primary.name}: ${progress} + ${jobs.length - 1} weitere Modelle in Warteschlange.`;
}

function showToast(message, tone = "success") {
  state.ui.toast = { message, tone };
  renderApp();
  if (state.ui.toastTimer) {
    window.clearTimeout(state.ui.toastTimer);
  }
  state.ui.toastTimer = window.setTimeout(() => {
    state.ui.toast = null;
    renderApp();
  }, 3200);
}

class ApiError extends Error {
  constructor(message, { status, payload = null, retryAfter = 0 } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
    this.retryAfter = retryAfter;
  }
}

async function fetchJSON(url, options = {}) {
  const requestOptions = buildRequestOptions(options);
  const response = await fetch(url, requestOptions);
  if (!response.ok) {
    const payload = await readResponsePayload(response);
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? String(payload.detail || "")
        : typeof payload === "string"
          ? payload
          : response.statusText;
    const error = new ApiError(detail || `HTTP ${response.status}`, {
      status: response.status,
      payload,
      retryAfter: Number(response.headers.get("retry-after") || 0),
    });
    if (response.status === 401 && shouldHandleUnauthorized(url)) {
      await handleUnauthorizedResponse(detail || "Sitzung abgelaufen. Bitte erneut anmelden.");
    }
    throw error;
  }
  if (response.status === 204) {
    return null;
  }
  return readResponsePayload(response);
}

function buildRequestOptions(options = {}) {
  const nextOptions = { ...options, credentials: options.credentials || "same-origin" };
  const method = String(nextOptions.method || "GET").toUpperCase();
  const headers = new Headers(nextOptions.headers || {});
  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json");
  }
  if (isMutatingMethod(method)) {
    const csrfToken = readCookieValue("__Host-marc_csrf") || readCookieValue("marc_csrf");
    if (csrfToken && !headers.has(state.auth.csrfHeaderName || "X-CSRF-Token")) {
      headers.set(state.auth.csrfHeaderName || "X-CSRF-Token", csrfToken);
    }
  }
  nextOptions.headers = headers;
  return nextOptions;
}

async function readResponsePayload(response) {
  const contentType = response.headers.get("content-type") || "";
  const text = await response.text();
  if (!text) {
    return null;
  }
  if (contentType.includes("application/json")) {
    return JSON.parse(text);
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    return text;
  }
}

function shouldHandleUnauthorized(url) {
  const target = String(url || "");
  return target.startsWith("/api/") && !target.startsWith("/api/auth/");
}

function isMutatingMethod(method) {
  return ["POST", "PUT", "PATCH", "DELETE"].includes(String(method || "").toUpperCase());
}

function readCookieValue(name) {
  const source = typeof document === "undefined" ? "" : String(document.cookie || "");
  for (const part of source.split(";")) {
    const [rawKey, ...rest] = part.trim().split("=");
    if (rawKey === name) {
      return decodeURIComponent(rest.join("="));
    }
  }
  return "";
}

function loginRetryAfterSeconds() {
  if (!state.auth.rateLimitedUntil) {
    return 0;
  }
  return Math.max(0, Math.ceil((state.auth.rateLimitedUntil - Date.now()) / 1000));
}

function hydrateRouteFromLocation() {
  const route = parseUiRoute(window.location.href);
  state.activeSessionId = route.sessionId;
  state.ui.page = route.page;
}

function syncHistory(sessionIdOrOptions = {}) {
  let sessionId = state.activeSessionId;
  let page = state.ui.page || "workspace";
  let replace = false;

  if (typeof sessionIdOrOptions === "string" || sessionIdOrOptions === null) {
    sessionId = sessionIdOrOptions;
  } else if (sessionIdOrOptions && typeof sessionIdOrOptions === "object") {
    if ("sessionId" in sessionIdOrOptions) {
      sessionId = sessionIdOrOptions.sessionId;
    }
    if ("page" in sessionIdOrOptions) {
      page = sessionIdOrOptions.page;
    }
    replace = Boolean(sessionIdOrOptions.replace);
  }

  const url = buildUiRoute({ sessionId, page });
  window.history[replace ? "replaceState" : "pushState"]({}, "", url);
}

function parseUiRoute(value) {
  const url = new URL(value, "http://localhost");
  const page = url.searchParams.get("view") === "settings" ? "settings" : "workspace";
  return {
    page,
    sessionId: url.searchParams.get("session"),
  };
}

function buildUiRoute({ sessionId = null, page = "workspace" } = {}) {
  const params = new URLSearchParams();
  if (sessionId) {
    params.set("session", sessionId);
  }
  if (page === "settings") {
    params.set("view", "settings");
  }
  const query = params.toString();
  return query ? `/?${query}` : "/";
}

function applyStoredPreferences() {
  const stored = readStoredPreferences();
  state.selectedWorkspaceId = stored.selectedWorkspaceId || state.selectedWorkspaceId;
  state.composer.agentProfile = stored.agentProfile || state.composer.agentProfile;
  state.composer.accessMode = stored.accessMode || state.config?.access_mode || state.composer.accessMode;
  state.composer.modelName = stored.modelName || state.config?.model_name || "";
  state.composer.executionProfile = stored.executionProfile || state.composer.executionProfile;
  state.composer.dryRun =
    typeof stored.dryRun === "boolean" ? stored.dryRun : Boolean(state.config?.dry_run);
}

function persistPreferences() {
  try {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        selectedWorkspaceId: state.selectedWorkspaceId,
        agentProfile: state.composer.agentProfile,
        accessMode: state.composer.accessMode,
        modelName: state.composer.modelName,
        executionProfile: state.composer.executionProfile,
        dryRun: state.composer.dryRun,
      }),
    );
  } catch (error) {
    console.error(error);
  }
}

function readStoredPreferences() {
  try {
    return JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "{}");
  } catch (error) {
    return {};
  }
}

function labelForAccessMode(mode) {
  if (mode === "safe") return "Nur Lesen";
  if (mode === "approval") return "Mit Freigabe";
  if (mode === "full") return "Voller Zugriff";
  return mode || "-";
}

function labelForStatus(status) {
  if (status === "queued") return "Vorbereitet";
  if (status === "running") return "Laeuft";
  if (status === "completed") return "Abgeschlossen";
  if (status === "partial") return "Teilweise";
  if (status === "failed") return "Fehler";
  return status || "-";
}

function labelForModelStatus(status) {
  if (status === "installed") return "Installiert";
  if (status === "queued") return "Wartet";
  if (status === "pulling") return "Laedt";
  if (status === "verifying") return "Prueft";
  if (status === "failed") return "Fehler";
  return "Fehlt";
}

function formatModelProgress(model) {
  if (!model) {
    return "";
  }

  if (model.status === "installed") {
    return "Bereit";
  }
  if (model.status === "failed") {
    return model.message || "Download fehlgeschlagen";
  }
  if (model.status === "queued") {
    return model.message || "Wartet auf Download";
  }

  const progress = Number(model.progress);
  const hasProgress = Number.isFinite(progress) && progress > 0;
  const progressText = hasProgress ? `${Math.round(progress * 100)}%` : "";
  const byteText = formatByteProgress(model.completed_bytes, model.total_bytes);
  const detail = [model.message, progressText, byteText].filter(Boolean).join(" · ");
  return detail || "Wird vorbereitet";
}

function formatByteProgress(completedBytes, totalBytes) {
  const completed = Number(completedBytes);
  const total = Number(totalBytes);
  if (!Number.isFinite(total) || total <= 0 || !Number.isFinite(completed) || completed < 0) {
    return "";
  }
  return `${formatBytes(completed)} / ${formatBytes(total)}`;
}

function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }

  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const precision = unitIndex >= 2 ? 1 : 0;
  return `${size.toFixed(precision)} ${units[unitIndex]}`;
}

function statusTone(status) {
  if (status === "completed") return "completed";
  if (status === "partial") return "warning";
  if (status === "failed") return "danger";
  return "running";
}

function roleLabel(role) {
  if (role === "assistant") return "Agent";
  if (role === "system") return "System";
  return "Du";
}

function renderRichText(text) {
  return renderMarkdownDocument(String(text || ""));
}

function renderMarkdownDocument(value) {
  const lines = String(value || "").replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  const paragraph = [];
  const quote = [];
  const code = [];
  let listItems = [];
  let listType = null;
  let inCode = false;

  const flushParagraph = () => {
    if (!paragraph.length) {
      return;
    }
    html.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph.length = 0;
  };

  const flushList = () => {
    if (!listItems.length || !listType) {
      return;
    }
    html.push(
      `<${listType} class="rich-list">${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</${listType}>`,
    );
    listItems = [];
    listType = null;
  };

  const flushQuote = () => {
    if (!quote.length) {
      return;
    }
    html.push(`<blockquote>${renderMarkdownDocument(quote.join("\n"))}</blockquote>`);
    quote.length = 0;
  };

  const flushCode = () => {
    if (!code.length) {
      return;
    }
    html.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
    code.length = 0;
  };

  for (const rawLine of lines) {
    const line = rawLine.replace(/\t/g, "  ");
    const trimmed = line.trim();

    if (/^```/.test(trimmed)) {
      flushParagraph();
      flushList();
      flushQuote();
      if (inCode) {
        flushCode();
        inCode = false;
      } else {
        inCode = true;
      }
      continue;
    }

    if (inCode) {
      code.push(rawLine);
      continue;
    }

    if (!trimmed) {
      flushParagraph();
      flushList();
      flushQuote();
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      flushQuote();
      const level = Math.min(headingMatch[1].length, 3);
      html.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
      continue;
    }

    if (/^>\s?/.test(trimmed)) {
      flushParagraph();
      flushList();
      quote.push(trimmed.replace(/^>\s?/, ""));
      continue;
    }

    const orderedMatch = trimmed.match(/^\d+\.\s+(.+)$/);
    if (orderedMatch) {
      flushParagraph();
      flushQuote();
      if (listType && listType !== "ol") {
        flushList();
      }
      listType = "ol";
      listItems.push(orderedMatch[1]);
      continue;
    }

    const unorderedMatch = trimmed.match(/^[-*]\s+(.+)$/);
    if (unorderedMatch) {
      flushParagraph();
      flushQuote();
      if (listType && listType !== "ul") {
        flushList();
      }
      listType = "ul";
      listItems.push(unorderedMatch[1]);
      continue;
    }

    flushQuote();
    flushList();
    paragraph.push(trimmed);
  }

  flushParagraph();
  flushList();
  flushQuote();
  flushCode();

  return html.join("");
}

function renderInlineMarkdown(text) {
  const codeTokens = [];
  let value = String(text || "").replace(/`([^`]+)`/g, (_, code) => {
    const token = `@@CODE${codeTokens.length}@@`;
    codeTokens.push(code);
    return token;
  });

  value = escapeHtml(value)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^)\s]+|\/[^)\s]+)\)/g,
      (_, label, href) =>
        `<a class="rich-link" href="${escapeAttribute(href)}"${String(href).startsWith("http") ? ' target="_blank" rel="noreferrer noopener"' : ""}>${label}</a>`,
    );

  return value.replace(/@@CODE(\d+)@@/g, (_, index) => `<code>${escapeHtml(codeTokens[Number(index)] || "")}</code>`);
}

function formatSessionTimestamp(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "-";
  }
  const now = new Date();
  const sameDay =
    now.getFullYear() === date.getFullYear() &&
    now.getMonth() === date.getMonth() &&
    now.getDate() === date.getDate();
  if (sameDay) {
    return formatTime(value);
  }
  return date.toLocaleDateString("de-DE", { day: "2-digit", month: "2-digit" });
}

function formatTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "-";
  }
  return date.toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" });
}

function formatSessionElapsed(session) {
  const startedAt = new Date(session?.created_at || "");
  const finishedAt = new Date(session?.updated_at || "");
  if (Number.isNaN(startedAt.getTime()) || Number.isNaN(finishedAt.getTime())) {
    return "";
  }
  const totalSeconds = Math.max(0, Math.round((finishedAt.getTime() - startedAt.getTime()) / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function shorten(text, limit = 72) {
  const value = String(text || "");
  if (value.length <= limit) {
    return value;
  }
  return `${value.slice(0, limit - 1)}…`;
}

function shortenPath(path, limit = 40) {
  const value = String(path || "");
  if (value.length <= limit) {
    return value;
  }
  const parts = value.split("/");
  if (parts.length <= 2) {
    return shorten(value, limit);
  }
  return `${parts.slice(0, 2).join("/")}/.../${parts[parts.length - 1]}`;
}

function hasCollectionChanged(previous, next) {
  return JSON.stringify(previous || []) !== JSON.stringify(next || []);
}

function logSignature(record) {
  return JSON.stringify([
    record.timestamp,
    record.event,
    record.payload || {},
  ]);
}

function isSessionRunning(session) {
  return Boolean(session && ["queued", "running"].includes(session.status));
}

function currentThoughtFrom(sourceState = state) {
  if (!isSessionRunning(sourceState.activeSession)) {
    return "";
  }

  if (sourceState.activeSession?.stop_requested) {
    return "Der laufende Schritt wird sauber beendet.";
  }

  if (sourceState.activeSession?.status === "queued" && (sourceState.logs || []).length === 0) {
    return "Der Lauf wird vorbereitet.";
  }

  for (const record of [...(sourceState.logs || [])].reverse()) {
    const thought = describeCurrentStep(record);
    if (thought) {
      return thought;
    }
  }

  const latestToolCall = [...(sourceState.activeSession?.tool_calls || [])]
    .reverse()
    .find((item) => item.tool_name || item.thought_summary || item.expected_outcome);
  if (latestToolCall) {
    const toolThought = describeToolActivity({
      tool: latestToolCall.tool_name,
      tool_name: latestToolCall.tool_name,
      args: latestToolCall.tool_args || {},
      thought_summary: latestToolCall.thought_summary,
      expected_outcome: latestToolCall.expected_outcome,
    });
    if (toolThought) {
      return toolThought;
    }
  }

  return "Der Agent arbeitet am aktuellen Auftrag.";
}

function currentThought() {
  return currentThoughtFrom(state);
}

function describeCurrentStep(record) {
  if (!record || !record.event) {
    return "";
  }

  const progress = describeAgentProgressActivity(record.event, record.payload || {});
  if (progress) {
    return progress;
  }

  if (record.event === "tool_requested") {
    return describeToolActivity(record.payload || {});
  }

  if (record.event === "decision") {
    return describeDecisionActivity(record.payload || {});
  }

  return "";
}

function describeAgentProgressActivity(event, payload) {
  const path = extractPathFromPayload(payload || {});

  if (event === "router_input") {
    return "Auftrag wird eingeordnet";
  }
  if (event === "router_retry_started") {
    return "Einordnung wird kompakter neu versucht";
  }
  if (event === "router_fast_path" || event === "router_timeout_fast_fallback") {
    return "Auftrag ist klar und geht direkt weiter";
  }
  if (event === "path_generation_started") {
    return "Zielpfad wird bestimmt";
  }
  if (event === "path_generation_skipped" || event === "path_generation_finished") {
    return path
      ? `Zieldatei festgelegt: ${shortenPath(path, 56)}`
      : "Zielpfad ist festgelegt";
  }
  if (event === "content_generation_started") {
    return path
      ? `Inhalt wird erstellt fuer ${shortenPath(path, 56)}`
      : "Dateiinhalt wird erstellt";
  }
  if (event === "content_generation_progress") {
    return describeStreamingModelProgress("content", payload);
  }
  if (event === "content_generation_retry_started") {
    return describeContentGenerationRetry(path, payload);
  }
  if (event === "content_generation_retry_error") {
    return describeContentGenerationRetryError(path, payload);
  }
  if (event === "content_generation_recovery_started") {
    return describeContentGenerationRecovery(path, payload);
  }
  if (event === "content_generation_recovery_finished") {
    return describeContentGenerationRecoveryFinished(path, payload);
  }
  if (event === "content_generation_recovery_unavailable") {
    return path
      ? `Kein sicherer lokaler Recovery-Pfad fuer ${shortenPath(path, 56)}`
      : "Kein sicherer lokaler Recovery-Pfad";
  }
  if (event === "content_generation_fallback_started") {
    const source = String(payload?.source || "").trim();
    if (source === "starter_scaffold") {
      return path
        ? `Lokales Starter-Grundgeruest wird vorbereitet fuer ${shortenPath(path, 56)}`
        : "Lokales Starter-Grundgeruest wird vorbereitet";
    }
    return path
      ? `Fallback aktiv fuer ${shortenPath(path, 56)}`
      : "Fallback fuer die Inhaltserstellung aktiv";
  }
  if (event === "content_generation_finished") {
    return path
      ? `Inhalt vorbereitet fuer ${shortenPath(path, 56)}`
      : "Dateiinhalt vorbereitet";
  }
  if (event === "content_generation_retry_finished") {
    return path
      ? `Kompakter Retry erfolgreich fuer ${shortenPath(path, 56)}`
      : "Kompakter Retry erfolgreich";
  }
  if (event === "content_generation_fallback_finished") {
    const source = String(payload?.source || "").trim();
    if (source === "starter_scaffold") {
      return path
        ? `Starter-Grundgeruest bereit fuer ${shortenPath(path, 56)}`
        : "Starter-Grundgeruest ist bereit";
    }
    return path
      ? `Fallback-Inhalt bereit fuer ${shortenPath(path, 56)}`
      : "Fallback-Inhalt ist bereit";
  }
  if (event === "final_block_reason") {
    return describeFinalBlock(payload);
  }
  if (event === "final_response_generation_started") {
    return "Antwort wird formuliert";
  }
  if (event === "final_response_generation_progress") {
    return describeStreamingModelProgress("final_response", payload);
  }

  return "";
}

function describeDecisionActivity(payload) {
  const toolThought = describeToolActivity(
    {
      tool: payload.tool_name,
      tool_name: payload.tool_name,
      tool_args: payload.tool_args || {},
      thought_summary: payload.thought_summary,
      expected_outcome: payload.expected_outcome,
    },
    { fromDecision: true },
  );
  if (toolThought) {
    return toolThought;
  }

  if (payload.action_type === "final") {
    return "Antwort wird formuliert";
  }

  const summary = normalizeProgressText(payload.thought_summary || payload.expected_outcome || "");
  if (summary) {
    return summary;
  }

  return "Naechster Schritt wird geplant";
}

function describeToolActivity(payload, options = {}) {
  const toolName = payload.tool || payload.tool_name || "";
  if (!toolName) {
    return normalizeProgressText(payload.thought_summary || payload.expected_outcome || payload.message || "");
  }

  const args = payload.args || payload.tool_args || {};
  const path = extractPathFromPayload(payload);
  const command = extractCommandFromPayload(payload);
  const query = typeof args.query === "string" ? args.query.trim() : "";
  const focus = typeof args.focus === "string" ? args.focus.trim() : "";
  const branchName = typeof args.name === "string" ? args.name.trim() : "";
  const genericSummary = normalizeProgressText(
    options.fromDecision
      ? payload.expected_outcome || payload.thought_summary || payload.message || ""
      : payload.thought_summary || payload.expected_outcome || payload.message || "",
  );

  if (toolName === "create_file") {
    return path
      ? `Neue Datei: ${shortenPath(path, 56)}`
      : "Neue Datei wird angelegt";
  }
  if (["write_file", "append_file", "replace_in_file", "patch_file"].includes(toolName)) {
    return path
      ? `Datei wird angepasst: ${shortenPath(path, 56)}`
      : "Datei wird angepasst";
  }
  if (toolName === "delete_file") {
    return path
      ? `Datei wird entfernt: ${shortenPath(path, 56)}`
      : "Datei wird entfernt";
  }
  if (toolName === "read_file") {
    return path
      ? `Datei wird gelesen: ${shortenPath(path, 56)}`
      : "Relevante Dateien werden gelesen";
  }
  if (toolName === "search_in_files") {
    return query
      ? `Projekt durchsuchen nach "${shorten(query, 32)}"`
      : "Projekt wird nach der passenden Stelle durchsucht";
  }
  if (toolName === "list_files") {
    return path && path !== "."
      ? `Projektstruktur wird geprueft: ${shortenPath(path, 56)}`
      : "Projektstruktur wird geprueft";
  }
  if (toolName === "inspect_workspace") {
    return focus
      ? `Projektkontext wird gesammelt: ${shorten(focus, 40)}`
      : "Projektkontext wird gesammelt";
  }
  if (toolName === "run_tests") {
    return command
      ? `Check laeuft: ${shorten(command, 56)}`
      : "Checks werden ausgefuehrt";
  }
  if (toolName === "run_shell") {
    return command
      ? `Befehl laeuft: ${shorten(command, 56)}`
      : "Befehl wird ausgefuehrt";
  }
  if (toolName === "show_diff" || toolName === "git_diff") {
    return "Diff wird geprueft";
  }
  if (toolName === "git_log") {
    return "Commit-Verlauf wird geprueft";
  }
  if (toolName === "git_create_branch") {
    return branchName
      ? `Branch wird erstellt: ${shorten(branchName, 32)}`
      : "Neuer Branch wird erstellt";
  }

  return genericSummary || "Naechster Schritt wird ausgefuehrt";
}

function describeStreamingModelProgress(kind, payload) {
  const progressType = String(payload?.type || "").trim();
  const path = extractPathFromPayload(payload || {});
  const tier = capabilityTierLabel(payload?.capability_tier);

  if (progressType === "status") {
    const stage = String(payload?.stage || "").trim();
    if (stage === "request_started") {
      if (kind === "content") {
        return path
          ? `Modellstart fuer ${shortenPath(path, 56)}${tier ? ` (${tier})` : ""}`
          : `Modellstart fuer den Dateiinhalt${tier ? ` (${tier})` : ""}`;
      }
      return `Modellstart fuer die Antwort${tier ? ` (${tier})` : ""}`;
    }
    if (stage === "waiting_for_first_chunk") {
      if (kind === "content") {
        return path
          ? `Wartet auf ersten Chunk fuer ${shortenPath(path, 56)}${tier ? ` (${tier})` : ""}`
          : `Wartet auf den ersten Chunk${tier ? ` (${tier})` : ""}`;
      }
      return `Wartet auf den ersten Antwort-Chunk${tier ? ` (${tier})` : ""}`;
    }
    if (stage === "startup_timeout_warning") {
      if (kind === "content") {
        return path
          ? `Wartet auf ersten Output fuer ${shortenPath(path, 56)}${tier ? ` (${tier})` : ""}`
          : `Wartet auf ersten Inhalt${tier ? ` (${tier})` : ""}`;
      }
      return `Wartet auf ersten Antwort-Output${tier ? ` (${tier})` : ""}`;
    }
  }

  if (progressType === "heartbeat") {
    if (payload?.phase === "waiting_for_start") {
      if (kind === "content") {
        return path
          ? `Modell startet noch fuer ${shortenPath(path, 56)}${tier ? ` (${tier})` : ""}`
          : `Modell startet noch${tier ? ` (${tier})` : ""}`;
      }
      return `Modell startet noch fuer die Antwort${tier ? ` (${tier})` : ""}`;
    }
    if (kind === "content") {
      return path
        ? `Inhalt wird weiter generiert fuer ${shortenPath(path, 56)}${tier ? ` (${tier})` : ""}`
        : `Dateiinhalt wird weiter generiert${tier ? ` (${tier})` : ""}`;
    }
    return `Antworttext wird weiter generiert${tier ? ` (${tier})` : ""}`;
  }

  if (progressType === "chunk") {
    if (kind === "content") {
      return path
        ? `Streamt weiter fuer ${shortenPath(path, 56)}${tier ? ` (${tier})` : ""}`
        : `Inhalt streamt weiter${tier ? ` (${tier})` : ""}`;
    }
    return `Antwort streamt weiter${tier ? ` (${tier})` : ""}`;
  }

  return "";
}

function describeContentGenerationRetry(path, payload) {
  const strategy = String(payload?.strategy || "").trim();
  const tier = capabilityTierLabel(payload?.capability_tier);

  if (strategy === "resume_same_model") {
    return path
      ? `Wird mit vorhandenem Fortschritt fortgesetzt: ${shortenPath(path, 56)}${tier ? ` (${tier})` : ""}`
      : `Wird mit vorhandenem Fortschritt fortgesetzt${tier ? ` (${tier})` : ""}`;
  }
  if (strategy === "resume_fallback_model") {
    return path
      ? `Fortsetzung mit kleinerem Modell: ${shortenPath(path, 56)}${tier ? ` (${tier})` : ""}`
      : `Fortsetzung mit kleinerem Modell${tier ? ` (${tier})` : ""}`;
  }
  if (strategy === "fallback_model") {
    return path
      ? `Retry mit kleinerem Modell: ${shortenPath(path, 56)}${tier ? ` (${tier})` : ""}`
      : `Retry mit kleinerem Modell${tier ? ` (${tier})` : ""}`;
  }
  if (strategy === "compact_fallback_model") {
    return path
      ? `Kompakter Retry: ${shortenPath(path, 56)}${tier ? ` (${tier})` : ""}`
      : `Kompakter Retry mit weniger Kontext${tier ? ` (${tier})` : ""}`;
  }
  if (strategy === "starter_scaffold") {
    return path
      ? `Lokales Starter-Grundgeruest: ${shortenPath(path, 56)}${tier ? ` (${tier})` : ""}`
      : `Lokales Starter-Grundgeruest${tier ? ` (${tier})` : ""}`;
  }
  if (strategy === "deterministic_template") {
    return path
      ? `Deterministische Vorlage: ${shortenPath(path, 56)}${tier ? ` (${tier})` : ""}`
      : `Deterministische Vorlage${tier ? ` (${tier})` : ""}`;
  }

  return path
    ? `Retry mit reduziertem Kontext: ${shortenPath(path, 56)}${tier ? ` (${tier})` : ""}`
    : `Retry mit reduziertem Kontext${tier ? ` (${tier})` : ""}`;
}

function describeContentGenerationRetryError(path, payload) {
  const strategy = String(payload?.strategy || "").trim();
  const reason = String(payload?.reason || "").trim();
  const hadProgress = Boolean(payload?.had_progress);

  if (reason === "startup_timeout" && !hadProgress) {
    if (strategy === "fallback_model") {
      return path
        ? `Auch das Fallback-Modell startet nicht sauber fuer ${shortenPath(path, 56)}`
        : "Auch das Fallback-Modell startet nicht sauber";
    }
    return path
      ? `Modellstart weiterhin blockiert fuer ${shortenPath(path, 56)}`
      : "Modellstart weiterhin blockiert";
  }

  return "";
}

function describeContentGenerationRecovery(path, payload) {
  const strategy = String(payload?.strategy || "").trim();

  if (strategy === "starter_scaffold") {
    return path
      ? `Recovery ueber lokales Starter-Grundgeruest fuer ${shortenPath(path, 56)}`
      : "Recovery ueber lokales Starter-Grundgeruest";
  }
  if (strategy === "deterministic_template") {
    return path
      ? `Recovery ueber lokale Vorlage fuer ${shortenPath(path, 56)}`
      : "Recovery ueber lokale Vorlage";
  }

  return path
    ? `Recovery gestartet fuer ${shortenPath(path, 56)}`
    : "Recovery gestartet";
}

function describeContentGenerationRecoveryFinished(path, payload) {
  const strategy = String(payload?.strategy || "").trim();

  if (strategy === "starter_scaffold") {
    return path
      ? `Starter-Recovery abgeschlossen fuer ${shortenPath(path, 56)}`
      : "Starter-Recovery abgeschlossen";
  }
  if (strategy === "deterministic_template") {
    return path
      ? `Vorlagen-Recovery abgeschlossen fuer ${shortenPath(path, 56)}`
      : "Vorlagen-Recovery abgeschlossen";
  }

  return "";
}

function describeFinalBlock(payload) {
  const stopReason = String(payload?.stop_reason || "").trim();

  if (stopReason === "model_start_failed") {
    return "Blockiert am Modellstart";
  }
  if (stopReason === "generation_failed_after_progress") {
    return "Blockiert nach Teilfortschritt";
  }
  if (stopReason === "repair_generation_failed") {
    return "Blockiert im Repair-Generierungspfad";
  }

  return "";
}

function extractPathFromPayload(payload) {
  const args = payload.args || payload.tool_args || {};
  if (typeof args.path === "string" && args.path.trim()) {
    return args.path.trim();
  }
  if (typeof payload.path === "string" && payload.path.trim()) {
    return payload.path.trim();
  }
  return "";
}

function normalizeProgressText(text) {
  const value = String(text || "")
    .replace(/\s+/g, " ")
    .trim();
  if (!value) {
    return "";
  }
  const shortened = shorten(value, 110);
  if (/[.!?]$/.test(shortened)) {
    return shortened;
  }
  return `${shortened}.`;
}

function capabilityTierLabel(value) {
  const normalized = String(value || "").trim().toLowerCase();
  if (!normalized) {
    return "";
  }
  if (normalized === "tier_a") return "Tier A";
  if (normalized === "tier_b") return "Tier B";
  if (normalized === "tier_c") return "Tier C";
  if (normalized === "tier_d") return "Tier D";
  if (normalized === "tier_e") return "Tier E";
  return "";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("\n", "&#10;");
}

function icon(name) {
  const icons = {
    plus:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M10 4v12M4 10h12" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/></svg>',
    folder:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M2.5 5.5h4.2l1.6 1.8h9.2v7.2a2 2 0 0 1-2 2H4.5a2 2 0 0 1-2-2z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.6"/></svg>',
    edit:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M4 13.8V16h2.2l7.2-7.2-2.2-2.2zM12.5 4.7l2.2 2.2 1-1a1.6 1.6 0 0 0-2.2-2.2z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.5"/></svg>',
    compose:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M3.5 14.5V16.5h2l8.9-8.9-2-2zM12.6 4.6l2 2 1-1a1.4 1.4 0 1 0-2-2z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.5"/></svg>',
    arrow:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M4 10h10M10.5 4.5 16 10l-5.5 5.5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"/></svg>',
    play:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M7 5.5v9l7-4.5z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.6"/></svg>',
    stop:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><rect x="5.5" y="5.5" width="9" height="9" rx="1.5" fill="currentColor"/></svg>',
    close:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/></svg>',
    trash:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M5.5 6.5h9M8 6.5V5a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v1.5M7 8.5v6M10 8.5v6M13 8.5v6M6.5 6.5l.6 9a1.8 1.8 0 0 0 1.8 1.5h2.2a1.8 1.8 0 0 0 1.8-1.5l.6-9" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"/></svg>',
    broom:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M11.8 3.7 16.3 8.2M10.2 5.3 14.7 9.8M8.7 6.8l4.5 4.5M5.7 9.8l4.5 4.5M4.4 11.1c2.4-.1 4.5.8 6.2 2.6l1.6 1.6c.4.4.4 1 0 1.4l-.6.6c-.4.4-1 .4-1.4 0l-1.6-1.6c-1.8-1.8-2.7-3.9-2.6-6.2z" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.4"/></svg>',
    "git-push":
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M6 14.5h8a2 2 0 0 0 2-2V9.5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6"/><path d="M10 12.5V4.5M6.8 7.7 10 4.5l3.2 3.2" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6"/><circle cx="6" cy="14.5" r="1.5" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="14" cy="14.5" r="1.5" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>',
    download:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M10 4.5v7.5M6.8 9.8 10 13l3.2-3.2M4.5 14.5h11" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6"/></svg>',
    sliders:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M4 5.5h12M4 10h12M4 14.5h12" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.6"/><circle cx="7" cy="5.5" r="1.6" fill="currentColor"/><circle cx="12.5" cy="10" r="1.6" fill="currentColor"/><circle cx="9" cy="14.5" r="1.6" fill="currentColor"/></svg>',
    "chevron-right":
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M7.5 5.5 12 10l-4.5 4.5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"/></svg>',
    chat:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M4 4.5h12v8H7.5L4 15z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.6"/></svg>',
    search:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><circle cx="8.5" cy="8.5" r="4.8" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="m12.2 12.2 3.6 3.6" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.6"/></svg>',
    spark:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M10 3.5 11.8 8.2 16.5 10l-4.7 1.8L10 16.5l-1.8-4.7L3.5 10l4.7-1.8z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.5"/></svg>',
    terminal:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M3.5 5.5h13v9h-13z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.5"/><path d="m6 8 2 2-2 2M9.8 12h4.2" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"/></svg>',
  };
  return icons[name] || icons.spark;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    buildActivityClusters,
    buildReferenceHeroView,
    buildRuntimeStatusItems,
    buildThreadPresentationView,
    buildUiRoute,
    buildWorkspaceShellView,
    buildPhaseSteps,
    buildSessionOverview,
    buildValidationSnapshot,
    createRefreshController,
    describeLogRecord,
    messageDisplayState,
    parseUiRoute,
    shouldStartRefresh,
    updateRefreshBackoff,
    formatSessionElapsed,
    phaseStepKey,
    renderRichText,
    sanitizeAssistantMessageContent,
    sessionBadgeText,
    sessionStatusTone,
  };
}
