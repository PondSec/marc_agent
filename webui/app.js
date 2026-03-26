const STORAGE_KEY = "marc_a1.workspace_ui.v3";
const COMMIT_AND_PUSH_PROMPT =
  "Bitte pruefe den aktuellen Git-Status in diesem Workspace, erstelle einen kleinen sinnvollen Commit mit einer kurzen passenden Message und pushe den aktuellen Branch zu origin. Wenn es nichts zu committen gibt oder der Push scheitert, erklaere kurz den Grund im Chat.";

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
  composer: {
    prompt: "",
    agentProfile: "core",
    accessMode: "approval",
    modelName: "",
    executionProfile: "balanced",
    dryRun: false,
  },
  ui: {
    booting: true,
    sessionLoading: false,
    workspaceModalOpen: false,
    settingsModalOpen: false,
    workspaceMode: "create",
    editingWorkspaceId: null,
    workspaceName: "",
    workspacePath: "",
    toast: null,
    toastTimer: null,
  },
};

document.addEventListener("DOMContentLoaded", () => {
  initializeApp().catch((error) => {
    console.error(error);
    showToast(`Initialisierung fehlgeschlagen: ${error.message}`, "error");
  });
});

async function initializeApp() {
  hydrateRouteFromLocation();
  bindEvents();
  renderApp();
  await boot();
}

function bindEvents() {
  document.addEventListener("click", handleClick);
  document.addEventListener("input", handleInput);
  document.addEventListener("change", handleChange);
  document.addEventListener("keydown", handleKeydown);
  window.addEventListener("popstate", handlePopState);
}

async function boot() {
  try {
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
  } finally {
    state.ui.booting = false;
    renderApp();
    window.setInterval(() => refreshSessions(), 8000);
    window.setInterval(() => refreshModelCatalog({ silent: true }), 4000);
  }
}

async function refreshHealth() {
  try {
    state.health = await fetchJSON("/api/health");
  } catch (error) {
    state.health = { ok: false, active_sessions: [] };
  }
}

async function refreshWorkspaces() {
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

async function refreshSessions() {
  const previousActiveSessionId = state.activeSessionId;
  const nextSessions = await fetchJSON("/api/sessions");
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
}

async function refreshModelCatalog({ silent = false } = {}) {
  try {
    const catalog = await fetchJSON("/api/models");
    const changed = applyModelCatalog(catalog);
    if (changed) {
      renderApp();
    }
  } catch (error) {
    if (!silent) {
      showToast(`Modelle konnten nicht geladen werden: ${error.message}`, "error");
    }
  }
}

async function ensureRecommendedModels({ silent = false } = {}) {
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
  if (!sessionId) {
    return;
  }

  disconnectStream();
  state.ui.sessionLoading = true;
  state.activeSessionId = sessionId;
  renderApp();

  try {
    const [session, logs] = await Promise.all([
      fetchJSON(`/api/sessions/${sessionId}`),
      fetchJSON(`/api/sessions/${sessionId}/logs`),
    ]);
    state.activeSession = session;
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
    state.activeSessionId = null;
    state.activeSession = null;
    disconnectStream();
    syncHistory(null);
  }
  persistPreferences();
  renderApp();
}

function startNewChat(workspaceId) {
  state.selectedWorkspaceId = workspaceId || state.selectedWorkspaceId;
  state.activeSessionId = null;
  state.activeSession = null;
  state.logs = [];
  state.composer.prompt = "";
  disconnectStream();
  syncHistory(null);
  persistPreferences();
  renderApp();
  focusComposer();
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
    showToast("Lege zuerst einen Workspace an oder waehle einen aus.", "error");
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
    renderApp();
    refreshSessions();
  });

  stream.addEventListener("log", (event) => {
    const record = JSON.parse(event.data);
    appendLogRecord(record);
    renderApp();
  });

  stream.addEventListener("done", () => {
    disconnectStream();
    refreshSessions();
    renderApp();
  });

  stream.onerror = () => {
    disconnectStream();
    renderApp();
  };
}

function disconnectStream() {
  if (state.stream) {
    state.stream.close();
    state.stream = null;
  }
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

  if (action === "open-settings-modal") {
    openSettingsModal();
    return;
  }

  if (action === "edit-workspace") {
    openWorkspaceModal("edit", target.dataset.workspaceId);
    return;
  }

  if (action === "close-workspace-modal") {
    closeWorkspaceModal();
    return;
  }

  if (action === "close-settings-modal") {
    closeSettingsModal();
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
  }
}

function handleInput(event) {
  if (event.target.id === "composerInput") {
    state.composer.prompt = event.target.value;
    autosizeTextarea(event.target);
    return;
  }
  if (event.target.id === "workspaceNameInput") {
    state.ui.workspaceName = event.target.value;
    return;
  }
  if (event.target.id === "workspacePathInput") {
    state.ui.workspacePath = event.target.value;
  }
}

function handleChange(event) {
  if (event.target.id === "agentProfileSelect") {
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
  }
  if (event.key === "Escape") {
    if (state.ui.workspaceModalOpen) {
      closeWorkspaceModal();
      return;
    }
    if (state.ui.settingsModalOpen) {
      closeSettingsModal();
    }
  }
}

function handlePopState() {
  hydrateRouteFromLocation();
  if (state.activeSessionId) {
    openSession(state.activeSessionId, { updateHistory: false });
  } else {
    state.activeSession = null;
    state.logs = [];
    disconnectStream();
    renderApp();
  }
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

function openSettingsModal() {
  state.ui.settingsModalOpen = true;
  renderApp();
  refreshModelCatalog({ silent: true });
}

function closeWorkspaceModal() {
  state.ui.workspaceModalOpen = false;
  state.ui.workspaceMode = "create";
  state.ui.editingWorkspaceId = null;
  renderApp();
}

function closeSettingsModal() {
  state.ui.settingsModalOpen = false;
  renderApp();
}

async function saveWorkspace() {
  const name = state.ui.workspaceName.trim();
  const path = state.ui.workspacePath.trim();
  if (!name || !path) {
    showToast("Name und Pfad werden beide benoetigt.", "error");
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
    showToast("Workspace gespeichert.", "success");
  } catch (error) {
    showToast(`Workspace konnte nicht gespeichert werden: ${error.message}`, "error");
  }
}

function renderApp() {
  const root = document.getElementById("app");
  if (!root) {
    return;
  }
  const uiSnapshot = captureUiSnapshot();
  root.innerHTML = `
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
    ${renderSettingsModal()}
    ${renderToast()}
  `;
  syncComposerControls();
  restoreUiSnapshot(uiSnapshot);
}

function renderSidebar() {
  const workspace = selectedWorkspace();
  const threadCount = workspace ? sessionsForWorkspace(workspace.id).length : 0;

  return `
    <div class="sidebar-inner">
      <section class="sidebar-section">
        <div class="sidebar-section-head">
          <p class="sidebar-label">Workspaces</p>
          <button
            class="icon-button"
            type="button"
            data-action="open-workspace-modal"
            aria-label="Workspace anlegen"
          >
            ${icon("plus")}
          </button>
        </div>
        ${renderWorkspaceList()}
      </section>
      <section class="sidebar-section">
        <div class="sidebar-section-head">
          <p class="sidebar-label">Threads</p>
          <span class="sidebar-count">${threadCount}</span>
        </div>
        ${renderThreadList()}
      </section>
    </div>
  `;
}

function renderWorkspaceList() {
  if (!state.workspaces.length) {
    return `
      <div class="sidebar-empty">
        Noch kein Workspace vorhanden. Lege zuerst links einen Projektordner an.
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
  const count = sessionsForWorkspace(workspace.id).length;

  return `
    <div class="workspace-item ${active ? "active" : ""}">
      <button
        class="workspace-button"
        type="button"
        data-action="select-workspace"
        data-workspace-id="${escapeHtml(workspace.id)}"
      >
        <span class="workspace-name">${escapeHtml(workspace.name)}</span>
        <span class="workspace-badge">${count}</span>
      </button>
      <button
        class="workspace-edit"
        type="button"
        data-action="edit-workspace"
        data-workspace-id="${escapeHtml(workspace.id)}"
        aria-label="Workspace bearbeiten"
      >
        ${icon("edit")}
      </button>
    </div>
  `;
}

function renderThreadList() {
  const workspace = selectedWorkspace();
  if (!workspace) {
    return `
      <div class="sidebar-empty">
        Waehle einen Workspace aus, damit hier die Threads erscheinen.
      </div>
    `;
  }

  const sessions = sessionsForWorkspace(workspace.id);
  if (!sessions.length) {
    return `
      <div class="sidebar-empty">
        Noch keine Threads in ${escapeHtml(workspace.name)}. Starte oben einen neuen Chat.
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

  return `
    <button
      class="thread-item ${active ? "active" : ""}"
      type="button"
      data-action="open-session"
      data-session-id="${escapeHtml(session.id)}"
    >
      <span class="thread-title">${escapeHtml(shorten(title, 34))}</span>
    </button>
  `;
}

function renderAgentProfileOptions() {
  const profiles = getAgentProfiles();
  if (!profiles.length) {
    return `<option value="${escapeHtml(state.composer.agentProfile || "core")}" selected>${escapeHtml(state.composer.agentProfile || "core")}</option>`;
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

function renderTopBar() {
  const workspace = workspaceForSession(state.activeSession) || selectedWorkspace();
  const commitDisabled = !workspace || isSessionRunning(state.activeSession);

  return `
    <header class="top-bar">
      <div class="top-bar-inner">
        <h1 class="workspace-title">${escapeHtml(workspace?.name || "Kein Workspace")}</h1>
        <div class="top-bar-actions">
          <button
            class="button-secondary"
            type="button"
            data-action="${workspace ? "new-chat" : "open-workspace-modal"}"
            ${workspace ? `data-workspace-id="${escapeHtml(workspace.id)}"` : ""}
          >
            Neuer Chat
          </button>
          <button
            class="icon-button top-bar-icon"
            type="button"
            data-action="commit-push"
            aria-label="Commit und Push an den Agenten senden"
            title="${escapeAttribute(
              commitDisabled
                ? "Commit und Push ist erst verfuegbar, wenn kein Agent-Schritt mehr laeuft."
                : "Commit und Push anfordern",
            )}"
            ${commitDisabled ? "disabled" : ""}
          >
            ${icon("git-push")}
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
        <div class="chat-container">
          <div class="message-list">
            ${renderChatStateMessages()}
          </div>
        </div>
      </div>
    </section>
  `;
}

function renderChatStateMessages() {
  if (state.ui.booting) {
    return renderInlineNote("Oberflaeche wird geladen");
  }

  if (state.ui.sessionLoading) {
    return renderInlineNote("Thread wird geladen");
  }

  if (!state.activeSession) {
    return "";
  }

  const entries = conversationTimeline(state.activeSession, state.logs);
  return [
    entries.map(renderTimelineEntry).join(""),
    ["queued", "running"].includes(state.activeSession.status) ? renderRunningMessage() : "",
  ].join("");
}

function renderInlineNote(text) {
  return `<div class="inline-note">${escapeHtml(text)}</div>`;
}

function renderMessageBubble(message) {
  return `
    <div class="message-row ${escapeHtml(message.role)}">
      <article class="message-bubble ${escapeHtml(message.role)}">
        <div class="message-head">
          <span class="message-author">${escapeHtml(roleLabel(message.role))}</span>
          <span class="message-time">${escapeHtml(formatTime(message.created_at))}</span>
        </div>
        <div class="message-body">${renderRichText(message.content)}</div>
      </article>
    </div>
  `;
}

function renderTimelineEntry(entry) {
  if (entry.type === "activity") {
    return renderActivityLine(entry.record);
  }
  return renderMessageBubble(entry.message);
}

function renderActivityLine(record) {
  const detail = describeLogRecord(record);
  if (!detail) {
    return "";
  }

  return `
    <div class="message-row activity">
      <div class="activity-line ${escapeHtml(detail.tone || "neutral")}">
        <div class="activity-main">
          <span class="activity-marker" aria-hidden="true"></span>
          <span class="activity-text">${escapeHtml(detail.text)}</span>
          ${detail.meta ? `<code class="activity-code">${escapeHtml(detail.meta)}</code>` : ""}
        </div>
        ${record.timestamp ? `
          <span class="activity-time">${escapeHtml(formatTime(record.timestamp))}</span>
        ` : ""}
      </div>
    </div>
  `;
}

function renderRunningMessage() {
  const stopping = Boolean(state.activeSession?.stop_requested);
  return `
    <div class="message-row activity">
      <div class="activity-line live ${stopping ? "danger" : "neutral"}">
        <div class="activity-main">
          <span class="activity-marker" aria-hidden="true"></span>
          <span class="activity-text">
            ${escapeHtml(stopping ? "Stop-Anfrage gesendet. Der Agent beendet den aktuellen Schritt." : "Agent arbeitet gerade")}
          </span>
          ${stopping ? "" : `<div class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></div>`}
        </div>
      </div>
    </div>
  `;
}

function renderChatInput() {
  const workspace = workspaceForSession(state.activeSession) || selectedWorkspace();
  const chatHint = composerHint(workspace);
  const thought = currentThought();
  const running = isSessionRunning(state.activeSession);
  const modelInstallNotice = currentModelInstallNotice();

  return `
    <footer class="chat-input-shell">
      <div class="chat-input-inner">
        <div class="chat-input-container">
          <div class="composer-toolbar">
            <span class="chat-hint">${escapeHtml(chatHint)}</span>
            <button class="button-ghost" type="button" data-action="open-settings-modal">
              Optionen
            </button>
          </div>
          ${modelInstallNotice ? renderModelInstallStrip(modelInstallNotice) : ""}
          ${thought ? renderThoughtStrip(thought) : ""}
          <div class="chat-input-row">
            <textarea
              id="composerInput"
              class="chat-input"
              rows="3"
              placeholder="${escapeAttribute(composerPlaceholder(workspace))}"
            ></textarea>
            <button
              class="send-button ${running ? "stop" : "send"}"
              type="button"
              data-action="${running ? "stop-session" : "submit-prompt"}"
              aria-label="${running ? "Stoppen" : "Senden"}"
            >
              ${icon(running ? "stop" : "arrow")}
            </button>
          </div>
        </div>
      </div>
    </footer>
  `;
}

function renderModelInstallStrip(notice) {
  return `
    <div class="model-install-strip">
      <span class="thought-label">Modelle</span>
      <span class="thought-text">${escapeHtml(notice)}</span>
    </div>
  `;
}

function renderThoughtStrip(thought) {
  return `
    <div class="thought-strip">
      <span class="thought-label">Arbeitet gerade</span>
      <span class="thought-text">${escapeHtml(thought)}</span>
    </div>
  `;
}

function renderSettingsModal() {
  if (!state.ui.settingsModalOpen) {
    return "";
  }

  return `
    <div class="modal-backdrop" data-action="close-settings-modal"></div>
    <div class="modal-layer">
      <section class="modal-card">
        <header class="modal-head">
          <div>
            <p class="modal-kicker">Chat-Optionen</p>
            <h3>Einstellungen</h3>
          </div>
          <button class="icon-button" type="button" data-action="close-settings-modal" aria-label="Schliessen">
            ${icon("close")}
          </button>
        </header>
        <div class="modal-body settings-form">
          <label class="modal-field">
            <span>Agent</span>
            <select id="agentProfileSelect">
              ${renderAgentProfileOptions()}
            </select>
          </label>
          <label class="modal-field">
            <span>Access Mode</span>
            <select id="accessModeSelect">
              ${renderAccessModeOptions()}
            </select>
          </label>
          <label class="modal-field">
            <span>Model</span>
            <select id="modelNameSelect">
              ${renderModelOptions()}
            </select>
            <small>${escapeHtml(modelFieldHelperText())}</small>
          </label>
          <label class="modal-field">
            <span>Execution</span>
            <select id="executionProfileSelect">
              ${renderExecutionProfileOptions()}
            </select>
          </label>
          <label class="settings-toggle">
            <span>
              <strong>Dry Run</strong>
              <small>Werkzeuge nur simulieren</small>
            </span>
            <input id="dryRunToggle" type="checkbox"${state.composer.dryRun ? " checked" : ""} />
          </label>
          <section class="model-panel">
            <div class="model-panel-head">
              <div>
                <strong>Empfohlene Modelle</strong>
                <small>Die App haelt die drei staerksten lokalen Modelle fuer Coding-Aufgaben bereit.</small>
              </div>
              <button class="button-ghost model-panel-action" type="button" data-action="ensure-models">
                Jetzt pruefen
              </button>
            </div>
            <div class="model-list">
              ${renderRecommendedModels()}
            </div>
          </section>
        </div>
      </section>
    </div>
  `;
}

function renderWorkspaceModal() {
  if (!state.ui.workspaceModalOpen) {
    return "";
  }

  const edit = state.ui.workspaceMode === "edit";
  return `
    <div class="modal-backdrop" data-action="close-workspace-modal"></div>
    <div class="modal-layer">
      <section class="modal-card">
        <header class="modal-head">
          <div>
            <p class="modal-kicker">Workspace</p>
            <h3>${edit ? "Workspace bearbeiten" : "Workspace anlegen"}</h3>
          </div>
          <button class="icon-button" type="button" data-action="close-workspace-modal" aria-label="Schliessen">
            ${icon("close")}
          </button>
        </header>
        <div class="modal-body">
          <label class="modal-field">
            <span>Name</span>
            <input id="workspaceNameInput" type="text" placeholder="z. B. agent_ai" value="${escapeAttribute(state.ui.workspaceName)}" />
          </label>
          <label class="modal-field">
            <span>Ordnerpfad</span>
            <input id="workspacePathInput" type="text" placeholder="/Users/.../projekt" value="${escapeAttribute(state.ui.workspacePath)}" />
          </label>
          <p class="modal-note">Der Workspace-Pfad wird in dieser Browser-Oberflaeche direkt als lokaler Ordnerpfad eingetragen.</p>
        </div>
        <footer class="modal-actions">
          <button class="button-secondary" type="button" data-action="close-workspace-modal">Abbrechen</button>
          <button class="button-primary" type="button" data-action="save-workspace">${edit ? "Speichern" : "Anlegen"}</button>
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

function syncComposerControls() {
  syncTextArea("composerInput", state.composer.prompt);
}

function captureUiSnapshot() {
  const activeElement = document.activeElement;
  const snapshot = {
    activeId: activeElement?.id || null,
    selectionStart: null,
    selectionEnd: null,
  };

  if (activeElement && typeof activeElement.selectionStart === "number") {
    snapshot.selectionStart = activeElement.selectionStart;
    snapshot.selectionEnd = activeElement.selectionEnd;
  }

  return snapshot;
}

function restoreUiSnapshot(snapshot) {
  if (!snapshot?.activeId) {
    return;
  }

  const target = document.getElementById(snapshot.activeId);
  if (!target) {
    return;
  }

  if (typeof target.focus === "function") {
    target.focus();
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

function conversationTimeline(session, logs) {
  const messageEntries = conversationMessages(session).map((message) => ({
    type: "message",
    timestamp: message.created_at,
    message,
  }));
  const activityEntries = (logs || [])
    .filter((record) => Boolean(describeLogRecord(record)))
    .map((record) => ({
      type: "activity",
      timestamp: record.timestamp,
      record,
    }));

  return [...messageEntries, ...activityEntries].sort((left, right) =>
    String(left.timestamp || "").localeCompare(String(right.timestamp || "")),
  );
}

function describeLogRecord(record) {
  if (!record || !record.event) {
    return null;
  }

  const payload = record.payload || {};
  const toolName = payload.tool || payload.tool_name || null;
  const message = String(payload.message || payload.error || "").trim();

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
    };
  }

  if (record.event === "tool_error" || record.event === "tool_execution_error" || record.event === "tool_validation_error") {
    return {
      text: `Fehler in ${humanizeToolName(toolName || "tool")}`,
      meta: message || null,
      tone: "danger",
    };
  }

  if (record.event === "task_stop_requested") {
    return {
      text: "Stop angefordert",
      meta: "Der Agent beendet den aktuellen Schritt und stoppt danach.",
      tone: "neutral",
    };
  }

  if (record.event === "task_stopped") {
    return {
      text: "Agent gestoppt",
      meta: "Die laufende Session wurde angehalten.",
      tone: "neutral",
    };
  }

  if (record.event === "task_crashed") {
    return {
      text: "Agent abgestuerzt",
      meta: message || "Unbekannter Laufzeitfehler",
      tone: "danger",
    };
  }

  if (record.event === "task_finished" && payload.status !== "completed") {
    return {
      text: `Task beendet: ${labelForStatus(payload.status || "")}`,
      meta: String(payload.stop_reason || payload.workflow_stage || "").trim() || null,
      tone: payload.status === "failed" ? "danger" : "neutral",
    };
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
      text: "Analysiert den Workspace",
      meta: focus ? shorten(focus, 52) : null,
      tone: "neutral",
    };
  }

  if (toolName === "search_in_files") {
    return {
      text: "Sucht im Projekt",
      meta: query || null,
      tone: "neutral",
    };
  }

  if (toolName === "read_file") {
    return {
      text: "Liest Datei",
      meta: path ? shortenPath(path, 56) : null,
      tone: "neutral",
    };
  }

  if (toolName === "list_files") {
    return {
      text: "Prueft die Projektstruktur",
      meta: path && path !== "." ? shortenPath(path, 56) : null,
      tone: "neutral",
    };
  }

  if (["create_file", "write_file", "append_file", "replace_in_file", "patch_file", "delete_file"].includes(toolName || "")) {
    return {
      text: "Bearbeitet Datei",
      meta: path ? shortenPath(path, 56) : null,
      tone: "neutral",
    };
  }

  if (toolName === "run_shell" || toolName === "run_tests") {
    return {
      text: "Fuehrt Befehl aus",
      meta: command ? shorten(command, 68) : humanizeToolName(toolName),
      tone: "command",
    };
  }

  if (toolName === "git_status") {
    return {
      text: "Prueft den Git-Status",
      meta: null,
      tone: "neutral",
    };
  }

  if (toolName === "git_diff" || toolName === "show_diff") {
    return {
      text: "Prueft Aenderungen im Diff",
      meta: null,
      tone: "neutral",
    };
  }

  if (toolName === "git_log") {
    return {
      text: "Liest die letzten Commits",
      meta: null,
      tone: "neutral",
    };
  }

  return {
    text: `Startet ${humanizeToolName(toolName || "Tool")}`,
    meta: null,
    tone: command ? "command" : "neutral",
  };
}

function describeToolResultLog(toolName, payload) {
  const message = String(payload.message || "").trim();
  const success = Boolean(payload.success);
  const changedFiles = Number(payload.changed_files || 0);

  if (!success) {
    return {
      text: `Fehler in ${humanizeToolName(toolName || "tool")}`,
      meta: message || null,
      tone: "danger",
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
      tone: "neutral",
    };
  }

  if (toolName === "run_shell" || toolName === "run_tests") {
    return {
      text: "Befehl abgeschlossen",
      meta: message ? shorten(message, 80) : null,
      tone: "success",
    };
  }

  if (changedFiles > 0) {
    return {
      text: changedFiles === 1 ? "1 Datei geaendert" : `${changedFiles} Dateien geaendert`,
      meta: message && !/changed|updated|created|deleted/i.test(message) ? shorten(message, 80) : null,
      tone: "success",
    };
  }

  if (message) {
    return {
      text: shorten(message, 88),
      meta: null,
      tone: "success",
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

function selectedWorkspace() {
  return state.workspaces.find((workspace) => workspace.id === state.selectedWorkspaceId) || null;
}

function activeWorkspaceId() {
  return state.activeSession?.workspace_id || state.selectedWorkspaceId;
}

function workspaceForSession(session) {
  if (!session) {
    return null;
  }
  return state.workspaces.find((workspace) => workspace.id === session.workspace_id) || null;
}

function sessionsForWorkspace(workspaceId) {
  return state.sessions
    .filter((session) => session.workspace_id === workspaceId && !session.archived)
    .sort((left, right) => new Date(right.updated_at) - new Date(left.updated_at));
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
    return "Lege zuerst links einen Workspace an und trage den Ordnerpfad ein";
  }
  if (state.activeSession) {
    return `Schreibe in diesem Chat weiter (${workspace.name})`;
  }
  return `Starte einen neuen Chat im Workspace ${workspace.name}`;
}

function composerHint(workspace) {
  if (!workspace) {
    return "Lege zuerst einen Workspace an";
  }
  if (!state.activeSession) {
    return "Starte einen neuen Chat im Workspace";
  }
  if (state.activeSession.stop_requested) {
    return "Stoppe laufende Session...";
  }
  return "Der Agent protokolliert seine Schritte direkt im Chat";
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

function fetchJSON(url, options) {
  return fetch(url, options).then(async (response) => {
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const payload = await response.json();
        detail = payload.detail || JSON.stringify(payload);
      } catch (error) {
        detail = await response.text();
      }
      throw new Error(detail || `HTTP ${response.status}`);
    }
    return response.json();
  });
}

function hydrateRouteFromLocation() {
  const url = new URL(window.location.href);
  state.activeSessionId = url.searchParams.get("session");
}

function syncHistory(sessionId) {
  const url = sessionId ? `/?session=${encodeURIComponent(sessionId)}` : "/";
  window.history.pushState({}, "", url);
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
  if (mode === "safe") return "Safe";
  if (mode === "approval") return "Approval";
  if (mode === "full") return "Full Access";
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
  return escapeHtml(String(text || "")).replaceAll("\n", "<br />");
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

function currentThought() {
  if (!isSessionRunning(state.activeSession)) {
    return "";
  }

  if (state.activeSession?.stop_requested) {
    return "Stop-Anfrage aktiv. Der laufende Schritt wird gerade sauber beendet.";
  }

  if (state.activeSession?.status === "queued" && state.logs.length === 0) {
    return "Ich bereite gerade die Aufgabe vor.";
  }

  for (const record of [...state.logs].reverse()) {
    const thought = describeCurrentStep(record);
    if (thought) {
      return thought;
    }
  }

  const latestToolCall = [...(state.activeSession?.tool_calls || [])]
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

  return "Ich arbeite gerade an deinem Auftrag.";
}

function describeCurrentStep(record) {
  if (!record || !record.event) {
    return "";
  }

  if (record.event === "tool_requested") {
    return describeToolActivity(record.payload || {});
  }

  if (record.event === "decision") {
    return describeDecisionActivity(record.payload || {});
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
    return "Ich formuliere gerade die Antwort.";
  }

  const summary = normalizeProgressText(payload.thought_summary || payload.expected_outcome || "");
  if (summary) {
    return summary;
  }

  return "Ich plane gerade den naechsten Schritt.";
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
      ? `Ich lege gerade ${shortenPath(path, 56)} an.`
      : "Ich lege gerade neue Dateien an.";
  }
  if (["write_file", "append_file", "replace_in_file", "patch_file"].includes(toolName)) {
    return path
      ? `Ich passe gerade ${shortenPath(path, 56)} an.`
      : "Ich passe gerade Dateien an.";
  }
  if (toolName === "delete_file") {
    return path
      ? `Ich entferne gerade ${shortenPath(path, 56)}.`
      : "Ich entferne gerade eine Datei.";
  }
  if (toolName === "read_file") {
    return path
      ? `Ich lese gerade ${shortenPath(path, 56)}.`
      : "Ich lese gerade relevante Dateien.";
  }
  if (toolName === "search_in_files") {
    return query
      ? `Ich suche gerade nach "${shorten(query, 32)}" im Projekt.`
      : "Ich suche gerade die passende Stelle im Projekt.";
  }
  if (toolName === "list_files") {
    return path && path !== "."
      ? `Ich pruefe gerade die Projektstruktur in ${shortenPath(path, 56)}.`
      : "Ich pruefe gerade die Projektstruktur.";
  }
  if (toolName === "inspect_workspace") {
    return focus
      ? `Ich verschaffe mir gerade einen Ueberblick ueber ${shorten(focus, 40)}.`
      : "Ich verschaffe mir gerade einen Ueberblick ueber das Projekt.";
  }
  if (toolName === "run_tests") {
    return command
      ? `Ich pruefe die Aenderungen gerade mit ${shorten(command, 56)}.`
      : "Ich pruefe gerade die Aenderungen mit Tests oder Checks.";
  }
  if (toolName === "run_shell") {
    return command
      ? `Ich fuehre gerade ${shorten(command, 56)} aus.`
      : "Ich fuehre gerade einen Befehl aus.";
  }
  if (toolName === "show_diff" || toolName === "git_diff") {
    return "Ich pruefe gerade die Dateiaenderungen im Diff.";
  }
  if (toolName === "git_log") {
    return "Ich schaue mir gerade die letzten Commits an.";
  }
  if (toolName === "git_create_branch") {
    return branchName
      ? `Ich erstelle gerade den Branch ${shorten(branchName, 32)}.`
      : "Ich erstelle gerade einen neuen Branch.";
  }

  return genericSummary || "Ich arbeite gerade am naechsten Schritt.";
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
    stop:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><rect x="5.5" y="5.5" width="9" height="9" rx="1.5" fill="currentColor"/></svg>',
    close:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M5 5l10 10M15 5 5 15" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"/></svg>',
    "git-push":
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M6 14.5h8a2 2 0 0 0 2-2V9.5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6"/><path d="M10 12.5V4.5M6.8 7.7 10 4.5l3.2 3.2" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.6"/><circle cx="6" cy="14.5" r="1.5" fill="none" stroke="currentColor" stroke-width="1.5"/><circle cx="14" cy="14.5" r="1.5" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>',
    chat:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M4 4.5h12v8H7.5L4 15z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.6"/></svg>',
    search:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><circle cx="8.5" cy="8.5" r="4.8" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="m12.2 12.2 3.6 3.6" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.6"/></svg>',
    spark:
      '<svg viewBox="0 0 20 20" class="icon" aria-hidden="true"><path d="M10 3.5 11.8 8.2 16.5 10l-4.7 1.8L10 16.5l-1.8-4.7L3.5 10l4.7-1.8z" fill="none" stroke="currentColor" stroke-linejoin="round" stroke-width="1.5"/></svg>',
  };
  return icons[name] || icons.spark;
}
