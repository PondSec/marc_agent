const state = {
  sessions: [],
  activeSessionId: null,
  activeSession: null,
  logs: [],
  workspace: null,
  config: null,
  selectedDiffPath: null,
  stream: null,
};

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  boot();
});

function bindEvents() {
  document.getElementById("taskForm").addEventListener("submit", startTask);
  document.getElementById("inspectButton").addEventListener("click", () => inspectWorkspace());
  document.getElementById("refreshSessionsButton").addEventListener("click", refreshSessions);
  document.getElementById("refreshConfigButton").addEventListener("click", refreshConfig);
  document.getElementById("newSessionButton").addEventListener("click", () => {
    state.activeSessionId = null;
    state.activeSession = null;
    state.logs = [];
    state.selectedDiffPath = null;
    disconnectStream();
    renderAll();
  });
}

async function boot() {
  await Promise.all([refreshConfig(), inspectWorkspace(), refreshSessions(), refreshHealth()]);
  window.setInterval(refreshSessions, 5000);
  window.setInterval(refreshHealth, 5000);
}

async function startTask(event) {
  event.preventDefault();
  const prompt = document.getElementById("taskInput").value.trim();
  if (!prompt) {
    return;
  }

  const continueSession = document.getElementById("continueToggle").checked;
  const body = {
    prompt,
    session_id: continueSession ? state.activeSessionId : null,
    access_mode: document.getElementById("accessModeSelect").value,
    dry_run: document.getElementById("dryRunToggle").checked,
    verbose: true,
  };

  try {
    const sessionSummary = await fetchJSON("/api/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    document.getElementById("taskInput").value = "";
    await refreshSessions();
    await loadSession(sessionSummary.id);
  } catch (error) {
    alert(error.message);
  }
}

async function refreshHealth() {
  const health = await fetchJSON("/api/health");
  document.getElementById("serverHealth").textContent = health.ok
    ? `Online - ${health.active_sessions.length} aktiv`
    : "Offline";
}

async function refreshConfig() {
  state.config = await fetchJSON("/api/config");
  document.getElementById("workspaceRoot").textContent = state.config.workspace_root;
  document.getElementById("helperDir").textContent = state.config.helper_dir;
  document.getElementById("configView").textContent = JSON.stringify(state.config, null, 2);

  const branding = state.config.branding || {};
  document.getElementById("brandEyebrow").textContent = branding.name || "M.A.R.C A1";
  document.getElementById("brandTitle").textContent =
    branding.full_name || "Modular Autonomous Runtime Core - Agent 1";
  document.getElementById("brandTagline").textContent =
    branding.tagline || "Lokaler Coding-Agent fuer echte Implementierungsarbeit.";

  const accessModeSelect = document.getElementById("accessModeSelect");
  accessModeSelect.innerHTML = (state.config.access_modes || ["safe", "approval", "full"])
    .map(
      (mode) =>
        `<option value="${escapeHtml(mode)}"${mode === state.config.access_mode ? " selected" : ""}>${escapeHtml(labelForAccessMode(mode))}</option>`,
    )
    .join("");
}

async function inspectWorkspace() {
  const focus = document.getElementById("taskInput").value.trim();
  const query = focus ? `?focus=${encodeURIComponent(focus)}` : "";
  state.workspace = await fetchJSON(`/api/workspace/inspect${query}`);
  renderWorkspace();
}

async function refreshSessions() {
  state.sessions = await fetchJSON("/api/sessions");
  renderSessions();
  if (!state.activeSessionId && state.sessions.length > 0) {
    await loadSession(state.sessions[0].id);
  } else if (state.activeSessionId) {
    const current = state.sessions.find((session) => session.id === state.activeSessionId);
    if (!current && state.sessions.length > 0) {
      await loadSession(state.sessions[0].id);
    }
  }
}

async function loadSession(sessionId) {
  state.activeSessionId = sessionId;
  state.activeSession = await fetchJSON(`/api/sessions/${sessionId}`);
  state.logs = await fetchJSON(`/api/sessions/${sessionId}/logs`);
  if ((state.activeSession.changed_files || []).length > 0) {
    const exists = state.activeSession.changed_files.some(
      (change) => change.path === state.selectedDiffPath,
    );
    if (!exists) {
      state.selectedDiffPath = state.activeSession.changed_files[0].path;
    }
  } else {
    state.selectedDiffPath = null;
  }
  renderAll();
  connectStream(sessionId);
}

function connectStream(sessionId) {
  disconnectStream();
  const stream = new EventSource(`/api/sessions/${sessionId}/events`);
  stream.addEventListener("session", (event) => {
    state.activeSession = JSON.parse(event.data);
    if ((state.activeSession.changed_files || []).length > 0 && !state.selectedDiffPath) {
      state.selectedDiffPath = state.activeSession.changed_files[0].path;
    }
    renderAll();
    refreshSessions();
  });
  stream.addEventListener("log", (event) => {
    state.logs.push(JSON.parse(event.data));
    renderLogs();
  });
  stream.addEventListener("done", () => {
    document.getElementById("liveHint").textContent = "Session abgeschlossen";
    document.getElementById("liveHint").classList.remove("live");
    refreshSessions();
    stream.close();
  });
  stream.onerror = () => {
    document.getElementById("liveHint").textContent = "Live-Stream getrennt";
    document.getElementById("liveHint").classList.remove("live");
    stream.close();
  };
  document.getElementById("liveHint").textContent = "Live verbunden";
  document.getElementById("liveHint").classList.add("live");
  state.stream = stream;
}

function disconnectStream() {
  if (state.stream) {
    state.stream.close();
    state.stream = null;
  }
  document.getElementById("liveHint").textContent = "Keine Live-Verbindung";
  document.getElementById("liveHint").classList.remove("live");
}

function renderAll() {
  renderSessions();
  renderSession();
  renderDiffs();
  renderLogs();
  renderWorkspace();
}

function renderSessions() {
  const container = document.getElementById("sessionList");
  if (!state.sessions.length) {
    container.innerHTML = `<div class="empty-state">Noch keine Sessions gespeichert.</div>`;
    return;
  }

  container.innerHTML = state.sessions
    .map((session) => {
      const isActive = session.id === state.activeSessionId;
      return `
        <article class="session-card ${isActive ? "active" : ""}" data-session-id="${escapeHtml(session.id)}">
          <header>
            <div>
              <h3>${escapeHtml(shorten(session.task, 72))}</h3>
              <p>${escapeHtml(formatDate(session.updated_at))}</p>
            </div>
            <span class="status-chip ${escapeHtml(session.status)}">${escapeHtml(session.status)}</span>
          </header>
          <p>${escapeHtml(session.current_phase)} - ${escapeHtml(session.validation_status)} - ${escapeHtml(labelForAccessMode(session.access_mode))}</p>
          <p>${session.tool_call_count} Tool-Calls | ${session.changed_file_count} Dateiaenderungen</p>
        </article>
      `;
    })
    .join("");

  container.querySelectorAll("[data-session-id]").forEach((card) => {
    card.addEventListener("click", () => {
      loadSession(card.dataset.sessionId);
    });
  });
}

function renderSession() {
  const title = document.getElementById("sessionTitle");
  const badge = document.getElementById("sessionBadge");
  const meta = document.getElementById("sessionMeta");
  const timeline = document.getElementById("timeline");

  if (!state.activeSession) {
    title.textContent = "Noch keine Session ausgewaehlt";
    badge.className = "status-chip neutral";
    badge.textContent = "Idle";
    meta.innerHTML = `<div class="empty-state">Starte einen Task oder waehle eine gespeicherte Session aus.</div>`;
    timeline.innerHTML = `<div class="empty-state">Hier erscheinen Plan, Tool-Aufrufe, Verifikation, Reparaturschleifen und Ergebnisse des Agenten.</div>`;
    return;
  }

  title.textContent = state.activeSession.task;
  badge.className = `status-chip ${state.activeSession.status}`;
  badge.textContent = state.activeSession.status;

  const options = state.activeSession.runtime_options || {};
  const blockers = state.activeSession.blockers || [];
  meta.innerHTML = `
    <div>
      <span class="meta-pill">Session ${escapeHtml(state.activeSession.id)}</span>
      <span class="meta-pill">Phase ${escapeHtml(state.activeSession.current_phase)}</span>
      <span class="meta-pill">Validation ${escapeHtml(state.activeSession.validation_status)}</span>
      <span class="meta-pill">${escapeHtml(labelForAccessMode(state.activeSession.access_mode))}</span>
      <span class="meta-pill">Iterationen ${state.activeSession.iterations}</span>
      <span class="meta-pill">Repair ${state.activeSession.repair_attempts}</span>
      <span class="meta-pill">Dry run ${options.dry_run ? "an" : "aus"}</span>
      <span class="meta-pill">Stop ${escapeHtml(state.activeSession.stop_reason || "-")}</span>
    </div>
    ${
      blockers.length
        ? `<div class="warning-box">${blockers.map((item) => escapeHtml(item)).join("<br />")}</div>`
        : ""
    }
  `;

  const summaryBits = [
    state.activeSession.plan_summary
      ? `<p>${escapeHtml(state.activeSession.plan_summary)}</p>`
      : "",
    state.activeSession.final_response
      ? `<p>${escapeHtml(state.activeSession.final_response)}</p>`
      : `<p>Session laeuft oder wartet auf das naechste Tool-Ergebnis.</p>`,
    renderFlatList("Plan", state.activeSession.plan?.map((item) => `${item.status} - ${item.step}`) || []),
    renderFlatList("Candidate Files", state.activeSession.candidate_files || []),
    renderFlatList("Verification", state.activeSession.verification_commands || []),
    renderFlatList("Completion Criteria", state.activeSession.completion_criteria || []),
    renderFlatList("Helper Artifacts", state.activeSession.helper_artifacts || []),
  ]
    .filter(Boolean)
    .join("");

  const cards = [];
  cards.push(`
    <article class="timeline-card summary-card">
      <header>
        <div>
          <h3>Session Summary</h3>
          <p class="muted-contrast">${escapeHtml(formatDate(state.activeSession.updated_at))}</p>
        </div>
      </header>
      ${summaryBits}
    </article>
  `);

  state.activeSession.tool_calls.forEach((call) => {
    cards.push(`
      <article class="timeline-card tool-card">
        <header>
          <div>
            <h3>${escapeHtml(call.tool_name)}</h3>
            <p>Iteration ${call.iteration} | Phase ${escapeHtml(call.phase || "-")} | ${escapeHtml(call.timestamp)}</p>
          </div>
          <span class="status-chip ${call.success ? "completed" : "failed"}">${call.success ? "ok" : "fail"}</span>
        </header>
        <p>${escapeHtml(call.summary)}</p>
        <p><strong>Ziel:</strong> ${escapeHtml(call.expected_outcome || "-")}</p>
        <p><strong>Gedanke:</strong> ${escapeHtml(call.thought_summary || "-")}</p>
        <pre>${escapeHtml(call.output_excerpt || "(kein Output-Exzerpt)")}</pre>
      </article>
    `);
  });

  if ((state.activeSession.notes || []).length) {
    cards.push(`
      <article class="timeline-card">
        <header>
          <div>
            <h3>Notizen</h3>
          </div>
        </header>
        <ul>${state.activeSession.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>
      </article>
    `);
  }

  timeline.innerHTML = cards.join("");
}

function renderFlatList(title, items) {
  if (!items || !items.length) {
    return "";
  }
  return `
    <div class="summary-block">
      <h4>${escapeHtml(title)}</h4>
      <ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    </div>
  `;
}

function renderDiffs() {
  const list = document.getElementById("diffList");
  const viewer = document.getElementById("diffViewer");
  const changes = state.activeSession?.changed_files || [];

  if (!changes.length) {
    list.innerHTML = `<div class="empty-state">Noch keine gespeicherten Dateiaenderungen fuer diese Session.</div>`;
    viewer.textContent = "Waehle eine Session mit Aenderungen aus.";
    return;
  }

  if (!state.selectedDiffPath) {
    state.selectedDiffPath = changes[0].path;
  }

  list.innerHTML = changes
    .map(
      (change) => `
        <button class="diff-button ${change.path === state.selectedDiffPath ? "active" : ""}" data-diff-path="${escapeHtml(change.path)}" type="button">
          ${escapeHtml(change.path)} | ${escapeHtml(change.operation)}
        </button>
      `,
    )
    .join("");

  list.querySelectorAll("[data-diff-path]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedDiffPath = button.dataset.diffPath;
      renderDiffs();
    });
  });

  const selected = changes.find((change) => change.path === state.selectedDiffPath) || changes[0];
  viewer.textContent = selected.diff || "(kein Diff gespeichert)";
}

function renderLogs() {
  const container = document.getElementById("logList");
  if (!state.logs.length) {
    container.innerHTML = `<div class="empty-state">Noch keine Log-Eintraege vorhanden.</div>`;
    return;
  }

  container.innerHTML = state.logs
    .slice()
    .reverse()
    .map(
      (record) => `
        <article class="log-card">
          <header>
            <div>
              <h3>${escapeHtml(record.event)}</h3>
              <p>${escapeHtml(formatDate(record.timestamp))}</p>
            </div>
          </header>
          <pre>${escapeHtml(JSON.stringify(record.payload, null, 2))}</pre>
        </article>
      `,
    )
    .join("");
}

function renderWorkspace() {
  const container = document.getElementById("workspaceSummary");
  if (!state.workspace) {
    container.innerHTML = `<div class="empty-state">Workspace-Analyse wird geladen...</div>`;
    return;
  }

  const snapshot = state.workspace.snapshot;
  const importantFiles = (snapshot.important_files || [])
    .slice(0, 8)
    .map((path) => `<li>${escapeHtml(path)}</li>`)
    .join("");
  const commands = (snapshot.likely_commands || [])
    .map((command) => `<span class="meta-pill">${escapeHtml(command)}</span>`)
    .join("");

  container.innerHTML = `
    <article class="workspace-card">
      <p>${escapeHtml(state.workspace.text)}</p>
      <div class="summary-block">
        <h4>Priorisierte Dateien</h4>
        <ul>${importantFiles}</ul>
      </div>
      <div class="summary-block">
        <h4>Vermutete Checks</h4>
        <div>${commands || '<span class="meta-pill">Keine Befehle erkannt</span>'}</div>
      </div>
    </article>
  `;
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (error) {
      detail = `${detail}`;
    }
    throw new Error(detail);
  }
  return response.json();
}

function shorten(text, maxLength = 80) {
  return text.length > maxLength ? `${text.slice(0, maxLength - 3)}...` : text;
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function labelForAccessMode(mode) {
  if (mode === "safe") return "Safe";
  if (mode === "approval") return "Approval";
  if (mode === "full") return "Full Access";
  return mode || "-";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
