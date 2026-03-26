# Local Codex-Style Agent

Eine lokal laufende Web-App fuer einen agentischen Coding-Assistenten mit Python, Ollama und `qwen3-coder:30b`.

## Architekturueberblick

Die Agent-Engine bleibt erhalten und wird jetzt von einem lokalen Web-Backend gekapselt. Dadurch wird die GUI zur Hauptschnittstelle, ohne dass Planung, Tool-Ausfuehrung, Safety oder Session-State neu geschrieben werden muessen.

Die Architektur besteht aus sieben Schichten:

1. Web-GUI
   `webui/` enthaelt die lokale Browseroberflaeche fuer Task-Eingabe, Session-Navigation, Live-Status, Tool-Calls, Logs und Diffs.
2. Web-Backend
   `server/app.py` stellt die HTTP-API bereit, `server/task_manager.py` startet Hintergrund-Tasks und kapselt Sessions, Logs und Live-Updates.
3. Agent-Schicht
   `agent/core.py`, `planner.py`, `executor.py`, `memory.py`, `session.py` bilden weiterhin den Plan-and-Execute-Loop.
4. LLM-Schicht
   `llm/ollama_client.py` spricht mit Ollama, `llm/schemas.py` erzwingt strukturierte JSON-Entscheidungen.
5. Runtime-Schicht
   `runtime/tool_dispatcher.py`, `runtime/logger.py`, `runtime/workspace.py` validieren Tool-Aufrufe, loggen Schritte und begrenzen alle Pfade auf den Workspace.
6. Tool-Schicht
   `tools/` kapselt Filesystem, Search, Shell, Git, Diffs und Safety-Regeln.
7. Optionale CLI
   `cli.py` bleibt fuer Skripting und Debugging verfuegbar, ist aber nicht mehr die primaere Nutzung.

## Was die GUI kann

Die Browseroberflaeche deckt die bisherigen CLI-Hauptfunktionen ab:

- Tasks starten
- bestehende Sessions ansehen
- Session-Verlauf verfolgen
- Live-Status waehrend der Ausfuehrung sehen
- Tool-Aufrufe und Ausgaben ansehen
- geaenderte Dateien und Diffs ansehen
- Workspace inspizieren
- Runtime-Konfiguration ansehen

Die Agent-Engine selbst wird nicht dupliziert. Die GUI steuert dieselbe `AgentCore`-Runtime wie vorher die CLI.

## Projektstruktur

```text
.
├── .env.example
├── .gitignore
├── README.md
├── cli.py
├── main.py
├── requirements.txt
├── agent
│   ├── __init__.py
│   ├── core.py
│   ├── executor.py
│   ├── memory.py
│   ├── models.py
│   ├── planner.py
│   ├── prompts.py
│   └── session.py
├── config
│   ├── __init__.py
│   ├── agent.json.example
│   └── settings.py
├── llm
│   ├── __init__.py
│   ├── ollama_client.py
│   └── schemas.py
├── runtime
│   ├── __init__.py
│   ├── logger.py
│   ├── tool_dispatcher.py
│   └── workspace.py
├── server
│   ├── __init__.py
│   ├── app.py
│   ├── schemas.py
│   └── task_manager.py
├── tests
│   ├── test_dispatcher.py
│   ├── test_filesystem.py
│   ├── test_safety.py
│   ├── test_shell.py
│   ├── test_web_api.py
│   └── test_workspace.py
├── tools
│   ├── __init__.py
│   ├── difftools.py
│   ├── filesystem.py
│   ├── gittools.py
│   ├── registry.py
│   ├── safety.py
│   ├── search.py
│   └── shell.py
└── webui
    ├── app.js
    ├── index.html
    └── styles.css
```

## Web-First Nutzung

Server starten:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Danach im Browser oeffnen:

```text
http://127.0.0.1:8000
```

Optionale Server-Parameter:

```bash
.venv/bin/python main.py --host 0.0.0.0 --port 8080
.venv/bin/python main.py --dry-run
.venv/bin/python main.py --read-only
.venv/bin/python main.py --approval-mode
```

## Ollama Setup

1. Ollama starten
2. Modell ziehen
3. Test-Run machen

```bash
ollama serve
ollama pull qwen3-coder:30b
ollama run qwen3-coder:30b
```

Offizielle Ollama-Modellseite:
[qwen3-coder:30b](https://ollama.com/library/qwen3-coder%3A30b-a3b-fp16)

Wichtige Defaults:

- `OLLAMA_HOST=http://127.0.0.1:11434`
- `MODEL_NAME=qwen3-coder:30b`
- `WORKSPACE_ROOT=.`

## API Ueberblick

Das Backend ist lokal gedacht und stellt unter anderem diese Endpunkte bereit:

- `GET /api/health`
- `GET /api/config`
- `GET /api/workspace/inspect`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `GET /api/sessions/{session_id}/logs`
- `GET /api/sessions/{session_id}/events`
- `POST /api/tasks`

`/api/sessions/{session_id}/events` liefert Live-Updates per Server-Sent Events.

## Sicherheitsmodell

Der Agent bleibt konservativ:

- Alle Pfade werden gegen `workspace_root` aufgeloest.
- Schreibzugriffe ausserhalb des Workspace werden blockiert.
- `read_only` blockiert mutierende File- und Shell-Aktionen.
- Shell-Kommandos werden klassifiziert als `low`, `medium`, `high`, `blocked`.
- Harte Blockregeln greifen fuer `sudo`, `shutdown`, `reboot`, `git reset --hard`, `git clean -fdx`, `git push --force`.
- Netzwerkkommandos bleiben standardmaessig deaktiviert.
- `approval_mode` blockiert riskante Shell- und Git-Aktionen.
- Alle Tool-Calls werden als JSONL geloggt.
- Diffs werden pro Session gespeichert und in der GUI sichtbar gemacht.

## Umgang mit grossen Projekten

Die Runtime ist weiterhin auf groessere Repositories ausgelegt:

- Repo-Snapshot statt blindem Komplett-Read
- Priorisierung von README, Manifests, Configs, Services, Routes und Tests
- `inspect_workspace`, `list_files`, `search_in_files` als gezielte Erkundungswerkzeuge
- Session-Memory mit kompakten Output-Exzerpten
- konfigurierbare Grenzen fuer Lesegroesse, Suchtreffer, Iterationen und Tool-Calls

## Optionale CLI

Die CLI existiert weiterhin fuer Automatisierung oder Debugging, ist aber nicht mehr die Hauptschnittstelle:

```bash
.venv/bin/python cli.py task "Fuege JWT Login hinzu"
.venv/bin/python cli.py inspect --focus auth
.venv/bin/python cli.py diff
.venv/bin/python cli.py config show
```

## Tests

Ausfuehren mit:

```bash
.venv/bin/python -m pytest
```

Abgedeckt sind aktuell:

- Workspace-Pfadsicherheit
- Safety-Rules fuer Shell-Kommandos
- Tool-Argumentvalidierung
- Patch- und Dateiaenderungslogik
- Shell-Timeout-Verhalten
- Web-API fuer GUI-Startpfad, Session-Liste und lokale Task-Ausloesung

## Erweiterungsmoeglichkeiten

Naechste sinnvolle Ausbaustufen:

- VS Code Extension auf Basis derselben HTTP-API
- Docker-Sandbox fuer Tool-Ausfuehrung
- SSH-Tool mit Host-Allowlist
- HTTP-Tools fuer interne APIs
- Repo-Index mit Dateisummaries und Symbol-Layer
- Approval-UI fuer riskante Aktionen
- Multi-Agent-Orchestrierung fuer Explore/Edit/Verify
- Background Queue fuer laengere Jobs

## Annahmen

- Python 3.11+ ist verfuegbar.
- Ollama ist lokal erreichbar.
- Das Modell liefert meist valides JSON; bei Problemen faellt die Agent-Runtime auf Heuristiken zurueck.
- Die Web-App ist lokal und single-user-orientiert, nicht als oeffentlich gehaertete Multi-User-Plattform gedacht.
