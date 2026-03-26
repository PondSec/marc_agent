# M.A.R.C A1

Modular Autonomous Runtime Core - Agent 1

M.A.R.C A1 ist ein lokaler, selbstgehosteter Coding-Agent fuer echte Entwicklungsarbeit. Die Runtime ist auf agentische Umsetzung optimiert: Repo verstehen, gezielt Dateien finden, Code aendern, Tests ausfuehren, Fehler analysieren, nachbessern und Ergebnisse mit Diffs, Logs und Session-State nachvollziehbar machen.

## Was Aus Dem Bestehenden Projekt Geworden Ist

Der vorherige Stand hatte bereits brauchbare Grundbausteine:

- Python-Backend mit FastAPI
- lokale Weboberflaeche
- Tool-Dispatcher fuer Filesystem, Search, Shell und Git
- Workspace-Sicherheitsgrenzen
- Session-Store und Logdateien
- Basistests fuer zentrale Runtime-Bausteine

Die groessten Luecken lagen im agentischen Verhalten:

- kein sauberes `access_mode` Modell
- zu lineare Agent-Schleife ohne starke Verify-/Repair-Logik
- zu schwache Abschlusskriterien nach Codeaenderungen
- wenig Sichtbarkeit fuer Phase, Validation-Status und Stop-Gruende
- keine explizite Ermutigung, bei Hindernissen Hilfsskripte oder Parser zu bauen
- Branding, Doku und UI noch nicht auf einen ernsthaften Agenten ausgerichtet

M.A.R.C A1 behebt genau diese Punkte, ohne das Projekt blind neu zu schreiben.

## Kernfaehigkeiten

M.A.R.C A1 ist auf diese Arbeitsweise ausgelegt:

1. Task verstehen und Repo gezielt analysieren
2. relevante Dateien priorisieren und lesen
3. bestehende Architektur respektieren
4. fokussierte Aenderungen ueber mehrere Dateien hinweg machen
5. Tests, Lint, Build oder andere Projektchecks gezielt ausfuehren
6. Fehler aus Logs und Tool-Outputs lesen
7. iterativ nachbessern
8. Status, Tool-Calls, Diffs, Blocker und Ergebnisse sichtbar machen

Wenn noetig darf M.A.R.C A1 auch kleine Hilfswerkzeuge bauen, zum Beispiel:

- Analyse-Skripte
- Parser oder Konverter
- Test-Harnesses
- temporaere Log- oder Format-Inspektoren

Das Ziel ist immer die Hauptaufgabe. Hilfsskripte sind ein Mittel zum Zweck, nicht der Fokus.

## Access Modes

Die Runtime hat jetzt ein klares Zugriffsmodell:

- `safe`
  Nur lesende Exploration und low-risk Verifikation. Schreibende Tools und mutierende Shell-Aktionen sind blockiert.
- `approval`
  Normale Coding-Edits ueber File-Tools sind erlaubt, medium/high-risk Shell- oder Git-Mutationen werden geblockt.
- `full`
  Systemweiter Modus: Dateien lesen und aendern auch ausserhalb von `workspace_root`, absolute Pfade verwenden, Shell-Kommandos auch ausserhalb des Repo ausfuehren und lokal autonom arbeiten. Harte Katastrophen-Blocker bleiben aktiv.

Die zentrale Einstellung ist:

```json
{
  "access_mode": "safe | approval | full"
}
```

Legacy-Flags wie `--read-only` und `--approval-mode` werden weiter akzeptiert, intern aber auf `access_mode` normalisiert.

## Agentischer Ablauf

Die Runtime arbeitet in einer expliziten Schleife:

1. `planning`
2. `exploring`
3. `editing`
4. `verifying`
5. `repairing`
6. `completed` oder `blocked`

Zusatzregeln:

- Kein verfruehtes Finish nach Codeaenderungen ohne Validation, falls ein sinnvoller Check verfuegbar ist
- wiederholte Verify-Fehlschlaege werden als Repair-Versuche gezaehlt
- sinnvolle Stop-Gruende werden gespeichert, zum Beispiel `validated`, `blocked`, `max_iterations_reached`
- Helper-Artefakte werden in der Session sichtbar gemacht

## Architektur

Die Architektur bleibt modular, ist aber jetzt staerker auf echte Agentik ausgerichtet:

1. Web-GUI
   `webui/` ist die Hauptoberflaeche fuer Tasks, Sessions, Live-Status, Tool-Calls, Logs und Diffs.
2. Web-Backend
   `server/` kapselt API, Session-Lifecycle und Live-Streams.
3. Agent-Core
   `agent/core.py`, `planner.py`, `memory.py`, `models.py` steuern Planung, Phasen, Verify-/Repair-Loop und Session-State.
4. Runtime
   `runtime/` validiert Tool-Calls, loggt Ereignisse und erzwingt Workspace-Grenzen.
5. Tools
   `tools/` enthaelt Filesystem-, Search-, Shell-, Git- und Safety-Logik.
6. LLM-Schicht
   `llm/ollama_client.py` spricht mit Ollama, `llm/schemas.py` erzwingt strukturierte JSON-Entscheidungen.
7. Optionale CLI
   `cli.py` ist fuer Debugging und Automatisierung da, aber nicht der Hauptfokus.

## Projektstruktur

```text
.
|-- .env.example
|-- README.md
|-- cli.py
|-- main.py
|-- requirements.txt
|-- agent
|   |-- core.py
|   |-- executor.py
|   |-- memory.py
|   |-- models.py
|   |-- planner.py
|   |-- prompts.py
|   `-- session.py
|-- config
|   |-- agent.json.example
|   `-- settings.py
|-- llm
|   |-- ollama_client.py
|   `-- schemas.py
|-- runtime
|   |-- logger.py
|   |-- tool_dispatcher.py
|   `-- workspace.py
|-- server
|   |-- app.py
|   |-- schemas.py
|   `-- task_manager.py
|-- tests
|   |-- test_access_modes.py
|   |-- test_dispatcher.py
|   |-- test_filesystem.py
|   |-- test_safety.py
|   |-- test_shell.py
|   |-- test_web_api.py
|   `-- test_workspace.py
|-- tools
|   |-- difftools.py
|   |-- filesystem.py
|   |-- gittools.py
|   |-- registry.py
|   |-- safety.py
|   |-- search.py
|   `-- shell.py
`-- webui
    |-- app.js
    |-- index.html
    `-- styles.css
```

## Starten

Du brauchst kein manuelles `venv`, wenn du das nicht willst. Auf einem frischen Rechner reicht:

```bash
python main.py
```

Beim ersten Start installiert M.A.R.C A1 fehlende Runtime-Pakete selbst aus `requirements-runtime.txt` und startet danach direkt die Web-App.

Unter Windows geht auch:

```bat
start_marc_a1.bat
```

Wenn du trotzdem isoliert arbeiten willst, bleibt ein `venv` weiterhin moeglich:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Dann im Browser:

```text
http://127.0.0.1:8000
```

Wichtige Optionen:

```bash
python main.py --host 0.0.0.0 --port 8080
python main.py --access-mode safe
python main.py --access-mode approval
python main.py --access-mode full
python main.py --dry-run
```

Hinweis:

- In `full` koennen Tools und Shell-Kommandos auch absolute Systempfade ausserhalb des Workspace verwenden.

## CLI

```bash
python cli.py task "Fuege JWT Login hinzu"
python cli.py inspect --focus auth
python cli.py diff
python cli.py config show
```

## Ollama Setup

```bash
ollama serve
ollama pull qwen3-coder:30b
ollama run qwen3-coder:30b
```

Wichtige Defaults:

- `OLLAMA_HOST=http://127.0.0.1:11434`
- `MODEL_NAME=qwen3-coder:30b`
- `WORKSPACE_ROOT=.`
- `ACCESS_MODE=approval`

## API

Wichtige Endpunkte:

- `GET /api/health`
- `GET /api/config`
- `GET /api/workspace/inspect`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `GET /api/sessions/{session_id}/logs`
- `GET /api/sessions/{session_id}/events`
- `POST /api/tasks`

`/api/sessions/{session_id}/events` streamt Live-Session-Updates per SSE.

## Git Workflow

Das Repo ist auf einen professionelleren Branching-Flow ausgelegt:

- `main` fuer stabile, releasbare Staende
- `develop` fuer laufende Integration
- `feature/<topic>` fuer neue Arbeit
- `fix/<topic>` fuer Fehlerbehebungen
- `hotfix/<topic>` fuer dringende Korrekturen auf `main`

Details stehen in [CONTRIBUTING.md](/Users/pond/Documents/agent_ai/CONTRIBUTING.md).

## Sicherheit

Die Runtime bleibt lokal, aber defensiv:

- `safe` und `approval` bleiben auf den Workspace begrenzt
- `full` hebt die Pfadbegrenzung auf und erlaubt systemweiten Dateizugriff
- harte Shell-Blocker fuer destruktive Kommandos bleiben aktiv
- Netzwerkzugriff ist standardmaessig aus
- `safe` blockiert Mutationen
- `approval` blockiert medium/high-risk Shell-Mutationen
- `full` erlaubt autonomes Arbeiten systemweit, aber keine Hard-Block-Aktionen
- alle Tool-Calls und Entscheidungen werden geloggt
- Session-Diffs werden gespeichert und in der GUI angezeigt

Interner Zustand liegt standardmaessig in:

```text
.marc_a1/
```

Darin befinden sich Sessions, Logs, Memory und ein Helper-Verzeichnis fuer agentische Hilfsskripte.

Bootstrap-Verhalten:

- `main.py` und `cli.py` installieren fehlende Runtime-Abhaengigkeiten automatisch
- ausserhalb eines `venv` wird standardmaessig per `pip install --user` installiert
- das Verhalten kann ueber `MARC_A1_PIP_SCOPE=user|global|auto` gesteuert werden
- zusaetzliche Pip-Argumente koennen ueber `MARC_A1_PIP_EXTRA_ARGS` gesetzt werden

## Tests

```bash
python -m pytest -q
```

Abgedeckt sind aktuell:

- Workspace-Sicherheit
- Tool-Argumentvalidierung
- File-Patching und Diffs
- Shell-Schutz und Timeout-Verhalten
- Access-Mode-Logik
- Web-API und Session-Metadaten

## Erweiterbarkeit

Sinnvolle naechste Ausbaustufen:

- Docker-Exec Tool
- SSH-Exec Tool mit Allowlist
- HTTP-Request Tool fuer interne APIs
- Approval-UI fuer medium/high-risk Aktionen
- Repo-Index mit Symbol-Layer
- Multi-Agent-Orchestrierung fuer Explore/Edit/Verify
- persistente Hintergrund-Queue
