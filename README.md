# M.A.R.C A1

Modular Autonomous Runtime Core - Agent 1

M.A.R.C A1 ist ein lokaler, selbstgehosteter Coding- und Entwicklungsagent. Er ist bewusst nicht als allgemeiner Assistent gedacht, sondern fuer echte Entwicklungsarbeit: Repository verstehen, relevante Dateien finden, Code aendern, Tests und Checks ausfuehren, Fehler diagnostizieren, gezielt nachbessern und das Ergebnis sauber dokumentieren.

Die Hauptoberflaeche ist die Web-GUI. Die CLI bleibt fuer Debugging und Automatisierung erhalten, ist aber bewusst sekundar.

## Zielbild

M.A.R.C A1 ist auf diesen Ablauf optimiert:

1. Aufgabe verstehen
2. Projektstruktur und bestehende Architektur analysieren
3. relevante Dateien und Muster priorisieren
4. gezielte Aenderungen an der richtigen Stelle vornehmen
5. passende Tests, Linter, Typechecks oder Builds auswaehlen
6. Fehlerausgaben und Logs diagnostizieren
7. nachbessern und erneut pruefen
8. Diffs, Logs, Diagnosen, Stop-Grund und Abschlussbericht liefern

Wenn die direkte Loesung blockiert ist, darf M.A.R.C A1 kleine Hilfsskripte, Parser oder Test-Harnesses im Workspace erzeugen, solange sie die eigentliche Entwicklungsaufgabe voranbringen und sauber nachvollziehbar bleiben.

## Was Im Vergleich Zum Frueheren Stand Staerker Ist

Das Projekt hatte bereits gute Grundbausteine:

- Python-Backend mit FastAPI
- lokale Weboberflaeche
- Tool-Dispatcher fuer Filesystem, Search, Shell und Git
- Workspace-Sicherheitsgrenzen
- Session-Store und Logdateien
- erste Basistests

Die groessten Schwaechen lagen im agentischen Verhalten:

- zu lineare Agent-Schleife
- zu duenne Repo-Inspektion
- zu schwache Dateiauswahl
- zu wenig projektbezogene Validation
- kaum strukturierte Fehlerdiagnose
- zu wenig Repair-Iteration
- zu wenig sichtbarer Session- und Abschlusszustand

M.A.R.C A1 hebt diese Punkte an, ohne das Repo blind neu zu schreiben.

## Kernverhalten

Die Runtime arbeitet entlang dieser Workflow-Stufen:

1. `discover`
2. `plan`
3. `act`
4. `verify`
5. `repair`
6. `report`

Intern werden diese Stufen ueber konkrete Laufphasen wie `planning`, `exploring`, `editing`, `verifying`, `repairing` und `reporting` abgebildet.

Wichtige Regeln:

- keine verfruehte Finalisierung nach Codeaenderungen, solange noch sinnvolle Validierung offen ist
- Validation-Runs werden pro Edit-Generation verfolgt
- fehlgeschlagene Checks werden als Diagnosen mit Datei- und Aktionshinweisen gespeichert
- Repair-Versuche werden gezaehlt
- jede Session erzeugt einen strukturierten Abschlussbericht
- Tool-Calls, Shell-Kommandos, Diffs und Logs bleiben nachvollziehbar

## Repo- und Kontextverstaendnis

M.A.R.C A1 baut vor relevanten Aenderungen gezielt Kontext auf, statt nur Dateien stumpf zu lesen.

Der Agent extrahiert unter anderem:

- priorisierte Dateien
- Fokus-Dateien passend zur Aufgabe
- File-Insights mit Kategorien und Gruenden
- Repo-Map und wichtige Top-Level-Bereiche
- Manifeste, Configs, Tests, Build- und Deploy-Dateien
- projektbezogene Validation- und Workflow-Kommandos

Damit wird das Verhalten deutlich naeher an einem starken Coding-Agenten als an einem einfachen Tool-Loop.

## Access Modes

Die Runtime kennt drei klare Zugriffsmodi:

- `safe`
  Nur lesende Exploration und low-risk Verifikation. Schreibende Tools und mutierende Shell-Kommandos werden blockiert.
- `approval`
  Normale Coding-Edits ueber File-Tools sind erlaubt. Medium- und High-Risk-Shell- oder Git-Mutationen werden blockiert.
- `full`
  Voller lokaler Modus. Dateien und Shell-Kommandos duerfen auch ausserhalb von `workspace_root` verwendet werden, solange keine Hard-Block-Sicherheitsregel greift.

Die zentrale Konfiguration ist:

```json
{
  "access_mode": "safe | approval | full"
}
```

Legacy-Flags wie `--read-only` und `--approval-mode` werden weiter akzeptiert, intern aber auf `access_mode` normalisiert.

Wichtig:

- `approval` hat aktuell noch keinen interaktiven Freigabe-Dialog in der GUI
- riskante Schritte werden dort derzeit sauber blockiert und im Session-State dokumentiert
- `full` ist der Modus fuer moeglichst autonome lokale Entwicklungsarbeit

## Web-GUI

Die Web-GUI ist die Hauptoberflaeche fuer echte Agentenarbeit.

Sie zeigt unter anderem:

- neue Tasks starten und Sessions fortsetzen
- Access Mode und Dry Run
- Live-Status und aktuelle Workflow-Stufe
- Tool-Calls und Log-Ereignisse
- geaenderte Dateien und gespeicherte Diffs
- Validation-Runs
- Diagnostik aus fehlgeschlagenen Checks
- Workspace-Analyse mit Repo-Map und Validation-Plan
- strukturierten Abschlussbericht

Standardadresse nach dem Start:

```text
http://127.0.0.1:8000
```

## Architektur

Die Architektur ist modular, aber klar auf agentische Entwicklungsarbeit ausgerichtet:

1. Web-GUI
   `webui/` ist die Hauptoberflaeche fuer Tasks, Sessions, Live-Status, Diffs, Logs und Reports.
2. Web-Backend
   `server/` kapselt API, Session-Lifecycle und Live-Streams.
3. Agent-Core
   `agent/core.py`, `planner.py`, `memory.py`, `verification.py`, `diagnostics.py` und `reporting.py` steuern Planung, Repo-Verstaendnis, Validation, Diagnose, Repair-Loop und Abschlussbericht.
4. Runtime
   `runtime/` erzwingt Workspace-Grenzen, Tool-Dispatching und Logging.
5. Tools
   `tools/` enthaelt Filesystem-, Search-, Shell-, Git- und Safety-Logik.
6. LLM-Schicht
   `llm/ollama_client.py` spricht mit Ollama, `llm/schemas.py` erzwingt strukturierte JSON-Entscheidungen.
7. Optionale CLI
   `cli.py` ist fuer Debugging, Automatisierung und schnelle lokale Nutzung da.

## Projektstruktur

```text
.
|-- README.md
|-- CONTRIBUTING.md
|-- bootstrap_runtime.py
|-- main.py
|-- cli.py
|-- start_marc_a1.bat
|-- requirements.txt
|-- requirements-runtime.txt
|-- agent
|   |-- core.py
|   |-- diagnostics.py
|   |-- executor.py
|   |-- memory.py
|   |-- models.py
|   |-- planner.py
|   |-- prompts.py
|   |-- reporting.py
|   |-- session.py
|   `-- verification.py
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
|-- tools
|   |-- difftools.py
|   |-- filesystem.py
|   |-- gittools.py
|   |-- registry.py
|   |-- safety.py
|   |-- search.py
|   `-- shell.py
|-- tests
|   |-- test_access_modes.py
|   |-- test_bootstrap.py
|   |-- test_diagnostics.py
|   |-- test_dispatcher.py
|   |-- test_filesystem.py
|   |-- test_repo_inspection.py
|   |-- test_safety.py
|   |-- test_shell.py
|   |-- test_validation_planner.py
|   |-- test_web_api.py
|   `-- test_workspace.py
`-- webui
    |-- app.js
    |-- index.html
    `-- styles.css
```

## Schnellstart

Minimal:

```bash
python main.py
```

Beim ersten Start installiert M.A.R.C A1 fehlende Runtime-Pakete aus `requirements-runtime.txt` und startet danach direkt die Web-App.

Unter Windows:

```bat
start_marc_a1.bat
```

Mit eigener virtueller Umgebung:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Nützliche Optionen:

```bash
python main.py --host 0.0.0.0 --port 8080
python main.py --access-mode safe
python main.py --access-mode approval
python main.py --access-mode full
python main.py --dry-run
```

## CLI

Die CLI ist vorhanden, aber bewusst nicht der Hauptfokus.

```bash
python cli.py task "Fuege JWT Login hinzu"
python cli.py inspect --focus auth
python cli.py diff
python cli.py config show
```

## Ollama Setup

Standardmodell:

- `qwen3-coder:30b`

Lokaler Ollama-Start:

```bash
ollama serve
ollama pull qwen3-coder:30b
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

Session-Antworten enthalten unter anderem:

- `workflow_stage`
- `validation_plan`
- `validation_runs`
- `diagnostics`
- `report`

`/api/sessions/{session_id}/events` streamt Live-Updates per SSE.

## Session-State Und Artefakte

Interner Zustand liegt standardmaessig in:

```text
.marc_a1/
```

Darin befinden sich unter anderem:

- `sessions/`
- `logs/`
- `memory/`
- `helpers/`
- `reports/`

Der Report-Bereich speichert fuer jede Session einen strukturierten Abschlussbericht mit Status, Stop-Grund, Commands, Validation-Runs, Diagnosen, Blockern und geaenderten Dateien.

## Sicherheit

Die Runtime bleibt lokal, aber defensiv:

- `safe` und `approval` bleiben auf den Workspace begrenzt
- `full` hebt die Pfadbegrenzung auf
- harte Shell-Blocker fuer destruktive Kommandos bleiben aktiv
- Netzwerkzugriff ist standardmaessig deaktiviert
- alle Tool-Calls und Entscheidungen werden geloggt
- Diffs und Session-Aenderungen bleiben sichtbar

Bootstrap-Verhalten:

- `main.py` und `cli.py` installieren fehlende Runtime-Abhaengigkeiten automatisch
- ausserhalb eines `venv` wird standardmaessig per `pip install --user` installiert
- das Verhalten kann ueber `MARC_A1_PIP_SCOPE=user|global|auto` gesteuert werden
- zusaetzliche Pip-Argumente koennen ueber `MARC_A1_PIP_EXTRA_ARGS` gesetzt werden

## Tests

Mit aktiver virtueller Umgebung oder installiertem `pytest`:

```bash
python -m pytest -q
```

Abgedeckt sind aktuell unter anderem:

- Workspace-Sicherheit
- Tool-Argumentvalidierung
- File-Patching und Diffs
- Shell-Schutz und Timeout-Verhalten
- Access-Mode-Logik
- Repo-Inspektion
- Fehlerdiagnostik
- Validation-Planung
- Web-API und Session-Metadaten

## Git Workflow

Das Repo ist auf einen professionelleren Branching-Flow ausgelegt:

- `main` fuer stabile, releasbare Staende
- `develop` fuer laufende Integration
- `feature/<topic>` fuer neue Arbeit
- `fix/<topic>` fuer Fehlerbehebungen
- `hotfix/<topic>` fuer dringende Korrekturen auf `main`

Details stehen in [CONTRIBUTING.md](CONTRIBUTING.md).

## Grenzen

M.A.R.C A1 ist jetzt deutlich naeher an einem starken lokalen Coding-Agenten, aber noch nicht fertig in jedem Punkt.

Aktuelle reale Grenzen:

- die eigentliche Codequalitaet haengt weiterhin stark vom lokalen Modell ab
- `approval` blockiert riskante Schritte derzeit, statt einen echten Freigabe-Dialog anzubieten
- die Repo-Inspektion ist heuristisch, nicht symbolisch indexiert
- es gibt noch keine Multi-Agent-Orchestrierung

## Erweiterbarkeit

Sinnvolle naechste Ausbaustufen:

- Approval-UI fuer medium/high-risk Aktionen
- Repo-Index mit Symbol-Layer
- Multi-Agent-Orchestrierung fuer Explore/Edit/Verify
- persistente Hintergrund-Queue
- weitere projektbewusste Exec-Tools fuer Build-, Container- und interne Infrastruktur-Workflows
