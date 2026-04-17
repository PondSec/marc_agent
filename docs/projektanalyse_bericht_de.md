# Projektanalysebericht – M.A.R.C A2

## 1) Executive Summary

M.A.R.C A2 ist kein allgemeiner Chatbot, sondern ein lokal selbstgehosteter **Coding-Agent** mit Fokus auf reale Entwicklungsarbeit im Repository (Verstehen, Planen, Editieren, Validieren, Reparieren, Reporting). Das Projekt ist bereits deutlich über dem Prototyp-Stadium: Es besitzt eine klare modulare Architektur, einen expliziten Agenten-Workflow, Sicherheits- und Access-Controls, eine vollständige Web-Konsole inklusive Authentifizierung sowie eine breite automatisierte Testbasis. Gleichzeitig gibt es typische Reifegrad-Grenzen eines ambitionierten lokalen Agentensystems (sehr große zentrale Module, Single-Tenant-Design, hohe Komplexität in Planer/Heuristik).【F:README.md†L5-L40】【F:agent/core.py†L45-L121】【F:server/app.py†L67-L112】

## 2) Produktzweck und Scope

Laut README ist das Zielbild klar: Der Agent soll Aufgaben verstehen, die Codebasis analysieren, gezielt ändern, passende Validierungen ausführen, Fehler diagnostizieren, iterativ nachbessern und einen strukturierten Abschluss liefern. Die Web-GUI ist die Hauptoberfläche; die CLI dient ergänzend für Debugging und Automatisierung.【F:README.md†L7-L31】【F:README.md†L119-L157】【F:cli.py†L11-L80】

Wesentliche Scope-Entscheidung: Das Projekt priorisiert **Engineering-Workflows** statt generischer Assistenzantworten. Das spiegelt sich in den Workflow-Stufen und in den ausführlichen Session-Artefakten wider (Plan, Tool Calls, Validation Runs, Diagnostics, Report).【F:README.md†L49-L81】【F:agent/models.py†L13-L50】

## 3) Architekturüberblick (technisch)

### 3.1 Hauptebenen

1. **Web-UI (`webui/`)**: Single-Page-Konsole mit Session-/Workspace-Verwaltung, Run-Queue, Diff-Panel, Terminal-Modal, Auth/Setup-Flows.
2. **Server (`server/`)**: FastAPI-Endpunkte, Task-Orchestrierung, Auth, Setup, Modellmanagement, Workspace- und Terminal-APIs.
3. **Agent-Kern (`agent/`)**: Routing, Planung, Tool-Ausführung, Validierung, Diagnostik, Repair-Loop, Reporting, Memory.
4. **Runtime (`runtime/`)**: Workspace-Grenzen, Tool-Dispatching, Logging.
5. **Tools (`tools/`)**: Dateisystem, Suche, Shell, Git, Safety.
6. **LLM-Schicht (`llm/`)**: Provider-Interface, Ollama-Client, Schemas und Resilience-Mechanik.

Diese Struktur ist sauber getrennt und in README sowie Code konsistent umgesetzt.【F:README.md†L191-L245】【F:server/app.py†L67-L112】【F:tools/registry.py†L96-L268】

### 3.2 Agentischer Lauf

`AgentCore.run_task()` implementiert den zentralen Lauf über Iterationen mit Phasen-/Workflow-Status, Planner-Entscheidung, Tool-Execution, Session-Update, Diagnose/Validation und finalem Report. Das ist eine robuste, zustandsorientierte Ausführung statt eines simplen Prompt-Loops.【F:agent/core.py†L74-L238】

### 3.3 Router-Contract und Safety

Die Intent-Interpretation ist über `RouterOutput` strikt typisiert (Pydantic), inklusive Intent, Action-Plan, Klarifizierungslogik, Confidence und Safety-Flag. Dadurch wird eine klare Trennung zwischen semantischer Interpretation und exekutiver Planung erreicht.【F:llm/schemas.py†L19-L159】【F:agent/router.py†L36-L109】

## 4) Funktionsumfang des Agents

### 4.1 Tooling-Fähigkeiten

Der Agent besitzt ein recht vollständiges Toolset:

- Repo-/Datei-Inspektion (`inspect_workspace`, `list_files`, `search_in_files`, `read_file`, `show_diff`)
- Schreiboperationen (`write_file`, `append_file`, `create_file`, `replace_in_file`, `patch_file`, `delete_file`)
- Ausführung/Verifikation (`run_shell`, `run_tests`)
- Git-Operationen (`git_status`, `git_diff`, `git_log`, `git_create_branch`)

Damit deckt der Agent den Kern eines lokalen Coding-Agenten ab (Analyse → Änderung → Prüfung).【F:tools/registry.py†L96-L268】

### 4.2 Validierung und Reparatur

Es existiert ein umfangreicher `ValidationPlanner` mit:

- Erkennung/Extraktion expliziter Test- und Prüfkommandos,
- Klassifikation von Validierungstypen,
- Failure-Evidence-Parsing (Tracebacks, Assertion-Mismatches, Import-/Name-Errors etc.),
- Repair-Briefing-Strukturen.

Das ist ein deutlicher Reifeindikator, weil die Korrekturschleife systematisch statt ad-hoc ist.【F:agent/verification.py†L23-L170】【F:agent/models.py†L80-L147】

### 4.3 Memory- und Session-Tiefe

Die Session-/Memory-Modelle enthalten viele technische Artefakte (Validation Runs, Diagnostics, Follow-up-Kontext, Repair Attempt Records, Runtime-Executions etc.). Das erlaubt Verlaufskonsistenz und nachvollziehbare Entscheidungen über mehrere Iterationen/Folgeaufgaben.【F:agent/models.py†L168-L272】【F:server/task_manager.py†L120-L211】

## 5) Sicherheits- und Betriebsreife

### 5.1 Access Modes und Guardrails

Das Projekt implementiert drei Access Modes (`safe`, `approval`, `full`) und differenziert Tool-/Shell-Risiken. Harte Blockregeln (z. B. `rm -rf /`, `git reset --hard`, `--force` Push-Muster) sind vorhanden. Shell-Kommandos werden risikoklassifiziert und abhängig vom Modus blockiert/zugelassen.【F:config/settings.py†L17-L23】【F:tools/safety.py†L31-L118】【F:tools/safety.py†L126-L240】

### 5.2 Authentifizierung und Security-Hardening

Die Web-Konsole ist mit serverseitiger Auth-Schicht abgesichert (Argon2id, Sessions mit Cookies, CSRF, Rate-Limits, optionale TOTP-2FA). Die Architektur und Restrisiken sind explizit dokumentiert. Das ist für ein lokales Agentensystem überdurchschnittlich ausgereift.【F:docs/auth-security.md†L5-L99】【F:server/auth_service.py†L65-L116】【F:server/auth_store.py†L10-L74】

## 6) Web-App und API-Reife

`server/app.py` zeigt eine breite API-Fläche: Auth, Setup-Assistent, Session-/Task-Steuerung, Workspace-Funktionen, Preview-Routen, Streaming, Terminal-Endpunkte. Die Web-UI (`webui/app.js`) bildet diese Funktionalität in vielen dedizierten Flows ab (Boot, Polling, Sessionsteuerung, Queueing, Terminal, Workspace-Management, Git-Sync-Hilfen).【F:server/app.py†L67-L320】【F:webui/app.js†L1-L140】【F:webui/app.js†L499-L1524】

## 7) Umfangsmetriken (Codebase-Snapshot)

Analysierter Stand (git-tracked Dateien):

- **127 Dateien** gesamt
- **~134.739 Zeilen** (inkl. Tests und Frontend-Assets)
- Größte Bereiche nach LOC:
  - `tests/` ~50.846
  - `agent/` ~47.866
  - `webui/` ~20.852
  - `server/` ~6.087

Interpretation: Das Projekt ist **umfangreich** und klar in Richtung „produktnahes Agentensystem“ statt „kleiner PoC“. Besonders auffällig sind die großen Agent- und Testpakete.【F:docs/projektanalyse_bericht_de.md†L1-L112】

## 8) Testabdeckung und Qualitätssignale

Es gibt 26 Testdateien mit insgesamt **876 erkannten Testfunktionen** (`def test_*`). Schwerpunkte liegen auf Planner, Task Understanding, Validation Planner, Web API, Core-Logik und Security. Diese Breite deutet auf eine aktiv abgesicherte Kernlogik hin – insbesondere für riskante Bereiche (Routing, Safety, Auth, Runtime Resilience).【F:tests/test_planner.py†L1-L120】【F:tests/test_task_understanding.py†L1-L120】【F:tests/test_web_api.py†L1-L120】

## 9) Reifegrad-Einschätzung

### Gesamteinschätzung

**Reifegrad: fortgeschritten (zwischen „MVP+“ und „früher Produktreife“).**

Begründung:

- + Klare modulare Architektur und strukturierter Agenten-Workflow.
- + Strikte Schemas, Sicherheitsmodi, Shell-Gatekeeping, Auth-Hardening.
- + Stark ausgebaute Verifikations-/Repair-Logik.
- + Große und thematisch breite Testbasis.
- ± Sehr große zentrale Dateien (insbesondere Planner/WebUI), die Wartungskomplexität erhöhen.
- ± Noch erkennbare Produktgrenzen (z. B. Single-Tenant-Fokus laut Security-Doku).

Die Richtung ist klar professionell; die nächste Reifestufe ist vor allem ein Thema von Modularisierung, operativer Beobachtbarkeit und Produktisierung über den lokalen Kern hinaus.【F:agent/planner.py†L1-L220】【F:webui/app.js†L1-L220】【F:docs/auth-security.md†L171-L214】

## 10) Was der Agent konkret kann (für Stakeholder verständlich)

Kurz gesagt kann M.A.R.C A2:

1. Entwicklungsaufgaben semantisch interpretieren und in planbare Arbeitsschritte übersetzen.
2. Repositories strukturiert untersuchen und relevante Artefakte priorisieren.
3. Dateien gezielt lesen/ändern und Diffs nachvollziehbar erzeugen.
4. Tests/Checks ausführen, Fehler systematisch auswerten und Reparaturschritte ableiten.
5. Sitzungszustände, Diagnosen und Reports persistent halten.
6. In einer Web-Konsole mit Auth, Workspace-Management und terminalnahen Workflows betrieben werden.

Damit ist es ein **lokaler Engineering-Agent** für iterative Implementierungsarbeit – deutlich mehr als ein reiner „Prompt-to-Text“-Assistent.【F:README.md†L7-L81】【F:tools/registry.py†L96-L268】【F:server/task_manager.py†L93-L211】

## 11) Risiken, Grenzen und nächste sinnvolle Schritte

### Hauptrisiken

- Komplexität/Konzentration in einzelnen großen Dateien kann langfristige Änderungsrisiken erzeugen.
- Single-Tenant-Ausrichtung und noch begrenzte Audit-Auswertung sind für größere Teams limitierend.
- Lokal-LLM-abhängige Semantik kann je nach Modellqualität variieren (trotz Fallback/Resilience).

### Konkrete nächste Schritte

1. Weitere Zerlegung großer Module (`agent/planner.py`, `webui/app.js`) in klar getrennte Teilkomponenten.
2. Erweiterte Observability (Metriken/Tracing) für Produktionsbetrieb.
3. Ausbau Multi-User/Rollenmodell, falls Teamnutzung geplant ist.
4. CI-Pipeline mit Security-/Dependency-Scans und regelmäßigen Integrationsläufen weiter härten.

Diese Schritte erhöhen Wartbarkeit und organisatorische Skalierbarkeit, ohne den bestehenden starken Kern neu zu bauen.【F:agent/planner.py†L1-L220】【F:webui/app.js†L1-L220】【F:docs/auth-security.md†L196-L214】

## 12) Methodik (was für diese Analyse gesichtet wurde)

- Architektur-/Produktdokumentation (`README.md`, `docs/auth-security.md`)
- Kernmodule Agent/Server/Tools/LLM/Runtime
- Umfangs- und Testinventur über Repository-Dateien
- Stichproben in zentralen Test-Suiten

Damit basiert der Bericht auf einer tatsächlichen Codebase-Analyse, nicht nur auf der README-Beschreibung.【F:README.md†L1-L40】【F:docs/auth-security.md†L1-L40】【F:agent/core.py†L45-L121】
