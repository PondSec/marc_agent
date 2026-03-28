# Login-System Und Security-Haertung

## Architekturueberblick

Die Web-Konsole verwendet jetzt eine serverseitige Authentifizierungsschicht mit folgenden Bausteinen:

- `server/auth_service.py`
  Zentrale Auth-Logik fuer Login, Logout, Session-Validierung, Cookie-Ausgabe, CSRF-Schutz, Rate-Limits, Passwort-Policy und Passwort-Reset-Tokens.
- `server/auth_store.py`
  Persistenter SQLite-Store fuer Benutzer, Sessions, Auth-Events und Rate-Limit-Status.
- `server/security.py`
  Sicherheits-Header, Trusted-Host-Pruefung und restriktive CORS-Konfiguration.
- `server/app.py`
  Oeffentliche Auth-Endpunkte (`/api/auth/session`, `/api/auth/login`, `/api/auth/logout`) und geschuetzte API-Routen mit Auth- und CSRF-Dependencies.
- `webui/app.js`
  Login-UI, Session-Status-Pruefung, Logout, automatische CSRF-Header fuer mutierende Requests und Handling abgelaufener Sessions.
- `webui/styles.css`
  Login-Layout im bestehenden Designsystem der Konsole.

## Threat Model

### Assets

- Admin-Zugang zur M.A.R.C-A1-Konsole
- Laufende Sessions, Workspaces, Reports und Logs
- Lokale Dateizugriffe ueber die Agent-Runtime
- Session-Cookies und CSRF-Tokens

### Angreiferprofile

- Externer Web-Angreifer ohne Zugangsdaten
- Angreifer mit geleakten oder geratenen Zugangsdaten
- Opportunistischer Bot fuer Credential Stuffing / Brute Force
- Nutzer mit Browser-Kontext auf anderer Origin, der CSRF versuchen wuerde

### Eintrittspunkte

- Login-Endpunkt `/api/auth/login`
- Alle geschuetzten `/api/*`-Routen
- Root-Dokument und statische Assets
- Session-Cookies im Browser

### Wahrscheinliche Angriffe

- Brute Force und Credential Stuffing
- Account Enumeration ueber differenzierte Fehlermeldungen
- CSRF gegen mutierende JSON-Endpunkte
- Session-Diebstahl oder Session-Fixation
- Reflected / stored XSS ueber UI-Daten
- Open Redirect ueber Ruecksprung-Parameter
- Clickjacking
- SQL-Injection-Versuche gegen Login

### Schwerwiegendste Auswirkungen

- Uebernahme der Admin-Konsole
- Starten oder Manipulieren von Agent-Laeufen
- Einblick in lokale Projekte, Reports und Logs
- Persistente Session-Missbraeuche

### Technische Gegenmassnahmen

- Argon2id fuer Passwort-Hashing
- Serverseitige Sessions mit zufaelligen Session-IDs und serverseitiger Invalidierung
- Session-Rotation bei Login und Invalidierung bei Logout/Ablauf
- Idle-Timeout und absolute Session-Laufzeit
- HttpOnly-/Secure-/SameSite-Cookies
- Double-Submit-CSRF-Token plus Origin-/Referer-Pruefung
- Generische Login-Fehlermeldungen gegen Enumeration
- IP-, Account- und Paar-bezogene Rate-Limits mit progressivem Backoff
- Restriktive CSP und weitere Security-Header gegen XSS, Clickjacking und Content-Sniffing
- Parametrisierte SQLite-Queries gegen Injection
- Keine Redirect-Weitergabe an externe Ziele; nur relative Ruecksprungpfade werden akzeptiert

## Security-Entscheidungen

- Argon2id statt Eigenbau:
  Passwort-Hashes werden mit `argon2-cffi` und `Type.ID` erzeugt.
- Serverseitiger Session-Store statt `localStorage`:
  Der Browser kennt nur ein opaques Session-Cookie. Auth-State liegt nicht ausschliesslich im Frontend.
- JSON-Login mit CSRF-Schutz:
  Da die SPA `fetch()` nutzt, werden mutierende Requests mit `X-CSRF-Token` abgesichert. Zusaetzlich muss `Origin` oder `Referer` zur erwarteten Origin passen.
- TOTP-2FA direkt vorbereitet und nutzbar:
  Wenn `AUTH_INITIAL_ADMIN_TOTP_SECRET` gesetzt ist, verlangt der Login zusaetzlich einen gueltigen Einmalcode.
- Restriktive Cookies:
  Session-Cookie `HttpOnly`, `Secure`, `SameSite=Lax`; CSRF-Cookie `Secure`, `SameSite=Strict`.
- Restriktive CORS-Policy:
  Standardmaessig keine Cross-Origin-Freigabe. Falls benoetigt, explizit ueber `CORS_ALLOWED_ORIGINS`.
- Trusted Hosts:
  Standardmaessig nur lokale Hosts; fuer Deployment auf Domain muss `ALLOWED_HOSTS` gesetzt werden.

## Sicherheits-Header Und Schutzmassnahmen

Aktiv gesetzt:

- `Content-Security-Policy`
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: no-referrer`
- `Permissions-Policy`
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Resource-Policy: same-origin`
- `X-Permitted-Cross-Domain-Policies: none`
- `Strict-Transport-Security` bei HTTPS
- `Cache-Control: no-store` fuer Root und API

Weitere aktive Schutzmassnahmen:

- Auth-Pflicht fuer die komplette API
- Session-Invalidierung bei Logout, Ablauf oder ungueltiger Session
- Session-Fixation-Schutz durch neue Session-ID bei Login
- Password-Policy mit Mindestlaenge und Schwachpasswort-Blockade
- Parametrisierte SQLite-Zugriffe
- TOTP-Validierung mit `pyotp`
- Signierte, kurzlebig pruefbare Passwort-Reset-Tokens als vorbereitete Utility

## Annahmen

- Die Konsole ist aktuell single-tenant und wird von einem Admin-Operator genutzt.
- Benutzerverwaltung ausser dem initialen Admin-Setup ist nicht Teil dieses Schritts.
- `AUTH_SECRET_KEY` und Initial-Credentials werden sicher ausserhalb des Repos gesetzt.
- `Secure`-Cookies setzen HTTPS voraus. Fuer lokales Debugging ohne TLS muss `AUTH_COOKIE_SECURE=false` bewusst gesetzt werden.

## Starten

1. `.env` mit starken Werten fuellen, mindestens:
   - `AUTH_SECRET_KEY`
   - `AUTH_INITIAL_ADMIN_EMAIL`
   - `AUTH_INITIAL_ADMIN_PASSWORD`
2. Abhaengigkeiten installieren:

```bash
.venv/bin/pip install -r requirements.txt
```

3. Server starten:

```bash
PYTHONPATH=/absolute/path/to/repo .venv/bin/uvicorn server.app:create_app --factory --host 127.0.0.1 --port 8000
```

4. Browser aufrufen:

```text
https://127.0.0.1:8000
```

Hinweis:
Fuer echtes `Secure`-Cookie-Verhalten sollte die Konsole hinter HTTPS oder einem lokalen TLS-Proxy laufen. In reinen HTTP-Entwicklungsumgebungen muss `AUTH_COOKIE_SECURE=false` bewusst dokumentiert gesetzt werden.

## Pruefen

Python-Tests:

```bash
PYTHONPATH=/absolute/path/to/repo .venv/bin/pytest tests/test_web_api.py tests/test_auth_security.py
```

Frontend-Logiktests:

```bash
node --test tests/test_webui.cjs
```

Dependency-Audit:

```bash
PYTHONPATH=/absolute/path/to/repo .venv/bin/python -m pip_audit -r requirements.txt
```

## Durchgefuehrte Testarten

- Unit-Tests fuer Passwort-Policy und Redirect-Sanitizing
- Integrationstests fuer Login, Logout, Session-Ablauf und geschuetzte API-Routen
- Security-Checks fuer:
  - fehlende Authentifizierung
  - Cookie-Flags
  - Security-Header
  - CSRF
  - falsche Origin
  - Enumeration
  - Brute Force / Rate-Limit
  - TOTP-2FA
  - Injection-aehnliche Login-Payloads
- Frontend-Logiktests fuer UI-Helfer und Thread-Darstellung

## Restrisiken

- Single-Tenant-Modell:
  Die Konsole trennt aktuell keine mehreren Benutzer oder Rollen.
- Kein produktiver Password-Reset-Transport:
  Signierte Tokens sind vorbereitet, aber es gibt noch keinen sicheren E-Mail- oder Out-of-Band-Versandpfad.
- Kein externer WAF/Abuse-Layer:
  Rate-Limits greifen in der App, aber nicht auf Reverse-Proxy- oder Edge-Ebene.
- Keine vollstaendige Audit-Log-Auswertung:
  Auth-Events werden gespeichert, aber noch nicht im UI ausgewertet oder alarmiert.
- Dev-Dependency-Befund:
  `pip-audit -r requirements.txt` meldet aktuell `CVE-2026-4539` in `pygments 2.19.2`, eingezogen ueber die Dev-Abhaengigkeit `pytest`. Fuer `requirements-runtime.txt` ist das Audit sauber; fuer `pygments` wurde zum Pruefzeitpunkt noch keine Fix-Version angeboten.

## Empfohlene Naechste Schritte

- Unabhaengigen Penetrationstest gegen das deployte System fahren
- Zweites Security-Review durch weiteren Engineer
- SAST und DAST im CI ergaenzen
- Secret-Scanning im Repo und in CI aktivieren
- Monitoring und Alerting fuer Auth-Events / Lockouts aufbauen
- MFA-Rollout fuer alle produktiven Operatoren verpflichtend machen
- Audit-Log-Ansicht und Alarmierung fuer auffaellige Login-Muster ergaenzen
- Optional Multi-User- / Rollenmodell einfuehren, falls das Produkt ueber Single-Tenant hinaus waechst
