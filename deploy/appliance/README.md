# MARC A1 Appliance

Dieses Verzeichnis baut ein angepasstes Debian-Installations-ISO fuer einen
M.A.R.C A1 Appliance-Host.

Zielbild:

- minimales Debian ohne Desktop
- Ollama lokal als Systemdienst
- M.A.R.C A1 als systemd-Dienst
- vorbereitete Runtime-Konfiguration fuer CPU-only Hardware
- automatischer First-Boot-Provisioner
- Web-Setup bleibt fuer den ersten Admin aktiv
- optionaler Offline-Import fuer lokale Ollama-Modelle
- autonomer Updater, der nur `origin/main` verfolgt

Die Voreinstellung ist bewusst auf den Dell OptiPlex 3070 Micro mit
`i5-9500T` und `32 GB RAM` zugeschnitten:

- Primärmodell: `qwen2.5-coder:14b`
- Routermodell: `qwen2.5-coder:14b`
- `OLLAMA_MAX_LOADED_MODELS=1`
- `OLLAMA_NUM_PARALLEL=1`
- `OLLAMA_CONTEXT_LENGTH=8192`
- CPU-Governor dauerhaft auf `performance`
- reduzierte Swap-Neigung und staerkerer VFS-Cache fuer Repo-/Datei-Last

Warum nicht `30b` als Standard:

- Auf dieser CPU ist nicht der Unterbau, sondern die Inferenz der eigentliche
  Flaschenhals.
- Ein `14b`-Profil liefert auf dem Geraet typischerweise die bessere
  Gesamtnutzbarkeit und deutlich weniger Wartezeit.

Auto-Update:

- Der Appliance-Installer legt einen separaten systemd-Timer an.
- Dieser prueft regelmaessig nur `origin/main`.
- Updates werden ausserhalb des laufenden Webprozesses eingespielt.
- Vor dem Umschalten wird ein Backup erstellt; bei einem Fehlstart erfolgt ein Rollback.

Build:

```bash
python3 deploy/appliance/build_custom_iso.py \
  --base-iso /Pfad/zum/debian-netinst.iso \
  --output-iso deploy/appliance/output/marc-a1-appliance.iso \
  --vendor-local-model
```

Danach kann das erzeugte ISO wie ein normales Debian-Image auf einen USB-Stick
geschrieben werden.
