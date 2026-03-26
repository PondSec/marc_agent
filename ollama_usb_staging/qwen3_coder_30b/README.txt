Qwen3 Coder 30B fuer Windows 11 mit Ollama

Inhalt:
- model\qwen3-coder-30b.blob
- qwen3-coder-30b.Modelfile.in
- install_qwen3_coder_30b.ps1
- install_qwen3_coder_30b.bat

So benutzt du den Stick auf Windows 11:
1. Ollama installieren, falls es noch nicht installiert ist.
2. Ollama einmal starten.
3. Auf dem Stick install_qwen3_coder_30b.bat per Doppelklick ausfuehren.
4. Danach zum Test in PowerShell oder CMD:
   ollama run qwen3-coder:30b

Hinweise:
- Das Script importiert das Modell lokal aus der Blob-Datei. Es wird dabei nichts aus dem Internet heruntergeladen.
- Auf dem Windows-Rechner muss genug freier Speicher fuer das Modell vorhanden sein.
- Wenn du Ollama auf Windows an einen benutzerdefinierten Modelle-Pfad gebunden hast, nutzt Ollama beim Import automatisch diesen Pfad.
