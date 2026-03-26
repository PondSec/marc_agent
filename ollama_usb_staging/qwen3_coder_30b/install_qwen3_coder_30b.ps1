$ErrorActionPreference = 'Stop'

$modelName = 'qwen3-coder:30b'
$expectedBytes = 18556688736
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$blobPath = Join-Path $scriptRoot 'model\qwen3-coder-30b.blob'
$templatePath = Join-Path $scriptRoot 'qwen3-coder-30b.Modelfile.in'
$tempModelfile = Join-Path $env:TEMP 'qwen3-coder-30b.import.Modelfile'

Write-Host ''
Write-Host 'Qwen3 Coder 30B wird in Ollama importiert...' -ForegroundColor Cyan
Write-Host ''

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw 'Ollama wurde auf diesem Windows-PC nicht gefunden. Bitte Ollama zuerst installieren und einmal starten.'
}

if (-not (Test-Path $blobPath -PathType Leaf)) {
    throw "Die Modelldatei fehlt: $blobPath"
}

if (-not (Test-Path $templatePath -PathType Leaf)) {
    throw "Die Modelfile-Vorlage fehlt: $templatePath"
}

$blobInfo = Get-Item $blobPath
if ($blobInfo.Length -ne $expectedBytes) {
    throw "Die Modelldatei hat nicht die erwartete Groesse. Erwartet: $expectedBytes Bytes, gefunden: $($blobInfo.Length) Bytes."
}

try {
    & ollama list | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw 'Ollama antwortet gerade nicht.'
    }
} catch {
    throw 'Ollama ist installiert, aber gerade nicht erreichbar. Bitte die Ollama-App unter Windows starten und dieses Script erneut ausfuehren.'
}

$blobPathForModelfile = (Resolve-Path $blobPath).Path
$modelfileContent = Get-Content $templatePath -Raw
$modelfileContent = $modelfileContent.Replace('__MODEL_BLOB_PATH__', $blobPathForModelfile)
Set-Content -Path $tempModelfile -Value $modelfileContent -Encoding Ascii

try {
    Write-Host "Importiere $modelName. Das kann ein paar Minuten dauern..." -ForegroundColor Yellow
    & ollama create $modelName -f $tempModelfile
    if ($LASTEXITCODE -ne 0) {
        throw 'ollama create ist fehlgeschlagen.'
    }

    & ollama show $modelName | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw 'Das Modell wurde nicht korrekt registriert.'
    }

    Write-Host ''
    Write-Host 'Import erfolgreich.' -ForegroundColor Green
    Write-Host 'Testen kannst du danach zum Beispiel mit:' -ForegroundColor Green
    Write-Host '  ollama run qwen3-coder:30b' -ForegroundColor Green
    Write-Host ''
} finally {
    if (Test-Path $tempModelfile) {
        Remove-Item $tempModelfile -Force
    }
}
