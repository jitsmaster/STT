param([string]$Config = (Join-Path $PSScriptRoot '..\config.toml'))
Set-Location (Join-Path $PSScriptRoot '..')
& .\.venv\Scripts\python.exe -m stt serve --config $Config
exit $LASTEXITCODE
