# Records the missing bake-off clips with the soundcore headset microphone (resumable).
# Turn the headset on first and wait until it is connected. Then: .\scripts\record-headset.ps1
param([string]$Name = 'soundcore', [string]$Out = 'samples\headset')
Set-Location (Join-Path $PSScriptRoot '..')
$py = '.\.venv\Scripts\python.exe'
$finder = @'
import sys
import sounddevice as sd
name = sys.argv[1].lower()
for i, d in enumerate(sd.query_devices()):
    api = sd.query_hostapis(d["hostapi"])["name"]
    if d["max_input_channels"] > 0 and name in d["name"].lower() and api != "Windows WDM-KS":
        print(i)
        break
'@
$idx = ($finder | & $py - $Name | Select-Object -First 1)
if (-not $idx) {
    Write-Host "No usable '$Name' microphone found. Turn the headset on, wait for it to connect, then run this script again."
    Write-Host "(Entries named Hands-Free under WDM-KS cannot be used. Check: .\.venv\Scripts\python -m stt devices)"
    exit 1
}
Write-Host "Recording with input device $idx"
& $py -m stt.bench.record --out $Out --device $idx
exit $LASTEXITCODE
