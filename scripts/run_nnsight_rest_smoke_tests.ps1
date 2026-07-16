param(
    [string]$Python = "python",
    [string]$LogDirectory = "results/_nnsight_rest_smoke/logs"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Logs = Join-Path $RepoRoot $LogDirectory
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$Tests = @(
    @{ Name = "E3"; Path = "tests/nnsight_e3_smoke.py" },
    @{ Name = "E4"; Path = "tests/nnsight_e4_smoke.py" },
    @{ Name = "E5"; Path = "tests/nnsight_e5_smoke.py" }
)

$OriginalLocation = Get-Location
try {
    Set-Location $RepoRoot
    foreach ($Test in $Tests) {
        $Log = Join-Path $Logs ("{0}_smoke.log" -f $Test.Name.ToLowerInvariant())
        Write-Host ("==== Running {0} offline NNsight smoke test ====" -f $Test.Name)
        & $Python $Test.Path 2>&1 | Tee-Object -FilePath $Log
        $Code = $LASTEXITCODE
        if ($Code -ne 0) {
            throw ("{0} smoke test failed with exit code {1}. Log: {2}" -f $Test.Name, $Code, $Log)
        }
    }
    Write-Host "==== All E3/E4/E5 offline NNsight smoke tests passed ===="
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Set-Location $OriginalLocation
}
exit 0
