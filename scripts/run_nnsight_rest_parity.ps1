param(
    [string]$Python = "python",
    [string]$LogDirectory = "results/_nnsight_rest_parity/logs",
    [string]$OutputDirectory = "results/_nnsight_rest_parity"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Logs = Join-Path $RepoRoot $LogDirectory
New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$Checks = @(
    @{
        Name = "E3"
        Arguments = @(
            "emergence_dynamics/emergence_dynamics_analysis.py",
            "--verify-parity", "--engine", "nnsight", "--model-name", "gpt2",
            "--mode", "all", "--sample-size", "5", "--cut-length", "16",
            "--output-dir", $OutputDirectory, "--experiment-name", "e3_gpt2_parity"
        )
    },
    @{
        Name = "E4"
        Arguments = @(
            "common/residual_sink_analysis.py",
            "--verify-parity", "--engine", "nnsight", "--model-name", "gpt2",
            "--seeds", "0", "--sample-size", "5", "--cut-length", "16",
            "--alphas", "0,1", "--output-dir", $OutputDirectory,
            "--experiment-name", "e4_gpt2_parity"
        )
    },
    @{
        Name = "E5"
        Arguments = @(
            "evaluation_robustness/evaluation_robustness_analysis.py",
            "--verify-parity", "--engine", "nnsight", "--model-name", "gpt2",
            "--seeds", "0", "--sample-size", "5", "--cut-length", "16",
            "--lengths", "8,16", "--alphas", "0,1", "--bootstrap-repetitions", "20",
            "--output-dir", $OutputDirectory, "--experiment-name", "e5_gpt2_parity"
        )
    }
)

$OriginalLocation = Get-Location
try {
    Set-Location $RepoRoot
    foreach ($Check in $Checks) {
        $Log = Join-Path $Logs ("{0}_parity.log" -f $Check.Name.ToLowerInvariant())
        Write-Host ("==== Running {0} GPT-2 parity/fidelity check ====" -f $Check.Name)
        $Arguments = $Check.Arguments
        & $Python @Arguments 2>&1 | Tee-Object -FilePath $Log
        $Code = $LASTEXITCODE
        if ($Code -ne 0) {
            throw ("{0} parity check failed with exit code {1}. Log: {2}" -f $Check.Name, $Code, $Log)
        }
    }
    Write-Host "==== All E3/E4/E5 parity commands completed ===="
}
catch {
    Write-Error $_
    exit 1
}
finally {
    Set-Location $OriginalLocation
}
exit 0
