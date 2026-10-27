$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
conda activate fs437-local
if ($env:CONDA_DEFAULT_ENV -ne "fs437-local") {
    throw "Environment activation failed. Open your existing conda PowerShell terminal."
}
python .\run_project.py
if ($LASTEXITCODE -ne 0) {
    throw "Project run failed. Send the terminal error."
}
