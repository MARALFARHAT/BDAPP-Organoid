$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
conda activate fs437-local
if ($env:CONDA_DEFAULT_ENV -ne "fs437-local") {
    throw "Could not activate the fs437-local conda environment."
}
python .\run_project.py
if ($LASTEXITCODE -ne 0) {
    throw "Project run failed."
}
