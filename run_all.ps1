# Uso: .\run_all.ps1  (com a .venv ativada, na raiz do projeto)
$ErrorActionPreference = "Stop"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
foreach ($cfg in "A_referencia", "B_taxa_menor", "C_l2", "D_vazamento", "E_dropout") {
    Write-Host "=== $cfg ==="
    python -m src.train --config "configs/$cfg.yaml"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
python -m src.comparar
