#!/usr/bin/env bash
# Uso: bash run_all.sh  (com a .venv ativada, na raiz do projeto)
set -e
export MLFLOW_DISABLE_AGENT_HINT=1
for cfg in A_referencia B_taxa_menor C_l2 D_vazamento E_dropout; do
    echo "=== $cfg ==="
    python -m src.train --config "configs/$cfg.yaml"
done
python -m src.comparar
