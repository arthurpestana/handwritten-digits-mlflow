import mlflow
import pandas as pd

from src.config import TRACKING_URI

COLUNAS = {
    "tags.mlflow.runName": "run",
    "tags.changed_param": "muda",
    "metrics.best_epoch": "melhor_epoca",
    "metrics.best_val_loss": "best_val_loss",
    "metrics.best_val_acc": "best_val_acc",
    "metrics.best_val_f1_macro": "best_val_f1",
    "metrics.final_gap_loss": "gap_final",
    "metrics.test_acc": "test_acc",
}


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(search_all_experiments=True, filter_string="attributes.status = 'FINISHED'")
    if runs.empty:
        raise SystemExit("nenhuma run finalizada; rode antes: python -m src.train --config configs/A_referencia.yaml")
    tabela = runs[[c for c in COLUNAS if c in runs.columns]].rename(columns=COLUNAS).sort_values("run")
    pd.set_option("display.width", 200)
    print(tabela.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
