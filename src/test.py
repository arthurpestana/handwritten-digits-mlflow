import argparse

import mlflow
import torch
import yaml
from sklearn.metrics import classification_report
from torch import nn

from src.config import TRACKING_URI
from src.data import carregar, particionar
from src.features import escalar
from src.train import figura_confusao, metricas


def executar(run_id):
    mlflow.set_tracking_uri(TRACKING_URI)
    if "test_acc" in mlflow.get_run(run_id).data.metrics:
        raise SystemExit(f"a run {run_id} já foi testada; o teste deve ser usado uma única vez")

    cfg = yaml.safe_load(mlflow.artifacts.load_text(f"runs:/{run_id}/config.yaml"))
    # pyfunc carrega o modelo salvo na run sem depender da classe MLP deste código
    modelo = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

    with mlflow.start_run(run_id=run_id), mlflow.start_span("testar") as span:
        # Refaça split e scaler com a mesma config: o teste é o mesmo conjunto reservado no treino
        X, y, _ = carregar()
        X_tr, X_va, X_te, *_, y_te = particionar(X, y, cfg)
        _, (_, _, Xt_te) = escalar(X, X_tr, X_va, X_te, cfg["data"]["leak_scaler"])

        logits = torch.tensor(modelo.predict(Xt_te.numpy()))
        res, pred = metricas(logits, torch.tensor(y_te), nn.CrossEntropyLoss())
        teste = {f"test_{k}": v for k, v in res.items()}
        mlflow.log_metrics(teste)
        mlflow.log_figure(figura_confusao(y_te, pred, "Matriz de confusão (teste)"), "confusao_teste.png")
        mlflow.log_text(classification_report(y_te, pred, digits=4), "classification_report_teste.txt")
        mlflow.set_tag("selecionado", "true")
        span.set_inputs({"run_id": run_id, "n_test": len(y_te)})
        span.set_outputs({k: round(v, 4) for k, v in teste.items()})

    print(f"{cfg['run_name']} no teste: " + " ".join(f"{k}={v:.4f}" for k, v in teste.items()))
    print(classification_report(y_te, pred, digits=4))


def main():
    parser = argparse.ArgumentParser(description="Avalia no teste, uma única vez, o modelo de uma run escolhida.")
    parser.add_argument("--run-id", required=True)
    executar(parser.parse_args().run_id)


if __name__ == "__main__":
    main()
