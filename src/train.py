import argparse
import copy
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import TRACKING_URI, achatar, carregar_config
from src.data import N_CLASSES, carregar, distribuicao_classes, particionar
from src.features import escalar, resumo_scaler
from src.model import MLP

# Split estratificado fica abaixo de 1 p.p.; acima de 2 p.p. algo quebrou na divisão
LIMITE_DESVIO_CLASSE = 0.02


def fixar_semente(seed):
    # Não fixe o random do Python: o MLflow gera os IDs de trace com ele e as runs colidiriam
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def metricas(logits, y, criterio):
    pred = logits.argmax(dim=1)
    return {
        "loss": criterio(logits, y).item(),
        "acc": (pred == y).float().mean().item(),
        "f1_macro": f1_score(y.numpy(), pred.numpy(), average="macro"),
    }, pred


def avaliar(modelo, X, y, criterio):
    modelo.eval()
    with torch.no_grad():
        return metricas(modelo(X), y, criterio)


def figura_curvas(hist, melhor_epoca):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    epocas = range(1, len(hist["train_loss"]) + 1)
    for ax, chave, titulo in ((ax1, "loss", "Perda (cross-entropy)"), (ax2, "acc", "Acurácia")):
        ax.plot(epocas, hist[f"train_{chave}"], label="treino")
        ax.plot(epocas, hist[f"val_{chave}"], label="validação")
        ax.axvline(melhor_epoca, color="gray", linestyle="--", label=f"melhor época ({melhor_epoca})")
        ax.set(xlabel="época", title=titulo)
        ax.legend()
    fig.tight_layout()
    return fig


def figura_confusao(y, pred, titulo):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax, colorbar=False)
    ax.set_title(titulo)
    fig.tight_layout()
    return fig


def relatorio(cfg, comando, melhor, artefatos):
    d = cfg["data"]
    return "\n".join([
        f"# {cfg['run_name']}",
        "",
        cfg["description"].strip(),
        "",
        "## Pipeline executado",
        "carregar → particionar → preparar → treinar → validar → registrar",
        "",
        f"## Métricas na melhor época ({melhor['epoca']})",
        *(f"- {k}: {v:.4f}" for k, v in melhor.items() if k != "epoca"),
        "",
        "## Artefatos",
        *(f"- {a}" for a in artefatos),
        "",
        "## Reprodutibilidade",
        f"- seed={cfg['seed']}, split estratificado "
        f"{1 - d['val_size'] - d['test_size']:.0%}/{d['val_size']:.0%}/{d['test_size']:.0%}",
        f"- python {platform.python_version()}, torch {torch.__version__}, mlflow {mlflow.__version__}, device cpu",
        f"- reexecutar: {comando}",
    ])


def executar(cfg, comando):
    cm, ct, cd = cfg["model"], cfg["train"], cfg["data"]
    fixar_semente(cfg["seed"])

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(cfg["experiment"])

    with mlflow.start_run(run_name=cfg["run_name"], description=cfg["description"], tags=cfg["tags"]) as run:
        mlflow.log_params({
            **achatar({k: cfg[k] for k in ("seed", "data", "model", "train")}),
            "torch_version": torch.__version__,
            "device": "cpu",
        })
        mlflow.log_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), "config.yaml")
        mlflow.set_tag("comando", comando)

        with mlflow.start_span("pipeline") as raiz:
            raiz.set_inputs({"comando": comando})

            with mlflow.start_span("carregar") as span:
                X, y, origem = carregar()
                mlflow.set_tag("origem_dados", origem)
                span.set_outputs({"origem": origem, "amostras": len(X), "atributos": X.shape[1], "classes": N_CLASSES})

            with mlflow.start_span("particionar") as span:
                X_tr, X_va, X_te, y_tr, y_va, y_te = particionar(X, y, cfg)
                dist, desvio = distribuicao_classes(y, {"treino": y_tr, "validacao": y_va, "teste": y_te})
                tamanhos = {"n_train": len(y_tr), "n_val": len(y_va), "n_test": len(y_te)}
                mlflow.log_params(tamanhos)
                mlflow.log_metric("split_desvio_max_classe", desvio)
                mlflow.log_dict(dist, "split_distribuicao.json")
                span.set_outputs({**tamanhos, "desvio_max_classe": round(desvio, 4)})
                if desvio > LIMITE_DESVIO_CLASSE:
                    raise ValueError(f"split desbalanceado: desvio máximo de classe {desvio:.3f}")

            with mlflow.start_span("preparar") as span:
                scaler, (Xt_tr, Xt_va, _) = escalar(X, X_tr, X_va, X_te, cd["leak_scaler"])
                yt_tr, yt_va = torch.tensor(y_tr), torch.tensor(y_va)
                resumo = resumo_scaler(scaler, cd["leak_scaler"])
                mlflow.log_dict(resumo, "preprocessamento.json")
                span.set_outputs({"scaler_ajustado_em": resumo["ajustado_em"]})

            with mlflow.start_span("treinar") as span:
                span.set_inputs({**cm, **ct})
                modelo = MLP(X.shape[1], cm["hidden"], N_CLASSES, cm["dropout"])
                criterio = nn.CrossEntropyLoss()
                otimizador = torch.optim.Adam(modelo.parameters(), lr=ct["lr"], weight_decay=ct["weight_decay"])
                loader = DataLoader(
                    TensorDataset(Xt_tr, yt_tr), batch_size=ct["batch_size"], shuffle=True,
                    generator=torch.Generator().manual_seed(cfg["seed"]),
                )
                mlflow.log_param("n_parametros", sum(p.numel() for p in modelo.parameters()))

                hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
                melhor = {"val_loss": float("inf")}
                for epoca in range(1, ct["epochs"] + 1):
                    modelo.train()
                    for xb, yb in loader:
                        otimizador.zero_grad()
                        criterio(modelo(xb), yb).backward()
                        otimizador.step()

                    tr, _ = avaliar(modelo, Xt_tr, yt_tr, criterio)
                    va, _ = avaliar(modelo, Xt_va, yt_va, criterio)
                    por_epoca = {"train_loss": tr["loss"], "val_loss": va["loss"], "train_acc": tr["acc"], "val_acc": va["acc"]}
                    for k, v in por_epoca.items():
                        hist[k].append(v)
                    mlflow.log_metrics(por_epoca, step=epoca)

                    # Guarde a melhor época, não a última: é ela que se quer recuperar depois
                    if va["loss"] < melhor["val_loss"]:
                        melhor = {"epoca": epoca, **por_epoca, "val_f1_macro": va["f1_macro"]}
                        melhores_pesos = copy.deepcopy(modelo.state_dict())

                    if epoca % 10 == 0:
                        print(f"época {epoca:3d} | train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} val_acc={va['acc']:.4f}")
                span.set_outputs({"melhor_epoca": melhor["epoca"], "melhor_val_loss": round(melhor["val_loss"], 4)})

            with mlflow.start_span("validar") as span:
                modelo.load_state_dict(melhores_pesos)
                _, pred_va = avaliar(modelo, Xt_va, yt_va, criterio)
                final = {
                    "best_epoch": melhor["epoca"],
                    "best_train_loss": melhor["train_loss"],
                    "best_val_loss": melhor["val_loss"],
                    "best_val_acc": melhor["val_acc"],
                    "best_val_f1_macro": melhor["val_f1_macro"],
                    "best_gap_loss": melhor["val_loss"] - melhor["train_loss"],
                    "final_gap_loss": hist["val_loss"][-1] - hist["train_loss"][-1],
                }
                mlflow.log_metrics(final)
                span.set_outputs({k: round(v, 4) for k, v in final.items()})

            with mlflow.start_span("registrar") as span:
                mlflow.log_figure(figura_curvas(hist, melhor["epoca"]), "curvas.png")
                mlflow.log_figure(figura_confusao(y_va, pred_va, "Matriz de confusão (validação)"), "confusao_validacao.png")
                mlflow.pytorch.log_model(modelo, name="model", input_example=Xt_va[:5].numpy())
                artefatos = ["config.yaml", "split_distribuicao.json", "preprocessamento.json",
                             "curvas.png", "confusao_validacao.png", "model/", "relatorio.md"]
                mlflow.log_text(relatorio(cfg, comando, melhor, artefatos), "relatorio.md")
                span.set_outputs({"artefatos": artefatos})

            raiz.set_outputs({k: round(v, 4) for k, v in final.items()})

        print(f"\n{cfg['run_name']} | run_id={run.info.run_id}")
        print(f"melhor época {melhor['epoca']} | val_loss={melhor['val_loss']:.4f} "
              f"val_acc={melhor['val_acc']:.4f} val_f1_macro={melhor['val_f1_macro']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Treina a MLP do Digits com rastreamento no MLflow.")
    parser.add_argument("--config", required=True, help="YAML da run, ex.: configs/A_referencia.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR",
                        help="sobrescreve uma chave do YAML, ex.: --set train.lr=0.001 (pode repetir)")
    args = parser.parse_args()

    comando = " ".join(["python -m src.train --config", args.config, *(f"--set {s}" for s in args.set)])
    executar(carregar_config(args.config, args.set), comando)


if __name__ == "__main__":
    main()
