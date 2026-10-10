import os

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

CAMINHO_LOCAL = "data/digits.csv"
N_CLASSES = 10


def carregar():
    # Cópia versionada no repositório: a run aponta para um arquivo fixo, não para a versão do sklearn
    if os.path.exists(CAMINHO_LOCAL):
        dados = np.loadtxt(CAMINHO_LOCAL, delimiter=",", skiprows=1)
        return dados[:, :-1], dados[:, -1].astype(int), f"arquivo local ({CAMINHO_LOCAL})"
    X, y = load_digits(return_X_y=True)
    os.makedirs(os.path.dirname(CAMINHO_LOCAL), exist_ok=True)
    cabecalho = ",".join([f"pixel_{i}" for i in range(X.shape[1])] + ["digito"])
    np.savetxt(CAMINHO_LOCAL, np.column_stack([X, y]), delimiter=",", fmt="%d", header=cabecalho, comments="")
    return X, y, "sklearn.datasets.load_digits"


def particionar(X, y, cfg):
    d, seed = cfg["data"], cfg["seed"]
    resto = d["val_size"] + d["test_size"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=resto, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=d["test_size"] / resto, stratify=y_tmp, random_state=seed
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def distribuicao_classes(y, partes):
    geral = np.bincount(y, minlength=N_CLASSES) / len(y)
    dist = {"dataset": geral.round(4).tolist()}
    desvio = 0.0
    for nome, y_parte in partes.items():
        p = np.bincount(y_parte, minlength=N_CLASSES) / len(y_parte)
        dist[nome] = p.round(4).tolist()
        desvio = max(desvio, float(np.abs(p - geral).max()))
    return dist, desvio
