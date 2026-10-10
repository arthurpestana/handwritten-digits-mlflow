import os

import yaml

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def carregar_config(caminho, sobrescritas=()):
    with open(caminho, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in sobrescritas:
        if "=" not in item:
            raise ValueError(f"--set inválido: '{item}'. Use chave.sub=valor")
        chave, texto = item.split("=", 1)
        *caminho_chave, folha = chave.split(".")
        alvo = cfg
        for parte in caminho_chave:
            alvo = alvo[parte]
        alvo[folha] = _numero(yaml.safe_load(texto))
    return cfg


def _numero(valor):
    # PyYAML lê "1e-5" como string; sem isso o Adam quebra ao comparar weight_decay
    if isinstance(valor, str):
        try:
            return float(valor)
        except ValueError:
            pass
    return valor


def achatar(cfg, prefixo=""):
    plano = {}
    for chave, valor in cfg.items():
        if isinstance(valor, dict):
            plano.update(achatar(valor, f"{prefixo}{chave}."))
        else:
            plano[f"{prefixo}{chave}"] = valor
    return plano
