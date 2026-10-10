import torch
from sklearn.preprocessing import StandardScaler


def escalar(X, X_tr, X_va, X_te, vazamento):
    # vazamento=True reproduz o bug do script original: média e desvio calculados com val e teste
    scaler = StandardScaler().fit(X if vazamento else X_tr)
    return scaler, [torch.tensor(scaler.transform(p), dtype=torch.float32) for p in (X_tr, X_va, X_te)]


def resumo_scaler(scaler, vazamento):
    return {
        "scaler": "StandardScaler",
        "ajustado_em": "dataset inteiro (VAZAMENTO)" if vazamento else "somente treino",
        "media": scaler.mean_.round(4).tolist(),
        "desvio": scaler.scale_.round(4).tolist(),
    }
