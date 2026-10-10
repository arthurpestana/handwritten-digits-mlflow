from torch import nn


class MLP(nn.Module):
    def __init__(self, n_entradas, ocultas, n_classes, dropout=0.0):
        super().__init__()
        camadas, anterior = [], n_entradas
        for n in ocultas:
            camadas += [nn.Linear(anterior, n), nn.ReLU(), nn.Dropout(dropout)]
            anterior = n
        # Sem softmax: a CrossEntropyLoss espera logits e aplica log-softmax internamente
        camadas.append(nn.Linear(anterior, n_classes))
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)
