# Pipeline de treino e avaliação com MLflow — Handwritten Digits

**Disciplina:** Inteligência Artificial 2026/2 — Sistemas de Informação / Unitins
**Avaliação:** A1 — Treinamento observável: como rastrear, comparar e reproduzir modelos
**Aluno:** Arthur Henrique
**Vídeo:** _(link do YouTube)_

MLP em PyTorch para classificar dígitos manuscritos (0–9), com treino, validação e teste rastreados no MLflow: parâmetros, métricas por época, artefatos, modelo e trace das etapas do pipeline.

## 1. Código de partida

O ponto de partida é o [`mlp_torch_avaliacao.py`](https://github.com/sousamaf/AI-Lab/blob/main/algorithms/neural_networks/mlp/mlp_torch_avaliacao.py) da disciplina (MLP `4→16→3` para o Iris). O [`mlp_torch_regularizacao_L2.py`](https://github.com/sousamaf/AI-Lab/blob/main/algorithms/neural_networks/mlp/mlp_torch_regularizacao_L2.py) difere dele em **uma linha** (`weight_decay=1e-5` no Adam). Aqui essa linha virou o parâmetro `train.weight_decay` da configuração: a **Run A** reproduz o primeiro script e a **Run C**, o segundo.

Mantido do original: `nn.Module` com `Linear → ReLU → Linear`, `CrossEntropyLoss`, Adam, o loop manual de treino/validação, F1 macro e a matriz de confusão.

| Script da aula | Este projeto | Por quê |
|---|---|---|
| Iris (150 amostras, 4 atributos, 3 classes) | Digits (1.797 amostras, 64 atributos, 10 classes) | Dataset atribuído; o Iris é pequeno demais para diferenciar configurações |
| `StandardScaler` ajustado em **todo** o dataset antes do split | Ajustado **só no treino** (`src/features.py`) | Corrige o vazamento de dados |
| Split sem `stratify`, sem `torch.manual_seed` | Split estratificado + sementes fixas | Reprodutibilidade (verificada: mesma config, mesmos números) |
| Full-batch, 1000 épocas | Mini-batches de 64 (`DataLoader`), 100 épocas | Treino padrão em lotes; 100 épocas bastam para o treino convergir |
| Teste com o modelo da **última** época, em toda execução | Teste com a **melhor** época, uma única vez, só na run escolhida | Evita escolher pelo teste |
| Hiperparâmetros no código | YAML por run + `--set` na linha de comando | Repetir sem editar o código |
| `plt.show()` e `print` | Figuras, métricas e modelo registrados no MLflow | Comparar e recuperar depois |

## 2. Dataset

[Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) (UCI, id 80). Na primeira execução, ele é obtido de `sklearn.datasets.load_digits()` (a cópia do conjunto de teste da UCI) e salvo em `data/digits.csv`, que fica versionado no repositório. As execuções seguintes leem esse arquivo, e a origem fica registrada na tag `origem_dados` de cada run.

Cada dígito foi escaneado em 32×32 pixels e dividido em blocos 4×4. Cada atributo é a **contagem de pixels acesos em um bloco** (0 a 16). São 64 atributos numéricos tabulares, não a imagem bruta, e por isso o problema cabe numa MLP. As 10 classes são balanceadas (≈180 amostras cada).

Split estratificado 70/15/15 com `seed=42`: **1.257 treino / 270 validação / 270 teste**.

## 3. Experimento e hipóteses

Um experimento no MLflow (`a1-digits-mlp`) com cinco runs. Arquitetura (`64→128→64→10`, 17.226 parâmetros), split, semente, batch e épocas são **iguais em todas**. Cada run muda **um único** parâmetro em relação à A.

| Run | Muda | Pergunta / hipótese |
|---|---|---|
| A_referencia | — (baseline: lr 0.01, sem regularização) | Sem regularização, a `val_loss` passa a subir enquanto a `train_loss` cai? |
| B_taxa_menor | `train.lr` 0.01 → 0.001 | Uma taxa 10× menor deixa a curva mais estável ou só mais lenta? |
| C_l2 | `train.weight_decay` 0 → 1e-5 | L2 reduz o gap entre `train_loss` e `val_loss` sem derrubar a `val_acc`? |
| D_vazamento | `data.leak_scaler` false → true | O vazamento do script original deixa a validação otimista? |
| E_dropout | `model.dropout` 0 → 0.3 | Dropout reduz o overfitting da A? Como se compara ao L2? |

Descrição, pergunta e hipótese ficam registradas em cada run (descrição + tags `changed_param` e `hypothesis`).

## 4. Estrutura

```
handwritten-digits-mlflow/
├── configs/                 # um YAML por run (A_referencia, B_taxa_menor, C_l2, D_vazamento, E_dropout)
├── data/
│   └── digits.csv           # cópia local do dataset (gerada na 1ª execução)
├── src/
│   ├── config.py            # lê o YAML e aplica --set chave.sub=valor
│   ├── data.py              # carrega o dataset, divide (estratificado) e mede o balanceamento do split
│   ├── features.py          # StandardScaler ajustado SÓ no treino
│   ├── model.py             # classe MLP (camadas ocultas e dropout configuráveis)
│   ├── train.py             # pipeline principal + MLflow (ponto de entrada)
│   ├── test.py              # avalia no teste, uma única vez, a run escolhida
│   └── comparar.py          # tabela comparativa das runs no terminal
├── run_all.ps1 / run_all.sh # roda todas as configs em sequência
├── MLproject                # permite rodar via `mlflow run`
├── environment.yml          # ambiente conda (alternativa à venv)
├── requirements.txt
└── README.md
```

Gerados na execução e fora do Git: `mlflow.db` (banco SQLite do MLflow) e `mlruns/` (artefatos).

## 5. Como reproduzir

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

.\run_all.ps1                                          # roda as 5 configs e imprime a tabela
mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://127.0.0.1:5000
python -m src.test --run-id <run_id_escolhido>         # teste final, só na run escolhida
```

Sempre a partir da raiz do projeto. Uma run avulsa:

```powershell
python -m src.train --config configs/C_l2.yaml
python -m src.train --config configs/A_referencia.yaml --set train.lr=0.005 --set run_name=A_lr0005
python -m src.comparar
```

Via MLflow Projects (o nome do experimento precisa ser o mesmo da config):

```powershell
mlflow run . -P config=configs/B_taxa_menor.yaml --env-manager local --experiment-name a1-digits-mlp
```

## 6. O que cada run registra

| Tipo | Conteúdo |
|---|---|
| Descrição e tags | pergunta/hipótese, `changed_param`, `hypothesis`, `comando` (a linha exata para reexecutar), `origem_dados`; após o teste, `selecionado=true` |
| Parâmetros | todos os campos da config (`train.lr`, `train.weight_decay`, `model.hidden`, `model.dropout`...), `n_train`/`n_val`/`n_test`, `n_parametros`, versão do torch, device |
| Métricas por época (`step` = época) | `train_loss`, `val_loss`, `train_acc`, `val_acc` |
| Métricas finais | `best_epoch`, `best_val_loss`, `best_val_acc`, `best_val_f1_macro`, `best_gap_loss`, `final_gap_loss`, `split_desvio_max_classe`; na run testada, `test_loss`, `test_acc`, `test_f1_macro` |
| Artefatos | `config.yaml` (configuração efetiva, já com os `--set`), `split_distribuicao.json`, `preprocessamento.json` (média/desvio do scaler e em que dados foi ajustado), `curvas.png`, `confusao_validacao.png`, `relatorio.md`, o modelo (`mlflow.pytorch.log_model`); na run testada, `confusao_teste.png` e `classification_report_teste.txt` |
| Trace | span raiz `pipeline` com os filhos `carregar → particionar → preparar → treinar → validar → registrar`, cada um com entradas, saídas e duração; o `src.test` gera um trace `testar` na mesma run |

O span `particionar` mede a **qualidade do split**: compara a proporção de cada classe em treino, validação e teste com a do dataset inteiro e falha a run se o desvio passar de 2 p.p. Neste split o desvio máximo foi de **0,24 p.p.**

A melhor época é escolhida pela menor `val_loss`, e os pesos dela são restaurados antes de salvar o modelo. Todas as runs treinam as mesmas 100 épocas, para que as curvas sejam comparáveis.

## 7. Resultados (seed 42)

| Run | Melhor época | `best_val_loss` | `best_val_acc` | `best_val_f1_macro` | `val_loss` final | Gap final (val − train) |
|---|---|---|---|---|---|---|
| A_referencia | 5 | 0,1530 | 0,9519 | 0,9516 | 0,4288 | 0,4288 |
| B_taxa_menor | 13 | 0,1452 | 0,9630 | 0,9626 | 0,1799 | 0,1797 |
| **C_l2** | **92** | 0,1278 | **0,9778** | **0,9776** | **0,1287** | **0,1287** |
| D_vazamento | 5 | 0,1980 | 0,9630 | 0,9626 | 0,2558 | 0,2558 |
| E_dropout | 13 | **0,1046** | **0,9778** | 0,9775 | 0,7489 | 0,7443 |

**Leitura das curvas** (artefato `curvas.png` de cada run, ou Compare no MLflow):

- **A:** a `train_loss` chega a zero em ~20 épocas (100% de acurácia no treino), e a `val_loss` atinge o mínimo na época 5. Depois sobe, com pico de 0,65 na época 19, e estaciona em 0,43. É overfitting clássico: a rede decorou o treino. Os picos também mostram instabilidade com lr 0,01.
- **B:** com lr 10× menor, a curva de validação fica **suave** e a melhor época vem mais tarde (13). Depois do mínimo, a `val_loss` sobe devagar (0,145 → 0,180). Hipótese confirmada: mais estável e mais lenta, com overfitting bem mais brando.
- **C:** a `train_loss` também zera por volta da época 10, mas a `val_loss` **continua caindo** até a época 92 (0,16 → 0,128). Quando o gradiente dos dados some, o termo de L2 passa a dominar a atualização do Adam e encolhe os pesos aos poucos, o que reduz o excesso de confiança da rede. Hipótese confirmada: gap final de 0,13 contra 0,43 da A.
- **D:** ver o diagnóstico na seção 9.
- **E:** tem o **melhor checkpoint isolado** (0,105 na época 13), mas depois a `val_loss` oscila muito (0,15 → 0,86 → 0,50 → 0,75). Com lr 0,01, o ruído do dropout desestabiliza o treino. Hipótese **não** confirmada: o gap final (0,74) é maior que o da A.

## 8. Escolha do modelo e teste

**Escolhida: C_l2.** A escolha foi feita pela validação, antes de olhar o teste.

A E tem `best_val_loss` menor, mas a mesma `val_acc` (0,9778), e o melhor ponto dela é uma queda isolada numa curva instável. Para checar se isso era sinal ou sorte, rodei as configurações com 5 sementes diferentes (mesmo split, verificação exploratória fora do MLflow):

| Config | `best_val_loss` | `val_acc` na melhor época | `val_loss` final |
|---|---|---|---|
| A | 0,160 ± 0,028 | 0,970 ± 0,007 | 0,221 ± 0,039 |
| B | 0,144 ± 0,015 | 0,967 ± 0,004 | 0,173 ± 0,021 |
| **C** | **0,137 ± 0,017** | **0,974 ± 0,005** | 0,196 ± 0,083 |
| D | 0,196 ± 0,032 | 0,964 ± 0,009 | 0,318 ± 0,163 |
| E | 0,174 ± 0,022 | 0,956 ± 0,010 | 1,070 ± 0,452 |

A C tem o melhor resultado médio; a E, que parecia a melhor na semente 42, é a segunda pior em média e a mais instável. Por isso a escolha ficou com a C, que tem desempenho consistente e curva estável, e não com o melhor número isolado.

Teste, rodado **uma única vez** com `python -m src.test --run-id <run_id da C>`:

| `test_loss` | `test_acc` | `test_f1_macro` |
|---|---|---|
| 0,0367 | **0,9926** | 0,9926 |

São 2 erros em 270 amostras, dois dígitos 8 classificados como 1. Os demais dígitos têm precisão e revocação de 100% (`classification_report_teste.txt`). O `src.test` recusa testar de novo uma run que já tem `test_acc`.

**Ressalvas:**
- Validação e teste têm 270 amostras cada, então **1 amostra ≈ 0,37 p.p.** de acurácia. O teste ficou acima da validação (99,3% contra 97,8%) por variação de amostra, não porque o modelo "melhorou".
- A vantagem da C sobre B é pequena (as faixas se sobrepõem). Registrar a varredura de sementes como um experimento no MLflow é o próximo passo natural.

## 9. Diagnóstico de incidentes

### 9.1 O vazamento que **piorou** a validação (Run D)

- **Hipótese:** o scaler ajustado em todo o dataset, como no script original, deixaria a validação otimista.
- **Sintoma:** aconteceu o contrário. A D teve `best_val_loss` de 0,198 contra 0,153 da A, e o padrão se repetiu nas 5 sementes.
- **Investigação:** o artefato `preprocessamento.json` das duas runs mostra a diferença. O Digits tem colunas de pixel **zeradas em todo o treino** e acesas em raras amostras de validação. A coluna 56, por exemplo, tem 0 em todo o treino e 1 pixel aceso na validação.
  - Ajustado só no treino, o scaler vê desvio zero, usa escala 1, e o pixel aceso vale 1.
  - Ajustado no dataset inteiro, o scaler vê esse pixel raro, calcula um desvio de 0,0236 e o transforma em **≈42 desvios-padrão**, um valor que a rede nunca viu no treino.
- **Conclusão:** o vazamento existe, e neste dataset ele **distorce** a validação em vez de inflá-la. Por isso o resultado de uma run com vazamento não é confiável em nenhuma direção.

### 9.2 Falha no teste localizada pelo trace

A primeira execução do teste na run C falhou, e o MLflow marcou a run como `FAILED`. Na aba **Traces** da run, o span `testar` aparece com status `ERROR` e a exceção `NotImplementedError: Calling eval() is not supported yet`.

- **Hipótese:** o MLflow 3.17 salva modelos PyTorch por padrão no formato **pt2** (`torch.export`), que não aceita `model.eval()`.
- **Correção:** o teste passou a carregar o modelo pela interface genérica `mlflow.pyfunc.load_model(...).predict(...)`. Ela não depende da classe `MLP` do código e é a forma como um sistema externo usaria o modelo.
- Na segunda tentativa, outro trace `ERROR` mostrou um `UnboundLocalError`: uma variável sobrescrevia o nome da função `metricas`. Corrigido, o teste rodou com sucesso, e a run voltou a `FINISHED` com a tag `selecionado=true`.

Os traces com erro foram mantidos na run como evidência. Nenhuma das tentativas com falha registrou métricas de teste.

> As runs A–D foram executadas antes da reorganização do código em `src/` (scripts `train.py`/`test.py` na raiz, mesma lógica). Depois da reorganização, A e C foram reexecutadas num banco temporário com `python -m src.train`, e os números foram idênticos.

## 10. Limitações e próximos passos

- Uma semente por configuração no MLflow (a varredura de sementes da seção 8 foi exploratória).
- O split é aleatório entre amostras, mas o mesmo escritor pode aparecer no treino e no teste. A acurácia mede generalização para **novos dígitos**, não necessariamente para **novos escritores**. A divisão original da UCI separa escritores (30 no treino, 13 no teste) e seria o teste mais honesto.
- 100 épocas fixas: a C ainda melhorava levemente na época 92. Uma varredura de `weight_decay` (1e-4, 1e-3) e dropout com lr menor (0,001) são as próximas perguntas.
- As colunas de pixel raras geram valores extremos após a padronização mesmo sem vazamento (até ~50 desvios na validação). Escalar por `/16` (MinMax) é uma hipótese a testar.
- Para imagens brutas, uma CNN seria a escolha natural (Aula 11). Aqui a entrada já são atributos agregados, e a MLP funciona como baseline.

## 11. Referências

- Script original: `sousamaf/AI-Lab` — `algorithms/neural_networks/mlp/mlp_torch_avaliacao.py` e `mlp_torch_regularizacao_L2.py`
- Alpaydin, E.; Kaynak, C. *Optical Recognition of Handwritten Digits*. UCI Machine Learning Repository, 1998
- MLflow Tracking e Tracing: <https://mlflow.org/docs/latest/>
- PyTorch: <https://pytorch.org/docs/stable/>
- Organização de pastas baseada em [rafaelsdiniz/credit-approval-mlflow](https://github.com/rafaelsdiniz/credit-approval-mlflow)
