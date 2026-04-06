"""
ETAPA 2: Regressao Logistica + PCA + Grid Search
==================================================
O que voce vai aprender neste script:
  - O que e PCA (Analise de Componentes Principais) e por que reduzir dimensoes
  - O que e Grid Search e como encontrar os melhores hiperparametros automaticamente
  - O que e Cross-Validation e por que e melhor que um unico train/test split
  - Como combinar varias tecnicas em um Pipeline do scikit-learn

Por que usar PCA?
  - No baseline, temos 784 dimensoes (pixels). Muitas sao redundantes!
  - Pixels no fundo da imagem sao quase sempre pretos (nao ajudam na classificacao)
  - PCA encontra as "direcoes mais importantes" nos dados e descarta o resto
  - Analogia: e como RESUMIR um livro -- voce perde detalhes, mas mantem a essencia
  - Outra analogia: imagine projetar a sombra de um objeto 3D numa parede.
    Voce perde uma dimensao, mas ainda consegue reconhecer o objeto!

Leia tambem: docs/03_pca_e_reducao.md
"""

import time
import numpy as np
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# ---------------------------------------------------------------------------
# 1. CARREGAR E PREPARAR OS DADOS
# ---------------------------------------------------------------------------
# Mesmo processo da Etapa 1: carregar MNIST e achatar as imagens.
ds = load_dataset("ylecun/mnist", download_mode="reuse_dataset_if_exists")


def to_numpy(split):
    X = np.stack([np.array(img).reshape(-1) for img in split["image"]])
    y = np.array(split["label"])
    return X, y


X_train, y_train = to_numpy(ds["train"])
X_test, y_test = to_numpy(ds["test"])

print(f"Dados de treino: {X_train.shape}")  # (60000, 784)
print(f"Dados de teste:  {X_test.shape}")    # (10000, 784)


# ---------------------------------------------------------------------------
# 2. CRIAR O PIPELINE
# ---------------------------------------------------------------------------
# O Pipeline encadeia 3 etapas:
#
#   1. StandardScaler: normaliza os dados (mesma escala)
#
#   2. PCA: reduz de 784 dimensoes para N dimensoes (N sera definido pelo Grid Search)
#      PCA funciona assim: ele encontra as "direcoes" onde os dados mais variam.
#      Depois, projeta os dados nessas direcoes, descartando as menos importantes.
#      Se n_components=100, comprimimos de 784 numeros para apenas 100,
#      mantendo a maior parte da informacao.
#
#   3. LogisticRegression: o classificador (mesmo da Etapa 1)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("model", LogisticRegression(max_iter=1000))
])


# ---------------------------------------------------------------------------
# 3. DEFINIR A GRADE DE HIPERPARAMETROS
# ---------------------------------------------------------------------------
# Hiperparametros sao configuracoes que NOS escolhemos (nao sao aprendidos pelo modelo).
# Exemplos: numero de componentes do PCA, o valor de C na Regressao Logistica.
#
# O problema: como saber qual combinacao e a melhor?
# Resposta: testar todas! Isso e o Grid Search (busca em grade).
#
# A sintaxe "pca__n_components" significa:
#   "o parametro n_components da etapa chamada 'pca' no pipeline"
#   O duplo underline (__) conecta o nome da etapa ao parametro.
param_grid = {
    # Quantas dimensoes manter apos PCA?
    # 50 = muito comprimido, 150 = menos comprimido
    "pca__n_components": [50, 100, 150],

    # C = "rigidez" do modelo de Regressao Logistica
    # C pequeno (0.1) = modelo mais simples, generaliza mais (pode "underfittar")
    # C grande (10)   = modelo mais complexo, se ajusta mais aos dados (pode "overfittar")
    "model__C": [0.1, 1, 10]
}
# Total de combinacoes: 3 x 3 = 9 modelos diferentes serao testados!


# ---------------------------------------------------------------------------
# 4. EXECUTAR O GRID SEARCH COM CROSS-VALIDATION
# ---------------------------------------------------------------------------
# Cross-Validation (cv=3) divide os dados de treino em 3 partes:
#   - Treina com 2 partes, testa com 1 parte
#   - Repete 3 vezes, alternando qual parte e usada para teste
#   - A nota final e a MEDIA das 3 tentativas
#
# Por que usar Cross-Validation?
#   Se testamos com um unico split, o resultado pode ser "sorte".
#   Com 3 splits diferentes, temos mais confianca no resultado.
#   E como fazer 3 provas diferentes em vez de uma so.
#
# n_jobs=-1 = usar TODOS os nucleos do processador (acelera o processo)
# verbose=2 = mostrar progresso detalhado no terminal
grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=2)

print("\nIniciando Grid Search (pode levar alguns minutos)...")
inicio = time.time()

grid.fit(X_train, y_train)

tempo = time.time() - inicio
print(f"\nGrid Search concluido em {tempo:.1f} segundos")


# ---------------------------------------------------------------------------
# 5. RESULTADOS
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("RESULTADOS DO GRID SEARCH")
print("=" * 50)

# Melhor combinacao de hiperparametros encontrada
print(f"\nMelhores parametros: {grid.best_params_}")

# Melhor acuracia obtida com cross-validation (media das 3 tentativas)
print(f"Melhor score (CV):   {grid.best_score_:.4f}")

# Acuracia no conjunto de teste (dados totalmente novos)
score_teste = grid.score(X_test, y_test)
print(f"Score no teste:      {score_teste:.4f}")

# Comparacao com o baseline
print(f"\nCompare com o baseline da Etapa 1!")
print(f"O PCA ajudou ou atrapalhou?")
