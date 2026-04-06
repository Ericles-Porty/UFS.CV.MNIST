"""
ETAPA 1: Regressao Logistica - Modelo Baseline
=================================================
O que voce vai aprender neste script:
  - Como carregar o dataset MNIST (70.000 imagens de digitos escritos a mao)
  - Como transformar uma IMAGEM 2D (28x28 pixels) em um VETOR 1D (784 numeros)
  - O que e Regressao Logistica e por que ela funciona para classificacao
  - Como avaliar um modelo usando acuracia e classification report
  - Como analisar os erros do modelo (imagens que ele erra)

Por que comecar com Regressao Logistica?
  - E o modelo mais simples que funciona razoavelmente bem (~92% de acerto)
  - Serve como "baseline" (ponto de referencia) para comparar com modelos mais complexos
  - Nos ajuda a entender o problema ANTES de partir para redes neurais

Leia tambem: docs/01_conceitos_basicos.md e docs/02_regressao_logistica.md
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# 1. CARREGAR O DATASET MNIST
# ---------------------------------------------------------------------------
# O MNIST e o "Hello World" da visao computacional.
# Contem 70.000 imagens em escala de cinza (28x28 pixels) de digitos 0-9,
# escritos a mao por pessoas reais. Foi criado por Yann LeCun em 1998.
#
# - train (60.000 imagens): usadas para o modelo APRENDER
# - test (10.000 imagens): usadas para AVALIAR o modelo (dados que ele nunca viu)
#
# download_mode="reuse_dataset_if_exists" evita baixar de novo se ja tiver em cache.
ds = load_dataset("ylecun/mnist", download_mode="reuse_dataset_if_exists")
train_ds = ds["train"]
test_ds = ds["test"]


# ---------------------------------------------------------------------------
# 2. CONVERTER IMAGENS PARA FORMATO NUMERICO (NUMPY)
# ---------------------------------------------------------------------------
def to_numpy(split):
    """
    Converte as imagens do dataset para arrays NumPy.

    Cada imagem e uma grade 2D de 28x28 pixels.
    Porem, a Regressao Logistica espera dados em formato de TABELA:
    cada amostra deve ser um vetor (uma linha).

    Entao fazemos o "flattening" (achatamento):
        Imagem 28x28 (matriz)  -->  Vetor de 784 numeros (uma linha)

        Visualmente:
        [[0, 0, 128],               [0, 0, 128, 0, 255, 64, ...]
         [0, 255, 64],    --->       (784 numeros em sequencia)
         [...        ]]

    PROBLEMA: ao achatar, perdemos a informacao de POSICAO dos pixels.
    O modelo nao sabe que o pixel 0 esta ao lado do pixel 1.
    Isso e uma limitacao que CNNs resolvem (veja Etapa 4).
    """
    # np.array(img) converte a imagem PIL para uma matriz 28x28 de numeros
    # .reshape(-1) achata a matriz para um vetor de 784 numeros
    # np.stack empilha todos os vetores em uma tabela (N linhas x 784 colunas)
    X = np.stack([np.array(img).reshape(-1) for img in split["image"]])
    y = np.array(split["label"])
    return X, y


X_train, y_train = to_numpy(train_ds)  # X_train.shape = (60000, 784)
X_test, y_test = to_numpy(test_ds)      # X_test.shape  = (10000, 784)

print(f"Dados de treino: {X_train.shape[0]} imagens, {X_train.shape[1]} pixels cada")
print(f"Dados de teste:  {X_test.shape[0]} imagens")


# ---------------------------------------------------------------------------
# 3. CRIAR E TREINAR O MODELO
# ---------------------------------------------------------------------------
# Usamos um Pipeline do scikit-learn, que encadeia etapas de preprocessamento + modelo.
# Vantagem: garante que as mesmas transformacoes sao aplicadas no treino E no teste.
#
# Etapa 1 - StandardScaler (normalizacao):
#   Os pixels vao de 0 a 255. Valores grandes podem confundir o modelo.
#   O StandardScaler transforma os dados para ter media=0 e desvio padrao=1.
#   Analogia: imagine comparar peso (kg) com altura (cm) -- as escalas sao
#   muito diferentes. A normalizacao coloca tudo na mesma escala.
#
# Etapa 2 - LogisticRegression (o modelo em si):
#   Apesar do nome "regressao", e usado para CLASSIFICACAO.
#   Ele encontra a melhor "fronteira" (hiperplano) para separar as classes.
#   max_iter=1000: numero maximo de iteracoes do otimizador para convergir.
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

print("\nTreinando modelo... (pode levar ~1 minuto)")
pipe.fit(X_train, y_train)  # O modelo aprende os padroes dos dados de treino


# ---------------------------------------------------------------------------
# 4. AVALIAR O MODELO
# ---------------------------------------------------------------------------
# Testamos com dados que o modelo NUNCA viu (X_test).
# Isso simula o "mundo real" -- se o modelo so acerta dados que ja viu,
# ele nao e util (isso se chama "overfitting").
y_pred = pipe.predict(X_test)

# Acuracia = porcentagem de acertos
print(f"\nAcuracia: {accuracy_score(y_test, y_pred):.4f}")

# Classification report mostra metricas POR DIGITO:
#   - precision: dos que o modelo disse ser "5", quantos realmente eram "5"?
#   - recall: de todos os "5" reais, quantos o modelo encontrou?
#   - f1-score: media harmonica entre precision e recall
print("\nRelatorio detalhado por digito:")
print(classification_report(y_test, y_pred))


# ---------------------------------------------------------------------------
# 5. ANALISAR OS ERROS DO MODELO
# ---------------------------------------------------------------------------
# Olhar os erros e TÃO importante quanto olhar a acuracia!
# Isso nos ajuda a entender as limitacoes do modelo.
# Erros comuns: 4 confundido com 9, 3 com 8, 7 com 1 (parecem similares).

# Caminho para salvar as imagens de erro (relativo ao script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "resultados", "erros_logreg")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_ERROS = 50  # Salvamos no maximo 50 imagens de erro (suficiente para analise)
erros_salvos = 0

for i in range(len(X_test)):
    if erros_salvos >= MAX_ERROS:
        break

    pred = y_pred[i]
    true = y_test[i]

    if pred == true:
        continue  # Acertou, pula para o proximo

    # Reconstruir a imagem 28x28 a partir do vetor de 784
    img = X_test[i].reshape(28, 28)

    plt.imshow(img, cmap="gray")
    plt.title(f"Predito: {pred} | Verdadeiro: {true}")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, f"erro_{erros_salvos+1}.png"))
    plt.close()  # Libera memoria (importante quando salvamos muitas imagens)

    erros_salvos += 1

total_erros = (y_pred != y_test).sum()
print(f"\nTotal de erros: {total_erros} de {len(y_test)} imagens")
print(f"Primeiros {erros_salvos} erros salvos em: resultados/erros_logreg/")
