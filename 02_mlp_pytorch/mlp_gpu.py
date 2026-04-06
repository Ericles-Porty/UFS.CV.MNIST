"""
ETAPA 3: Rede Neural MLP (Multi-Layer Perceptron) com PyTorch
==============================================================
O que voce vai aprender neste script:
  - O que e uma rede neural e como ela "aprende"
  - A arquitetura MLP: camadas empilhadas de neuronios conectados
  - Como usar PyTorch para construir e treinar redes neurais
  - O que e GPU e por que ela acelera o treinamento
  - O loop de treinamento: forward pass -> loss -> backward pass -> otimizacao

Arquitetura deste modelo:
  Imagem (784 pixels) --> Camada Oculta (128 neuronios) --> ReLU --> Saida (10 classes)

  Entrada:  [0.0, 0.1, 0.8, ...]  (784 numeros = pixels achatados)
  Oculta:   [0.3, 0.7, 0.1, ...]  (128 numeros = "features" aprendidas)
  Saida:    [0.01, 0.02, ..., 0.9] (10 numeros = score para cada digito 0-9)

Por que MLP depois de Regressao Logistica?
  - Regressao Logistica so encontra fronteiras LINEARES (retas)
  - MLP pode aprender relacoes NAO-LINEARES (curvas, padroes complexos)
  - Resultado: acuracia sobe de ~92% para ~97%!
  - Mas MLP ainda ACHATA a imagem (perde estrutura 2D) --> isso motiva CNNs (Etapa 4)

Leia tambem: docs/04_redes_neurais_mlp.md
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset


# ---------------------------------------------------------------------------
# 1. CONFIGURAR DISPOSITIVO (CPU ou GPU)
# ---------------------------------------------------------------------------
# GPU (Graphics Processing Unit) = placa de video.
# GPUs foram criadas para processar graficos de jogos, mas sao otimas para
# redes neurais porque fazem MILHARES de calculos em paralelo.
#
# "cuda" e a tecnologia da NVIDIA para usar GPU em computacao.
# Se voce nao tem GPU NVIDIA, o codigo roda na CPU (mais lento, mas funciona).
# Para o MNIST, a diferenca e pequena (~30s CPU vs ~10s GPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------
# 2. CARREGAR E PREPARAR OS DADOS
# ---------------------------------------------------------------------------
ds = load_dataset("ylecun/mnist", download_mode="reuse_dataset_if_exists")


def to_tensor(split):
    """
    Converte as imagens para tensores PyTorch.

    Diferente do scikit-learn (que usa NumPy arrays), PyTorch usa TENSORES.
    Tensores sao como arrays NumPy, mas podem rodar na GPU e calcular gradientes.

    Etapas:
    1. Converter imagens para NumPy array
    2. Dividir por 255.0 (NORMALIZACAO: pixels 0-255 --> 0.0-1.0)
       Redes neurais aprendem melhor com valores pequenos entre 0 e 1.
    3. Converter para tensor PyTorch
    4. unsqueeze(1): adiciona uma dimensao de "canal"
       (N, 28, 28) --> (N, 1, 28, 28)
       O "1" significa 1 canal (escala de cinza).
       Imagens coloridas (RGB) teriam 3 canais.
       Essa dimensao e obrigatoria no PyTorch, mesmo para MLP.
    """
    X = np.stack([np.array(img) for img in split["image"]]) / 255.0
    y = np.array(split["label"])

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, 28, 28)
    y = torch.tensor(y, dtype=torch.long)

    return X, y


X_train, y_train = to_tensor(ds["train"])
X_test, y_test = to_tensor(ds["test"])

# Enviar os dados para o dispositivo (GPU ou CPU)
# Se for GPU, os dados sao copiados para a memoria da placa de video
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

print(f"Shape dos dados de treino: {X_train.shape}")  # (60000, 1, 28, 28)


# ---------------------------------------------------------------------------
# 3. DEFINIR O MODELO (ARQUITETURA DA REDE NEURAL)
# ---------------------------------------------------------------------------
# nn.Sequential empilha camadas em sequencia (a saida de uma e a entrada da proxima).
#
# Camada 1: nn.Flatten()
#   Achata a imagem: (1, 28, 28) --> (784)
#   Mesma ideia do reshape(-1) que fizemos na Etapa 1.
#   O MLP nao "entende" imagens 2D, so vetores 1D.
#
# Camada 2: nn.Linear(784, 128)
#   Camada "totalmente conectada" (fully connected / dense).
#   Cada um dos 784 pixels se conecta a cada um dos 128 neuronios.
#   Total de conexoes (pesos): 784 x 128 = 100.352 numeros que o modelo aprende!
#   Cada neuronio calcula: saida = soma(peso_i * entrada_i) + bias
#
# Camada 3: nn.ReLU()
#   Funcao de ativacao: ReLU(x) = max(0, x)
#   - Se x > 0: mantem o valor
#   - Se x <= 0: vira zero
#   POR QUE PRECISAMOS DISSO?
#   Sem ativacao, empilhar camadas lineares seria igual a UMA camada so
#   (porque linear * linear = linear). O ReLU introduz NAO-LINEARIDADE,
#   permitindo que a rede aprenda padroes complexos.
#
# Camada 4: nn.Linear(128, 10)
#   Camada de saida: 128 neuronios --> 10 numeros (um para cada digito 0-9).
#   O digito com o maior valor e a predicao do modelo.
model = nn.Sequential(
    nn.Flatten(),            # (1, 28, 28) --> (784)
    nn.Linear(28 * 28, 128), # 784 --> 128 neuronios ocultos
    nn.ReLU(),               # Ativacao nao-linear
    nn.Linear(128, 10)       # 128 --> 10 classes (digitos 0-9)
).to(device)  # .to(device) envia o modelo para GPU (se disponivel)


# ---------------------------------------------------------------------------
# 4. DEFINIR FUNCAO DE PERDA E OTIMIZADOR
# ---------------------------------------------------------------------------
# CrossEntropyLoss: mede "quanto o modelo errou".
#   Combina duas operacoes:
#   1. Softmax: transforma os 10 numeros de saida em PROBABILIDADES (somam 1.0)
#      Ex: [2.0, 0.1, 0.5, ...] --> [0.85, 0.01, 0.02, ...]
#   2. Negative Log Likelihood: penaliza o modelo quando ele da BAIXA probabilidade
#      para a classe correta. Se o modelo diz "90% chance de ser 5" e era 5, perda baixa.
#      Se diz "10% chance de ser 5" e era 5, perda ALTA.
criterion = nn.CrossEntropyLoss()

# Adam: otimizador que ajusta os pesos do modelo para REDUZIR a perda.
#   E uma versao melhorada do SGD (Stochastic Gradient Descent).
#   lr=0.001 (learning rate) = tamanho do "passo" de ajuste.
#   - lr muito alto: o modelo "pula" demais e nao converge
#   - lr muito baixo: o modelo aprende muito devagar
#   0.001 e um bom valor padrao para comecar.
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ---------------------------------------------------------------------------
# 5. LOOP DE TREINAMENTO
# ---------------------------------------------------------------------------
# O treinamento funciona assim:
# 1. Pega um LOTE (batch) de imagens (64 de cada vez, nao todas 60.000 de uma vez)
# 2. Passa pelo modelo (forward pass) --> obtem predicoes
# 3. Calcula a perda (loss) --> quanto errou
# 4. Backpropagation --> calcula como ajustar cada peso para errar menos
# 5. Atualiza os pesos --> o modelo fica um pouquinho melhor
# 6. Repete para o proximo lote
#
# Uma EPOCH = uma passada completa por todos os 60.000 dados de treino.
# Fazemos 5 epochs = 5 passadas completas.

epochs = 5
batch_size = 64  # Processar 64 imagens de cada vez

print(f"\nIniciando treinamento ({epochs} epochs, batch_size={batch_size})...")

for epoch in range(epochs):
    model.train()  # Modo treinamento (habilita dropout/batchnorm se houver)

    for i in range(0, len(X_train), batch_size):
        # Pegar um mini-lote de 64 imagens
        xb = X_train[i:i + batch_size]
        yb = y_train[i:i + batch_size]

        # === O TRIO DO TREINAMENTO ===

        # Passo 1: Zerar os gradientes da iteracao anterior
        # (senao eles acumulam e o treinamento fica errado)
        optimizer.zero_grad()

        # Passo 2: Forward pass -- passar os dados pelo modelo
        outputs = model(xb)  # Predicoes do modelo

        # Passo 3: Calcular a perda (quanto errou)
        loss = criterion(outputs, yb)

        # Passo 4: Backward pass (backpropagation)
        # Calcula o GRADIENTE: "para cada peso, em qual direcao e quanto
        # eu devo ajustar para reduzir a perda?"
        loss.backward()

        # Passo 5: Atualizar os pesos na direcao que reduz a perda
        optimizer.step()

    # Ao final de cada epoch, mostrar a perda e acuracia de treino
    model.eval()
    with torch.no_grad():
        train_out = model(X_train)
        train_preds = torch.argmax(train_out, dim=1)
        train_acc = (train_preds == y_train).float().mean()
    print(f"  Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f} - Acuracia treino: {train_acc:.4f}")


# ---------------------------------------------------------------------------
# 6. AVALIACAO NO CONJUNTO DE TESTE
# ---------------------------------------------------------------------------
# model.eval() = modo avaliacao (desabilita dropout/batchnorm)
# torch.no_grad() = nao calcular gradientes (economiza memoria, pois nao estamos treinando)
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    preds = torch.argmax(outputs, dim=1)  # Pegar o indice do maior valor = digito predito
    acc = (preds == y_test).float().mean()

print(f"\nAcuracia no teste: {acc.item():.4f}")
print("Compare com a Regressao Logistica da Etapa 1 (~0.92)!")


# ---------------------------------------------------------------------------
# 7. SALVAR IMAGENS DE ERROS
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "resultados", "erros_mlp")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_ERROS = 50
erros_salvos = 0

for i in range(len(X_test)):
    if erros_salvos >= MAX_ERROS:
        break

    pred = preds[i].item()
    true = y_test[i].item()

    if pred == true:
        continue

    # .cpu() = trazer da GPU para CPU (necessario para plotar com matplotlib)
    # .squeeze() = remover a dimensao de canal: (1, 28, 28) --> (28, 28)
    # .numpy() = converter tensor PyTorch para array NumPy
    img = X_test[i].cpu().squeeze().numpy()

    plt.imshow(img, cmap="gray")
    plt.title(f"Predito: {pred} | Verdadeiro: {true}")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_DIR, f"erro_{erros_salvos + 1}.png"))
    plt.close()

    erros_salvos += 1

total_erros = (preds != y_test).sum().item()
print(f"\nTotal de erros: {total_erros} de {len(y_test)} imagens")
print(f"Primeiros {erros_salvos} erros salvos em: resultados/erros_mlp/")
