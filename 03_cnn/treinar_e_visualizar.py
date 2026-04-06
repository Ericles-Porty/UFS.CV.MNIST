"""
ETAPA 4b: Treinar uma CNN e Visualizar Filtros e Ativacoes
============================================================
O que voce vai aprender neste script:
  - Como treinar uma CNN completa no MNIST
  - Como visualizar os FILTROS aprendidos pela rede
  - Como visualizar os MAPAS DE ATIVACAO (o que a rede "ve" em uma imagem)

O que sao FILTROS?
  - Sao as "janelinhas" 3x3 que a rede APRENDE durante o treinamento
  - Antes do treino: filtros aleatorios (nao detectam nada util)
  - Depois do treino: cada filtro se especializa em detectar um padrao!
    Ex: um detecta bordas verticais, outro bordas horizontais, outro curvas...
  - Voce pode ver isso nos graficos que este script gera!

O que sao MAPAS DE ATIVACAO?
  - E o RESULTADO de aplicar um filtro em uma imagem
  - Mostra ONDE na imagem cada padrao foi encontrado
  - Pixels claros = "aqui tem o padrao que eu procuro!"
  - Pixels escuros = "aqui nao tem nada relevante"

Saidas geradas:
  - resultados/visualizacoes_cnn/filtros_conv1.png  (filtros aprendidos)
  - resultados/visualizacoes_cnn/ativacoes_conv1.png (mapas de ativacao)

Leia tambem: docs/05_redes_convolucionais.md
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. CONFIGURACAO
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "resultados", "visualizacoes_cnn")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. CARREGAR E PREPARAR DADOS
# ---------------------------------------------------------------------------
# Mesmo processo das etapas anteriores: carregar, normalizar, converter para tensor
ds = load_dataset("ylecun/mnist", download_mode="reuse_dataset_if_exists")


def to_tensor(split):
    X = np.stack([np.array(img) for img in split["image"]]) / 255.0
    y = np.array(split["label"])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, 28, 28)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


X_train, y_train = to_tensor(ds["train"])
X_test, y_test = to_tensor(ds["test"])

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# ---------------------------------------------------------------------------
# 3. DEFINIR A CNN
# ---------------------------------------------------------------------------
# Mesma arquitetura do forward_debug.py, mas sem os prints.
# Para entender cada camada em detalhe, veja 03_cnn/forward_debug.py
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Bloco convolucional 1: detecta padroes simples (bordas, cantos)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)    # 1 canal --> 8 filtros

        # Bloco convolucional 2: combina padroes simples em complexos (formas, curvas)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)   # 8 canais --> 16 filtros

        self.relu = nn.ReLU()       # Ativacao nao-linear
        self.pool = nn.MaxPool2d(2) # Reduz tamanho pela metade

        # Camadas densas para classificacao final
        self.fc1 = nn.Linear(16 * 5 * 5, 64)  # 400 --> 64
        self.fc2 = nn.Linear(64, 10)           # 64 --> 10 digitos

    def forward(self, x):
        # Bloco 1: Conv --> ReLU --> Pool
        # (N, 1, 28, 28) --> (N, 8, 26, 26) --> (N, 8, 13, 13)
        x = self.pool(self.relu(self.conv1(x)))

        # Bloco 2: Conv --> ReLU --> Pool
        # (N, 8, 13, 13) --> (N, 16, 11, 11) --> (N, 16, 5, 5)
        x = self.pool(self.relu(self.conv2(x)))

        # Achatar: (N, 16, 5, 5) --> (N, 400)
        x = x.view(x.size(0), -1)

        # Camadas densas: (N, 400) --> (N, 64) --> (N, 10)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ---------------------------------------------------------------------------
# 4. TREINAMENTO
# ---------------------------------------------------------------------------
# O treinamento e igual ao da Etapa 3 (MLP).
# A diferenca e que a CNN preserva a estrutura 2D da imagem,
# permitindo detectar padroes ESPACIAIS (bordas, curvas, formas).
epochs = 3
batch_size = 64

print(f"\nTreinando CNN ({epochs} epochs, batch_size={batch_size})...")

for epoch in range(epochs):
    model.train()

    for i in range(0, len(X_train), batch_size):
        xb = X_train[i:i + batch_size]
        yb = y_train[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

    print(f"  Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")


# ---------------------------------------------------------------------------
# 5. AVALIACAO
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    preds = torch.argmax(model(X_test), dim=1)
    acc = (preds == y_test).float().mean()

print(f"\nAcuracia no teste: {acc.item():.4f}")
print("Compare: LogReg ~0.92 | MLP ~0.97 | CNN ~0.98-0.99")


# ---------------------------------------------------------------------------
# 6. VISUALIZAR FILTROS APRENDIDOS
# ---------------------------------------------------------------------------
# model.conv1.weight.data contem os 8 filtros da primeira camada convolucional.
# Shape: (8, 1, 3, 3) = 8 filtros, 1 canal de entrada, 3x3 pixels cada
#
# Cada filtro e uma "janelinha" de 3x3 que o modelo APRENDEU.
# Pixels CLAROS (brancos) = pesos positivos (regioes que o filtro "procura")
# Pixels ESCUROS (pretos) = pesos negativos (regioes que o filtro "evita")
#
# Voce provavelmente vera filtros parecidos com:
#   - Detectores de bordas horizontais (linha clara no meio)
#   - Detectores de bordas verticais (coluna clara no meio)
#   - Detectores de bordas diagonais
#   - Detectores de cantos
weights = model.conv1.weight.data.cpu()

plt.figure(figsize=(10, 4))
plt.suptitle("Filtros Aprendidos pela Conv1 (3x3 cada)", fontsize=14)

for i in range(weights.shape[0]):
    plt.subplot(2, 4, i + 1)
    plt.imshow(weights[i][0], cmap="gray")
    plt.title(f"Filtro {i}")
    plt.axis("off")

plt.tight_layout()
caminho_filtros = os.path.join(OUTPUT_DIR, "filtros_conv1.png")
plt.savefig(caminho_filtros)
plt.close()
print(f"\nFiltros salvos em: {caminho_filtros}")


# ---------------------------------------------------------------------------
# 7. VISUALIZAR MAPAS DE ATIVACAO
# ---------------------------------------------------------------------------
# Vamos pegar UMA imagem de teste e ver o que acontece quando
# ela passa pela primeira camada convolucional (conv1).
#
# O resultado sao 8 "mapas de ativacao" -- um para cada filtro.
# Cada mapa mostra ONDE naquela imagem o padrao do filtro foi encontrado.
#
# Exemplo: se o filtro 0 detecta bordas verticais,
# o mapa de ativacao 0 vai ter pixels claros nas bordas verticais do digito
# e pixels escuros no resto da imagem.
img = X_test[0].unsqueeze(0)  # Pegar 1 imagem: (1, 1, 28, 28)

with torch.no_grad():
    # Passar so pela conv1 (sem ReLU, pool, etc.) para ver o resultado "puro"
    activation = model.conv1(img).cpu()

plt.figure(figsize=(10, 4))
digito = y_test[0].item()
plt.suptitle(f"Mapas de Ativacao da Conv1 (imagem do digito '{digito}')", fontsize=14)

for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(activation[0, i], cmap="gray")
    plt.title(f"Mapa {i}")
    plt.axis("off")

plt.tight_layout()
caminho_ativacoes = os.path.join(OUTPUT_DIR, "ativacoes_conv1.png")
plt.savefig(caminho_ativacoes)
plt.close()
print(f"Ativacoes salvas em: {caminho_ativacoes}")

print(f"\nAbra as imagens em resultados/visualizacoes_cnn/ para ver o que a CNN aprendeu!")
