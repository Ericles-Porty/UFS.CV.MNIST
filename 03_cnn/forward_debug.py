"""
ETAPA 4a: Debug do Forward Pass de uma CNN
============================================
O que voce vai aprender neste script:
  - Como uma imagem PASSA por cada camada de uma CNN (Rede Neural Convolucional)
  - Como o TAMANHO (shape) do tensor muda em cada camada e POR QUE
  - O que cada tipo de camada FAZ com a imagem

Este script NAO treina o modelo!
  Os pesos sao ALEATORIOS, entao a predicao sera aleatoria.
  O objetivo e VISUALIZAR como os dados fluem pela rede.

Arquitetura da CNN (execute e observe os shapes!):

  Entrada:    (1, 1, 28, 28)   -- 1 imagem, 1 canal, 28x28 pixels
      |
  Conv1:      (1, 8, 26, 26)   -- 8 filtros de 3x3 (28-3+1=26)
      |
  ReLU:       (1, 8, 26, 26)   -- mesmo shape (so zera negativos)
      |
  MaxPool:    (1, 8, 13, 13)   -- reduz pela metade (26/2=13)
      |
  Conv2:      (1, 16, 11, 11)  -- 16 filtros de 3x3 (13-3+1=11)
      |
  ReLU:       (1, 16, 11, 11)  -- mesmo shape
      |
  MaxPool:    (1, 16, 5, 5)    -- reduz pela metade (11/2=5, arredonda pra baixo)
      |
  Flatten:    (1, 400)         -- achata: 16 * 5 * 5 = 400
      |
  FC1:        (1, 64)          -- camada densa: 400 --> 64
      |
  ReLU:       (1, 64)          -- ativacao
      |
  FC2:        (1, 10)          -- saida: 64 --> 10 (um score por digito)

Formulas uteis:
  Tamanho apos convolucao (sem padding): saida = entrada - kernel_size + 1
  Tamanho apos MaxPool(2):               saida = entrada / 2 (arredonda pra baixo)

Leia tambem: docs/05_redes_convolucionais.md
"""

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset

# ---------------------------------------------------------------------------
# 1. CONFIGURAR DISPOSITIVO
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# ---------------------------------------------------------------------------
# 2. PEGAR UMA UNICA IMAGEM DO DATASET
# ---------------------------------------------------------------------------
ds = load_dataset("ylecun/mnist", download_mode="reuse_dataset_if_exists")

# Pegar a primeira imagem de teste
amostra = ds["test"][0]
label_verdadeiro = amostra["label"]

# Converter para tensor e normalizar (0-255 --> 0.0-1.0)
img = np.array(amostra["image"]) / 255.0

# Adicionar dimensoes de batch e canal:
# (28, 28) --> (1, 1, 28, 28)
#              ^  ^
#              |  |-- 1 canal (escala de cinza)
#              |-- 1 imagem (batch de 1)
x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

print(f"\nImagem de teste: digito '{label_verdadeiro}'")
print(f"Shape de entrada: {x.shape}  -->  (batch=1, canais=1, altura=28, largura=28)")


# ---------------------------------------------------------------------------
# 3. DEFINIR A CNN COM DEBUG (imprime shape em cada camada)
# ---------------------------------------------------------------------------
class DebugCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # CONVOLUCAO 1: 1 canal de entrada --> 8 filtros de saida, kernel 3x3
        # O que e convolucao?
        #   Imagine uma "janelinha" de 3x3 pixels que DESLIZA pela imagem.
        #   Em cada posicao, ela faz uma multiplicacao ponto-a-ponto e soma tudo.
        #   Cada filtro detecta um PADRAO diferente (bordas, curvas, cantos...).
        #   8 filtros = 8 detectores de padroes diferentes.
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)

        # CONVOLUCAO 2: 8 canais de entrada --> 16 filtros de saida, kernel 3x3
        # Os 8 canais vem da conv1. Agora, cada filtro combina padroes
        # da camada anterior para detectar padroes MAIS COMPLEXOS.
        # Ex: conv1 detecta bordas --> conv2 combina bordas em formas.
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)

        # ReLU: ativacao nao-linear (zera valores negativos)
        self.relu = nn.ReLU()

        # MAX POOLING: reduz o tamanho pela metade.
        #   Em cada janela 2x2, pega o MAIOR valor e descarta os outros 3.
        #   Isso torna o modelo mais ROBUSTO a pequenos deslocamentos.
        #   Se um digito esta 1 pixel para a direita, o resultado do pooling
        #   e quase o mesmo (o maximo nao muda tanto).
        self.pool = nn.MaxPool2d(2)

        # CAMADAS FULLY CONNECTED (iguais ao MLP da Etapa 3)
        # Apos as convolucoes, temos um tensor 3D (16 x 5 x 5 = 400).
        # Achatamos para 1D e passamos por camadas densas.
        self.fc1 = nn.Linear(16 * 5 * 5, 64)   # 400 --> 64
        self.fc2 = nn.Linear(64, 10)            # 64 --> 10 (digitos 0-9)

    def forward(self, x):
        """Forward pass com prints de debug em cada camada."""
        print("\n" + "=" * 50)
        print("FORWARD PASS (camada por camada)")
        print("=" * 50)

        # --- Bloco 1: Conv1 + ReLU + Pool ---
        x = self.conv1(x)
        print(f"Apos conv1:   {x.shape}  -->  28 - 3 + 1 = 26, entao 8 mapas de 26x26")

        x = self.relu(x)
        print(f"Apos ReLU:    {x.shape}  -->  mesmo shape (so zera valores negativos)")

        x = self.pool(x)
        print(f"Apos pool1:   {x.shape}  -->  26 / 2 = 13, entao 8 mapas de 13x13")

        # --- Bloco 2: Conv2 + ReLU + Pool ---
        x = self.conv2(x)
        print(f"Apos conv2:   {x.shape}  -->  13 - 3 + 1 = 11, entao 16 mapas de 11x11")

        x = self.relu(x)
        print(f"Apos ReLU:    {x.shape}  -->  mesmo shape")

        x = self.pool(x)
        print(f"Apos pool2:   {x.shape}  -->  11 / 2 = 5 (arredonda), entao 16 mapas de 5x5")

        # --- Flatten ---
        # x.view(batch, -1): achata tudo exceto a dimensao de batch
        # 16 canais * 5 * 5 pixels = 400 numeros
        x = x.view(x.size(0), -1)
        print(f"Apos flatten: {x.shape}  -->  16 * 5 * 5 = 400 numeros")

        # --- Camadas densas (iguais ao MLP) ---
        x = self.fc1(x)
        print(f"Apos fc1:     {x.shape}  -->  400 --> 64 neuronios")

        x = self.relu(x)
        print(f"Apos ReLU:    {x.shape}  -->  ativacao")

        x = self.fc2(x)
        print(f"Apos fc2:     {x.shape}  -->  64 --> 10 scores (um por digito)")

        return x


# ---------------------------------------------------------------------------
# 4. EXECUTAR O FORWARD PASS
# ---------------------------------------------------------------------------
model = DebugCNN().to(device)

# torch.no_grad() = nao calcular gradientes (nao estamos treinando)
with torch.no_grad():
    output = model(x)

# Mostrar a predicao (sera ALEATORIA porque os pesos nao foram treinados!)
predicao = torch.argmax(output, dim=1).item()
print(f"\n{'=' * 50}")
print(f"Predicao do modelo: {predicao}")
print(f"Valor verdadeiro:   {label_verdadeiro}")
print(f"{'=' * 50}")
print(f"\nNOTA: Os pesos sao ALEATORIOS (modelo nao foi treinado)!")
print(f"A predicao e aleatoria. O objetivo era ver os SHAPES das camadas.")
print(f"Para treinar e obter resultados reais, execute: 03_cnn/treinar_e_visualizar.py")
