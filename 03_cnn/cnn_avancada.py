"""
ETAPA 5: CNN Avancada - Buscando 99%+ de Acuracia
====================================================
O que voce vai aprender neste script:
  - Data Augmentation: "criar" dados novos transformando os existentes
  - Batch Normalization: estabilizar e acelerar o treinamento
  - Dropout: evitar overfitting (quando o modelo "decora" os dados)
  - Learning Rate Scheduler: ajustar a taxa de aprendizado durante o treino
  - Arquitetura mais profunda: mais filtros = mais capacidade de aprender

Comparacao com a CNN basica (treinar_e_visualizar.py):

  CNN Basica:                        CNN Avancada:
  ─────────────────                  ─────────────────
  Conv1: 8 filtros                   Conv1: 32 filtros
  Conv2: 16 filtros                  Conv2: 64 filtros
  Sem BatchNorm                      Com BatchNorm (estabiliza treino)
  Sem Dropout                        Com Dropout 25% (evita overfitting)
  Sem Data Augmentation              Com rotacao + deslocamento
  lr fixo = 0.001                    lr com scheduler (diminui ao longo do tempo)
  3 epochs                           15 epochs
  Acuracia: ~98%                     Acuracia: ~99%+

Por que essas tecnicas funcionam?
  - Data Augmentation: o modelo vê o "7" reto, torto, deslocado... aprende que
    todos sao "7", independente da posicao. Generaliza melhor!
  - BatchNorm: normaliza as ativacoes entre camadas. Sem isso, as ativacoes
    podem ficar muito grandes ou muito pequenas, desestabilizando o treino.
  - Dropout: "desliga" neuronios aleatorios durante o treino. Isso FORCA a rede
    a nao depender de nenhum neuronio individual. E como estudar com colegas
    diferentes -- voce nao pode depender so de um.
  - LR Scheduler: no inicio, passos grandes para aprender rapido. No final,
    passos pequenos para "afinar" os pesos com precisao.

Leia tambem: docs/05_redes_convolucionais.md
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
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
# 2. CARREGAR O DATASET
# ---------------------------------------------------------------------------
ds = load_dataset("ylecun/mnist", download_mode="reuse_dataset_if_exists")


def to_tensor(split):
    X = np.stack([np.array(img) for img in split["image"]]) / 255.0
    y = np.array(split["label"])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, 28, 28)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


X_train, y_train = to_tensor(ds["train"])
X_test, y_test = to_tensor(ds["test"])

# Dados de teste vao direto para o device (nao precisam de augmentation)
X_test, y_test = X_test.to(device), y_test.to(device)


# ---------------------------------------------------------------------------
# 3. DATA AUGMENTATION
# ---------------------------------------------------------------------------
# Data Augmentation = aplicar transformacoes ALEATORIAS nas imagens de treino
# a cada epoch, para que o modelo veja versoes ligeiramente diferentes.
#
# Isso SIMULA ter mais dados de treino sem coletar dados novos!
#
# Transformacoes usadas:
#   - RandomRotation(10): rotaciona ate 10 graus (digitos podem ser escritos tortos)
#   - RandomAffine(translate): desloca ate 10% horizontal/vertical
#     (digitos podem estar deslocados na imagem)
#
# IMPORTANTE: Data Augmentation so e aplicado nos dados de TREINO!
# Nos dados de teste, queremos avaliar nas imagens originais.
#
# Como funciona na pratica:
#   Epoch 1: modelo ve o "7" original
#   Epoch 2: modelo ve o "7" rotacionado 5° para a direita
#   Epoch 3: modelo ve o "7" deslocado 2 pixels para cima
#   ... e assim por diante. Cada vez e uma versao diferente!
augmentation = transforms.Compose([
    transforms.RandomRotation(10),                          # Rotacao aleatoria ate 10°
    transforms.RandomAffine(0, translate=(0.1, 0.1)),       # Deslocamento ate 10%
])


# ---------------------------------------------------------------------------
# 4. DATALOADER COM AUGMENTATION
# ---------------------------------------------------------------------------
# DataLoader e a forma padrao do PyTorch para iterar sobre os dados em batches.
# Vantagens sobre o loop manual (for i in range(0, len(X), batch_size)):
#   - shuffle=True: embaralha os dados a cada epoch (ajuda o treino)
#   - Gerencia batches automaticamente (inclusive o ultimo batch menor)
#
# Para aplicar augmentation, criamos um Dataset customizado que transforma
# cada imagem antes de entrega-la ao modelo.

class MNISTAugmented(torch.utils.data.Dataset):
    """
    Dataset que aplica Data Augmentation nas imagens de treino.

    A cada vez que o DataLoader pede uma imagem, este dataset:
    1. Pega a imagem original
    2. Aplica as transformacoes aleatorias (rotacao, deslocamento)
    3. Retorna a imagem transformada

    Como as transformacoes sao ALEATORIAS, a mesma imagem fica diferente
    toda vez que e acessada. E como se tivessemos infinitos dados!
    """
    def __init__(self, images, labels, transform=None):
        self.images = images    # Tensor (N, 1, 28, 28)
        self.labels = labels    # Tensor (N,)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]     # (1, 28, 28)
        label = self.labels[idx]

        if self.transform:
            # transforms espera imagem PIL ou tensor, nosso tensor ja funciona
            img = self.transform(img)

        return img, label


# Criar dataset de treino COM augmentation e DataLoader
train_dataset = MNISTAugmented(X_train, y_train, transform=augmentation)
train_loader = DataLoader(
    train_dataset,
    batch_size=128,       # Batch maior = treino mais estavel (temos BatchNorm agora)
    shuffle=True,         # Embaralhar dados a cada epoch
)


# ---------------------------------------------------------------------------
# 5. DEFINIR A CNN AVANCADA
# ---------------------------------------------------------------------------
class CNNAvancada(nn.Module):
    """
    CNN com BatchNorm e Dropout para alcançar 99%+ no MNIST.

    Arquitetura:
      Entrada (1, 28, 28)
          |
      Conv1 (32 filtros 3x3, padding=1) --> BatchNorm --> ReLU --> MaxPool(2)
      Shape: (32, 28, 28) --> (32, 14, 14)
          |
      Conv2 (64 filtros 3x3, padding=1) --> BatchNorm --> ReLU --> MaxPool(2)
      Shape: (64, 14, 14) --> (64, 7, 7)
          |
      Flatten: (64 * 7 * 7) = 3136
          |
      FC1: 3136 --> 128 --> BatchNorm --> ReLU --> Dropout(25%)
          |
      FC2: 128 --> 10 (saida)
    """

    def __init__(self):
        super().__init__()

        # --- Bloco Convolucional 1 ---
        # Conv2d com padding=1: mantem o tamanho da imagem apos a convolucao!
        #   Sem padding: 28 - 3 + 1 = 26 (diminui)
        #   Com padding=1: adiciona 1 pixel de borda (zeros) em cada lado
        #                  (28 + 2) - 3 + 1 = 28 (mantem!)
        #   Isso permite empilhar mais camadas sem encolher demais.
        #
        # 32 filtros (em vez de 8): mais filtros = mais padroes detectados.
        #   A CNN basica com 8 filtros so pode detectar 8 padroes diferentes.
        #   Com 32, ela pode detectar bordas, cantos, curvas, texturas...
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # BatchNorm2d: normaliza as ativacoes DENTRO de cada camada.
        #   Sem BatchNorm: ativacoes podem ficar enormes (ex: 500.0) ou minusculas (0.0001)
        #   Com BatchNorm: ativacoes ficam centradas em 0 com desvio padrao ~1
        #
        #   Por que isso ajuda?
        #   - Treino mais RAPIDO (gradientes fluem melhor)
        #   - Treino mais ESTAVEL (menos chance de "explodir" ou "sumir")
        #   - Permite usar learning rates maiores
        #
        #   O "32" e o numero de canais (deve bater com a saida do conv anterior)
        self.bn1 = nn.BatchNorm2d(32)

        # --- Bloco Convolucional 2 ---
        # 64 filtros: a segunda camada combina os 32 padroes da primeira
        # em padroes MAIS COMPLEXOS (formas de digitos, lacos, intersecoes)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # --- Camadas compartilhadas ---
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        # --- Dropout ---
        # Dropout DESLIGA neuronios aleatorios durante o treino.
        #
        # Com Dropout(0.25): a cada forward pass, 25% dos neuronios sao
        # zerados aleatoriamente. Neuronios diferentes sao desligados cada vez.
        #
        # Por que isso evita overfitting?
        #   Sem Dropout: alguns neuronios "dominam" e o modelo depende deles.
        #   Se esses neuronios decoraram algo especifico do treino, o modelo
        #   nao generaliza para dados novos.
        #
        #   Com Dropout: o modelo e FORCADO a espalhar o conhecimento entre
        #   todos os neuronios. Nenhum neuronio individual e indispensavel.
        #   E como uma equipe onde todos precisam saber fazer tudo.
        #
        # IMPORTANTE: Dropout so funciona durante model.train().
        # Durante model.eval() (avaliacao), todos os neuronios ficam ativos.
        self.dropout = nn.Dropout(0.25)

        # --- Camadas Densas (Fully Connected) ---
        # Apos as convolucoes: 64 canais x 7 x 7 = 3136 features
        # (28 / 2 / 2 = 7, por causa dos dois MaxPool)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn_fc = nn.BatchNorm1d(128)  # BatchNorm1d para camadas densas (1D)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Bloco 1: Conv --> BatchNorm --> ReLU --> Pool
        # (N, 1, 28, 28) --> (N, 32, 28, 28) --> (N, 32, 14, 14)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))

        # Bloco 2: Conv --> BatchNorm --> ReLU --> Pool
        # (N, 32, 14, 14) --> (N, 64, 14, 14) --> (N, 64, 7, 7)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        # Achatar: (N, 64, 7, 7) --> (N, 3136)
        x = x.view(x.size(0), -1)

        # Camada densa 1: 3136 --> 128 + BatchNorm + ReLU + Dropout
        x = self.dropout(self.relu(self.bn_fc(self.fc1(x))))

        # Camada de saida: 128 --> 10
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# 6. CRIAR MODELO, LOSS, OTIMIZADOR E SCHEDULER
# ---------------------------------------------------------------------------
model = CNNAvancada().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler: reduz o lr ao longo do treinamento.
#
# StepLR(step_size=5, gamma=0.5):
#   A cada 5 epochs, o learning rate e multiplicado por 0.5 (cortado pela metade).
#
#   Epoch 1-5:   lr = 0.001    (passos grandes, aprende rapido)
#   Epoch 6-10:  lr = 0.0005   (passos medios, refinando)
#   Epoch 11-15: lr = 0.00025  (passos pequenos, ajuste fino)
#
# Analogia: quando voce esta aprendendo a estacionar um carro,
# primeiro faz manobras grandes para se posicionar, depois ajusta
# centimetro por centimetro para ficar perfeito.
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Contar parametros do modelo (para comparar com a CNN basica)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parametros do modelo: {total_params:,} (CNN basica tinha ~11.000)")


# ---------------------------------------------------------------------------
# 7. LOOP DE TREINAMENTO
# ---------------------------------------------------------------------------
epochs = 15
print(f"\nTreinando CNN Avancada ({epochs} epochs)...")
print(f"{'─' * 60}")

melhor_acc = 0.0

for epoch in range(epochs):
    model.train()  # Ativa Dropout e BatchNorm em modo treino
    total_loss = 0.0
    batches = 0

    # DataLoader itera automaticamente em batches embaralhados
    # E aplica Data Augmentation em cada imagem!
    for xb, yb in train_loader:
        # Enviar batch para o dispositivo (GPU/CPU)
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    # Atualizar o learning rate ao final de cada epoch
    scheduler.step()
    lr_atual = scheduler.get_last_lr()[0]

    # --- Avaliar no conjunto de teste ao final de cada epoch ---
    model.eval()  # Desativa Dropout, BatchNorm usa estatisticas acumuladas
    with torch.no_grad():
        outputs_test = model(X_test)
        preds = torch.argmax(outputs_test, dim=1)
        acc = (preds == y_test).float().mean().item()

    # Guardar a melhor acuracia
    if acc > melhor_acc:
        melhor_acc = acc
        epoca_melhor = epoch + 1

    avg_loss = total_loss / batches
    print(f"  Epoch {epoch + 1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
          f"Acc teste: {acc:.4f} | lr: {lr_atual:.6f}")

print(f"{'─' * 60}")
print(f"\nMelhor acuracia: {melhor_acc:.4f} (epoch {epoca_melhor})")


# ---------------------------------------------------------------------------
# 8. AVALIACAO FINAL DETALHADA
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    outputs_test = model(X_test)
    preds = torch.argmax(outputs_test, dim=1)
    acc_final = (preds == y_test).float().mean().item()

total_erros = (preds != y_test).sum().item()

print(f"\n{'=' * 60}")
print(f"RESULTADO FINAL")
print(f"{'=' * 60}")
print(f"Acuracia: {acc_final:.4f} ({10000 - total_erros}/10000 corretos)")
print(f"Erros:    {total_erros} imagens de 10000")
print(f"\nComparacao:")
print(f"  Regressao Logistica:  ~0.92  (~800 erros)")
print(f"  MLP:                  ~0.97  (~300 erros)")
print(f"  CNN basica:           ~0.98  (~200 erros)")
print(f"  CNN avancada:         {acc_final:.4f}  ({total_erros} erros) ← voce esta aqui!")


# ---------------------------------------------------------------------------
# 9. VISUALIZAR FILTROS DA PRIMEIRA CAMADA
# ---------------------------------------------------------------------------
# Agora com 32 filtros (em vez de 8), podemos ver uma variedade maior
# de padroes que a rede aprendeu.
weights = model.conv1.weight.data.cpu()

plt.figure(figsize=(12, 6))
plt.suptitle("Filtros Aprendidos pela Conv1 - CNN Avancada (3x3 cada)", fontsize=14)

for i in range(32):
    plt.subplot(4, 8, i + 1)
    plt.imshow(weights[i][0], cmap="gray")
    plt.title(f"{i}", fontsize=8)
    plt.axis("off")

plt.tight_layout()
caminho = os.path.join(OUTPUT_DIR, "filtros_cnn_avancada.png")
plt.savefig(caminho, dpi=150)
plt.close()
print(f"\nFiltros salvos em: {caminho}")


# ---------------------------------------------------------------------------
# 10. VISUALIZAR ERROS RESTANTES
# ---------------------------------------------------------------------------
# Com 99%+, sobraram muito poucos erros. Vamos ver TODOS eles!
# Esses sao os casos mais dificeis do MNIST -- muitas vezes ate humanos
# teriam dificuldade em ler esses digitos.
erros_idx = (preds != y_test).nonzero(as_tuple=True)[0]

# Mostrar ate 50 erros (provavelmente sao menos que isso!)
n_mostrar = min(len(erros_idx), 50)

if n_mostrar > 0:
    cols = 10
    rows = (n_mostrar + cols - 1) // cols
    plt.figure(figsize=(cols * 1.5, rows * 1.8))
    plt.suptitle(f"Erros da CNN Avancada ({total_erros} no total)", fontsize=14)

    for i in range(n_mostrar):
        idx = erros_idx[i].item()
        img = X_test[idx].cpu().squeeze().numpy()
        pred = preds[idx].item()
        true = y_test[idx].item()

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"P:{pred} V:{true}", fontsize=8, color="red")
        plt.axis("off")

    plt.tight_layout()
    caminho = os.path.join(OUTPUT_DIR, "erros_cnn_avancada.png")
    plt.savefig(caminho, dpi=150)
    plt.close()
    print(f"Erros salvos em: {caminho}")
    print(f"\nObserve os erros: muitos sao digitos que ate HUMANOS teriam dificuldade!")
