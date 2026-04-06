# Visao Computacional com MNIST - Guia de Aprendizado

Um guia pratico para aprender **visao computacional do zero**, usando o dataset MNIST (digitos escritos a mao, 0-9).

O projeto implementa 5 abordagens diferentes, do mais simples ao mais avancado, para que voce entenda a evolucao dos modelos e **por que** cada tecnica existe.

## Pre-requisitos

- **Python 3.8+**
- Conhecimento basico de Python (variaveis, loops, funcoes)
- **Nao precisa saber Machine Learning** -- este projeto ensina!

## Instalacao

```bash
# Criar ambiente virtual (recomendado)
python -m venv .venv

# Ativar o ambiente virtual
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Estrutura do Projeto

```
UFS.CV.MNIST/
├── README.md                           ← Voce esta aqui!
├── requirements.txt                    ← Dependencias do projeto
│
├── docs/                               ← Documentacao para aprender
│   ├── 01_conceitos_basicos.md         ← Imagens, pixels, tensores, MNIST
│   ├── 02_regressao_logistica.md       ← Classificacao linear, metricas
│   ├── 03_pca_e_reducao.md             ← PCA, Grid Search, Cross-Validation
│   ├── 04_redes_neurais_mlp.md         ← Neuronios, backpropagation, PyTorch
│   ├── 05_redes_convolucionais.md      ← Convolucao, filtros, pooling, CNNs
│   └── 06_tecnicas_avancadas.md       ← Data Augmentation, BatchNorm, Dropout, Scheduler
│
├── 01_regressao_logistica/             ← Modelos classicos (scikit-learn)
│   ├── baseline.py                     ← Regressao Logistica basica
│   └── pca_grid_search.py             ← + PCA + busca de hiperparametros
│
├── 02_mlp_pytorch/                     ← Rede neural simples (PyTorch)
│   └── mlp_gpu.py                      ← Perceptron multicamadas com GPU
│
├── 03_cnn/                             ← Rede convolucional (PyTorch)
│   ├── forward_debug.py                ← Debug: ver shapes camada por camada
│   ├── treinar_e_visualizar.py         ← Treino + visualizacao de filtros
│   └── cnn_avancada.py                 ← CNN otimizada: BatchNorm, Dropout, Augmentation
│
└── resultados/                         ← Outputs gerados pelos scripts
    ├── erros_logreg/                   ← Imagens que LogReg errou
    ├── erros_mlp/                      ← Imagens que MLP errou
    └── visualizacoes_cnn/              ← Filtros e ativacoes da CNN
```

## Roteiro de Aprendizado

**Siga nesta ordem!** Cada etapa constroi sobre a anterior.

---

### Etapa 1: Regressao Logistica (o mais simples)

> Primeiro, entenda o problema e o modelo mais basico.

1. Leia: [docs/01_conceitos_basicos.md](docs/01_conceitos_basicos.md)
2. Leia: [docs/02_regressao_logistica.md](docs/02_regressao_logistica.md)
3. Execute: `python 01_regressao_logistica/baseline.py`
4. Olhe as imagens de erro em `resultados/erros_logreg/` -- quais digitos o modelo confunde?

### Etapa 2: PCA + Grid Search (melhorando o baseline)

> Aprenda a comprimir dados e otimizar hiperparametros.

1. Leia: [docs/03_pca_e_reducao.md](docs/03_pca_e_reducao.md)
2. Execute: `python 01_regressao_logistica/pca_grid_search.py`
3. Compare o resultado com o baseline -- PCA ajudou?

### Etapa 3: Rede Neural MLP (entrando no Deep Learning)

> Seu primeiro modelo de Deep Learning com PyTorch.

1. Leia: [docs/04_redes_neurais_mlp.md](docs/04_redes_neurais_mlp.md)
2. Execute: `python 02_mlp_pytorch/mlp_gpu.py`
3. Compare os erros em `resultados/erros_mlp/` com os da LogReg -- quais erros sumiram?

### Etapa 4: CNN (o estado da arte para imagens)

> Entenda por que CNNs dominam a visao computacional.

1. Leia: [docs/05_redes_convolucionais.md](docs/05_redes_convolucionais.md)
2. Execute: `python 03_cnn/forward_debug.py` -- observe como os shapes mudam camada por camada
3. Execute: `python 03_cnn/treinar_e_visualizar.py` -- treine e veja os filtros aprendidos
4. Abra `resultados/visualizacoes_cnn/filtros_conv1.png` -- voce consegue identificar detectores de bordas?
5. Abra `resultados/visualizacoes_cnn/ativacoes_conv1.png` -- veja o que a rede "enxerga" em uma imagem

### Etapa 5: CNN Avancada (buscando 99%+)

> Combine todas as tecnicas modernas para extrair o maximo do MNIST.

1. Leia: [docs/06_tecnicas_avancadas.md](docs/06_tecnicas_avancadas.md)
2. Execute: `python 03_cnn/cnn_avancada.py` -- treinamento mais longo (~2 min), mas resultado superior
3. Abra `resultados/visualizacoes_cnn/filtros_cnn_avancada.png` -- 32 filtros aprendidos (vs 8 da CNN basica)
4. Abra `resultados/visualizacoes_cnn/erros_cnn_avancada.png` -- veja os poucos erros restantes (muitos sao dificeis ate para humanos!)

---

## Comparacao de Resultados

| Modelo | Acuracia Esperada | Tempo (aprox.) | Tecnicas usadas |
|--------|:-----------------:|:--------------:|:---------------:|
| Regressao Logistica | ~92% | ~1 min | StandardScaler |
| LogReg + PCA + Grid Search | ~92-93% | ~3 min | + PCA, Cross-Validation |
| MLP (PyTorch) | ~97% | ~30s (GPU) | Rede neural simples |
| CNN basica | ~98% | ~30s (GPU) | Convolucao, Pooling |
| **CNN avancada** | **~99%+** | **~2 min (GPU)** | **+ BatchNorm, Dropout, Augmentation, Scheduler** |

## Proximos Passos

Depois de completar todas as etapas, voce pode:

1. **Outros datasets** -- Fashion-MNIST (roupas) ou CIFAR-10 (fotos reais coloridas)
2. **Transfer Learning** -- usar redes pre-treinadas (ResNet, EfficientNet) como base
3. **Segmentacao e Deteccao** -- ir alem da classificacao (detectar objetos em imagens)
4. **Frameworks modernos** -- PyTorch Lightning (simplifica o loop de treino)
5. **Competicoes** -- Kaggle tem desafios de visao computacional para praticar
