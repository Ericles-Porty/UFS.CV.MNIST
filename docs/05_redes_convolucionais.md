# Redes Neurais Convolucionais (CNNs)

## Por que CNNs?

O MLP (Etapa 3) achata a imagem em um vetor de 784 numeros. Isso **destroi a estrutura espacial**: o modelo nao sabe que o pixel (0,0) esta ao lado do pixel (0,1).

CNNs resolvem isso processando a imagem como uma **grade 2D**, preservando a relacao entre pixels vizinhos. Por isso, sao o padrao para tarefas de visao computacional.

## O que e uma convolucao?

Uma convolucao e uma operacao onde um **filtro** (kernel) pequeno desliza pela imagem, fazendo uma conta em cada posicao.

### Passo a passo (filtro 3x3):

```
Imagem (5x5):                Filtro (3x3):
┌──────────────────┐         ┌──────────┐
│  1   0   1   0   1│         │  1  0  1 │
│  0   1   0   1   0│         │  0  1  0 │
│  1   0   1   0   1│         │  1  0  1 │
│  0   1   0   1   0│         └──────────┘
│  1   0   1   0   1│
└──────────────────┘
```

**Posicao (0,0):** o filtro cobre os pixels do canto superior esquerdo:

```
Imagem:     Filtro:      Multiplicacao:
1  0  1     1  0  1      1*1 + 0*0 + 1*1
0  1  0  ×  0  1  0  =   0*0 + 1*1 + 0*0  = SOMA = 5
1  0  1     1  0  1      1*1 + 0*0 + 1*1
```

O filtro desliza uma posicao para a direita e repete. Depois desce uma linha. Assim por toda a imagem.

### Tamanho da saida

```
Saida = Entrada - Kernel + 1

Exemplo: imagem 28x28, filtro 3x3
Saida = 28 - 3 + 1 = 26x26
```

## O que os filtros detectam?

Cada filtro se especializa em detectar um **padrao especifico**:

### Filtro detector de bordas verticais:
```
-1  0  1
-1  0  1
-1  0  1
```
Este filtro da valores ALTOS onde ha uma transicao claro/escuro na horizontal (= borda vertical).

### Filtro detector de bordas horizontais:
```
-1  -1  -1
 0   0   0
 1   1   1
```
Este filtro da valores ALTOS onde ha uma transicao claro/escuro na vertical (= borda horizontal).

### O que torna CNNs especiais?

Os filtros NAO sao definidos por nos -- a rede os **aprende durante o treinamento**! Ela descobre sozinha quais padroes sao uteis para classificar digitos.

No script `treinar_e_visualizar.py`, voce pode VER os filtros aprendidos!

## ReLU nas CNNs

Apos cada convolucao, aplicamos ReLU:
- Valores positivos (padrao encontrado): **mantem**
- Valores negativos (padrao nao encontrado): **zera**

Isso "limpa" o resultado, mantendo so as ativacoes relevantes.

## Max Pooling: reduzindo o tamanho

Max Pooling percorre o mapa de ativacao com uma janela (2x2) e pega o **valor maximo** de cada janela:

```
Antes (4x4):              Depois (2x2):
┌─────┬─────┐
│ 1 3 │ 2 1 │             ┌─────┐
│ 4 2 │ 0 3 │   MaxPool   │ 4 3 │
├─────┼─────┤   ──────→   │ 6 5 │
│ 6 1 │ 5 2 │             └─────┘
│ 0 3 │ 1 4 │
└─────┴─────┘
```

**Por que usar pooling?**

1. **Reduz o tamanho:** menos pixels = menos calculos = mais rapido
2. **Invariancia a translacao:** se o digito se mover 1 pixel, o maximo da janela provavelmente nao muda. O modelo fica mais robusto a pequenos deslocamentos.
3. **Abstrai detalhes:** mantem a informacao "tem um padrao aqui" sem se importar com a posicao exata dentro da janela.

## Arquitetura completa da nossa CNN

```
┌─────────────────────────────────────────────────────────┐
│  EXTRACAO DE FEATURES (aprende padroes na imagem)       │
│                                                         │
│  Entrada (1, 28, 28)                                    │
│      │                                                  │
│      ▼                                                  │
│  Conv1 (8 filtros 3x3) → ReLU → MaxPool(2)             │
│  Shape: (8, 26, 26) → (8, 26, 26) → (8, 13, 13)       │
│      │                                                  │
│      ▼                                                  │
│  Conv2 (16 filtros 3x3) → ReLU → MaxPool(2)            │
│  Shape: (16, 11, 11) → (16, 11, 11) → (16, 5, 5)      │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  CLASSIFICACAO (decide qual digito e)                   │
│                                                         │
│      │                                                  │
│      ▼                                                  │
│  Flatten: (16, 5, 5) → (400)                            │
│      │                                                  │
│      ▼                                                  │
│  FC1: 400 → 64 → ReLU                                  │
│      │                                                  │
│      ▼                                                  │
│  FC2: 64 → 10 (scores para cada digito)                 │
└─────────────────────────────────────────────────────────┘
```

### O que cada parte faz:

- **Conv1:** detecta padroes SIMPLES (bordas, cantos)
- **Conv2:** combina padroes simples em padroes COMPLEXOS (curvas, lacos)
- **Flatten + FC:** olha todos os padroes detectados e decide "baseado nisso tudo, e o digito 7"

## Mapas de ativacao: o que a rede "ve"

Quando uma imagem passa por um filtro, o resultado e um **mapa de ativacao**:

- **Pixels claros:** "aqui o filtro encontrou o padrao que procura!"
- **Pixels escuros:** "aqui nao tem nada relevante"

O script `treinar_e_visualizar.py` gera essas visualizacoes. Voce vai ver, por exemplo, que um filtro "acende" nas bordas do digito enquanto outro "acende" no interior.

## Por que CNN e melhor que MLP para imagens?

| | MLP | CNN |
|---|-----|-----|
| **Estrutura da imagem** | Perdida (achata para 1D) | Preservada (processa em 2D) |
| **Compartilhamento de pesos** | Nao (cada pixel tem seus pesos) | Sim (mesmo filtro 3x3 e reusado em toda a imagem) |
| **Parametros** | ~101.632 | ~11.000 (muito menos!) |
| **Invariancia a posicao** | Nao (se o digito se mover, confunde) | Parcial (pooling ajuda) |
| **Acuracia MNIST** | ~97% | ~98-99% |

### Compartilhamento de pesos (a sacada genial!)

No MLP, cada pixel tem sua propria conexao. Se o modelo aprendeu a detectar uma borda no canto superior, ele NAO sabe detectar a mesma borda no canto inferior.

Na CNN, o **mesmo filtro 3x3 desliza por toda a imagem**. Se ele aprendeu a detectar uma borda, detecta em QUALQUER posicao. Isso e chamado de **compartilhamento de pesos** e torna a CNN muito mais eficiente.

## Padding: mantendo o tamanho da imagem

Na CNN basica, a convolucao **diminui** o tamanho (28 → 26). Mas e possivel **manter** o tamanho adicionando uma borda de zeros ao redor da imagem. Isso se chama **padding**.

```
Sem padding (CNN basica):
  Entrada: 28x28 → Conv 3x3 → Saida: 26x26  (diminuiu!)
  Formula: 28 - 3 + 1 = 26

Com padding=1 (CNN avancada):
  Entrada: 28x28 → Adiciona borda de 1 pixel → 30x30 → Conv 3x3 → Saida: 28x28  (manteve!)
  Formula: (28 + 2) - 3 + 1 = 28
```

**Vantagem:** permite empilhar mais camadas sem que a imagem encolha rapido demais. A reducao de tamanho fica por conta do MaxPool, que voce controla.

## Resumo de todos os modelos

| Modelo | Tipo | Acuracia | Vantagem | Limitacao |
|--------|------|----------|----------|-----------|
| Regressao Logistica | Linear | ~92% | Simples, rapido | So fronteiras lineares |
| LogReg + PCA | Linear | ~92-93% | Comprime dados | Ainda linear |
| MLP | Rede Neural | ~97% | Nao-linear | Perde estrutura 2D |
| CNN basica | Rede Neural Conv. | ~98% | Preserva estrutura 2D | Poucos filtros, sem regularizacao |
| **CNN avancada** | **Rede Neural Conv.** | **~99%+** | **BatchNorm + Dropout + Augmentation** | Treina mais devagar |

## Proximo passo

A CNN basica ja e boa (~98%), mas com tecnicas avancadas podemos chegar a 99%+!

- **docs/06_tecnicas_avancadas.md** -- Data Augmentation, BatchNorm, Dropout, Scheduler
- **03_cnn/cnn_avancada.py** -- o codigo correspondente
