# PCA e Reducao de Dimensionalidade

## O problema: dimensoes demais

No MNIST, cada imagem tem **784 pixels** (dimensoes). Mas sera que todos sao uteis?

Pense assim: os pixels nas **bordas** da imagem sao quase sempre pretos (valor 0). Eles nao ajudam a distinguir um "3" de um "7". Sao dimensoes **redundantes**.

Alem disso, muitos pixels carregam informacao **parecida** -- se um pixel e claro, o vizinho provavelmente tambem e. Isso e redundancia.

### A maldicao da dimensionalidade

Quanto mais dimensoes, mais dificil fica para o modelo encontrar padroes:
- Com 2 dimensoes, voce precisa de poucos pontos para "preencher" o espaco
- Com 784 dimensoes, precisaria de uma quantidade ASTRONOMICA de dados
- Os dados ficam "diluidos" no espaco de alta dimensao

**Solucao:** reduzir o numero de dimensoes, mantendo a informacao mais importante.

## O que e PCA?

**PCA (Principal Component Analysis / Analise de Componentes Principais)** e uma tecnica que encontra as **direcoes mais importantes** nos dados e descarta o resto.

### Analogia: sombra de um objeto

Imagine um objeto 3D (uma caneca, por exemplo). Se voce projetar a sombra dela numa parede:
- Voce **perde uma dimensao** (3D → 2D)
- Mas ainda **reconhece** que e uma caneca!
- A sombra captura a "essencia" do formato

PCA faz algo parecido: projeta dados de 784D para, digamos, 100D, mantendo a "essencia".

### Como funciona (simplificado):

1. PCA olha os dados e encontra as **direcoes onde os dados mais variam**
2. A primeira direcao (PC1) captura a MAIOR variacao
3. A segunda (PC2) captura a segunda maior, e assim por diante
4. Voce escolhe quantas direcoes manter (n_components)

```
784 dimensoes originais
    ↓ PCA(n_components=100)
100 dimensoes "resumidas"
    (mantendo ~95% da informacao!)
```

### Quanto comprimir?

| n_components | Compressao | Informacao mantida (aprox.) |
|-------------|------------|---------------------------|
| 50 | 784 → 50 (94% menor) | ~85% |
| 100 | 784 → 100 (87% menor) | ~92% |
| 150 | 784 → 150 (81% menor) | ~95% |

Mais componentes = mais informacao mantida, mas modelo mais lento.
Menos componentes = mais compressao, mas pode perder detalhes importantes.

## Grid Search: encontrando os melhores hiperparametros

### Parametros vs Hiperparametros

| | Parametros | Hiperparametros |
|---|-----------|-----------------|
| **Quem define?** | O modelo aprende sozinho | **Nos** escolhemos |
| **Exemplo** | Pesos da Regressao Logistica | n_components do PCA, C da LogReg |
| **Como otimizar?** | Treinamento (backpropagation) | Grid Search, tentativa e erro |

### O que e Grid Search?

E uma busca **exaustiva** por todas as combinacoes de hiperparametros:

```
n_components: [50, 100, 150]    (3 opcoes)
C:            [0.1,   1,  10]   (3 opcoes)
                                ─────────
                                9 combinacoes no total!
```

O Grid Search treina e avalia **cada uma das 9 combinacoes** e diz qual foi a melhor.

### O hiperparametro C

C controla a "rigidez" da Regressao Logistica:

- **C pequeno (0.1):** modelo mais SIMPLES, generaliza mais. Pode ser simples demais (underfitting).
- **C grande (10):** modelo mais COMPLEXO, se ajusta mais aos dados. Pode decorar (overfitting).
- **C medio (1):** equilibrio (geralmente um bom ponto de partida).

## Cross-Validation: avaliacao mais confiavel

### O problema de um unico split

Se avaliamos o modelo em apenas um conjunto de teste, o resultado pode ser "sorte" -- talvez aquele conjunto especifico era facil ou dificil.

### A solucao: Cross-Validation (cv=3)

Com 3-fold Cross-Validation, os dados de treino sao divididos em 3 partes:

```
Rodada 1: [TREINO] [TREINO] [TESTE]  → Score: 0.91
Rodada 2: [TREINO] [TESTE]  [TREINO] → Score: 0.92
Rodada 3: [TESTE]  [TREINO] [TREINO] → Score: 0.90
                                        ───────────
                                        Media: 0.91
```

Cada parte serve como teste uma vez. A nota final e a **media das 3 rodadas**.

**Vantagem:** resultado mais CONFIAVEL, pois testamos em 3 conjuntos diferentes (como fazer 3 provas em vez de 1).

## Proximo passo

Agora que voce entende tecnicas classicas de ML, vamos para redes neurais:
- **docs/04_redes_neurais_mlp.md** -- redes neurais com PyTorch
- **02_mlp_pytorch/mlp_gpu.py** -- o codigo correspondente
