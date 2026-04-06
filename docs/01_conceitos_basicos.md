# Conceitos Basicos de Visao Computacional

## O que e uma imagem digital?

Uma imagem digital e uma **grade (matriz) de numeros** chamados **pixels**.

Cada pixel representa a intensidade de luz naquele ponto:
- **0** = preto (sem luz)
- **255** = branco (luz maxima)
- Valores intermediarios = tons de cinza

Uma imagem de 28x28 pixels e uma tabela com 28 linhas e 28 colunas, totalizando **784 numeros**.

### Exemplo visual: o digito "5"

Imagine que cada numero e a cor de um pixel (simplificado para 8x8):

```
  0   0   0   0   0   0   0   0
  0   0 200 200 200 200   0   0
  0   0 200   0   0   0   0   0
  0   0 200 200 200   0   0   0
  0   0   0   0   0 200   0   0
  0   0   0   0   0 200   0   0
  0   0 200 200 200   0   0   0
  0   0   0   0   0   0   0   0
```

Os valores altos (200) formam o desenho do "5". Os zeros sao o fundo preto.

## O que e o MNIST?

MNIST (Modified National Institute of Standards and Technology) e um dataset criado por **Yann LeCun** em 1998. E considerado o **"Hello World" da visao computacional** -- quase todo mundo que aprende sobre o tema comeca por aqui.

O dataset contem:
- **70.000 imagens** de digitos escritos a mao (0 a 9)
- Cada imagem tem **28x28 pixels** em escala de cinza
- **60.000** para treino e **10.000** para teste
- As imagens foram coletadas de formularios escritos por funcionarios dos correios americanos e estudantes

### Por que o MNIST e tao popular?

1. **Simples o suficiente** para rodar em qualquer computador (sem GPU potente)
2. **Complexo o suficiente** para testar diferentes algoritmos
3. **Bem estudado** -- existem milhares de artigos e tutoriais sobre ele
4. **Benchmark padrao** -- permite comparar modelos de forma justa

## O que e um tensor?

No contexto de Machine Learning, **tensor** e uma generalizacao de vetores e matrizes:

| Tipo | Dimensoes | Exemplo | Em Python |
|------|-----------|---------|-----------|
| Escalar | 0D | O numero `7` | `x = 7` |
| Vetor | 1D | Uma lista de numeros `[1, 2, 3]` | `np.array([1, 2, 3])` |
| Matriz | 2D | Uma tabela/grade | `np.array([[1,2],[3,4]])` |
| Tensor 3D | 3D | Varios "andares" de matrizes | Uma imagem colorida (3 canais) |
| Tensor 4D | 4D | Um lote de imagens | `(batch, canais, altura, largura)` |

### No MNIST, trabalhamos com tensores 4D:

```
Shape: (N, C, H, W)
        |  |  |  |
        |  |  |  +-- W = Width (largura) = 28
        |  |  +----- H = Height (altura) = 28
        |  +-------- C = Channels (canais) = 1 (escala de cinza)
        +----------- N = numero de imagens no lote (batch)
```

Uma imagem colorida RGB teria C=3 (um canal para vermelho, um para verde, um para azul).

## Normalizacao: por que dividir por 255?

Os pixels vao de 0 a 255. Redes neurais aprendem MELHOR com valores pequenos (entre 0 e 1).

```python
X = X / 255.0
# Antes: [0, 128, 255]
# Depois: [0.0, 0.502, 1.0]
```

**Por que isso ajuda?**
- Valores grandes fazem os gradientes "explodirem" (ficarem enormes), desestabilizando o treinamento
- Com valores entre 0 e 1, o treinamento e mais estavel e mais rapido
- Analogia: e como ajustar uma receita. Se um ingrediente esta em gramas e outro em toneladas, fica dificil balancear. Normalizando, tudo fica na mesma escala.

## Train/Test Split: por que separar os dados?

Separamos os dados em dois conjuntos:

- **Treino (train):** 60.000 imagens usadas para o modelo APRENDER
- **Teste (test):** 10.000 imagens usadas para AVALIAR o modelo

### Por que nao usar tudo para treinar?

Imagine que voce estuda para uma prova. Se a prova tiver exatamente as mesmas questoes do caderno de exercicios, voce pode tirar 10 mesmo sem entender a materia (so decorou). Mas se as questoes forem NOVAS, ai sim sabemos se voce realmente aprendeu.

O conjunto de teste sao as "questoes novas" -- dados que o modelo **nunca viu**. Se ele acerta nesses dados, sabemos que ele realmente aprendeu os padroes, e nao so decorou.

Quando um modelo "decora" os dados de treino mas erra nos novos, chamamos isso de **overfitting** (sobreajuste).

## Proximo passo

Agora que voce entende os conceitos basicos, va para:
- **docs/02_regressao_logistica.md** -- seu primeiro modelo de classificacao!
- **01_regressao_logistica/baseline.py** -- o codigo correspondente
