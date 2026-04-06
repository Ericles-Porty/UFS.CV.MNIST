# Redes Neurais: MLP (Multi-Layer Perceptron)

## O que e uma rede neural?

Uma rede neural e um modelo matematico **inspirado** no cerebro humano (mas NAO e um cerebro!). E composta por **neuronios artificiais** organizados em **camadas**.

### O neuronio artificial

Cada neuronio faz uma conta simples:

```
entradas: [x1, x2, x3]     (os dados)
pesos:    [w1, w2, w3]     (importancia de cada entrada)
bias:     b                 (um ajuste fino)

saida = ativacao(w1*x1 + w2*x2 + w3*x3 + b)
```

1. **Multiplica** cada entrada pelo seu peso
2. **Soma** tudo + o bias
3. Passa por uma **funcao de ativacao** (ex: ReLU)

### Camadas

Os neuronios sao organizados em camadas:

```
ENTRADA (784)     OCULTA (128)      SAIDA (10)
  ○                  ○                 ○  ← digito 0
  ○ ─────────────→   ○ ──────────→     ○  ← digito 1
  ○   (cada pixel    ○  (features      ○  ← digito 2
  ○    conecta com   ○   aprendidas)   ...
  ...  todos os      ○                 ○  ← digito 9
  ○    neuronios)    ○
 784 pixels         128 neuronios     10 scores
```

Cada seta e uma conexao com um **peso** que o modelo aprende. No nosso caso:
- Entrada → Oculta: 784 × 128 = **100.352 pesos**
- Oculta → Saida: 128 × 10 = **1.280 pesos**
- Total: ~101.632 numeros que o modelo precisa aprender!

## ReLU: a funcao de ativacao

**ReLU (Rectified Linear Unit)** e a funcao de ativacao mais popular:

```
ReLU(x) = max(0, x)

Se x = 3.5  →  ReLU = 3.5  (positivo: mantem)
Se x = -2.0 →  ReLU = 0.0  (negativo: zera)
Se x = 0.0  →  ReLU = 0.0  (zero: mantem)
```

### Por que precisamos de ativacao?

Sem ativacao, empilhar camadas lineares seria igual a **uma unica camada** (porque linear vezes linear = linear). A ativacao introduz **nao-linearidade**, permitindo que a rede aprenda fronteiras curvas e padroes complexos.

**Analogia:** Regressao Logistica = regua (so traca retas). MLP com ReLU = regua flexivel (pode curvar).

## Como a rede "aprende"?

O treinamento segue um ciclo que se repete milhares de vezes:

### 1. Forward Pass (passagem para frente)
Os dados entram pela camada de entrada, passam pela oculta, e saem na saida.

```
Imagem → [784 pixels] → [128 neuronios] → [10 scores] → Predicao: "3"
```

### 2. Calcular a Loss (perda/erro)
Comparamos a predicao com a resposta correta:

```
Predicao: [0.1, 0.05, 0.05, 0.7, 0.02, ...]  ← modelo diz "3" (70%)
Verdade:  digito 5
Loss: ALTA (errou feio!)
```

**CrossEntropyLoss** e a funcao que calcula esse erro. Ela:
1. Transforma os scores em probabilidades (softmax)
2. Penaliza o modelo quando da baixa probabilidade para a resposta certa

### 3. Backward Pass (backpropagation)
O algoritmo calcula: **"para cada peso, em qual direcao e quanto devo ajustar para reduzir o erro?"**

Isso e feito com calculo de **gradientes** (derivadas). O gradiente aponta na direcao que AUMENTA o erro, entao andamos na direcao OPOSTA.

**Analogia:** Imagine que voce esta cego numa montanha e quer descer ao vale. Voce tateia o chao em volta e da um passo na direcao que desce mais. Repete ate chegar ao ponto mais baixo.

### 4. Atualizar os pesos (optimizer.step)
O otimizador ajusta todos os pesos um pouquinho na direcao calculada pelo backpropagation.

### Resumo do ciclo em codigo:

```python
optimizer.zero_grad()   # 1. Limpa gradientes anteriores
outputs = model(xb)     # 2. Forward pass
loss = criterion(outputs, yb)  # 3. Calcula erro
loss.backward()         # 4. Backpropagation (calcula gradientes)
optimizer.step()        # 5. Atualiza pesos
```

## Epochs e Batches

### Por que nao usar todos os dados de uma vez?

60.000 imagens de 784 pixels = muita memoria! Alem disso, mini-lotes ajudam o treinamento:

- **Batch (lote):** grupo de imagens processadas de uma vez (ex: 64)
- **Epoch:** uma passada completa por TODOS os dados de treino

```
60.000 imagens / 64 por batch = 937 batches por epoch
5 epochs = 5 × 937 = 4.685 atualizacoes de pesos!
```

## Adam: o otimizador

**Adam (Adaptive Moment Estimation)** e uma versao melhorada do SGD (Stochastic Gradient Descent).

- SGD simples: todos os pesos sao ajustados com o mesmo "tamanho de passo"
- Adam: **adapta** o tamanho do passo para cada peso individualmente
- Pesos que mudam pouco recebem passos maiores
- Pesos que mudam muito recebem passos menores

### Learning Rate (taxa de aprendizado)

`lr=0.001` controla o tamanho do passo:

```
lr muito alto (0.1):    o modelo "pula" demais e nunca converge ✗
lr muito baixo (0.00001): o modelo aprende extremamente devagar ✗
lr adequado (0.001):     equilibrio entre velocidade e estabilidade ✓
```

## GPU vs CPU

| | CPU | GPU |
|---|-----|-----|
| **O que e** | Processador principal do computador | Placa de video (ex: NVIDIA) |
| **Nucleos** | 4-16 nucleos poderosos | Milhares de nucleos simples |
| **Bom para** | Tarefas sequenciais | Tarefas paralelas (como multiplicacao de matrizes!) |
| **No MNIST** | ~30 segundos | ~10 segundos |

Redes neurais fazem MUITAS multiplicacoes de matrizes. GPUs sao perfeitas para isso porque fazem milhares de multiplicacoes ao mesmo tempo.

**CUDA** e a tecnologia da NVIDIA que permite usar GPUs para computacao. Se voce nao tem GPU NVIDIA, o codigo roda normalmente na CPU (so demora um pouco mais).

## Por que MLP e melhor que Regressao Logistica?

| | Regressao Logistica | MLP |
|---|-------------------|-----|
| **Fronteiras** | Lineares (retas) | Nao-lineares (curvas) |
| **Camadas** | 1 (entrada → saida) | 2+ (entrada → oculta → saida) |
| **Acuracia MNIST** | ~92% | ~97% |
| **Parametros** | ~7.850 | ~101.632 |

## Limitacao do MLP

O MLP ainda **achata** a imagem (28x28 → 784). Ele nao sabe que pixels vizinhos estao proximos. Isso e como ler um livro com as letras embaralhadas -- da pra entender alguma coisa, mas e muito mais dificil.

**Solucao:** CNNs (Redes Convolucionais) que preservam a estrutura 2D!

## Proximo passo

- **docs/05_redes_convolucionais.md** -- CNNs: o estado da arte para imagens
- **03_cnn/forward_debug.py** -- veja como os dados fluem por uma CNN
- **03_cnn/treinar_e_visualizar.py** -- treine uma CNN e veja o que ela aprende
