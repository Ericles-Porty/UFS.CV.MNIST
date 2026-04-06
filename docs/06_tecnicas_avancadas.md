# Tecnicas Avancadas: De 98% para 99%+

A CNN basica (Etapa 4) ja atinge ~98% no MNIST. Parece otimo, mas significa ~200 erros em 10.000 imagens. Com as tecnicas deste documento, reduzimos para menos de 100 erros (~99%+).

Cada tecnica ataca um problema diferente. Juntas, elas se complementam.

## 1. Data Augmentation (Aumento de Dados)

### O problema

O modelo so ve cada imagem **exatamente igual** durante o treino. Se alguem escreve um "7" ligeiramente torto, o modelo pode nao reconhecer porque so viu "7" retos.

### A solucao

Aplicar **transformacoes aleatorias** nas imagens de treino a cada epoch:

```
Imagem original:     Rotacao 8°:      Deslocamento:
    ┌─────┐           ┌─────┐           ┌─────┐
    │  7  │    →      │  7  │    →      │   7 │
    │     │           │ /   │           │     │
    └─────┘           └─────┘           └─────┘
```

### Transformacoes usadas no script

```python
transforms.RandomRotation(10)                      # Rotacao ate 10°
transforms.RandomAffine(0, translate=(0.1, 0.1))   # Desloca ate 10% horizontal/vertical
```

### Como funciona na pratica

```
Epoch 1:  modelo ve o "7" original
Epoch 2:  modelo ve o "7" rotacionado 5° para a direita
Epoch 3:  modelo ve o "7" deslocado 2 pixels para cima
Epoch 4:  modelo ve o "7" rotacionado 3° para a esquerda e deslocado para baixo
...cada vez e uma versao DIFERENTE!
```

E como se tivessemos **infinitos dados de treino**, sem precisar coletar nenhuma imagem nova.

### Por que funciona

- O modelo aprende que "7 reto" e "7 torto" sao o MESMO digito
- Generaliza melhor para escrita de pessoas diferentes
- Reduz overfitting (o modelo nao pode "decorar" imagens se elas mudam a cada vez)

### Cuidado importante

Data Augmentation so e aplicado nos dados de **TREINO**! Os dados de teste devem permanecer originais, senao a avaliacao nao seria justa.

### Que transformacoes NAO usar no MNIST

- **Espelhamento horizontal/vertical:** um "6" espelhado vira "6" ao contrario, que nao existe. Um "9" espelhado vira algo parecido com "6"!
- **Rotacao grande (>15°):** um "6" rotacionado 180° vira "9"
- **Zoom excessivo:** pode cortar partes importantes do digito

A escolha das transformacoes depende do **dominio**. Para fotos de gatos, espelhamento horizontal e otimo. Para digitos, nao.

---

## 2. Batch Normalization (Normalizacao por Lote)

### O problema

Durante o treino, as ativacoes (saidas de cada camada) podem ficar **muito grandes** ou **muito pequenas**. Isso causa:

- Gradientes que "explodem" (valores enormes → treino instavel)
- Gradientes que "somem" (valores minusculos → modelo para de aprender)

Esse fenomeno se chama **Internal Covariate Shift**: a distribuicao das ativacoes muda a cada atualizacao de pesos, dificultando o aprendizado das camadas seguintes.

### A solucao

Batch Normalization (BatchNorm) **normaliza as ativacoes** dentro de cada mini-batch:

```
Antes do BatchNorm:               Depois do BatchNorm:
Ativacoes de um batch:             Ativacoes normalizadas:
[150.0, -200.0, 80.0, 0.5]  →    [0.8, -1.5, 0.3, -0.6]

As ativacoes ficam centradas em ~0 com desvio padrao ~1
```

### A formula (simplificada)

```
Para cada feature no batch:
  1. Calcular a media:        μ = media(valores)
  2. Calcular o desvio padrao: σ = std(valores)
  3. Normalizar:               x_norm = (x - μ) / σ
  4. Escalar e deslocar:       saida = γ * x_norm + β

  γ (gamma) e β (beta) sao parametros APRENDIVEIS!
  Isso permite que a rede "desfaca" a normalizacao se for melhor.
```

### No codigo

```python
self.bn1 = nn.BatchNorm2d(32)   # Para camadas convolucionais (2D)
self.bn_fc = nn.BatchNorm1d(128) # Para camadas densas (1D)

# Uso no forward:
x = self.relu(self.bn1(self.conv1(x)))   # Conv → BatchNorm → ReLU
```

### Ordem importa: Conv → BatchNorm → ReLU

1. **Conv** produz as ativacoes
2. **BatchNorm** normaliza (centra em 0, desvio padrao 1)
3. **ReLU** zera negativos

Se fizessemos ReLU antes do BatchNorm, metade dos valores ja seriam 0, e a normalizacao nao funcionaria bem.

### Beneficios

- **Treino mais rapido:** gradientes fluem melhor, o modelo converge mais cedo
- **Mais estavel:** permite usar learning rates maiores sem desestabilizar
- **Leve regularizacao:** a normalizacao por mini-batch introduz um ruido (cada batch tem media/std ligeiramente diferente), o que ajuda a evitar overfitting

### Train vs Eval

- **model.train():** BatchNorm calcula media e desvio padrao do mini-batch atual
- **model.eval():** BatchNorm usa a media e desvio padrao **acumulados** durante todo o treinamento (mais estavel para predicoes)

Por isso e **essencial** chamar `model.eval()` antes de avaliar!

---

## 3. Dropout (Regularizacao por Desligamento)

### O problema: overfitting

Quando o modelo e grande e os dados sao limitados, ele pode **decorar** os dados de treino em vez de aprender padroes gerais. Isso se chama **overfitting**.

Sinais de overfitting:
- Acuracia no treino: 99.9% (quase perfeita)
- Acuracia no teste: 97% (bem pior)
- O modelo "decora" o treino mas nao generaliza

### A solucao

Dropout **desliga neuronios aleatorios** durante o treinamento:

```
Sem Dropout:                    Com Dropout(0.25):
○ → ○ → ○ → ○ → ○              ○ → ○ → ✗ → ○ → ○
○ → ○ → ○ → ○ → ○              ○ → ✗ → ○ → ○ → ✗
○ → ○ → ○ → ○ → ○              ○ → ○ → ○ → ✗ → ○
○ → ○ → ○ → ○ → ○              ○ → ○ → ○ → ○ → ○
Todos ativos                    25% desligados (✗) aleatoriamente
```

A cada forward pass, neuronios DIFERENTES sao desligados. O modelo nunca sabe quais estarao ativos.

### Por que funciona

**Analogia da equipe de trabalho:**
- Sem Dropout: voce tem uma equipe de 4 pessoas, mas so 1 faz todo o trabalho. Se essa pessoa faltar, a equipe nao funciona.
- Com Dropout: todos precisam saber fazer tudo, porque qualquer um pode "faltar" (ser desligado). A equipe fica mais ROBUSTA.

Resultado: o conhecimento e **espalhado** entre todos os neuronios. Nenhum neuronio individual se torna indispensavel.

### No codigo

```python
self.dropout = nn.Dropout(0.25)   # 25% de chance de desligar cada neuronio

# Uso no forward:
x = self.dropout(self.relu(self.bn_fc(self.fc1(x))))
```

### Valores comuns de Dropout

| Valor | Efeito | Quando usar |
|-------|--------|-------------|
| 0.1 - 0.2 | Leve | Modelos pequenos ou pouco overfitting |
| **0.25** | **Moderado** | **Bom ponto de partida (nosso caso)** |
| 0.5 | Forte | Modelos grandes com muito overfitting |
| > 0.5 | Muito forte | Raramente usado (pode "underfittar") |

### Train vs Eval

- **model.train():** Dropout esta ATIVO (desliga neuronios)
- **model.eval():** Dropout esta DESATIVADO (todos os neuronios ativos, mas com pesos escalados)

---

## 4. Learning Rate Scheduler

### O problema

Com learning rate (lr) fixo:
- **lr alto** durante todo o treino: no inicio aprende rapido, mas no final fica "saltando" em torno do otimo sem convergir
- **lr baixo** durante todo o treino: muito lento para aprender, pode ficar preso em minimos locais

### A solucao

Comecar com lr alto e **reduzir gradualmente**:

```
Epochs 1-5:   lr = 0.001     ████████████  Passos grandes (aprende rapido)
Epochs 6-10:  lr = 0.0005    ██████        Passos medios (refinando)
Epochs 11-15: lr = 0.00025   ███           Passos pequenos (ajuste fino)
```

### Analogia

Imagine que voce esta procurando um endereco:
1. **Inicio (lr alto):** voce dirige rapido ate o bairro certo
2. **Meio (lr medio):** dirige mais devagar ate achar a rua
3. **Final (lr baixo):** anda devagar olhando os numeros das casas

Se voce dirigisse devagar o tempo todo, demoraria horas. Se dirigisse rapido o tempo todo, passaria do endereco.

### No codigo

```python
# StepLR: a cada 5 epochs, multiplica lr por 0.5 (corta pela metade)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# No loop de treino, apos cada epoch:
scheduler.step()   # Atualiza o lr
```

### Outros schedulers populares

| Scheduler | Comportamento | Quando usar |
|-----------|--------------|-------------|
| **StepLR** | Corta lr a cada N epochs | Simples e eficaz (nosso caso) |
| CosineAnnealingLR | lr segue uma curva cosseno (sobe e desce) | Treinos longos |
| ReduceLROnPlateau | Corta lr quando a loss para de melhorar | Quando nao sabe quantas epochs usar |
| OneCycleLR | lr sobe e depois desce | Treinos rapidos, state-of-the-art |

---

## 5. Mais Filtros = Mais Capacidade

### CNN basica vs CNN avancada

```
CNN Basica:                     CNN Avancada:
Conv1: 1 → 8 filtros            Conv1: 1 → 32 filtros
Conv2: 8 → 16 filtros           Conv2: 32 → 64 filtros
FC: 400 → 64 → 10              FC: 3136 → 128 → 10
~11.000 parametros              ~240.000 parametros
```

### Por que mais filtros ajudam

- **8 filtros** so podem detectar 8 padroes diferentes (poucas bordas e curvas)
- **32 filtros** detectam 32 padroes: bordas em todas as direcoes, cantos, curvas, texturas...
- Na segunda camada, os 64 filtros **combinam** os 32 padroes anteriores em formas mais complexas

**Analogia:** e como ter mais "olhos" especializados. Um time de 32 especialistas encontra mais padroes do que um time de 8.

### Cuidado: mais parametros = mais risco de overfitting

E por isso que usamos BatchNorm + Dropout + Data Augmentation! Eles trabalham juntos:
- **Mais filtros:** mais capacidade de aprender (pode overfittar)
- **BatchNorm:** estabiliza o treino
- **Dropout:** impede que neuronios individuais dominem
- **Data Augmentation:** mais variedade nos dados

---

## 6. DataLoader: a forma correta de iterar sobre dados

### Na CNN basica (loop manual)

```python
for i in range(0, len(X_train), batch_size):
    xb = X_train[i:i+batch_size]
    yb = y_train[i:i+batch_size]
```

Problema: os dados sao sempre processados na **mesma ordem**. O modelo pode "viciar" na sequencia.

### Na CNN avancada (DataLoader)

```python
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

for xb, yb in train_loader:
    ...
```

Vantagens:
- **shuffle=True:** embaralha os dados a cada epoch (ordem diferente toda vez)
- **Gerencia batches** automaticamente (inclusive o ultimo batch menor)
- **Integra com Data Augmentation** (transformacoes aplicadas on-the-fly)

---

## Resumo: como cada tecnica contribui

```
Problema                          Solucao                      Impacto
─────────────────────────────     ──────────────────────       ──────────
Poucos dados de treino        →   Data Augmentation            +0.5-1.0%
Treino instavel               →   Batch Normalization          +0.3-0.5%
Overfitting                   →   Dropout                      +0.2-0.5%
LR fixo nao e otimo           →   Learning Rate Scheduler      +0.1-0.3%
Poucos filtros                →   Mais filtros (32→64)         +0.3-0.5%
Dados sempre na mesma ordem   →   DataLoader com shuffle       +0.1-0.2%
                                                               ─────────
                                  Total combinado:             ~98% → 99%+
```

**IMPORTANTE:** os valores acima sao aproximados e variam a cada execucao. O ponto e que cada tecnica contribui um pouco, e **juntas** fazem a diferenca.

## Proximo passo

Execute o script e veja o resultado!
```bash
python 03_cnn/cnn_avancada.py
```

Depois de completar este projeto, veja as sugestoes no README.md para continuar aprendendo (outros datasets, Transfer Learning, etc).
