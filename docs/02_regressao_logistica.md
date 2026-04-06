# Regressao Logistica para Classificacao de Imagens

## O que e classificacao?

Classificacao e a tarefa de **atribuir uma categoria** a um dado de entrada.

No nosso caso:
- **Entrada:** uma imagem de 28x28 pixels
- **Saida:** um digito de 0 a 9
- **Tarefa:** o modelo olha a imagem e diz "isso e um 7"

## O que e Regressao Logistica?

Apesar do nome ter "regressao", e um modelo de **classificacao**. O nome vem da funcao logistica (sigmoide) usada internamente.

### Como funciona (de forma simples):

1. O modelo recebe 784 numeros (os pixels achatados)
2. Cada pixel recebe um **peso** (um numero que diz "quanto este pixel importa")
3. O modelo calcula uma **soma ponderada**: `resultado = peso_1 * pixel_1 + peso_2 * pixel_2 + ... + peso_784 * pixel_784`
4. Faz isso **10 vezes** (uma para cada digito 0-9)
5. A classe com o maior resultado e a predicao

### Analogia

Imagine que voce esta julgando se um desenho e um "0" ou um "1":
- Se os pixels no CENTRO da imagem estao acesos, provavelmente e um "0" (a barriga do zero)
- Se os pixels numa LINHA VERTICAL estao acesos, provavelmente e um "1"

O modelo aprende exatamente isso -- quais pixels importam para cada digito.

## Flattening: por que achatar a imagem?

A Regressao Logistica espera dados em formato de **tabela** (cada amostra e uma linha).

```
Imagem 2D (28x28):          Vetor 1D (784):
┌────────────────┐
│ 0  0  128  200 │           [0, 0, 128, 200, 0, 255, 64, 0, ...]
│ 0  255  64   0 │  ──→      (784 numeros em sequencia)
│ ...            │
└────────────────┘
```

**Problema:** ao achatar, perdemos a nocao de que pixels vizinhos estao proximos. O modelo nao sabe que o pixel da posicao (0,0) esta ao lado do pixel (0,1). Isso e uma limitacao que CNNs resolvem (Etapa 4).

## StandardScaler: normalizacao de features

O StandardScaler transforma os dados para ter:
- **Media = 0** (centrado no zero)
- **Desvio padrao = 1** (espalhamento padronizado)

```
Antes:  pixel pode ser 0, 50, 128, 255...  (escala grande)
Depois: pixel fica entre -2 e +2 aproximadamente (escala padrao)
```

**Por que isso e necessario?**
Imagine comparar o peso (em kg) e a altura (em cm) de uma pessoa. 70kg e 170cm estao em escalas muito diferentes. O StandardScaler coloca tudo na mesma escala para que nenhuma feature domine as outras.

## Metricas de avaliacao

### Acuracia (Accuracy)
A metrica mais simples: **porcentagem de acertos**.

```
Acuracia = acertos / total = 9200 / 10000 = 0.92 (92%)
```

### Classification Report (Relatorio por classe)

| Metrica | O que mede | Exemplo |
|---------|-----------|---------|
| **Precision** | Dos que o modelo disse ser "5", quantos realmente eram "5"? | Se disse "5" 100 vezes e acertou 95: precision = 0.95 |
| **Recall** | De todos os "5" verdadeiros, quantos o modelo encontrou? | Se existiam 100 "cincos" e achou 90: recall = 0.90 |
| **F1-Score** | Media harmonica entre precision e recall | Equilibrio entre as duas metricas |

Essas metricas por classe sao uteis porque alguns digitos sao mais dificeis que outros. Por exemplo, 4 e 9 sao frequentemente confundidos (parecem similares!).

## Limitacoes da Regressao Logistica

1. **Modelo linear:** so consegue tracar fronteiras retas entre classes. Padroes complexos (curvas, combinacoes) sao dificeis.
2. **Ignora estrutura espacial:** trata a imagem como uma lista de numeros, perdendo a relacao entre pixels vizinhos.
3. **Acuracia limitada:** ~92% no MNIST. Parece bom, mas significa ~800 erros em 10.000 imagens.

## Proximo passo

- **docs/03_pca_e_reducao.md** -- como melhorar a Regressao Logistica com reducao de dimensionalidade
- **01_regressao_logistica/pca_grid_search.py** -- o codigo correspondente
