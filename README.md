# Pipeline Integrado de Segmentação, Representação e Análise de Características em Imagens Histopatológicas

**PPGCC - Processamento de Imagens Digitais (PID)**
**Desafio 6 - Final**
**Prof. Leandro**

## Descrição

Pipeline completo para análise de imagens histopatológicas, incluindo:
- Conversão e análise de canais HSV
- Segmentação por Multi-Otsu / Máxima Entropia
- Refinamento com Morfologia Matemática
- Extração de características (Haralick + Fractal)
- Normalização e Seleção de Features (RELIEF)
- Visualização e Classificação Supervisionada

## Estrutura do Projeto

```
PID_final/
├── config.py                    # Configurações globais do pipeline
├── utils.py                     # Funções utilitárias
├── main.py                      # Script principal (pipeline completo)
├── experimento1_hsv.py          # Experimento 1: Representação HSV
├── experimento2_segmentacao.py  # Experimento 2: Segmentação
├── experimento3_morfologia.py   # Experimento 3: Morfologia Matemática
├── experimento4_features.py     # Experimento 4: Extração de Características
├── experimento5_relief.py       # Experimento 5: Normalização e Seleção
├── experimento6_classificacao.py# Experimento 6: Visualização e Classificação
├── discussao_final.py           # Seção 6.7: Discussão Final
├── requirements.txt             # Dependências Python
├── README.md                    # Este arquivo
└── dataset/LC25000/             # Dataset (250 imagens por classe)
    ├── colon_aca/               # Colon adenocarcinoma
    ├── colon_n/                 # Colon benign tissue
    ├── lung_aca/                # Lung adenocarcinoma
    ├── lung_n/                  # Lung benign tissue
    └── lung_scc/                # Lung squamous cell carcinoma
```

## Instalação

### 1. Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### 2. Criar ambiente virtual (recomendado)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Dataset

O dataset LC25000 deve estar em `dataset/LC25000/` com 250 imagens por classe (5% do total de 5000, conforme exigido pela atividade).

## Uso

### Pipeline Completo

```bash
python main.py
```

### Escolher Classes

```bash
python main.py --class1 colon_aca --class2 colon_n
python main.py --class1 lung_aca --class2 lung_n
python main.py --class1 lung_aca --class2 lung_scc
```

### Executar Experimentos Específicos

```bash
python main.py --exp 1 2 3
python main.py --exp 4 5 6
```

## Configuração

Edite `config.py` para personalizar:

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `CLASS_1`, `CLASS_2` | Classes do dataset | `colon_aca`, `colon_n` |
| `MAX_IMAGES_PER_CLASS` | Imagens por classe | `250` (5% de 5000) |
| `HSV_CHANNEL_SELECTED` | Canal HSV para processamento | `S` (Saturação) |
| `SEGMENTATION_METHOD` | Método de segmentação | `multi_otsu` |
| `STRUCTURING_ELEMENT_TYPE` | Elemento estruturante | `disk` |
| `NORMALIZATION_METHOD` | Método de normalização | `zscore` |
| `RELIEF_N_FEATURES` | Nº de features selecionadas | 6 |
| `CLASSIFIER` | Classificador | `svm` |
| `TEST_SIZE` | Proporção teste | 0.3 |
| `CV_FOLDS` | Folds validação cruzada | 5 |

## Experimentos

### Experimento 1 — Representação HSV
- Converte imagens RGB → HSV
- Analisa canais H, S, V separadamente
- Seleciona canal mais discriminativo (S por padrão)

### Experimento 2 — Segmentação
- Aplica Multi-Otsu (3 classes) ou Máxima Entropia
- Gera máscara binária (tecido vs. fundo)
- Avalia: fração de área, número de regiões, circularidade

### Experimento 3 — Morfologia Matemática
- Abertura (remove ruído pequeno)
- Fechamento (preenche buracos)
- Elemento estruturante: disco

### Experimento 4 — Extração de Características
- **Haralick (GLCM)**: 6 descritores × 3 distâncias × 4 ângulos
- **Fractal**: Box-counting multiescala (6 escalas)
- **Cor**: Média e desvio de R, G, B
- **Geométrica**: Área, solidez, excentricidade

### Experimento 5 — Normalização e Seleção
- Normalização: Z-score ou Min-Max
- RELIEF-F: ranqueamento considerando k vizinhos
- Seleciona top-N features mais relevantes

### Experimento 6 — Visualização e Classificação
- Projeções PCA (2D/3D) e t-SNE
- Classificadores: SVM, Random Forest, KNN, LDA
- Validação cruzada estratificada
- Curvas ROC, matriz de confusão

### Seção 6.7 — Discussão Final
- Integração crítica de todos os experimentos
- Análise de limitações e complementaridade dos descritores

## Resultados

Após execução, os resultados são salvos em `results/`:

```
results/
├── figures/
│   ├── experimento1/    # Figuras HSV
│   ├── experimento2/    # Figuras segmentação
│   ├── experimento3/    # Figuras morfologia
│   ├── experimento4/    # Figuras features
│   ├── experimento5/    # Figuras normalização/seleção
│   └── experimento6/    # Figuras classificação
└── resumo.txt           # Resumo numérico
```

## Dependências

| Pacote | Versão | Uso |
|--------|--------|-----|
| opencv-python | ≥4.8.0 | Processamento de imagens |
| scikit-image | ≥0.21.0 | GLCM, segmentação, morfologia |
| scikit-learn | ≥1.3.0 | Classificação, normalização |
| numpy | ≥1.24.0 | Computação numérica |
| scipy | ≥1.11.0 | Estatísticas, regressão |
| matplotlib | ≥3.7.0 | Visualização |
| seaborn | ≥0.12.0 | Gráficos estatísticos |
| Pillow | ≥10.0.0 | Carregamento de imagens |
| tqdm | ≥4.65.0 | Barras de progresso |
| pandas | ≥2.0.0 | Manipulação de dados |

## Referências

1. Haralick, R.M., Shanmugam, K., Dinstein, I. (1973). "Textural Features for Image Classification." *IEEE Transactions on Systems, Man, and Cybernetics*.

2. Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms." *IEEE Transactions on Systems, Man, and Cybernetics*.

3. Li, C.H., Lee, C.K. (1993). "Minimum Cross Entropy Thresholding." *Pattern Recognition*.

4. Kononenko, I. (1994). "Estimating Attributes: Analysis and Extensions of RELIEF." *ECML-94*.

5. Buda, M., Maki, A., Mazurowski, M.A. (2018). "A systematic study of the class imbalance problem in convolutional neural networks." *Neural Networks*.

6. Kather, J.N. et al. (2019). "Predicting survival from colorectal cancer histology slides using deep learning." *PLOS ONE*.

## Autores

Grupo - PPGCC (Programa de Pós-Graduação em Ciência da Computação)
Disciplina: Processamento de Imagens Digitais (PID)