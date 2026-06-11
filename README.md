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
├── requirements.txt             # Dependências Python
├── README.md                    # Este arquivo
├── dataset/                     # Dataset LC25000 (baixar separadamente)
│   └── LC25000/
│       ├── colon_aca/           # Colon adenocarcinoma
│       ├── colon_n/             # Colon benign tissue
│       ├── lung_aca/            # Lung adenocarcinoma
│       ├── lung_n/              # Lung benign tissue
│       └── lung_scc/            # Lung squamous cell carcinoma
└── results/                     # Resultados gerados automaticamente
    ├── figures/                 # Figuras geradas
    │   ├── experimento1/
    │   ├── experimento2/
    │   ├── experimento3/
    │   ├── experimento4/
    │   ├── experimento5/
    │   └── experimento6/
    ├── masks/                   # Máscaras de segmentação
    ├── features/                # Features extraídas
    ├── classification/          # Resultados de classificação
    └── resumo.txt               # Resumo dos resultados
```

## Instalação

### 1. Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### 2. Clonar/Copiar o projeto

```bash
cd PID_final
```

### 3. Criar ambiente virtual (recomendado)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 4. Instalar dependências

```bash
pip install -r requirements.txt
```

### 5. Baixar o dataset

Baixar o dataset **LC25000** do Kaggle:
https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images/data

Extrair para a estrutura:
```
dataset/LC25000/
    colon_aca/
    colon_n/
    lung_aca/
    lung_n/
    lung_scc/
```

## Uso

### Pipeline Completo

```bash
python main.py
```

### Executar experimentos específicos

```bash
# Apenas experimento 1
python main.py --exp 1

# Experimentos 1 a 3
python main.py --exp 1 2 3

# Experimentos 4 a 6 (requer resultados anteriores)
python main.py --exp 4 5 6
```

### Opções adicionais

```bash
# Limitar número de imagens por classe
python main.py --max-images 100

# Selecionar classes diferentes
python main.py --class1 lung_aca --class2 lung_n

# Combinar opções
python main.py --exp 1 2 3 --max-images 50 --class1 colon_aca --class2 colon_n
```

## Configuração

Edite `config.py` para personalizar:

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `CLASS_1`, `CLASS_2` | Classes do dataset | `colon_aca`, `colon_n` |
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
- Gera: histogramas, visualizações dos canais

### Experimento 2 — Segmentação
- Aplica Multi-Otsu (3 classes) ou Máxima Entropia
- Gera máscara binária (tecido vs. fundo)
- Avalia: fração de área, número de regiões, circularidade
- Analisa impacto de textura e heterogeneidade

### Experimento 3 — Morfologia Matemática
- Abertura (remove ruído pequeno)
- Fechamento (preenche buracos)
- Elemento estruturante: disco (preserva formas arredondadas)
- Quantifica melhoria: solidez, regiões removidas, buracos preenchidos

### Experimento 4 — Extração de Características
- **Haralick (GLCM)**: 6 descritores (contrast, dissimilarity, homogeneity, energy, correlation, ASM)
  - Distâncias: [1, 3, 5] pixels
  - Ângulos: [0°, 45°, 90°, 135°]
- **Fractal**: Box-counting multiescala
  - Escalas: [2, 4, 8, 16, 32, 64]
  - Regressão linear log-log para dimensão fractal
- **Cor**: Média e desvio de R, G, B nas regiões segmentadas
- **Geométrica**: Área, solidez, excentricidade

### Experimento 5 — Normalização e Seleção
- Normalização: Z-score ou Min-Max
- RELIEF-F: ranqueamento considerando k vizinhos
- Seleciona top-N features mais relevantes
- Gera: distribuições, scores, correlações

### Experimento 6 — Visualização e Classificação
- Projeções PCA (2D/3D) e t-SNE
- Classificadores: SVM, Random Forest, KNN, LDA
- Validação cruzada estratificada
- Curvas ROC, matriz de confusão
- Fronteira de decisão visual
- Estatísticas por classe (média, desvio, IC)

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
