# TUTORIAL - Pipeline de Analise de Imagens Histopatologicas

## 1. Visao Geral

O pipeline processa imagens histopatologicas coradas com H&E (Hematoxilina e Eosina) e realiza classificacao binaria entre dois tipos de tecido. O fluxo completo e:

```
Imagem RGB
    |
    v
[Exp 1] Converte RGB -> HSV, seleciona canal S (Saturacao)
    |
    v
[Exp 2] Segmenta tecido vs fundo (Multi-Otsu 3 classes)
    |
    v
[Exp 3] Refina mascara com morfologia (abertura + fechamento)
    |
    v
[Exp 4] Extrai features: Haralick + Fractal + Cor + Geometria
    |
    v
[Exp 5] Normaliza (Z-score) e seleciona features (RELIEF)
    |
    v
[Exp 6] Classifica (SVM, RF, KNN) + visualiza (PCA, t-SNE)
    |
    v
[6.7]  Discussao final integrando todos os resultados
```

---

## 2. Arquivos do Projeto

| Arquivo | Responsabilidade |
|---------|-----------------|
| `config.py` | Central de configuracao - todos os parametros ajustaveis |
| `utils.py` | Funcoes auxiliares - carregar imagens, salvar figuras, normalizar, classificar, estatisticas |
| `main.py` | Orquestrador - carrega dados e chama cada experimento em sequencia |
| `experimento1_hsv.py` | Converte RGB->HSV, analisa canais H/S/V, seleciona o mais discriminativo |
| `experimento2_segmentacao.py` | Segmenta tecido vs fundo (Multi-Otsu ou Maxima Entropia) |
| `experimento3_morfologia.py` | Refina mascara com abertura + fechamento (elemento disco) |
| `experimento4_features.py` | Extrai ~48 features: Haralick (GLCM), Fractal, Cor, Geometricas |
| `experimento5_relief.py` | Normaliza (Z-score) e ranqueia features com RELIEF-F |
| `experimento6_classificacao.py` | Classifica (SVM/RF/KNN), visualiza (PCA/t-SNE), curvas ROC |
| `discussao_final.py` | Gera relatorio textual integrando todos os resultados |

---

## 3. Como Executar

### 3.1 Pipeline Completo (padrao)

```bash
python main.py
```

Usa as classes padrao definidas em `config.py`:
- **Classe 1:** `colon_aca` (Colon Adenocarcinoma)
- **Classe 2:** `colon_n` (Colon Benign Tissue)
- **Imagens:** 250 por classe (5% do total, conforme exigido)

### 3.2 Escolher Classes

```bash
# Lung cancer vs benigno
python main.py --class1 lung_aca --class2 lung_n

# Dois tipos de cancer de pulmao
python main.py --class1 lung_aca --class2 lung_scc
```

### 3.3 Executar Experimentos Especificos

Os experimentos sao sequenciais - cada um depende do anterior. **Nao e possivel pular etapas**, pois os resultados ficam em memoria durante a execucao:

```bash
# So HSV + Segmentacao + Morfologia
python main.py --exp 1 2 3

# Pipeline completo (experimentos 1 a 6 + discussao)
python main.py

# Para rodar 4-5-6, e preciso incluir 1-2-3 na mesma execucao:
python main.py --exp 1 2 3 4 5 6
```

> **Nota:** `--exp 4 5 6` sozinho falha porque o Exp 4 precisa das mascaras refinadas do Exp 3, que precisa da segmentacao do Exp 2, que precisa do canal HSV do Exp 1. Sempre inclua os experimentos anteriores.

---

## 4. Configuracao (config.py)

Tudo que pode ser ajustado esta em `config.py`. Os parametros estao organizados por experimento:

### 4.1 Classes e Dataset

```python
CLASS_1 = "colon_aca"           # Pasta da classe 1
CLASS_2 = "colon_n"             # Pasta da classe 2
CLASS_1_NAME = "Colon Adenocarcinoma"  # Nome legivel
CLASS_2_NAME = "Colon Benign Tissue"
MAX_IMAGES_PER_CLASS = 250      # Fixo: 5% de 5000
```

**Para mudar as classes:** altere `CLASS_1` e `CLASS_2` OU use `--class1` / `--class2` na linha de comando.

### 4.2 Experimento 1 - HSV

```python
HSV_CHANNEL_SELECTED = "S"  # "H", "S" ou "V"
```

O canal S (Saturacao) e o padrao porque tecido corado tem alta saturacao e o fundo branco tem saturacao ~0.

### 4.3 Experimento 2 - Segmentacao

```python
SEGMENTATION_METHOD = "multi_otsu"  # ou "max_entropy"
MULTI_OTSU_CLASSES = 3              # Classes do Multi-Otsu
```

**Multi-Otsu (3 classes):** separa fundo branco, tecido rosa (eosina) e tecido roxo (hematoxilina). A mascara binaria e "tecido vs fundo".

**Maxima Entropia:** alternativa quando a distribuicao de intensidades nao e claramente multimodal.

### 4.4 Experimento 3 - Morfologia

```python
STRUCTURING_ELEMENT_TYPE = "disk"   # "disk", "square", "diamond"
STRUCTURING_ELEMENT_SIZE = 5        # Tamanho base do elemento
MORPHOLOGY_OPERATIONS = [
    ("opening", 3),   # Erosao -> Dilatacao (remove ruido pequeno)
    ("closing", 5),   # Dilatacao -> Erosao (preenche buracos)
]
```

**Abertura:** remove pequenas regioes espurias (ruido).
**Fechamento:** preenche buracos internos no tecido segmentado.
**Disco:** preserva formas arredondadas de nucleos celulares.

### 4.5 Experimento 4 - Features

```python
GLCM_DISTANCES = [1, 3, 5]                    # Distancias em pixels
GLCM_ANGLES = [0, 0.7854, 1.5708, 2.3561]     # 0, 45, 90, 135 graus
GLCM_LEVELS = 32                               # Niveis de quantizacao
HARALICK_FEATURES = ["contrast", "dissimilarity", "homogeneity",
                      "energy", "correlation", "ASM"]
FRACTAL_SCALES = [2, 4, 8, 16, 32, 64]        # Escalas do box-counting
```

**Haralick (GLCM):** 6 descritores x 3 distancias x 4 angulos = media -> 6 valores por imagem.

**Fractal:** box-counting em 6 escalas -> regressao log-log -> dimensao fractal + R2 + contagens por escala + dimensoes locais.

**Cor:** media e desvio de R, G, B nas regioes segmentadas -> 6 valores.

**Geometrica:** numero de regioes, area media/desvio/maxima, solidez media -> 5 valores.

**Total:** ~48 features por imagem.

### 4.6 Experimento 5 - Normalizacao e Selecao

```python
NORMALIZATION_METHOD = "zscore"  # ou "minmax"
RELIEF_N_FEATURES = 6            # Top-N features selecionadas
RELIEF_N_NEIGHBORS = 10          # Vizinhos do RELIEF-F
```

**Z-score:** transforma cada feature para media=0 e desvio=1. Necessario porque as features tem escalas muito diferentes.

**RELIEF-F:** ranqueia features comparando vizinhos da mesma classe (hits) vs classes diferentes (misses).

### 4.7 Experimento 6 - Classificacao

```python
CLASSIFIER = "svm"          # "svm", "random_forest", "knn"
SVM_KERNEL = "rbf"          # Kernel SVM
SVM_C = 1.0                 # Regularizacao
TEST_SIZE = 0.3             # 30% para teste
CV_FOLDS = 5                # Validacao cruzada 5-fold
RANDOM_STATE = 42           # Reprodutibilidade
```

**SVM (RBF):** classificador principal.
**Comparacao:** tambem treina Random Forest e KNN.
**Validacao cruzada:** 5-fold estratificado.

---

## 5. Fluxo de Dados Detalhado

### 5.1 main.py - Orquestracao

```
1. Parse argumentos (--exp, --class1, --class2)
2. Cria diretorios de resultados
3. Chama load_dataset() -> images_rgb, labels_dict, paths_dict
4. Para cada experimento (1-6):
   - Verifica se deve executar
   - Chama run_experiment_N()
   - Passa resultados para o proximo experimento
5. Chama run_discussao_final()
6. Salva resumo em results/resumo.txt
```

### 5.2 Dependencias entre Experimentos

```
Exp 1 (HSV) --> selected_channels --> Exp 2 (Segmentacao)
                                              |
                                              v
                                        segmentation_results --> Exp 3 (Morfologia)
                                                                    |
                                                                    v
                                                              refined_results --> Exp 4 (Features)
                                                                                    |
                                                                                    v
                                                                              all_features --> Exp 5 (RELIEF)
                                                                                                |
                                                                                                v
                                                                                          selected_features --> Exp 6 (Classificacao)
                                                                                                                    |
                                                                                                                    v
                                                                                                              exp6_results --> 6.7 (Discussao)
```

---

## 6. Saidas Geradas

Apos a execucao, os resultados sao salvos em `results/`:

```
results/
├── figures/
│   ├── experimento1/
│   │   ├── exp1_canais_hsv_*.png       # Canais H, S, V
│   │   ├── exp1_histogramas_hsv_*.png  # Histogramas dos canais
│   │   └── exp1_canal_selecionado_*.png
│   ├── experimento2/
│   │   ├── exp2_segmentacao_*_img*.png # Original + Segmentacao + Mascara
│   │   ├── exp2_histograma_limiares_*.png
│   │   └── exp2_metricas_boxplot.png
│   ├── experimento3/
│   │   ├── exp3_morfologia_*_img*.png  # Antes/depois morfologia
│   │   ├── exp3_regioes_extraidas_*.png
│   │   ├── exp3_pipeline_morfologico_*.png
│   │   └── exp3_metricas_melhoria.png
│   ├── experimento4/
│   │   ├── exp4_fractal_*_img*.png     # Assinatura fractal
│   │   ├── exp4_glcm_*.png             # Heatmap da GLCM
│   │   └── exp4_fractal_boxplot.png
│   ├── experimento5/
│   │   ├── exp5_normalizacao_distribuicoes.png
│   │   ├── exp5_relief_scores.png      # Ranqueamento RELIEF
│   │   ├── exp5_distribuicoes_selecionadas.png
│   │   ├── exp5_comparacao_escalas.png
│   │   └── exp5_correlacao_selecionadas.png
│   ├── experimento6/
│   │   ├── exp6_pca_2d_*.png           # Projecao PCA 2D
│   │   ├── exp6_pca_3d_*.png           # Projecao PCA 3D
│   │   ├── exp6_tsne_2d_*.png          # Projecao t-SNE
│   │   ├── exp6_pca_variancia_*.png    # Variancia explicada
│   │   ├── exp6_matriz_confusao.png
│   │   ├── exp6_curvas_roc.png
│   │   ├── exp6_comparacao_classificadores.png
│   │   ├── exp6_impacto_dimensionalidade.png
│   │   └── exp6_fronteira_decisao.png
│   └── discussao_final_resumo.png
├── discussao_final.txt
└── resumo.txt
```

---

## 7. Guia Rapido de Modificacao

### Mudar classes
```bash
python main.py --class1 lung_aca --class2 lung_scc
```

### Mudar metodo de segmentacao
Edite `config.py`:
```python
SEGMENTATION_METHOD = "max_entropy"
```

### Mudar numero de features selecionadas
```python
RELIEF_N_FEATURES = 10  # era 6
```

### Mudar classificador
```python
CLASSIFIER = "random_forest"
```

### Mudar parametros GLCM
```python
GLCM_DISTANCES = [1, 2, 3, 4, 5]
GLCM_LEVELS = 64
```

### Mudar operacoes morfologicas
```python
MORPHOLOGY_OPERATIONS = [
    ("opening", 5),
    ("closing", 8),
]
STRUCTURING_ELEMENT_TYPE = "square"
```

---

## 8. Resumo Tecnico de Cada Experimento

### Experimento 1 - HSV
- **Entrada:** imagens RGB (250 por classe)
- **Processo:** converte RGB->HSV, calcula estatisticas de cada canal
- **Saida:** canal S selecionado para cada imagem
- **Justificativa:** Saturacao separa tecido corado (alto S) de fundo branco (S~0)

### Experimento 2 - Segmentacao
- **Entrada:** canal S de cada imagem
- **Processo:** Multi-Otsu com 3 classes -> limiares -> mascara binaria
- **Metricas:** fracao de area, numero de regioes, circularidade, variancia tecidual
- **Saida:** mascara binaria + metricas + analise de textura

### Experimento 3 - Morfologia
- **Entrada:** mascara binaria do Exp 2
- **Processo:** abertura (remove ruido) -> fechamento (preenche buracos)
- **Elemento:** disco (preserva formas arredondadas)
- **Metricas:** melhoria de solidez, regioes removidas, buracos preenchidos
- **Saida:** mascara refinada + regioes coloridas extraidas

### Experimento 4 - Features
- **Entrada:** canal S + mascara refinada + imagem RGB
- **Processo:**
  - GLCM: 6 descritores x 3 distancias x 4 angulos
  - Fractal: box-counting em 6 escalas -> regressao log-log
  - Cor: media e desvio de R,G,B na regiao segmentada
  - Geometria: areas, solidez, numero de regioes
- **Saida:** vetor de ~48 features por imagem

### Experimento 5 - Normalizacao e Selecao
- **Entrada:** matriz de features (n_amostras x ~48)
- **Processo:** Z-score -> RELIEF-F com k=10 vizinhos -> ranqueamento
- **Saida:** top-6 features selecionadas + scores

### Experimento 6 - Classificacao
- **Entrada:** features selecionadas (n_amostras x 6)
- **Processo:** Split 70/30, treina SVM/RF/KNN, validacao cruzada 5-fold, PCA, t-SNE, ROC
- **Saida:** acuracia, matriz de confusao, relatorio de classificacao

### Secao 6.7 - Discussao Final
- **Entrada:** todos os resultados dos experimentos 1-6
- **Processo:** gera relatorio textual analisando criticamente cada etapa
- **Saida:** results/discussao_final.txt + figura resumo
