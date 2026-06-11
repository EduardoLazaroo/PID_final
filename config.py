"""
=============================================================================
config.py - Configurações globais do Pipeline de Análise de Imagens Histopatológicas
=============================================================================
Centraliza todos os parâmetros utilizados nos experimentos, permitindo
reprodutibilidade e ajuste fino do pipeline.
"""

import os

# =============================================================================
# CAMINHOS E DIRETÓRIOS
# =============================================================================

# Diretório raiz do dataset LC25000
# Ajuste conforme sua estrutura local. Espera-se:
#   DATASET_ROOT/
#       colon_aca/       (Colon adenocarcinoma)
#       colon_n/         (Colon benign tissue)
#       lung_aca/        (Lung adenocarcinoma)
#       lung_n/          (Lung benign tissue)
#       lung_scc/        (Lung squamous cell carcinoma)
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "dataset", "LC25000")

# Diretório para salvar resultados (figuras, tabelas, máscaras)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Subdiretórios de resultados
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MASKS_DIR = os.path.join(RESULTS_DIR, "masks")
FEATURES_DIR = os.path.join(RESULTS_DIR, "features")
CLASSIFICATION_DIR = os.path.join(RESULTS_DIR, "classification")

# =============================================================================
# SELEÇÃO DE CLASSES
# =============================================================================
# Escolha duas classes do dataset LC25000.
# Opções disponíveis:
#   - "colon_aca"   : Colon adenocarcinoma (2500 imagens)
#   - "colon_n"     : Colon benign tissue (2500 imagens)
#   - "lung_aca"    : Lung adenocarcinoma (2500 imagens)
#   - "lung_n"      : Lung benign tissue (2500 imagens)
#   - "lung_scc"    : Lung squamous cell carcinoma (2500 imagens)
#
# Padrão: colon_aca vs colon_n (estudo de câncer de cólon)

CLASS_1 = "colon_aca"
CLASS_2 = "colon_n"

# Nomes legíveis para relatórios
CLASS_1_NAME = "Colon Adenocarcinoma"
CLASS_2_NAME = "Colon Benigno"

MIN_FRACTION = 0.05
MIN_FRACTION = 0.05

# Número máximo de imagens por classe (None = usar todas disponíveis)
MAX_IMAGES_PER_CLASS = None

# =============================================================================
# EXPERIMENTO 1 - REPRESENTAÇÃO HSV
# =============================================================================
# Canal HSV selecionado para processamento posterior
# Opções: "H" (Hue), "S" (Saturation), "V" (Value)
# Justificativa: O canal S (Saturation) captura a pureza da cor, sendo
# particularmente útil para diferenciar regiões coradas (tecido) de fundo
# branco em imagens histopatológicas H&E.
HSV_CHANNEL_SELECTED = "S"

# =============================================================================
# EXPERIMENTO 2 - SEGMENTAÇÃO
# =============================================================================
# Método de segmentação: "multi_otsu" ou "max_entropy"
SEGMENTATION_METHOD = "multi_otsu"

# Número de classes para Multi-Otsu (3 = fundo, tecido claro, tecido escuro)
MULTI_OTSU_CLASSES = 3

# =============================================================================
# EXPERIMENTO 3 - MORFOLOGIA MATEMÁTICA
# =============================================================================
# Tipo de elemento estruturante: "disk", "square", "diamond", "ellipse"
STRUCTURING_ELEMENT_TYPE = "disk"

# Tamanho do elemento estusturante (raio para disco)
STRUCTURING_ELEMENT_SIZE = 5

# Operações morfológicas aplicadas em sequência
# Cada tupla: (operação, tamanho_se)
# Operações: "opening", "closing", "erosion", "dilation"
MORPHOLOGY_OPERATIONS = [
    ("opening", 3),   # Remove ruído pequeno (sal)
    ("closing", 5),   # Preenche pequenos buracos (pimenta)
]

# =============================================================================
# EXPERIMENTO 4 - EXTRAÇÃO DE CARACTERÍSTICAS
# =============================================================================

# --- Haralick (GLCM) ---
# Distâncias (em pixels) para cálculo da GLCM
GLCM_DISTANCES = [1, 3, 5]

# Ângulos (em radianos) para cálculo da GLCM
GLCM_ANGLES = [0, 0.7854, 1.5708, 2.3561]  # 0°, 45°, 90°, 135°

# Níveis de quantização para GLCM (redução de níveis de cinza)
GLCM_LEVELS = 32

# Descritores de Haralick a extrair (4+ conforme especificado)
HARALICK_FEATURES = [
    "contrast",       # Contraste - mede variações locais
    "dissimilarity",  # Dissimilaridade - similaridade ao contraste
    "homogeneity",    # Homogeneidade - uniformidade local
    "energy",         # Energia (segundo momento angular) - uniformidade
    "correlation",    # Correlação - dependência linear
    "ASM",            # Segundo momento angular - uniformidade da GLCM
]

# --- Dimensão Fractal (Box-Counting) ---
# Escalas (tamanhos de caixa) para box-counting
FRACTAL_SCALES = [2, 4, 8, 16, 32, 64]

# Número de escalas para regressão linear multiescala
FRACTAL_MIN_SCALES = 4

# =============================================================================
# EXPERIMENTO 5 - NORMALIZAÇÃO E SELEÇÃO
# =============================================================================
# Método de normalização: "zscore" ou "minmax"
NORMALIZATION_METHOD = "zscore"

# Número de features mais relevantes a selecionar via RELIEF
RELIEF_N_FEATURES = 6

# Número de vizinhos para o algoritmo RELIEF
RELIEF_N_NEIGHBORS = 10

# =============================================================================
# EXPERIMENTO 6 - CLASSIFICAÇÃO
# =============================================================================
# Classificador: "svm", "random_forest", "knn", "lda"
CLASSIFIER = "svm"

# Parâmetros do classificador
SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_GAMMA = "scale"

RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_FOREST_MAX_DEPTH = None

KNN_N_NEIGHBORS = 5

# Proporção treino/teste
TEST_SIZE = 0.3

# Semente aleatória para reprodutibilidade
RANDOM_STATE = 42

# Número de folds para validação cruzada
CV_FOLDS = 5

# =============================================================================
# VISUALIZAÇÃO
# =============================================================================
# DPI para salvar figuras
FIGURE_DPI = 150

# Estilo de plot
PLOT_STYLE = "seaborn-v0_8-whitegrid"

# Mapa de cores para segmentação
SEGMENTATION_CMAP = "viridis"

# =============================================================================
# CRIAÇÃO AUTOMÁTICA DE DIRETÓRIOS
# =============================================================================
def create_directories():
    """Cria todos os diretórios necessários para resultados."""
    dirs = [
        RESULTS_DIR, FIGURES_DIR, MASKS_DIR,
        FEATURES_DIR, CLASSIFICATION_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[CONFIG] Diretórios de resultados criados em: {RESULTS_DIR}")
