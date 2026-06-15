"""
=============================================================================
config.py - Configuracoes globais do Pipeline de Analise de Imagens Histopatologicas
=============================================================================
Centraliza todos os parametros utilizados nos experimentos, permitindo
reprodutibilidade e ajuste fino do pipeline.
"""

import os

# =============================================================================
# CAMINHOS E DIRETORIOS
# =============================================================================

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "dataset", "LC25000")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MASKS_DIR = os.path.join(RESULTS_DIR, "masks")
FEATURES_DIR = os.path.join(RESULTS_DIR, "features")
CLASSIFICATION_DIR = os.path.join(RESULTS_DIR, "classification")

# =============================================================================
# SELECAO DE CLASSES
# =============================================================================

CLASS_1 = "colon_aca"
CLASS_2 = "colon_n"

CLASS_1_NAME = "Colon Adenocarcinoma"
CLASS_2_NAME = "Colon Benign Tissue"

# Fracao minima de imagens por classe (5% conforme especificado)
MIN_FRACTION = 0.05
# Minimo absoluto de imagens por classe (250 = 5% de 5000)
MIN_IMAGES_ABSOLUTE = 250

# Numero fixo de imagens por classe (250 = 5% de 5000, conforme exigido)
MAX_IMAGES_PER_CLASS = 250

# =============================================================================
# EXPERIMENTO 1 - REPRESENTACAO HSV
# =============================================================================

HSV_CHANNEL_SELECTED = "S"

# =============================================================================
# EXPERIMENTO 2 - SEGMENTACAO
# =============================================================================

SEGMENTATION_METHOD = "multi_otsu"

MULTI_OTSU_CLASSES = 3

# =============================================================================
# EXPERIMENTO 3 - MORFOLOGIA MATEMATICA
# =============================================================================

STRUCTURING_ELEMENT_TYPE = "disk"

STRUCTURING_ELEMENT_SIZE = 5

MORPHOLOGY_OPERATIONS = [
    ("opening", 3),
    ("closing", 5),
]

# =============================================================================
# EXPERIMENTO 4 - EXTRACAO DE CARACTERISTICAS
# =============================================================================

GLCM_DISTANCES = [1, 3, 5]

GLCM_ANGLES = [0, 0.7854, 1.5708, 2.3561]

GLCM_LEVELS = 32

HARALICK_FEATURES = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
]

FRACTAL_SCALES = [2, 4, 8, 16, 32, 64]

FRACTAL_MIN_SCALES = 4

# =============================================================================
# EXPERIMENTO 5 - NORMALIZACAO E SELECAO
# =============================================================================

NORMALIZATION_METHOD = "zscore"

RELIEF_N_FEATURES = 6

RELIEF_N_NEIGHBORS = 10

# =============================================================================
# EXPERIMENTO 6 - CLASSIFICACAO
# =============================================================================

CLASSIFIER = "svm"

SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_GAMMA = "scale"

RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_FOREST_MAX_DEPTH = None

KNN_N_NEIGHBORS = 5

TEST_SIZE = 0.3

RANDOM_STATE = 42

CV_FOLDS = 5

# =============================================================================
# VISUALIZACAO
# =============================================================================

FIGURE_DPI = 150

SEGMENTATION_CMAP = "viridis"

# =============================================================================
# CRIACAO AUTOMATICA DE DIRETORIOS
# =============================================================================

def create_directories():
    """Cria todos os diretorios necessarios para resultados."""
    dirs = [
        RESULTS_DIR, FIGURES_DIR, MASKS_DIR,
        FEATURES_DIR, CLASSIFICATION_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[CONFIG] Diretorios de resultados criados em: {RESULTS_DIR}")
