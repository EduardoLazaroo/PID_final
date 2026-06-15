"""
=============================================================================
utils.py - Funcoes utilitarias para o Pipeline de Analise de Imagens
=============================================================================
Contem funcoes auxiliares para carregamento de dados, visualizacao,
calculos estatisticos e operacoes comuns aos experimentos.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import config


# =============================================================================
# CARREGAMENTO DE DADOS
# =============================================================================

def load_dataset():
    """
    Carrega imagens das duas classes selecionadas no config.
    Usa config.MAX_IMAGES_PER_CLASS como limite por classe (padrao: 250).

    Retorna:
        images: dict {classe: [lista de arrays numpy]}
        labels: dict {classe: label numerico}
        paths: dict {classe: [lista de caminhos]}
    """
    classes = [config.CLASS_1, config.CLASS_2]
    class_names = [config.CLASS_1_NAME, config.CLASS_2_NAME]

    images = {}
    labels = {}
    paths = {}

    print("[UTILS] Carregando dataset...")
    for idx, (cls, name) in enumerate(zip(classes, class_names)):
        cls_dir = os.path.join(config.DATASET_ROOT, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(
                f"Diretorio nao encontrado: {cls_dir}\n"
                f"Verifique config.DATASET_ROOT e a estrutura do dataset."
            )

        all_files = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])

        # Seed para reprodutibilidade
        rng = np.random.RandomState(config.RANDOM_STATE)
        rng.shuffle(all_files)

        n_total = len(all_files)
        # Limite por classe definido no config (padrao: 250 = 5% de 5000)
        n_select = min(config.MAX_IMAGES_PER_CLASS, n_total)

        selected = all_files[:n_select]
        labels[cls] = idx
        images[cls] = []
        paths[cls] = []

        print(f"  Classe '{name}': {len(selected)}/{n_total} imagens carregadas")

        for fname in tqdm(selected, desc=f"  Carregando {name}"):
            fpath = os.path.join(cls_dir, fname)
            try:
                img = np.array(Image.open(fpath).convert("RGB"))
                images[cls].append(img)
                paths[cls].append(fpath)
            except Exception as e:
                print(f"    [AVISO] Erro ao carregar {fname}: {e} - ignorando")

    return images, labels, paths


def prepare_feature_matrix(images_dict, labels_dict):
    """
    Converte dicionario de imagens em matriz de features e vetor de labels.

    Retorna:
        X: array (n_samples, n_features) - inicialmente vazio
        y: array (n_samples,) - labels
        image_list: lista de imagens na ordem correspondente
    """
    image_list = []
    y = []
    for cls in sorted(images_dict.keys()):
        for img in images_dict[cls]:
            image_list.append(img)
            y.append(labels_dict[cls])
    return np.array(y), image_list


# =============================================================================
# VISUALIZACAO
# =============================================================================

def save_figure(fig, filename, subdir=None, dpi=None):
    """Salva figura matplotlib no diretorio de resultados."""
    if dpi is None:
        dpi = config.FIGURE_DPI
    if subdir:
        save_dir = os.path.join(config.FIGURES_DIR, subdir)
    else:
        save_dir = config.FIGURES_DIR
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [FIGURA SALA] {filepath}")

def plot_channel_analysis(hsv_channels, channel_names, selected_idx):
    """Plota analise dos canais HSV com destaque para o selecionado."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (ch, name) in enumerate(zip(hsv_channels, channel_names)):
        axes[i].imshow(ch, cmap='gray')
        title = f"Canal {name}"
        if i == selected_idx:
            title += " (SELECIONADO)"
            for spine in axes[i].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
        axes[i].set_title(title, fontsize=12, fontweight='bold' if i == selected_idx else 'normal')
        axes[i].axis('off')
    fig.tight_layout()
    return fig

def plot_segmentation_comparison(original, segmented, mask, title="Segmentacao"):
    """Plota comparacao: original, segmentacao e mascara."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(segmented, cmap=config.SEGMENTATION_CMAP)
    axes[1].set_title("Segmentacao")
    axes[1].axis('off')
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title("Mascara Binaria")
    axes[2].axis('off')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig

def plot_morphology_comparison(before, after, mask_before, mask_after):
    """Plota comparacao antes e depois da morfologia matematica."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(before, cmap='gray')
    axes[0, 0].set_title("Antes - Segmentacao")
    axes[0, 0].axis('off')
    axes[0, 1].imshow(after, cmap='gray')
    axes[0, 1].set_title("Depois - Morfologia")
    axes[0, 1].axis('off')
    axes[1, 0].imshow(mask_before, cmap='gray')
    axes[1, 0].set_title("Antes - Mascara")
    axes[1, 0].axis('off')
    axes[1, 1].imshow(mask_after, cmap='gray')
    axes[1, 1].set_title("Depois - Mascara Final")
    axes[1, 1].axis('off')
    fig.suptitle("Refinamento com Morfologia Matematica", fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig

def plot_feature_distribution(features, y, feature_names, class_names):
    """Plota distribuicao de features por classe."""
    n_features = features.shape[1]
    ncols = min(3, n_features)
    nrows = (n_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_features):
        r, c = divmod(i, ncols)
        for cls_idx, cls_name in enumerate(class_names):
            mask = y == cls_idx
            axes[r, c].hist(
                features[mask, i], bins=30, alpha=0.6, label=cls_name
            )
        axes[r, c].set_title(feature_names[i], fontsize=10)
        axes[r, c].legend(fontsize=8)
        axes[r, c].set_xlabel("Valor")
        axes[r, c].set_ylabel("Frequencia")

    for i in range(n_features, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis('off')

    fig.suptitle("Distribuicao de Features por Classe", fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig

def plot_feature_space(features, y, class_names, method="pca", n_components=2):
    """
    Visualiza espaco de caracteristicas via PCA ou t-SNE.
    """
    if method == "pca":
        reducer = PCA(n_components=n_components)
        label_prefix = "PC"
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=config.RANDOM_STATE)
        label_prefix = "t-SNE"
    else:
        raise ValueError(f"Metodo desconhecido: {method}")

    reduced = reducer.fit_transform(features)

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        for cls_idx, cls_name in enumerate(class_names):
            mask = y == cls_idx
            ax.scatter(
                reduced[mask, 0], reduced[mask, 1],
                label=cls_name, alpha=0.6, s=30
            )
        ax.set_xlabel(f"{label_prefix} 1", fontsize=12)
        ax.set_ylabel(f"{label_prefix} 2", fontsize=12)
        ax.legend(fontsize=11)
        ax.set_title(
            f"Projecao {method.upper()} do Espaco de Caracteristicas",
            fontsize=14, fontweight='bold'
        )
    elif n_components == 3:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        for cls_idx, cls_name in enumerate(class_names):
            mask = y == cls_idx
            ax.scatter(
                reduced[mask, 0], reduced[mask, 1], reduced[mask, 2],
                label=cls_name, alpha=0.6, s=30
            )
        ax.set_xlabel(f"{label_prefix} 1", fontsize=10)
        ax.set_ylabel(f"{label_prefix} 2", fontsize=10)
        ax.set_zlabel(f"{label_prefix} 3", fontsize=10)
        ax.legend(fontsize=11)
        ax.set_title(
            f"Projecao {method.upper()} 3D do Espaco de Caracteristicas",
            fontsize=14, fontweight='bold'
        )

    fig.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names):
    """Plota matriz de confusao."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='Classe Real',
        xlabel='Classe Predita',
        title='Matriz de Confusao'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

def plot_relief_scores(feature_names, scores, n_selected):
    """Plota ranqueamento RELIEF das features."""
    sorted_idx = np.argsort(scores)[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.4)))
    colors = ['green' if i < n_selected else 'gray' for i in range(len(scores))]
    ax.barh(
        range(len(scores)),
        scores[sorted_idx][::-1],
        color=colors[::-1]
    )
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx][::-1])
    ax.set_xlabel("Score RELIEF", fontsize=12)
    ax.set_title(
        f"Ranqueamento RELIEF (top {n_selected} em verde)",
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout()
    return fig

# =============================================================================
# NORMALIZACAO
# =============================================================================

def normalize_features(X_train, X_test=None, method="zscore"):
    """
    Normaliza features usando z-score ou min-max.
    """
    if method == "zscore":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Metodo de normalizacao desconhecido: {method}")

    X_train_norm = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_norm = scaler.transform(X_test)
        return X_train_norm, X_test_norm, scaler
    return X_train_norm, scaler

# =============================================================================
# CLASSIFICACAO
# =============================================================================

def get_classifier(name=None):
    """Retorna classificador conforme configuracao."""
    if name is None:
        name = config.CLASSIFIER

    if name == "svm":
        return SVC(
            kernel=config.SVM_KERNEL,
            C=config.SVM_C,
            gamma=config.SVM_GAMMA,
            random_state=config.RANDOM_STATE
        )
    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.RANDOM_FOREST_N_ESTIMATORS,
            max_depth=config.RANDOM_FOREST_MAX_DEPTH,
            random_state=config.RANDOM_STATE
        )
    elif name == "knn":
        return KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS)
    elif name == "lda":
        return LinearDiscriminantAnalysis()
    else:
        raise ValueError(f"Classificador desconhecido: {name}")

def evaluate_classifier(X, y, classifier=None, cv=None):
    """
    Avalia classificador com validacao cruzada.
    """
    if classifier is None:
        classifier = get_classifier()
    if cv is None:
        cv = StratifiedKFold(
            n_splits=config.CV_FOLDS,
            shuffle=True,
            random_state=config.RANDOM_STATE
        )

    scores = cross_val_score(classifier, X, y, cv=cv, scoring='accuracy')
    return scores, scores.mean(), scores.std()

def train_and_evaluate(X_train, X_test, y_train, y_test, classifier=None, class_names=None):
    """
    Treina classificador e avalia no conjunto de teste.
    """
    if classifier is None:
        classifier = get_classifier()

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)
    cm = confusion_matrix(y_test, y_pred)

    return classifier, y_pred, accuracy, report, cm

# =============================================================================
# ESTATISTICAS
# =============================================================================

def compute_class_statistics(features, y, feature_names, class_names, confidence=0.95):
    """
    Calcula media, desvio padrao e intervalo de confianca por classe.
    """
    from scipy import stats as scipy_stats

    alpha = 1 - confidence
    stats = {}

    for cls_idx, cls_name in enumerate(class_names):
        mask = y == cls_idx
        cls_features = features[mask]
        stats[cls_name] = {}

        for i, fname in enumerate(feature_names):
            values = cls_features[:, i]
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            n = len(values)
            se = std / np.sqrt(n)
            t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=n - 1)
            ci_low = mean - t_crit * se
            ci_high = mean + t_crit * se

            stats[cls_name][fname] = {
                "mean": mean,
                "std": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": n
            }

    return stats

def print_statistics_table(stats):
    """Imprime tabela formatada de estatisticas."""
    print("\n" + "=" * 80)
    print("ESTATISTICAS POR CLASSE")
    print("=" * 80)

    for cls_name, features in stats.items():
        print(f"\n--- {cls_name} ---")
        print(f"{'Feature':<30} {'Media':>10} {'Desvio':>10} {'IC Inf':>10} {'IC Sup':>10}")
        print("-" * 72)
        for fname, vals in features.items():
            print(
                f"{fname:<30} "
                f"{vals['mean']:>10.4f} "
                f"{vals['std']:>10.4f} "
                f"{vals['ci_low']:>10.4f} "
                f"{vals['ci_high']:>10.4f}"
            )