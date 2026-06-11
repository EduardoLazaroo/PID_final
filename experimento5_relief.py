"""
=============================================================================
experimento5_relief.py - Experimento 5: Normalização e Seleção de Características
=============================================================================

Procedimentos:
    a) Normalizar características usando z-score ou min-max
    b) Aplicar algoritmo RELIEF para ranqueamento
    c) Selecionar features mais relevantes

Questões:
    - Como a normalização influenciou a distribuição e comparabilidade?
    - Quais características foram mais relevantes pelo RELIEF e por quê?

Justificativa da normalização:
    - Z-score: Transforma features para média 0 e desvio 1, preservando
      a forma da distribuição. Ideal quando features têm escalas muito
      diferentes (ex: contraste vs. dimensão fractal).
    - Min-max: Comprime para [0, 1], útil quando se deseja preservar
      a relação linear entre valores.

Justificativa do RELIEF:
    O algoritmo RELIEF avalia a qualidade de cada feature medindo quão bem
    ela distingue amostras próximas de classes diferentes. Features que
    têm valores similares para amostras da mesma classe e diferentes para
    classes distintas recebem scores altos.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
import config
from utils import (
    save_figure, plot_feature_distribution, plot_relief_scores,
    normalize_features
)


# =============================================================================
# ALGORITMO RELIEF (implementação manual)
# =============================================================================

def relief_algorithm(X, y, n_neighbors=10, n_iterations=100):
    """
    Implementação do algoritmo RELIEF para ranqueamento de features.
    
    O RELIEF atribui pesos às features baseado na capacidade de distinguir
    entre classes. Para cada amostra:
    1. Encontra o hit (vizinho mais próximo da mesma classe)
    2. Encontra o miss (vizinho mais próximo de classe diferente)
    3. Atualiza pesos: w_i -= diff(x, hit_i) / m + diff(x, miss_i) / m
    
    Parâmetros:
        X: matriz (n_samples, n_features) - JÁ NORMALIZADA
        y: vetor (n_samples,) - labels
        n_neighbors: número de vizinhos a considerar
        n_iterations: número de iterações (amostras a avaliar)
    
    Retorna:
        scores: array (n_features,) - score de cada feature
        feature_ranking: índices das features ordenadas por relevância
    """
    n_samples, n_features = X.shape
    scores = np.zeros(n_features)
    
    # Classes únicas
    classes = np.unique(y)
    
    # Número de iterações (limitado pelo tamanho do dataset)
    m = min(n_iterations, n_samples)
    
    # Selecionar amostras aleatórias para avaliação
    rng = np.random.RandomState(config.RANDOM_STATE)
    sample_indices = rng.choice(n_samples, size=m, replace=False)
    
    for idx in sample_indices:
        current_sample = X[idx]
        current_class = y[idx]
        
        # Calcular distâncias para todas as outras amostras
        distances = np.sqrt(np.sum((X - current_sample) ** 2, axis=1))
        
        # Encontrar hits (mesma classe) e misses (classe diferente)
        same_class_mask = (y == current_class)
        same_class_mask[idx] = False  # Excluir a própria amostra
        
        diff_class_mask = (y != current_class)
        
        # Hit: vizinho mais próximo da mesma classe
        if np.any(same_class_mask):
            hit_distances = distances.copy()
            hit_distances[~same_class_mask] = np.inf
            hit_idx = np.argmin(hit_distances)
            hit = X[hit_idx]
        else:
            hit = current_sample
        
        # Miss: vizinho mais próximo de classe diferente
        if np.any(diff_class_mask):
            miss_distances = distances.copy()
            miss_distances[~diff_class_mask] = np.inf
            miss_idx = np.argmin(miss_distances)
            miss = X[miss_idx]
        else:
            miss = current_sample
        
        # Atualizar scores
        # Diferença absoluta para cada feature
        diff_hit = np.abs(current_sample - hit)
        diff_miss = np.abs(current_sample - miss)
        
        scores = scores - diff_hit / m + diff_miss / m
    
    # Ranking (ordem decrescente de score)
    feature_ranking = np.argsort(scores)[::-1]
    
    return scores, feature_ranking


def relief_f_algorithm(X, y, n_neighbors=10):
    """
    RELIEF-F: extensão que considera múltiplos vizinhos (k hits e k misses).
    
    Mais robusto que o RELIEF original para datasets com ruído.
    """
    n_samples, n_features = X.shape
    scores = np.zeros(n_features)
    classes = np.unique(y)
    
    rng = np.random.RandomState(config.RANDOM_STATE)
    m = min(100, n_samples)
    sample_indices = rng.choice(n_samples, size=m, replace=False)
    
    for idx in sample_indices:
        current_sample = X[idx]
        current_class = y[idx]
        
        distances = np.sqrt(np.sum((X - current_sample) ** 2, axis=1))
        
        # k nearest hits
        same_class_mask = (y == current_class)
        same_class_mask[idx] = False
        
        hits = []
        if np.any(same_class_mask):
            hit_indices = np.where(same_class_mask)[0]
            hit_distances = distances[hit_indices]
            k_hits = min(n_neighbors, len(hit_indices))
            nearest_hit_idx = hit_indices[np.argsort(hit_distances)[:k_hits]]
            hits = [X[i] for i in nearest_hit_idx]
        
        # k nearest misses (de cada classe diferente)
        misses = []
        for cls in classes:
            if cls == current_class:
                continue
            diff_mask = (y == cls)
            if np.any(diff_mask):
                miss_indices = np.where(diff_mask)[0]
                miss_distances = distances[miss_indices]
                k_misses = min(n_neighbors, len(miss_indices))
                nearest_miss_idx = miss_indices[np.argsort(miss_distances)[:k_misses]]
                misses.extend([X[i] for i in nearest_miss_idx])
        
        # Atualizar scores
        if hits:
            mean_hit = np.mean(hits, axis=0)
            scores -= np.abs(current_sample - mean_hit) / (m * len(hits))
        
        if misses:
            mean_miss = np.mean(misses, axis=0)
            scores += np.abs(current_sample - mean_miss) / (m * len(misses))
    
    feature_ranking = np.argsort(scores)[::-1]
    return scores, feature_ranking


# =============================================================================
# ANÁLISE DE NORMALIZAÇÃO
# =============================================================================

def analyze_normalization_effect(X_original, X_normalized, feature_names, method):
    """
    Analisa o efeito da normalização nas distribuições.
    
    Retorna:
        stats_before: estatísticas antes da normalização
        stats_after: estatísticas depois da normalização
    """
    stats_before = {}
    stats_after = {}
    
    for i, name in enumerate(feature_names):
        stats_before[name] = {
            "mean": np.mean(X_original[:, i]),
            "std": np.std(X_original[:, i]),
            "min": np.min(X_original[:, i]),
            "max": np.max(X_original[:, i]),
            "range": np.max(X_original[:, i]) - np.min(X_original[:, i]),
        }
        stats_after[name] = {
            "mean": np.mean(X_normalized[:, i]),
            "std": np.std(X_normalized[:, i]),
            "min": np.min(X_normalized[:, i]),
            "max": np.max(X_normalized[:, i]),
            "range": np.max(X_normalized[:, i]) - np.min(X_normalized[:, i]),
        }
    
    return stats_before, stats_after


def run_experiment_5(all_features, all_labels, feature_names, class_names):
    """
    Executa o Experimento 5 completo.
    
    Parâmetros:
        all_features: matriz (n_samples, n_features)
        all_labels: vetor de labels
        feature_names: nomes das features
        class_names: nomes das classes
    
    Retorna:
        X_normalized: features normalizadas
        selected_features: features selecionadas
        selected_names: nomes das features selecionadas
        relief_scores: scores do RELIEF
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 5 - Normalização e Seleção de Características")
    print("=" * 70)
    
    # a) Normalização
    print(f"\n[Método de Normalização: {config.NORMALIZATION_METHOD}]")
    
    if config.NORMALIZATION_METHOD == "zscore":
        print("  Z-score: transforma para média=0, desvio=1")
        print("  Preserva a forma da distribuição original.")
        print("  Vantagem: não é afetado por outliers extremos.")
    elif config.NORMALIZATION_METHOD == "minmax":
        print("  Min-Max: comprime para intervalo [0, 1]")
        print("  Preserva relações lineares entre valores.")
        print("  Vantagem: garante que todas as features tenham a mesma escala absoluta.")
    
    X_normalized, scaler = normalize_features(
        all_features, method=config.NORMALIZATION_METHOD
    )
    
    # Analisar efeito da normalização
    stats_before, stats_after = analyze_normalization_effect(
        all_features, X_normalized, feature_names, config.NORMALIZATION_METHOD
    )
    
    print("\n[Efeito da Normalização]")
    print(f"{'Feature':<30} {'Antes (média±desvio)':<25} {'Depois (média±desvio)':<25}")
    print("-" * 80)
    for name in feature_names:
        before = stats_before[name]
        after = stats_after[name]
        print(
            f"{name:<30} "
            f"{before['mean']:>10.4f} ± {before['std']:<10.4f}   "
            f"{after['mean']:>10.4f} ± {after['std']:<10.4f}"
        )
    
    # b) Aplicar RELIEF
    print(f"\n[Algoritmo RELIEF]")
    print(f"  Vizinhos: {config.RELIEF_N_NEIGHBORS}")
    print(f"  Features a selecionar: {config.RELIEF_N_FEATURES}")
    
    relief_scores, ranking = relief_f_algorithm(
        X_normalized, all_labels, n_neighbors=config.RELIEF_N_NEIGHBORS
    )
    
    print("\n  Ranqueamento RELIEF:")
    for rank, idx in enumerate(ranking):
        marker = " ✓" if rank < config.RELIEF_N_FEATURES else ""
        print(f"    {rank+1}. {feature_names[idx]:<30} score={relief_scores[idx]:.6f}{marker}")
    
    # c) Selecionar features mais relevantes
    selected_indices = ranking[:config.RELIEF_N_FEATURES]
    selected_features = X_normalized[:, selected_indices]
    selected_names = [feature_names[i] for i in selected_indices]
    
    print(f"\n  Features selecionadas ({config.RELIEF_N_FEATURES}):")
    for name in selected_names:
        print(f"    - {name}")
    
    # Visualizações
    print("\n[GERAÇÃO DE FIGURAS]")
    
    # Figura 1: Distribuição antes e depois da normalização
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Antes
    for i, name in enumerate(feature_names[:6]):  # Mostrar primeiras 6
        axes[0].hist(all_features[:, i], bins=30, alpha=0.5, label=name)
    axes[0].set_title("Antes da Normalização", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Valor")
    axes[0].set_ylabel("Frequência")
    axes[0].legend(fontsize=8)
    
    # Depois
    for i, name in enumerate(feature_names[:6]):
        axes[1].hist(X_normalized[:, i], bins=30, alpha=0.5, label=name)
    axes[1].set_title(
        f"Depois da Normalização ({config.NORMALIZATION_METHOD})",
        fontsize=13, fontweight='bold'
    )
    axes[1].set_xlabel("Valor")
    axes[1].set_ylabel("Frequência")
    axes[1].legend(fontsize=8)
    
    fig.suptitle(
        "Efeito da Normalização nas Distribuições",
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout()
    save_figure(fig, "exp5_normalizacao_distribuicoes.png", subdir="experimento5")
    
    # Figura 2: Scores do RELIEF
    fig = plot_relief_scores(feature_names, relief_scores, config.RELIEF_N_FEATURES)
    save_figure(fig, "exp5_relief_scores.png", subdir="experimento5")
    
    # Figura 3: Distribuição das features selecionadas por classe
    fig = plot_feature_distribution(
        selected_features, all_labels, selected_names, class_names
    )
    save_figure(fig, "exp5_distribuicoes_selecionadas.png", subdir="experimento5")
    
    # Figura 4: Comparação de escalas (antes vs depois)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot antes
    bp1 = axes[0].boxplot(
        [all_features[:, i] for i in range(len(feature_names))],
        labels=[f[:15] for f in feature_names],
        patch_artist=True
    )
    axes[0].set_title("Antes da Normalização", fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45, labelsize=7)
    
    # Boxplot depois
    bp2 = axes[1].boxplot(
        [X_normalized[:, i] for i in range(len(feature_names))],
        labels=[f[:15] for f in feature_names],
        patch_artist=True
    )
    axes[1].set_title(
        f"Depois ({config.NORMALIZATION_METHOD})",
        fontsize=12, fontweight='bold'
    )
    axes[1].tick_params(axis='x', rotation=45, labelsize=7)
    
    fig.suptitle(
        "Comparação de Escalas das Features",
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout()
    save_figure(fig, "exp5_comparacao_escalas.png", subdir="experimento5")
    
    # Figura 5: Heatmap de correlação entre features selecionadas
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = np.corrcoef(selected_features.T)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(selected_names)))
    ax.set_yticks(range(len(selected_names)))
    ax.set_xticklabels([n[:20] for n in selected_names], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([n[:20] for n in selected_names], fontsize=8)
    plt.colorbar(im, ax=ax, label="Correlação de Pearson")
    ax.set_title(
        "Correlação entre Features Selecionadas",
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout()
    save_figure(fig, "exp5_correlacao_selecionadas.png", subdir="experimento5")
    
    print("\n[EXPERIMENTO 5 CONCLUÍDO]")
    return X_normalized, selected_features, selected_names, relief_scores
