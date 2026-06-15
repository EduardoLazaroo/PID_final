"""
=============================================================================
experimento5_relief.py - Experimento 5: Normalizacao e Selecao de Caracteristicas
=============================================================================

Procedimentos:
    a) Normalizar caracteristicas usando z-score ou min-max
    b) Aplicar algoritmo RELIEF para ranqueamento
    c) Selecionar features mais relevantes
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import config
from utils import save_figure, plot_feature_distribution, plot_relief_scores


def relief_f_algorithm(X, y, n_neighbors=10):
    """
    RELIEF-F: extensao que considera multiplos vizinhos (k hits e k misses).
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
        
        # k nearest misses
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
        
        if hits:
            mean_hit = np.mean(hits, axis=0)
            scores -= np.abs(current_sample - mean_hit) / (m * len(hits))
        
        if misses:
            mean_miss = np.mean(misses, axis=0)
            scores += np.abs(current_sample - mean_miss) / (m * len(misses))
    
    feature_ranking = np.argsort(scores)[::-1]
    return scores, feature_ranking


def analyze_normalization_effect(X_original, X_normalized, feature_names, method):
    """Analisa o efeito da normalizacao nas distribuicoes."""
    stats_before = {}
    stats_after = {}
    
    for i, name in enumerate(feature_names):
        stats_before[name] = {
            "mean": np.mean(X_original[:, i]),
            "std": np.std(X_original[:, i]),
        }
        stats_after[name] = {
            "mean": np.mean(X_normalized[:, i]),
            "std": np.std(X_normalized[:, i]),
        }
    
    return stats_before, stats_after


def run_experiment_5(all_features, all_labels, feature_names, class_names):
    """
    Executa o Experimento 5 completo.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 5 - Normalizacao e Selecao de Caracteristicas")
    print("=" * 70)
    
    # a) Normalizacao
    print(f"\n[Metodo de Normalizacao: {config.NORMALIZATION_METHOD}]")
    
    if config.NORMALIZATION_METHOD == "zscore":
        scaler = StandardScaler()
    elif config.NORMALIZATION_METHOD == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Metodo desconhecido: {config.NORMALIZATION_METHOD}")
    
    X_normalized = scaler.fit_transform(all_features)
    
    # Analisar efeito
    stats_before, stats_after = analyze_normalization_effect(
        all_features, X_normalized, feature_names, config.NORMALIZATION_METHOD
    )
    
    print("\n[Efeito da Normalizacao]")
    print(f"{'Feature':<30} {'Antes (media+/-desvio)':<25} {'Depois (media+/-desvio)':<25}")
    print("-" * 80)
    for name in feature_names:
        before = stats_before[name]
        after = stats_after[name]
        print(f"{name:<30} {before['mean']:>10.4f} +/- {before['std']:<10.4f}   {after['mean']:>10.4f} +/- {after['std']:<10.4f}")
    
    # b) Aplicar RELIEF
    print(f"\n[Algoritmo RELIEF]")
    print(f"  Vizinhos: {config.RELIEF_N_NEIGHBORS}")
    print(f"  Features a selecionar: {config.RELIEF_N_FEATURES}")
    
    relief_scores, ranking = relief_f_algorithm(
        X_normalized, all_labels, n_neighbors=config.RELIEF_N_NEIGHBORS
    )
    
    print("\n  Ranqueamento RELIEF:")
    for rank, idx in enumerate(ranking):
        marker = " [OK]" if rank < config.RELIEF_N_FEATURES else ""
        print(f"    {rank+1}. {feature_names[idx]:<30} score={relief_scores[idx]:.6f}{marker}")
    
    # c) Selecionar features
    selected_indices = ranking[:config.RELIEF_N_FEATURES]
    selected_features = X_normalized[:, selected_indices]
    selected_names = [feature_names[i] for i in selected_indices]
    
    print(f"\n  Features selecionadas ({config.RELIEF_N_FEATURES}):")
    for name in selected_names:
        print(f"    - {name}")
    
    # Visualizacoes
    print("\n[GERACAO DE FIGURAS]")
    
    # Distribuicao antes e depois
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, name in enumerate(feature_names[:6]):
        axes[0].hist(all_features[:, i], bins=30, alpha=0.5, label=name)
    axes[0].set_title("Antes da Normalizacao", fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=8)
    for i, name in enumerate(feature_names[:6]):
        axes[1].hist(X_normalized[:, i], bins=30, alpha=0.5, label=name)
    axes[1].set_title(f"Depois ({config.NORMALIZATION_METHOD})", fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=8)
    fig.suptitle("Efeito da Normalizacao", fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, "exp5_normalizacao_distribuicoes.png", subdir="experimento5")
    
    # Scores RELIEF
    fig = plot_relief_scores(feature_names, relief_scores, config.RELIEF_N_FEATURES)
    save_figure(fig, "exp5_relief_scores.png", subdir="experimento5")
    
    # Distribuicao das selecionadas
    fig = plot_feature_distribution(selected_features, all_labels, selected_names, class_names)
    save_figure(fig, "exp5_distribuicoes_selecionadas.png", subdir="experimento5")
    
    # Comparacao de escalas
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bp1 = axes[0].boxplot(
        [all_features[:, i] for i in range(len(feature_names))],
        labels=[f[:15] for f in feature_names], patch_artist=True
    )
    axes[0].set_title("Antes", fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45, labelsize=7)
    bp2 = axes[1].boxplot(
        [X_normalized[:, i] for i in range(len(feature_names))],
        labels=[f[:15] for f in feature_names], patch_artist=True
    )
    axes[1].set_title(f"Depois ({config.NORMALIZATION_METHOD})", fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45, labelsize=7)
    fig.suptitle("Comparacao de Escalas", fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, "exp5_comparacao_escalas.png", subdir="experimento5")
    
    # Correlacao
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = np.corrcoef(selected_features.T)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(selected_names)))
    ax.set_yticks(range(len(selected_names)))
    ax.set_xticklabels([n[:20] for n in selected_names], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([n[:20] for n in selected_names], fontsize=8)
    plt.colorbar(im, ax=ax, label="Correlacao de Pearson")
    ax.set_title("Correlacao entre Features Selecionadas", fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, "exp5_correlacao_selecionadas.png", subdir="experimento5")
    
    print("\n[EXPERIMENTO 5 CONCLUIDO]")
    return X_normalized, selected_features, selected_names, relief_scores
