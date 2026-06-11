"""
=============================================================================
experimento6_classificacao.py - Experimento 6: Visualização e Classificação
=============================================================================

Procedimentos:
    a) Construir visualização do espaço de características (PCA/t-SNE)
    b) Realizar classificação supervisionada
    c) Avaliar impacto da redução de dimensionalidade

Questões:
    - As features selecionadas permitiram separabilidade adequada?
    - Qual o impacto da redução de dimensionalidade no desempenho?

Justificativa do classificador:
    SVM com kernel RBF é escolhido como classificador principal porque:
    1. Eficaz em espaços de alta dimensão (após seleção de features)
    2. Kernel RBF captura relações não-lineares entre features
    3. Robusto com amostras de tamanho moderado
    4. Amplamente utilizado em classificação de imagens médicas
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import config
from utils import (
    save_figure, plot_feature_space, plot_confusion_matrix,
    get_classifier, evaluate_classifier, train_and_evaluate,
    compute_class_statistics, print_statistics_table
)


def visualize_feature_space(X, y, class_names, feature_names, selected_names=None):
    """
    Visualiza o espaço de características usando PCA e t-SNE.
    
    Parâmetros:
        X: matriz de features
        y: labels
        class_names: nomes das classes
        feature_names: nomes de todas as features
        selected_names: nomes das features selecionadas (se aplicável)
    """
    print("\n[Visualização do Espaço de Características]")
    
    # PCA 2D
    fig = plot_feature_space(X, y, class_names, method="pca", n_components=2)
    suffix = "selecionadas" if selected_names else "todas"
    save_figure(fig, f"exp6_pca_2d_{suffix}.png", subdir="experimento6")
    
    # PCA 3D
    fig = plot_feature_space(X, y, class_names, method="pca", n_components=3)
    save_figure(fig, f"exp6_pca_3d_{suffix}.png", subdir="experimento6")
    
    # t-SNE 2D
    print("  Calculando t-SNE (pode demorar...)")
    fig = plot_feature_space(X, y, class_names, method="tsne", n_components=2)
    save_figure(fig, f"exp6_tsne_2d_{suffix}.png", subdir="experimento6")
    
    # Variância explicada pelos componentes PCA
    pca = PCA()
    pca.fit(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Variância explicada individual
    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_, color='steelblue')
    axes[0].set_xlabel("Componente Principal", fontsize=12)
    axes[0].set_ylabel("Variância Explicada", fontsize=12)
    axes[0].set_title("Variância por Componente", fontsize=13, fontweight='bold')
    
    # Variância acumulada
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, len(cumvar) + 1), cumvar, 'bo-', markersize=6)
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% variância')
    axes[1].axhline(y=0.90, color='orange', linestyle='--', label='90% variância')
    axes[1].set_xlabel("Número de Componentes", fontsize=12)
    axes[1].set_ylabel("Variância Acumulada", fontsize=12)
    axes[1].set_title("Variância Acumulada", fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    
    fig.suptitle(
        "Análise de Componentes Principais",
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout()
    save_figure(fig, f"exp6_pca_variancia_{suffix}.png", subdir="experimento6")
    
    # Encontrar número de componentes para 95% da variância
    n_95 = np.argmax(cumvar >= 0.95) + 1
    n_90 = np.argmax(cumvar >= 0.90) + 1
    print(f"  Componentes para 90% variância: {n_90}")
    print(f"  Componentes para 95% variância: {n_95}")
    
    return pca


def compare_classifiers(X_train, X_test, y_train, y_test, class_names):
    """
    Compara múltiplos classificadores.
    
    Retorna:
        results: dict com resultados de cada classificador
    """
    print("\n[Comparação de Classificadores]")
    
    classifiers = {
        "SVM (RBF)": SVC(kernel='rbf', C=1.0, gamma='scale', random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=config.RANDOM_STATE
        ),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "LDA": LinearDiscriminantAnalysis(),
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Validação cruzada
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                             random_state=config.RANDOM_STATE)
        cv_scores = cross_val_score(clf, 
                                    np.vstack([X_train, X_test]),
                                    np.concatenate([y_train, y_test]),
                                    cv=cv, scoring='accuracy')
        
        results[name] = {
            "accuracy": acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "y_pred": y_pred,
            "classifier": clf,
        }
        
        print(f"  {name}:")
        print(f"    Acurácia (teste): {acc:.4f}")
        print(f"    CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results


def evaluate_dimensionality_impact(X_full, X_selected, y, class_names):
    """
    Avalia o impacto da redução de dimensionalidade.
    
    Compara classificação com todas as features vs. features selecionadas.
    """
    print("\n[Impacto da Redução de Dimensionalidade]")
    
    results = {}
    
    for name, X in [("Todas as features", X_full), ("Features selecionadas", X_selected)]:
        print(f"\n  {name} ({X.shape[1]} features):")
        
        clf = get_classifier()
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                             random_state=config.RANDOM_STATE)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        
        results[name] = {
            "n_features": X.shape[1],
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
            "scores": scores,
        }
        
        print(f"    Acurácia CV: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Comparação
    all_acc = results["Todas as features"]["mean_accuracy"]
    sel_acc = results["Features selecionadas"]["mean_accuracy"]
    diff = sel_acc - all_acc
    
    print(f"\n  Diferença de acurácia: {diff:+.4f}")
    if diff > 0:
        print("  → A seleção de features MELHOROU o desempenho!")
    elif diff > -0.02:
        print("  → A seleção manteve desempenho similar com menos features.")
    else:
        print("  → A seleção reduziu o desempenho (pode indicar perda de informação).")
    
    return results


def plot_roc_curves(X_train, X_test, y_train, y_test, class_names):
    """Plota curvas ROC para cada classificador."""
    classifiers = {
        "SVM": SVC(kernel='rbf', probability=True, random_state=config.RANDOM_STATE),
        "RF": RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatório')
    ax.set_xlabel("Taxa de Falso Positivo", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiro Positivo", fontsize=12)
    ax.set_title("Curvas ROC", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig


def run_experiment_6(X_normalized, selected_features, selected_names,
                     all_labels, feature_names, class_names):
    """
    Executa o Experimento 6 completo.
    
    Parâmetros:
        X_normalized: todas as features normalizadas
        selected_features: features selecionadas pelo RELIEF
        selected_names: nomes das features selecionadas
        all_labels: labels
        feature_names: nomes de todas as features
        class_names: nomes das classes
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 6 - Visualização e Classificação")
    print("=" * 70)
    
    # Mapear labels numéricos para índices de class_names
    unique_labels = np.unique(all_labels)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(sorted(unique_labels))}
    y = np.array([label_to_idx[lbl] for lbl in all_labels])
    
    # =========================================================================
    # a) Visualização do espaço de características
    # =========================================================================
    print("\n--- Visualização com features selecionadas ---")
    pca_selected = visualize_feature_space(
        selected_features, y, class_names, feature_names, selected_names
    )
    
    print("\n--- Visualização com todas as features ---")
    pca_all = visualize_feature_space(
        X_normalized, y, class_names, feature_names
    )
    
    # =========================================================================
    # b) Classificação
    # =========================================================================
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        selected_features, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n[Treino/Teste] {len(y_train)} treino, {len(y_test)} teste")
    
    # Classificador principal
    print(f"\n[Classificador Principal: {config.CLASSIFIER.upper()}]")
    
    classifier = get_classifier()
    classifier, y_pred, accuracy, report, cm = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        classifier=classifier,
        class_names=class_names
    )
    
    print(f"\n  Acurácia: {accuracy:.4f}")
    print(f"\n  Relatório de Classificação:")
    print(report)
    
    # Validação cruzada
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    cv_scores = cross_val_score(classifier, selected_features, y, cv=cv,
                                scoring='accuracy')
    print(f"  Validação Cruzada ({config.CV_FOLDS} folds):")
    print(f"    Acurácia: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    Scores: {[f'{s:.4f}' for s in cv_scores]}")
    
    # Comparação de classificadores
    clf_results = compare_classifiers(X_train, X_test, y_train, y_test, class_names)
    
    # =========================================================================
    # c) Impacto da redução de dimensionalidade
    # =========================================================================
    dim_impact = evaluate_dimensionality_impact(
        X_normalized, selected_features, y, class_names
    )
    
    # =========================================================================
    # Estatísticas por classe
    # =========================================================================
    print("\n[Estatísticas por Classe - Features Selecionadas]")
    stats = compute_class_statistics(selected_features, y, selected_names, class_names)
    print_statistics_table(stats)
    
    # =========================================================================
    # Visualizações
    # =========================================================================
    print("\n[GERAÇÃO DE FIGURAS]")
    
    # Matriz de confusão
    fig = plot_confusion_matrix(cm, class_names)
    save_figure(fig, "exp6_matriz_confusao.png", subdir="experimento6")
    
    # Curvas ROC
    fig = plot_roc_curves(X_train, X_test, y_train, y_test, class_names)
    save_figure(fig, "exp6_curvas_roc.png", subdir="experimento6")
    
    # Comparação de classificadores (barras)
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(clf_results.keys())
    accuracies = [clf_results[n]["accuracy"] for n in names]
    cv_means = [clf_results[n]["cv_mean"] for n in names]
    cv_stds = [clf_results[n]["cv_std"] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Teste', color='steelblue')
    bars2 = ax.bar(x + width/2, cv_means, width, yerr=cv_stds,
                   label='CV (média)', color='coral', capsize=5)
    
    ax.set_ylabel("Acurácia", fontsize=12)
    ax.set_title("Comparação de Classificadores", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # Anotações
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    fig.tight_layout()
    save_figure(fig, "exp6_comparacao_classificadores.png", subdir="experimento6")
    
    # Impacto da redução de dimensionalidade
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = list(dim_impact.keys())
    means = [dim_impact[c]["mean_accuracy"] for c in categories]
    stds = [dim_impact[c]["std_accuracy"] for c in categories]
    n_feats = [dim_impact[c]["n_features"] for c in categories]
    
    bars = ax.bar(range(len(categories)), means, yerr=stds,
                  color=['steelblue', 'coral'], capsize=10)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([f"{c}\n({n} feats)" for c, n in zip(categories, n_feats)],
                       fontsize=10)
    ax.set_ylabel("Acurácia CV", fontsize=12)
    ax.set_title(
        "Impacto da Redução de Dimensionalidade",
        fontsize=14, fontweight='bold'
    )
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, mean in zip(bars, means):
        ax.annotate(f'{mean:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, mean),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    fig.tight_layout()
    save_figure(fig, "exp6_impacto_dimensionalidade.png", subdir="experimento6")
    
    # Scatter plot 2D com fronteira de decisão (usando PCA para visualização)
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(selected_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plotar pontos
    for cls_idx, cls_name in enumerate(class_names):
        mask = y == cls_idx
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  label=cls_name, alpha=0.6, s=40)
    
    # Treinar SVM no espaço PCA para visualizar fronteira
    svm_2d = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_2d.fit(X_pca, y)
    
    # Criar mesh para fronteira
    h = 0.5
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)
    
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax.set_title(
        "Fronteira de Decisão (SVM sobre PCA)",
        fontsize=14, fontweight='bold'
    )
    ax.legend(fontsize=11)
    fig.tight_layout()
    save_figure(fig, "exp6_fronteira_decisao.png", subdir="experimento6")
    
    print("\n[EXPERIMENTO 6 CONCLUÍDO]")
    
    return {
        "accuracy": accuracy,
        "cv_scores": cv_scores,
        "confusion_matrix": cm,
        "classification_report": report,
        "classifier_results": clf_results,
        "dimensionality_impact": dim_impact,
        "class_statistics": stats,
    }
