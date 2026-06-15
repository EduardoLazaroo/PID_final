"""
=============================================================================
experimento6_classificacao.py - Experimento 6: Visualizacao e Classificacao
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import config
from utils import (
    save_figure, plot_feature_space, plot_confusion_matrix,
    get_classifier, train_and_evaluate, compute_class_statistics,
    print_statistics_table
)


def visualize_feature_space(X, y, class_names, suffix=""):
    """Visualiza o espaco de caracteristicas usando PCA e t-SNE."""
    print("\n[Visualizacao do Espaco de Caracteristicas]")

    fig = plot_feature_space(X, y, class_names, method="pca", n_components=2)
    save_figure(fig, f"exp6_pca_2d_{suffix}.png", subdir="experimento6")

    fig = plot_feature_space(X, y, class_names, method="pca", n_components=3)
    save_figure(fig, f"exp6_pca_3d_{suffix}.png", subdir="experimento6")

    print("  Calculando t-SNE (pode demorar...)")
    fig = plot_feature_space(X, y, class_names, method="tsne", n_components=2)
    save_figure(fig, f"exp6_tsne_2d_{suffix}.png", subdir="experimento6")

    pca = PCA()
    pca.fit(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_, color='steelblue')
    axes[0].set_xlabel("Componente Principal", fontsize=12)
    axes[0].set_ylabel("Variancia Explicada", fontsize=12)
    axes[0].set_title("Variancia por Componente", fontsize=13, fontweight='bold')

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, len(cumvar) + 1), cumvar, 'bo-', markersize=6)
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95%')
    axes[1].axhline(y=0.90, color='orange', linestyle='--', label='90%')
    axes[1].set_xlabel("Numero de Componentes", fontsize=12)
    axes[1].set_ylabel("Variancia Acumulada", fontsize=12)
    axes[1].set_title("Variancia Acumulada", fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)

    fig.suptitle("Analise de Componentes Principais", fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, f"exp6_pca_variancia_{suffix}.png", subdir="experimento6")

    n_95 = np.argmax(cumvar >= 0.95) + 1
    n_90 = np.argmax(cumvar >= 0.90) + 1
    print(f"  Componentes para 90%: {n_90}, 95%: {n_95}")


def compare_classifiers(X_train, X_test, y_train, y_test, class_names):
    """Compara multiplos classificadores."""
    print("\n[Comparacao de Classificadores]")

    classifiers = {
        "SVM (RBF)": SVC(kernel='rbf', C=1.0, gamma='scale', random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }

    results = {}

    for name, clf in classifiers.items():
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

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
            print(f"    Acuracia (teste): {acc:.4f}")
            print(f"    CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        except Exception as e:
            print(f"  {name}: ERRO - {e}")

    return results


def evaluate_dimensionality_impact(X_full, X_selected, y, class_names):
    """Avalia o impacto da reducao de dimensionalidade."""
    print("\n[Impacto da Reducao de Dimensionalidade]")

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

        print(f"    Acuracia CV: {scores.mean():.4f} +/- {scores.std():.4f}")

    all_acc = results["Todas as features"]["mean_accuracy"]
    sel_acc = results["Features selecionadas"]["mean_accuracy"]
    diff = sel_acc - all_acc

    print(f"\n  Diferenca de acuracia: {diff:+.4f}")
    if diff > 0:
        print("  [MELHOROU] A selecao melhorou o desempenho!")
    elif diff > -0.02:
        print("  [MANTEVE] A selecao manteve desempenho similar.")
    else:
        print("  [REDUZIU] A selecao reduziu o desempenho.")

    return results


def plot_roc_curves(X_train, X_test, y_train, y_test, class_names):
    """Plota curvas ROC."""
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

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aleatorio')
    ax.set_xlabel("Taxa de Falso Positivo", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiro Positivo", fontsize=12)
    ax.set_title("Curvas ROC", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def run_experiment_6(X_normalized, selected_features, selected_names,
                     all_labels, feature_names, class_names):
    """Executa o Experimento 6 completo."""
    print("\n" + "=" * 70)
    print("EXPERIMENTO 6 - Visualizacao e Classificacao")
    print("=" * 70)

    unique_labels = np.unique(all_labels)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(sorted(unique_labels))}
    y = np.array([label_to_idx[lbl] for lbl in all_labels])

    # a) Visualizacao
    print("\n--- Visualizacao com features selecionadas ---")
    visualize_feature_space(selected_features, y, class_names, "selecionadas")

    print("\n--- Visualizacao com todas as features ---")
    visualize_feature_space(X_normalized, y, class_names, "todas")

    # b) Classificacao
    X_train, X_test, y_train, y_test = train_test_split(
        selected_features, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    print(f"\n[Treino/Teste] {len(y_train)} treino, {len(y_test)} teste")

    print(f"\n[Classificador Principal: {config.CLASSIFIER.upper()}]")

    classifier = get_classifier()
    classifier, y_pred, accuracy, report, cm = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        classifier=classifier,
        class_names=class_names
    )

    print(f"\n  Acuracia: {accuracy:.4f}")
    print(f"\n  Relatorio de Classificacao:")
    print(report)

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    cv_scores = cross_val_score(classifier, selected_features, y, cv=cv,
                                scoring='accuracy')
    print(f"  Validacao Cruzada ({config.CV_FOLDS} folds):")
    print(f"    Acuracia: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    clf_results = compare_classifiers(X_train, X_test, y_train, y_test, class_names)

    # c) Impacto da reducao
    dim_impact = evaluate_dimensionality_impact(
        X_normalized, selected_features, y, class_names
    )

    # Estatisticas por classe
    print("\n[Estatisticas por Classe]")
    stats = compute_class_statistics(selected_features, y, selected_names, class_names)
    print_statistics_table(stats)

    # Visualizacoes
    print("\n[GERACAO DE FIGURAS]")

    fig = plot_confusion_matrix(cm, class_names)
    save_figure(fig, "exp6_matriz_confusao.png", subdir="experimento6")

    fig = plot_roc_curves(X_train, X_test, y_train, y_test, class_names)
    save_figure(fig, "exp6_curvas_roc.png", subdir="experimento6")

    # Comparacao de classificadores
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(clf_results.keys())
    accuracies = [clf_results[n]["accuracy"] for n in names]
    cv_means = [clf_results[n]["cv_mean"] for n in names]
    cv_stds = [clf_results[n]["cv_std"] for n in names]

    x_pos = np.arange(len(names))
    width = 0.35
    ax.bar(x_pos - width/2, accuracies, width, label='Teste', color='steelblue')
    ax.bar(x_pos + width/2, cv_means, width, yerr=cv_stds,
           label='CV (media)', color='coral', capsize=5)
    ax.set_ylabel("Acuracia", fontsize=12)
    ax.set_title("Comparacao de Classificadores", fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "exp6_comparacao_classificadores.png", subdir="experimento6")

    # Impacto da reducao
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
    ax.set_ylabel("Acuracia CV", fontsize=12)
    ax.set_title("Impacto da Reducao de Dimensionalidade", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "exp6_impacto_dimensionalidade.png", subdir="experimento6")

    # Fronteira de decisao
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(selected_features)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx, cls_name in enumerate(class_names):
        mask = y == cls_idx
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  label=cls_name, alpha=0.6, s=40)

    svm_2d = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_2d.fit(X_pca, y)

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
    ax.set_title("Fronteira de Decisao (SVM sobre PCA)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    fig.tight_layout()
    save_figure(fig, "exp6_fronteira_decisao.png", subdir="experimento6")

    print("\n[EXPERIMENTO 6 CONCLUIDO]")

    return {
        "accuracy": accuracy,
        "cv_scores": cv_scores,
        "confusion_matrix": cm,
        "classification_report": report,
        "classifier_results": clf_results,
        "dimensionality_impact": dim_impact,
        "class_statistics": stats,
    }
