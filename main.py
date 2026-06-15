"""
=============================================================================
main.py - Pipeline Integrado de Segmentação, Representação e Análise
         de Características em Imagens Histopatológicas
=============================================================================

PPGCC - Processamento de Imagens Digitais (PID)
Desafio 6 - Final
Prof. Leandro

Este script executa o pipeline completo:
  Experimento 1: Representação HSV e análise dos canais
  Experimento 2: Segmentação das estruturas histopatológicas
  Experimento 3: Refinamento com morfologia matemática
  Experimento 4: Extração e análise de características
  Experimento 5: Normalização e seleção de características
  Experimento 6: Visualização e classificação
  Seção 6.7: Discussão Final

Uso:
  python main.py                    # Executa pipeline completo
  python main.py --exp 1            # Executa apenas experimento 1
  python main.py --exp 1 2 3        # Executa experimentos 1 a 3
  python main.py --class1 colon_aca --class2 colon_n  # Escolher classes
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica
import matplotlib.pyplot as plt

import config
from utils import load_dataset, save_figure
from experimento1_hsv import run_experiment_1
from experimento2_segmentacao import run_experiment_2
from experimento3_morfologia import run_experiment_3
from experimento4_features import run_experiment_4
from experimento5_relief import run_experiment_5
from experimento6_classificacao import run_experiment_6
from discussao_final import run_discussao_final


def parse_arguments():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Análise de Imagens Histopatológicas"
    )
    parser.add_argument(
        '--exp', nargs='+', type=int, choices=range(1, 7),
        help='Experimentos a executar (1-6). Default: todos'
    )
    parser.add_argument(
        '--class1', type=str, default=None,
        help='Primeira classe (ex: colon_aca). Default: colon_aca'
    )
    parser.add_argument(
        '--class2', type=str, default=None,
        help='Segunda classe (ex: colon_n). Default: colon_n'
    )
    return parser.parse_args()


def build_class_name_map():
    """
    Cria mapeamento de chave de classe para nome legível.
    Retorna dict: {'colon_aca': 'Colon Adenocarcinoma', 'colon_n': 'Colon Benigno'}
    """
    return {
        config.CLASS_1: config.CLASS_1_NAME,
        config.CLASS_2: config.CLASS_2_NAME,
    }


def print_header():
    """Imprime cabeçalho do pipeline."""
    print("=" * 70)
    print("  PIPELINE INTEGRADO DE ANÁLISE DE IMAGENS HISTOPATOLÓGICAS")
    print("  PPGCC - Processamento de Imagens Digitais (PID)")
    print("  Desafio 6 - Final")
    print("=" * 70)
    print()
    print(f"  Dataset: LC25000 (Lung and Colon Cancer)")
    print(f"  Classe 1: {config.CLASS_1_NAME} ({config.CLASS_1})")
    print(f"  Classe 2: {config.CLASS_2_NAME} ({config.CLASS_2})")
    print(f"  Canal HSV: {config.HSV_CHANNEL_SELECTED}")
    print(f"  Segmentação: {config.SEGMENTATION_METHOD}")
    print(f"  Normalização: {config.NORMALIZATION_METHOD}")
    print(f"  Classificador: {config.CLASSIFIER}")
    print()


def print_summary(results):
    """Imprime resumo dos resultados."""
    print("\n" + "=" * 70)
    print("  RESUMO DOS RESULTADOS")
    print("=" * 70)

    if "exp6" in results:
        exp6 = results["exp6"]
        print(f"\n  Acuracia final: {exp6['accuracy']:.4f}")
        print(f"  CV ({config.CV_FOLDS} folds): {exp6['cv_scores'].mean():.4f} +/- {exp6['cv_scores'].std():.4f}")

        print("\n  Comparacao de Classificadores:")
        for name, res in exp6["classifier_results"].items():
            print(f"    {name}: {res['accuracy']:.4f} (CV: {res['cv_mean']:.4f} +/- {res['cv_std']:.4f})")

        print("\n  Impacto da Reducao de Dimensionalidade:")
        for name, res in exp6["dimensionality_impact"].items():
            print(f"    {name} ({res['n_features']} feats): {res['mean_accuracy']:.4f} +/- {res['std_accuracy']:.4f}")

    print("\n" + "=" * 70)
    print("  PIPELINE CONCLUIDO COM SUCESSO")
    print("=" * 70)
    print(f"\n  Resultados salvos em: {config.RESULTS_DIR}")
    print(f"  Figuras salvas em: {config.FIGURES_DIR}")
    print()


def main():
    """Função principal do pipeline."""
    args = parse_arguments()

    # Aplicar argumentos
    if args.class1:
        config.CLASS_1 = args.class1
    if args.class2:
        config.CLASS_2 = args.class2

    experiments = args.exp  # None = todos

    # Criar diretórios
    config.create_directories()

    # Cabeçalho
    print_header()

    # Mapeamento de nomes de classes (chave -> nome legível)
    class_name_map = build_class_name_map()
    class_names_list = [config.CLASS_1_NAME, config.CLASS_2_NAME]

    # Iniciar timer
    start_time = time.time()

    # =========================================================================
    # CARREGAR DATASET
    # =========================================================================
    print("[1/8] Carregando dataset...")
    images_rgb, labels_dict, paths_dict = load_dataset()

    # Verificar se há imagens
    for cls in images_rgb:
        if len(images_rgb[cls]) == 0:
            print(f"[ERRO] Nenhuma imagem encontrada para {cls}")
            print(f"Verifique o caminho: {os.path.join(config.DATASET_ROOT, cls)}")
            sys.exit(1)

    print(f"  Total de imagens: {sum(len(v) for v in images_rgb.values())}")

    # =========================================================================
    # EXPERIMENTO 1: Representação HSV
    # =========================================================================
    if experiments is None or 1 in experiments:
        print("\n[2/8] Executando Experimento 1: Representacao HSV...")
        exp1_start = time.time()
        selected_channels, exp1_stats = run_experiment_1(images_rgb, class_name_map)
        exp1_time = time.time() - exp1_start
        print(f"  Tempo: {exp1_time:.1f}s")
    else:
        selected_channels = None

    # =========================================================================
    # EXPERIMENTO 2: Segmentação
    # =========================================================================
    if experiments is None or 2 in experiments:
        if selected_channels is None:
            print("\n[ERRO] Experimento 2 requer saida do Experimento 1.")
            print("Execute: python main.py --exp 1 2")
            sys.exit(1)

        print("\n[3/8] Executando Experimento 2: Segmentacao...")
        exp2_start = time.time()
        segmentation_results, exp2_metrics, texture_analysis = run_experiment_2(
            selected_channels, class_name_map
        )
        exp2_time = time.time() - exp2_start
        print(f"  Tempo: {exp2_time:.1f}s")
    else:
        segmentation_results = None

    # =========================================================================
    # EXPERIMENTO 3: Morfologia Matemática
    # =========================================================================
    if experiments is None or 3 in experiments:
        if segmentation_results is None:
            print("\n[ERRO] Experimento 3 requer saida do Experimento 2.")
            print("Execute: python main.py --exp 1 2 3")
            sys.exit(1)

        print("\n[4/8] Executando Experimento 3: Morfologia Matematica...")
        exp3_start = time.time()
        refined_results, morphology_improvements = run_experiment_3(
            images_rgb, segmentation_results, class_name_map
        )
        exp3_time = time.time() - exp3_start
        print(f"  Tempo: {exp3_time:.1f}s")
    else:
        refined_results = None

    # =========================================================================
    # EXPERIMENTO 4: Extração de Características
    # =========================================================================
    if experiments is None or 4 in experiments:
        if refined_results is None or selected_channels is None:
            print("\n[ERRO] Experimento 4 requer saidas dos Experimentos 1-3.")
            print("Execute: python main.py --exp 1 2 3 4")
            sys.exit(1)

        print("\n[5/8] Executando Experimento 4: Extracao de Caracteristicas...")
        exp4_start = time.time()
        all_features, all_labels, feature_names = run_experiment_4(
            images_rgb, selected_channels, refined_results, class_name_map
        )
        exp4_time = time.time() - exp4_start
        print(f"  Tempo: {exp4_time:.1f}s")
    else:
        all_features = None

    # =========================================================================
    # EXPERIMENTO 5: Normalização e Seleção
    # =========================================================================
    if experiments is None or 5 in experiments:
        if all_features is None:
            print("\n[ERRO] Experimento 5 requer saida do Experimento 4.")
            print("Execute: python main.py --exp 1 2 3 4 5")
            sys.exit(1)

        print("\n[6/8] Executando Experimento 5: Normalizacao e Selecao...")
        exp5_start = time.time()
        X_normalized, selected_features, selected_names, relief_scores = run_experiment_5(
            all_features, all_labels, feature_names, class_names_list
        )
        exp5_time = time.time() - exp5_start
        print(f"  Tempo: {exp5_time:.1f}s")
    else:
        X_normalized = None
        selected_features = None
        selected_names = None

    # =========================================================================
    # EXPERIMENTO 6: Visualização e Classificação
    # =========================================================================
    exp6_results = None
    if experiments is None or 6 in experiments:
        if X_normalized is None or selected_features is None:
            print("\n[ERRO] Experimento 6 requer saidas dos Experimentos 1-5.")
            print("Execute: python main.py --exp 1 2 3 4 5 6")
            sys.exit(1)

        print("\n[7/8] Executando Experimento 6: Visualizacao e Classificacao...")
        exp6_start = time.time()
        exp6_results = run_experiment_6(
            X_normalized, selected_features, selected_names,
            all_labels, feature_names, class_names_list
        )
        exp6_time = time.time() - exp6_start
        print(f"  Tempo: {exp6_time:.1f}s")

    # =========================================================================
    # SEÇÃO 6.7: Discussão Final
    # =========================================================================
    if experiments is None and exp6_results is not None:
        print("\n[8/8] Gerando Discussao Final (Secao 6.7)...")
        exp7_start = time.time()
        run_discussao_final(
            exp1_stats=exp1_stats,
            exp2_metrics=exp2_metrics,
            texture_analysis=texture_analysis,
            morphology_improvements=morphology_improvements,
            feature_names=feature_names,
            selected_names=selected_names,
            relief_scores=relief_scores,
            exp6_results=exp6_results,
            class_names=class_names_list,
        )
        exp7_time = time.time() - exp7_start
        print(f"  Tempo: {exp7_time:.1f}s")

    # =========================================================================
    # RESUMO
    # =========================================================================
    total_time = time.time() - start_time
    print(f"\n[TEMPO TOTAL] {total_time:.1f}s ({total_time/60:.1f} min)")

    if exp6_results is not None:
        results = {"exp6": exp6_results}
        print_summary(results)

    # Salvar resumo em arquivo
    summary_path = os.path.join(config.RESULTS_DIR, "resumo.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("RESUMO DO PIPELINE DE ANALISE DE IMAGENS HISTOPATOLOGICAS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: LC25000\n")
        f.write(f"Classe 1: {config.CLASS_1_NAME} ({config.CLASS_1})\n")
        f.write(f"Classe 2: {config.CLASS_2_NAME} ({config.CLASS_2})\n")
        f.write(f"Total de imagens: {sum(len(v) for v in images_rgb.values())}\n")
        f.write(f"Tempo total: {total_time:.1f}s\n\n")

        if exp6_results is not None:
            f.write(f"Acuracia: {exp6_results['accuracy']:.4f}\n")
            f.write(f"CV: {exp6_results['cv_scores'].mean():.4f} +/- {exp6_results['cv_scores'].std():.4f}\n")
            f.write(f"\nFeatures selecionadas:\n")
            for name in selected_names:
                f.write(f"  - {name}\n")

    print(f"Resumo salvo em: {summary_path}")


if __name__ == "__main__":
    main()
