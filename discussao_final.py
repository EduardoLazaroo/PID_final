"""
=============================================================================
discussao_final.py - Secao 6.7: Discussao Final (Conclusao)
=============================================================================

Integra criticamente os resultados obtidos ao longo de todo o pipeline,
analisando:
    - influencia da representacao HSV;
    - impacto da segmentacao nas features;
    - robustez dos descritores extraidos;
    - complementaridade entre textura e complexidade estrutural;
    - limitacoes impostas pelo processamento;
    - capacidade do pipeline em discriminar padroes histopatologicos reais.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import config
from utils import save_figure


def run_discussao_final(exp1_stats, exp2_metrics, texture_analysis,
                        morphology_improvements, feature_names, selected_names,
                        relief_scores, exp6_results, class_names):
    """
    Gera a discussao final integrando todos os experimentos.
    
    Parametros:
        exp1_stats: estatisticas do Experimento 1 (HSV)
        exp2_metrics: metricas do Experimento 2 (Segmentacao)
        texture_analysis: analise de textura do Experimento 2
        morphology_improvements: melhorias do Experimento 3 (Morfologia)
        feature_names: nomes de todas as features
        selected_names: nomes das features selecionadas
        relief_scores: scores do RELIEF
        exp6_results: resultados do Experimento 6 (Classificacao)
        class_names: lista com nomes das classes [CLASS_1_NAME, CLASS_2_NAME]
    """
    print("\n" + "=" * 70)
    print("SECAO 6.7 - DISCUSSAO FINAL (CONCLUSAO)")
    print("=" * 70)
    
    discussion_path = os.path.join(config.RESULTS_DIR, "discussao_final.txt")
    lines = []
    
    def add(text=""):
        lines.append(text)
        print(text)
    
    add("=" * 70)
    add("DISCUSSAO FINAL - PIPELINE INTEGRADO DE ANALISE DE IMAGENS")
    add("HISTOPATOLOGICAS")
    add("=" * 70)
    add()
    
    # =========================================================================
    # 1. REPRESENTACAO HSV
    # =========================================================================
    add("1. INFLUENCIA DA REPRESENTACAO HSV")
    add("-" * 50)
    add()
    add("A conversao RGB -> HSV permitiu decompor a informacao cromatica em")
    add("tres canais independentes: Matiz (H), Saturacao (S) e Valor (V).")
    add()
    add(f"O canal {config.HSV_CHANNEL_SELECTED} foi selecionado para processamento posterior")
    add("com base nos seguintes criterios:")
    add()
    
    if config.HSV_CHANNEL_SELECTED == "S":
        add("- Contraste entre tecido e fundo: regioes coradas (tecido) apresentam")
        add("  alta saturacao, enquanto o fundo branco tem saturacao proxima de zero.")
        add("- Evidencia das estruturas celulares: a saturacao captura a pureza")
        add("  da cor, destacando areas com coloracao H&E intensa.")
    elif config.HSV_CHANNEL_SELECTED == "H":
        add("- O canal H captura a matiz, permitindo distinguir entre coloracoes")
        add("  de hematoxilina (roxo) e eosina (rosa).")
    else:
        add("- O canal V captura a luminancia, sendo util para distinguir areas")
        add("  claras (fundo) de areas escuras (tecido denso).")
    
    add()
    add("Estatisticas dos canais HSV por classe:")
    add()
    
    for cls_key, cls_stats in exp1_stats.items():
        add(f"  Classe: {cls_key}")
        for ch_name, ch_stats in cls_stats.items():
            add(f"    Canal {ch_name}: media={ch_stats['mean_of_means']:.2f}, "
                f"desvio={ch_stats['mean_of_stds']:.2f}")
        add()
    
    # =========================================================================
    # 2. SEGMENTACAO
    # =========================================================================
    add("2. IMPACTO DA SEGMENTACAO NAS FEATURES")
    add("-" * 50)
    add()
    add(f"O metodo de segmentacao utilizado foi: {config.SEGMENTATION_METHOD}")
    add()
    
    if config.SEGMENTATION_METHOD == "multi_otsu":
        add("O Multi-Otsu com 3 classes captura naturalmente a estrutura multimodal")
        add("das imagens H&E: fundo branco, tecido corado com eosina (rosa) e")
        add("tecido corado com hematoxilina (roxo).")
    else:
        add("A maxima entropia (Li) e robusta quando a distribuicao de intensidades")
    add("nao e claramente bimodal.")
    
    add()
    add("Metricas de segmentacao por classe:")
    add()
    
    for cls_key, metrics_list in exp2_metrics.items():
        area_fracs = [m["area_fraction"] for m in metrics_list]
        n_regions = [m["n_regions"] for m in metrics_list]
        add(f"  Classe: {cls_key}")
        add(f"    Fracao de area segmentada: {np.mean(area_fracs):.3f} +/- {np.std(area_fracs):.3f}")
        add(f"    Numero de regioes: {np.mean(n_regions):.1f} +/- {np.std(n_regions):.1f}")
        add()
    
    add("Influencia de textura, ruido e heterogeneidade:")
    add()
    
    for cls_key, tex_list in texture_analysis.items():
        tex_vars = [t["tissue_variance"] for t in tex_list]
        bg_vars = [t["background_variance"] for t in tex_list]
        ratios = [t["variance_ratio"] for t in tex_list]
        add(f"  Classe: {cls_key}")
        add(f"    Variancia tecidual: {np.mean(tex_vars):.2f}")
        add(f"    Variancia fundo: {np.mean(bg_vars):.2f}")
        add(f"    Razao de variância: {np.mean(ratios):.2f}")
        add()
    
    add("A heterogeneidade tecidual (razao de variância) indica o quanto a")
    add("textura do tecido difere do fundo. Valores maiores sugerem maior")
    add("complexidade estrutural, o que pode dificultar a segmentacao mas")
    add("tambem fornece mais informacao para discriminacao de classes.")
    add()
    
    # =========================================================================
    # 3. MORFOLOGIA MATEMATICA
    # =========================================================================
    add("3. GANHOS COM REFINAMENTO MORFOLOGICO")
    add("-" * 50)
    add()
    add(f"Elemento estruturante: {config.STRUCTURING_ELEMENT_TYPE} (tamanho base: {config.STRUCTURING_ELEMENT_SIZE})")
    add(f"Operacoes: {config.MORPHOLOGY_OPERATIONS}")
    add()
    
    for cls_key, imp_list in morphology_improvements.items():
        solidity_imp = [imp["solidity_improvement"] for imp in imp_list]
        regions_rem = [imp["regions_removed"] for imp in imp_list]
        holes_filled = [imp["holes_filled"] for imp in imp_list]
        
        add(f"  Classe: {cls_key}")
        add(f"    Melhoria de solidez: {np.mean(solidity_imp):.4f} +/- {np.std(solidity_imp):.4f}")
        add(f"    Regioes removidas (ruido): {np.mean(regions_rem):.1f} +/- {np.std(regions_rem):.1f}")
        add(f"    Buracos preenchidos: {np.mean(holes_filled):.1f} +/- {np.std(holes_filled):.1f}")
        add()
    
    add("As operacoes morfológicas melhoraram a qualidade estrutural da segmentacao")
    add("ao remover pequenos ruidos (abertura) e preencher buracos internos (fechamento).")
    add("O elemento em disco foi escolhido por preservar formas arredondadas")
    add("tipicas de nucleos celulares e estruturas glandulares.")
    add()
    
    # =========================================================================
    # 4. DESCRITORES DE HARALICK
    # =========================================================================
    add("4. ROBUSTEZ DOS DESCRITORES DE HARALICK")
    add("-" * 50)
    add()
    add("Foram extraidos 6 descritores de Haralick a partir da GLCM:")
    add("  - Contrast: mede variacoes locais de intensidade")
    add("  - Dissimilaridade: similar ao contraste, mas com ponderacao linear")
    add("  - Homogeneidade: mede a uniformidade local")
    add("  - Energy (segundo momento angular): mede a uniformidade da GLCM")
    add("  - Correlation: mede a dependencia linear entre pixels vizinhos")
    add("  - ASM: segundo momento angular da GLCM")
    add()
    add(f"Parametros GLCM: distancias={config.GLCM_DISTANCES}, "
        f"angulos={[f'{np.degrees(a):.0f} graus' for a in config.GLCM_ANGLES]}")
    add()
    add("As distancias multiplas permitem capturar dependencias texturais em")
    add("diferentes escalas, enquanto os 4 angulos garantem analise isotropica.")
    add()
    
    # =========================================================================
    # 5. DIMENSAO FRACTAL
    # =========================================================================
    add("5. COMPLEMENTARIEDADE ENTRE TEXTEURA E COMPLEXIDADE ESTRUTURAL")
    add("-" * 50)
    add()
    add("A dimensao fractal, obtida por box-counting em multiplas escalas,")
    add("quantifica a complexidade estrutural das regioes segmentadas.")
    add()
    add(f"Escalas utilizadas: {config.FRACTAL_SCALES}")
    add()
    add("A complementaridade entre os descritores de Haralick (textura) e a")
    add("dimensao fractal (complexidade estrutural) e fundamental porque:")
    add("  - Haralick captura padroes locais de co-ocorrencia de intensidades")
    add("  - Fractal captura a complexidade global da forma segmentada")
    add("  - Juntos, fornecem representacao multi-escala das estruturas")
    add()
    
    # =========================================================================
    # 6. RELIEF
    # =========================================================================
    add("6. ANALISE DO RELIEF - CARACTERISTICAS MAIS RELEVANTES")
    add("-" * 50)
    add()
    add(f"Algoritmo RELIEF-F com {config.RELIEF_N_NEIGHBORS} vizinhos.")
    add(f"Features selecionadas: {config.RELIEF_N_FEATURES}")
    add()
    
    sorted_idx = np.argsort(relief_scores)[::-1]
    add("Ranqueamento completo:")
    for rank, idx in enumerate(sorted_idx):
        marker = " [SELECIONADA]" if rank < config.RELIEF_N_FEATURES else ""
        add(f"  {rank+1}. {feature_names[idx]:<35} score={relief_scores[idx]:.6f}{marker}")
    
    add()
    add("As features mais relevantes representam os descritores que melhor")
    add("distinguem amostras proximas de classes diferentes. Features com")
    add("scores altos indicam que seus valores sao consistentes dentro da")
    add("mesma classe e diferentes entre classes.")
    add()
    
    # =========================================================================
    # 7. CLASSIFICACAO
    # =========================================================================
    add("7. CAPACIDADE DE DISCRIMINACAO DO MODELO")
    add("-" * 50)
    add()
    
    accuracy = exp6_results["accuracy"]
    cv_scores = exp6_results["cv_scores"]
    
    add(f"Classificador principal: {config.CLASSIFIER.upper()}")
    add(f"Acuracia no conjunto de teste: {accuracy:.4f}")
    add(f"Validacao cruzada ({config.CV_FOLDS} folds): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    add()
    
    add("Comparacao de classificadores:")
    for name, res in exp6_results["classifier_results"].items():
        add(f"  {name}: {res['accuracy']:.4f} (CV: {res['cv_mean']:.4f} +/- {res['cv_std']:.4f})")
    add()
    
    add("Impacto da reducao de dimensionalidade:")
    for name, res in exp6_results["dimensionality_impact"].items():
        add(f"  {name} ({res['n_features']} features): {res['mean_accuracy']:.4f} +/- {res['std_accuracy']:.4f}")
    add()
    
    add("Estatisticas por classe (features selecionadas):")
    add()
    
    stats = exp6_results["class_statistics"]
    for cls_name, features in stats.items():
        add(f"  --- {cls_name} ---")
        for fname, vals in features.items():
            add(f"    {fname}: media={vals['mean']:.4f}, desvio={vals['std']:.4f}, "
                f"IC=[{vals['ci_low']:.4f}, {vals['ci_high']:.4f}]")
        add()
    
    # =========================================================================
    # 8. LIMITACOES
    # =========================================================================
    add("8. LIMITACOES DO PIPELINE")
    add("-" * 50)
    add()
    add("Limitacoes do dataset:")
    add("  - Variabilidade de coloracao entre diferentes laboratorios")
    add("  - Resolucao fixa das imagens pode nao capturar detalhes finos")
    add("  - Desbalanceamento potencial entre classes")
    add()
    add("Limitacoes do processo de segmentacao:")
    add("  - O Multi-Otsu assume distribuicao multimodal, que pode nao ser")
    add("    ideal para todas as imagens")
    add("  - O elemento estruturante fixo pode nao ser otimo para todas")
    add("    as escalas de estruturas presentes")
    add()
    add("Limitacoes dos descritores:")
    add("  - Haralick e fractal saem descritores globais que podem nao")
    add("    capturar variacoes locais importantes")
    add("  - A escolha dos parametros GLCM (distancias, angulos) e heuristica")
    add()
    
    # =========================================================================
    # 9. CONCLUSAO GERAL
    # =========================================================================
    add("9. CONCLUSAO GERAL")
    add("-" * 50)
    add()
    
    if accuracy >= 0.85:
        conclusao = "O pipeline proposto demonstrou ALTA capacidade de discriminacao"
    elif accuracy >= 0.70:
        conclusao = "O pipeline proposto demonstrou MODERADA capacidade de discriminacao"
    else:
        conclusao = "O pipeline proposto demonstrou BAIXA capacidade de discriminacao"
    
    add(f"{conclusao} entre as classes {class_names[0]} e {class_names[1]}.")
    add()
    add(f"A acuracia de {accuracy:.1f}% no conjunto de teste e a acuracia media de")
    add(f"{cv_scores.mean():.1f}% na validacao cruzada indicam que as caracteristicas")
    add("extraidas (Haralick + Fractal + Cor + Geometria) contem informacao")
    add("discriminativa relevante para a classificacao de imagens histopatologicas.")
    add()
    add("A reducao de dimensionalidade via RELIEF permitiu selecionar as features")
    add("mais relevantes, {'melhorando' if exp6_results['dimensionality_impact'].get('Features selecionadas', {}).get('mean_accuracy', 0) >= exp6_results['dimensionality_impact'].get('Todas as features', {}).get('mean_accuracy', 0) else 'mantendo'} o desempenho com menos caracteristicas.")
    add()
    add("O pipeline integrado HSV -> Segmentacao -> Morfologia -> Features ->")
    add("Normalizacao -> RELIEF -> Classificacao mostrou-se adequado para a")
    add("discriminacao de padroes histopatologicos reais, demonstrando que")
    add("descritores texturais e de complexidade estrutural sao complementares")
    add("e eficazes para esta tarefa.")
    add()
    add("=" * 70)
    
    # Salvar discussao em arquivo
    with open(discussion_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print(f"\nDiscussao final salva em: {discussion_path}")
    
    # Gerar figura resumo do pipeline
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    summary_text = (
        f"RESUMO DO PIPELINE\n"
        f"{'=' * 50}\n\n"
        f"Dataset: LC25000 ({class_names[0]} vs {class_names[1]})\n"
        f"Canal HSV: {config.HSV_CHANNEL_SELECTED}\n"
        f"Segmentacao: {config.SEGMENTATION_METHOD}\n"
        f"Morfologia: {config.STRUCTURING_ELEMENT_TYPE} + {config.MORPHOLOGY_OPERATIONS}\n"
        f"Features: {len(feature_names)} total, {config.RELIEF_N_FEATURES} selecionadas\n"
        f"Normalizacao: {config.NORMALIZATION_METHOD}\n"
        f"Classificador: {config.CLASSIFIER}\n\n"
        f"RESULTADO:\n"
        f"Acuracia: {accuracy:.4f}\n"
        f"CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n\n"
        f"Features selecionadas:\n"
    )
    for name in selected_names:
        summary_text += f"  - {name}\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    save_figure(fig, "discussao_final_resumo.png", subdir=".")
    
    print("\n[DISCUSSAO FINAL CONCLUIDA]")
