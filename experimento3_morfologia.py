"""
=============================================================================
experimento3_morfologia.py - Experimento 3: Refinamento com Morfologia Matematica
=============================================================================

Procedimentos:
    a) Aplicar operacoes de morfologia matematica
    b) Definir e justificar elemento(s) estruturante(s)
    c) Comparar imagens antes e depois do refinamento
    d) Gerar mascara binaria final

Justificativa dos elementos estruturantes:
    - Elemento estruturante em disco (disk): Preserva formas arredondadas
      tipicas de nucleos celulares e estruturas glandulares.
    - Abertura (erosao seguida de dilatacao): Remove pequenos ruidos.
    - Fechamento (dilatacao seguida de erosao): Preenche pequenos buracos.

Questao:
    As operacoes morfologicas melhoraram a qualidade estrutural da segmentacao?
    Quanto?
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import config
from utils import save_figure, plot_morphology_comparison


def create_structuring_element(shape, size):
    """
    Cria elemento estruturante para operacoes morfologicas.
    """
    if shape == "disk":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size + 1, 2 * size + 1))
    elif shape == "square":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * size + 1, 2 * size + 1))
    elif shape == "diamond":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * size + 1, 2 * size + 1))
    elif shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size + 1, 2 * size + 1))
    else:
        raise ValueError(f"Forma desconhecida: {shape}")
    
    return kernel


def apply_morphology(binary_mask, operations, se_type=None, se_size=None):
    """
    Aplica sequencia de operacoes morfologicas.
    """
    if se_type is None:
        se_type = config.STRUCTURING_ELEMENT_TYPE
    if se_size is None:
        se_size = config.STRUCTURING_ELEMENT_SIZE
    
    mask = binary_mask.copy()
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    intermediate_results = {"original": mask.copy()}
    current = mask.copy()
    
    for op_name, op_size in operations:
        kernel = create_structuring_element(se_type, op_size)
        
        if op_name == "opening":
            current = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel)
        elif op_name == "closing":
            current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, kernel)
        elif op_name == "erosion":
            current = cv2.erode(current, kernel, iterations=1)
        elif op_name == "dilation":
            current = cv2.dilate(current, kernel, iterations=1)
        else:
            raise ValueError(f"Operacao desconhecida: {op_name}")
        
        intermediate_results[f"{op_name}_{op_size}"] = current.copy()
    
    refined_mask = (current > 0).astype(np.uint8)
    intermediate_results["final"] = refined_mask.copy()
    
    return refined_mask, intermediate_results


def evaluate_morphology_improvement(mask_before, mask_after):
    """
    Quantifica a melhoria da segmentacao apos morfologia.
    """
    improvement = {}
    
    before = mask_before.copy()
    after = mask_after.copy()
    if before.max() <= 1:
        before = (before * 255).astype(np.uint8)
    if after.max() <= 1:
        after = (after * 255).astype(np.uint8)
    
    # Mudanca na area
    area_before = np.sum(before > 0)
    area_after = np.sum(after > 0)
    improvement["area_change"] = area_after - area_before
    improvement["area_change_pct"] = (
        (area_after - area_before) / max(area_before, 1) * 100
    )
    
    # Numero de regioes conectadas
    labeled_before = measure.label(before > 0)
    regions_before = measure.regionprops(labeled_before)
    improvement["n_regions_before"] = len(regions_before)
    
    labeled_after = measure.label(after > 0)
    regions_after = measure.regionprops(labeled_after)
    improvement["n_regions_after"] = len(regions_after)
    improvement["regions_removed"] = len(regions_before) - len(regions_after)
    
    # Solidez media
    if len(regions_before) > 0:
        solidities_before = [r.solidity for r in regions_before if r.area > 10]
        improvement["mean_solidity_before"] = (
            np.mean(solidities_before) if solidities_before else 0
        )
    else:
        improvement["mean_solidity_before"] = 0
    
    if len(regions_after) > 0:
        solidities_after = [r.solidity for r in regions_after if r.area > 10]
        improvement["mean_solidity_after"] = (
            np.mean(solidities_after) if solidities_after else 0
        )
    else:
        improvement["mean_solidity_after"] = 0
    
    improvement["solidity_improvement"] = (
        improvement["mean_solidity_after"] - improvement["mean_solidity_before"]
    )
    
    # Buracos preenchidos
    holes_before = count_holes(before > 0)
    holes_after = count_holes(after > 0)
    improvement["holes_before"] = holes_before
    improvement["holes_after"] = holes_after
    improvement["holes_filled"] = holes_before - holes_after
    
    return improvement


def count_holes(binary_mask):
    """Conta numero de buracos (regioes de fundo dentro de objeto)."""
    inverted = ~binary_mask.astype(bool)
    labeled = measure.label(inverted)
    regions = measure.regionprops(labeled)
    h, w = binary_mask.shape
    holes = 0
    for r in regions:
        minr, minc, maxr, maxc = r.bbox
        if minr > 0 and minc > 0 and maxr < h and maxc < w:
            holes += 1
    return holes


def extract_colored_regions(image_rgb, binary_mask):
    """
    Extrai regioes coloridas da imagem original usando a mascara final.
    """
    mask_3ch = np.stack([binary_mask] * 3, axis=-1)
    extracted = image_rgb * mask_3ch
    return extracted


def run_experiment_3(images_rgb, segmentation_results, class_name_map):
    """
    Executa o Experimento 3 completo.
    
    Parametros:
        images_rgb: dict {classe: [lista de imagens RGB]}
        segmentation_results: resultados do Experimento 2
        class_name_map: dict {chave_classe: nome_legivel}
    
    Retorna:
        refined_results: dict com mascaras refinadas e regioes extraidas
        improvements: metricas de melhoria
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 3 - Refinamento com Morfologia Matematica")
    print("=" * 70)
    
    refined_results = {}
    improvements = {}
    
    print(f"\nElemento estruturante: {config.STRUCTURING_ELEMENT_TYPE}")
    print(f"Tamanho base: {config.STRUCTURING_ELEMENT_SIZE}")
    print(f"Operacoes: {config.MORPHOLOGY_OPERATIONS}")
    
    print("\nJustificativa:")
    print("  - Elemento em disco: preserva formas arredondadas de nucleos")
    print("    celulares e estruturas glandulares, evitando artefatos angulares.")
    print("  - Abertura: remove pequenos ruidos (regioes espúrias pequenas).")
    print("  - Fechamento: preenche buracos internos no tecido segmentado.")
    
    for cls in sorted(segmentation_results.keys()):
        cls_name = class_name_map.get(cls, cls)
        print(f"\n--- Classe: {cls_name} ---")
        refined_results[cls] = []
        improvements[cls] = []
        
        for i, seg in enumerate(segmentation_results[cls]):
            mask_before = seg["binary_mask"]
            
            # a) Aplicar morfologia
            mask_after, intermediates = apply_morphology(
                mask_before,
                operations=config.MORPHOLOGY_OPERATIONS
            )
            
            # b) Avaliar melhoria
            improvement = evaluate_morphology_improvement(mask_before, mask_after)
            improvements[cls].append(improvement)
            
            # d) Extrair regioes coloridas
            extracted = extract_colored_regions(images_rgb[cls][i], mask_after)
            
            refined_results[cls].append({
                "mask_before": mask_before,
                "mask_after": mask_after,
                "intermediates": intermediates,
                "extracted": extracted,
            })
        
        # Resumo
        solidity_imp = [imp["solidity_improvement"] for imp in improvements[cls]]
        regions_rem = [imp["regions_removed"] for imp in improvements[cls]]
        holes_filled = [imp["holes_filled"] for imp in improvements[cls]]
        
        print(f"  Melhoria de solidez: {np.mean(solidity_imp):.4f} +/- {np.std(solidity_imp):.4f}")
        print(f"  Regioes removidas: {np.mean(regions_rem):.1f} +/- {np.std(regions_rem):.1f}")
        print(f"  Buracos preenchidos: {np.mean(holes_filled):.1f} +/- {np.std(holes_filled):.1f}")
    
    # Visualizacoes
    print("\n[GERACAO DE FIGURAS]")
    
    # Figura: Comparacao antes/depois
    for cls in sorted(refined_results.keys()):
        n_show = min(3, len(refined_results[cls]))
        for i in range(n_show):
            res = refined_results[cls][i]
            fig = plot_morphology_comparison(
                res["mask_before"],
                res["mask_after"],
                res["mask_before"],
                res["mask_after"]
            )
            save_figure(
                fig,
                f"exp3_morfologia_{cls}_img{i+1}.png",
                subdir="experimento3"
            )
    
    # Figura: Regioes extraidas (coloridas)
    for cls in sorted(refined_results.keys()):
        cls_name = class_name_map.get(cls, cls)
        n_show = min(4, len(refined_results[cls]))
        fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
        for i in range(n_show):
            res = refined_results[cls][i]
            axes[0, i].imshow(images_rgb[cls][i])
            axes[0, i].set_title("Original", fontsize=9)
            axes[0, i].axis('off')
            axes[1, i].imshow(res["extracted"])
            axes[1, i].set_title("Extraido", fontsize=9)
            axes[1, i].axis('off')
        
        fig.suptitle(
            f"Regioes Extraidas - {cls_name}",
            fontsize=14, fontweight='bold'
        )
        fig.tight_layout()
        save_figure(
            fig,
            f"exp3_regioes_extraidas_{cls}.png",
            subdir="experimento3"
        )
    
    # Figura: Pipeline completo de morfologia
    for cls in sorted(refined_results.keys()):
        cls_name = class_name_map.get(cls, cls)
        res = refined_results[cls][0]
        intermediates = res["intermediates"]
        n_steps = len(intermediates)
        
        fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))
        if n_steps == 1:
            axes = [axes]
        
        for j, (step_name, step_mask) in enumerate(intermediates.items()):
            axes[j].imshow(step_mask, cmap='gray')
            axes[j].set_title(step_name, fontsize=9)
            axes[j].axis('off')
        
        fig.suptitle(
            f"Pipeline Morfologico - {cls_name}",
            fontsize=14, fontweight='bold'
        )
        fig.tight_layout()
        save_figure(
            fig,
            f"exp3_pipeline_morfologico_{cls}.png",
            subdir="experimento3"
        )
    
    # Figura: Grafico de barras com metricas de melhoria
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_keys = ["solidity_improvement", "regions_removed", "holes_filled"]
    metric_titles = ["Melhoria de Solidez", "Regioes Removidas", "Buracos Preenchidos"]
    
    for i, (key, title) in enumerate(zip(metric_keys, metric_titles)):
        data = []
        labels = []
        for cls in sorted(improvements.keys()):
            values = [imp[key] for imp in improvements[cls]]
            data.append(values)
            labels.append(class_name_map.get(cls, cls))
        
        bp = axes[i].boxplot(data, labels=labels, patch_artist=True)
        colors = ['lightcoral', 'lightblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        axes[i].set_title(title, fontsize=10)
        axes[i].tick_params(axis='x', rotation=15)
    
    fig.suptitle(
        "Impacto das Operacoes Morfologicas",
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout()
    save_figure(fig, "exp3_metricas_melhoria.png", subdir="experimento3")
    
    print("\n[EXPERIMENTO 3 CONCLUIDO]")
    return refined_results, improvements
