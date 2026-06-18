"""
=============================================================================
experimento2_segmentacao.py - Experimento 2: Segmentacao de Estruturas
=============================================================================

Procedimentos:
    a) Aplicar segmentacao utilizando Multi-Otsu ou maxima entropia
    b) Avaliar quantitativamente as regioes segmentadas
    
Questao:
    Como textura, ruido e heterogeneidade tecidual influenciaram a segmentacao?

Justificativa do metodo:
    O Multi-Otsu e escolhido porque imagens histopatologicas H&E tipicamente
    apresentam uma distribuicao multimodal de intensidades: fundo branco,
    tecido corado com eosina (rosa) e tecido corado com hematoxilina (roxo).
    O Multi-Otsu com 3 classes captura naturalmente essa estrutura.
    
    A maxima entropia e oferecida como alternativa, sendo robusta quando
    a distribuicao de intensidades nao e claramente bimodal.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu, threshold_li
from skimage import measure
import config
from utils import save_figure


def multi_otsu_segmentation(image_gray, n_classes=3):
    """
    Segmenta imagem usando metodo Multi-Otsu.
    
    Parametros:
        image_gray: imagem em escala de cinza (2D, valores 0-255)
        n_classes: numero de classes de intensidade
    
    Retorna:
        thresholds: limiares calculados
        segmented: imagem segmentada (labels por pixel)
        binary_mask: mascara binaria (tecido vs fundo)
    """
    if image_gray.max() <= 1.0:
        image_uint8 = (image_gray * 255).astype(np.uint8)
    else:
        image_uint8 = image_gray.astype(np.uint8)
    
    thresholds = threshold_multiotsu(image_uint8, classes=n_classes)
    segmented = np.digitize(image_uint8, bins=thresholds)
    binary_mask = (segmented < (n_classes - 1)).astype(np.uint8)
    
    return thresholds, segmented, binary_mask


def max_entropy_segmentation(image_gray):
    """
    Segmenta imagem usando metodo de maxima entropia (Li's method).
    """
    if image_gray.max() <= 1.0:
        image_uint8 = (image_gray * 255).astype(np.uint8)
    else:
        image_uint8 = image_gray.astype(np.uint8)
    
    threshold = threshold_li(image_uint8.astype(np.float64))
    binary_mask = (image_uint8 < threshold).astype(np.uint8)
    segmented = binary_mask.copy()
    
    return threshold, segmented, binary_mask


def evaluate_segmentation(binary_mask, original_gray=None):
    """
    Avalia quantitativamente a segmentacao.
    """
    metrics = {}
    
    # Area segmentada (fracao de pixels de tecido)
    metrics["area_fraction"] = np.sum(binary_mask > 0) / binary_mask.size
    
    # Numero de regioes conectadas
    labeled = measure.label(binary_mask)
    regions = measure.regionprops(labeled)
    metrics["n_regions"] = len(regions)
    
    # Area media das regioes
    if len(regions) > 0:
        areas = [r.area for r in regions]
        metrics["mean_region_area"] = np.mean(areas)
        metrics["std_region_area"] = np.std(areas)
        metrics["max_region_area"] = np.max(areas)
        metrics["min_region_area"] = np.min(areas)
    else:
        metrics["mean_region_area"] = 0
        metrics["std_region_area"] = 0
        metrics["max_region_area"] = 0
        metrics["min_region_area"] = 0
    
    # Perimetro total
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
    metrics["total_perimeter"] = total_perimeter
    metrics["n_contours"] = len(contours)
    
    # Circularidade media
    circularities = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            circularities.append(circularity)
    metrics["mean_circularity"] = np.mean(circularities) if circularities else 0
    
    return metrics


def analyze_texture_impact(image_gray, binary_mask):
    """
    Analisa como textura e heterogeneidade afetam a segmentacao.
    Versao simplificada para evitar problemas de memoria.
    """
    analysis = {}
    
    # Converter para float32 para reduzir memoria
    img_f32 = image_gray.astype(np.float32)
    
    # Calcular variancia local usando kernel menor para reduzir memoria
    kernel_size = 15
    local_mean = cv2.blur(img_f32, (kernel_size, kernel_size))
    local_sq_mean = cv2.blur(img_f32 * img_f32, (kernel_size, kernel_size))
    local_variance = local_sq_mean - local_mean * local_mean
    local_variance = np.maximum(local_variance, 0)
    
    # Criar mascaras
    tissue_mask = binary_mask > 0
    n_tissue = np.count_nonzero(tissue_mask)
    n_total = binary_mask.size
    n_bg = n_total - n_tissue
    
    # Calcular estatisticas para tecido usando indexacao direta
    if n_tissue > 0:
        # Usar np.where para selecionar pixels sem criar sub-array grande
        tissue_var_mean = np.sum(local_variance * tissue_mask) / n_tissue
        analysis["tissue_variance"] = float(tissue_var_mean)
        
        # Calcular media e std do tecido
        tissue_sum = np.sum(img_f32 * tissue_mask)
        tissue_mean = tissue_sum / n_tissue
        # Calcular variancia usando E[X^2] - (E[X])^2
        tissue_sq_sum = np.sum((img_f32 * img_f32) * tissue_mask)
        tissue_var = (tissue_sq_sum / n_tissue) - (tissue_mean * tissue_mean)
        analysis["tissue_std"] = float(np.sqrt(max(tissue_var, 0)))
    else:
        analysis["tissue_variance"] = 0.0
        analysis["tissue_std"] = 0.0
    
    # Calcular estatisticas para fundo
    if n_bg > 0:
        bg_mask = ~tissue_mask
        bg_var_mean = np.sum(local_variance * bg_mask) / n_bg
        analysis["background_variance"] = float(bg_var_mean)
        
        # Calcular media e std do fundo
        bg_sum = np.sum(img_f32 * bg_mask)
        bg_mean = bg_sum / n_bg
        # Calcular variancia usando E[X^2] - (E[X])^2
        bg_sq_sum = np.sum((img_f32 * img_f32) * bg_mask)
        bg_var = (bg_sq_sum / n_bg) - (bg_mean * bg_mean)
        analysis["background_std"] = float(np.sqrt(max(bg_var, 0)))
    else:
        analysis["background_variance"] = 0.0
        analysis["background_std"] = 0.0
    
    if analysis["background_variance"] > 0:
        analysis["variance_ratio"] = (
            analysis["tissue_variance"] / analysis["background_variance"]
        )
    else:
        analysis["variance_ratio"] = float('inf')
    
    return analysis


def run_experiment_2(selected_channels, class_name_map):
    """
    Executa o Experimento 2 completo.
    
    Parametros:
        selected_channels: dict {classe: [lista de canais selecionados]}
        class_name_map: dict {chave_classe: nome_legivel}
    
    Retorna:
        segmentation_results: dict com resultados da segmentacao
        all_metrics: metricas quantitativas
        texture_analysis: analise de textura
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 2 - Segmentacao das Estruturas Histopatologicas")
    print("=" * 70)
    
    segmentation_results = {}
    all_metrics = {}
    texture_analysis = {}
    
    for cls in sorted(selected_channels.keys()):
        cls_name = class_name_map.get(cls, cls)
        print(f"\n--- Classe: {cls_name} ---")
        segmentation_results[cls] = []
        all_metrics[cls] = []
        texture_analysis[cls] = []
        
        for i, channel in enumerate(selected_channels[cls]):
            # a) Aplicar segmentacao
            if config.SEGMENTATION_METHOD == "multi_otsu":
                thresholds, segmented, binary_mask = multi_otsu_segmentation(
                    channel, n_classes=config.MULTI_OTSU_CLASSES
                )
                if i == 0:
                    print(f"  Metodo: Multi-Otsu ({config.MULTI_OTSU_CLASSES} classes)")
                    print(f"  Limiares: {thresholds}")
            elif config.SEGMENTATION_METHOD == "max_entropy":
                threshold, segmented, binary_mask = max_entropy_segmentation(channel)
                if i == 0:
                    print(f"  Metodo: Maxima Entropia (Li)")
                    print(f"  Limiar: {threshold:.2f}")
            else:
                raise ValueError(
                    f"Metodo de segmentacao desconhecido: {config.SEGMENTATION_METHOD}"
                )
            
            # b) Avaliar quantitativamente
            metrics = evaluate_segmentation(binary_mask, channel)
            all_metrics[cls].append(metrics)
            
            # Analise de textura
            tex_analysis = analyze_texture_impact(channel, binary_mask)
            texture_analysis[cls].append(tex_analysis)
            
            segmentation_results[cls].append({
                "segmented": segmented,
                "binary_mask": binary_mask,
                "thresholds": thresholds if config.SEGMENTATION_METHOD == "multi_otsu" else threshold,
            })
        
        # Resumo das metricas
        area_fracs = [m["area_fraction"] for m in all_metrics[cls]]
        n_regions = [m["n_regions"] for m in all_metrics[cls]]
        print(f"  Fracao de area segmentada: {np.mean(area_fracs):.3f} +/- {np.std(area_fracs):.3f}")
        print(f"  Numero de regioes: {np.mean(n_regions):.1f} +/- {np.std(n_regions):.1f}")
    
    # Visualizacoes
    print("\n[GERACAO DE FIGURAS]")
    
    # Figura: Comparacao de segmentacao para amostras de cada classe
    for cls in sorted(selected_channels.keys()):
        cls_name = class_name_map.get(cls, cls)
        n_show = min(3, len(selected_channels[cls]))
        for i in range(n_show):
            channel = selected_channels[cls][i]
            seg = segmentation_results[cls][i]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(channel, cmap='gray')
            axes[0].set_title(f"Canal {config.HSV_CHANNEL_SELECTED}", fontsize=11)
            axes[0].axis('off')
            
            axes[1].imshow(seg["segmented"], cmap=config.SEGMENTATION_CMAP)
            axes[1].set_title("Segmentacao", fontsize=11)
            axes[1].axis('off')
            
            axes[2].imshow(seg["binary_mask"], cmap='gray')
            axes[2].set_title("Mascara Binaria", fontsize=11)
            axes[2].axis('off')
            
            fig.suptitle(
                f"Segmentacao - {cls_name} (Imagem {i+1})",
                fontsize=13, fontweight='bold'
            )
            fig.tight_layout()
            save_figure(
                fig,
                f"exp2_segmentacao_{cls}_img{i+1}.png",
                subdir="experimento2"
            )
    
    # Figura: Distribuicao de intensidades com limiares
    for cls in sorted(selected_channels.keys()):
        cls_name = class_name_map.get(cls, cls)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        sample = selected_channels[cls][0].ravel()
        ax.hist(sample, bins=100, density=True, alpha=0.7, color='steelblue')
        
        seg = segmentation_results[cls][0]
        if config.SEGMENTATION_METHOD == "multi_otsu":
            for t in seg["thresholds"]:
                ax.axvline(t, color='red', linestyle='--', linewidth=2,
                          label=f'Limiar: {t:.1f}')
        else:
            ax.axvline(seg["thresholds"], color='red', linestyle='--', linewidth=2,
                      label=f'Limiar: {seg["thresholds"]:.1f}')
        
        ax.set_xlabel("Intensidade", fontsize=12)
        ax.set_ylabel("Densidade", fontsize=12)
        ax.set_title(
            f"Histograma com Limiares - {cls_name}",
            fontsize=14, fontweight='bold'
        )
        ax.legend(fontsize=10)
        fig.tight_layout()
        save_figure(
            fig,
            f"exp2_histograma_limiares_{cls}.png",
            subdir="experimento2"
        )
    
    # Figura: Boxplot das metricas por classe
    metric_names = ["area_fraction", "n_regions", "mean_region_area", "mean_circularity"]
    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))
    
    for i, metric_name in enumerate(metric_names):
        data = []
        labels = []
        for cls in sorted(all_metrics.keys()):
            values = [m[metric_name] for m in all_metrics[cls]]
            data.append(values)
            labels.append(class_name_map.get(cls, cls))
        
        bp = axes[i].boxplot(data, labels=labels, patch_artist=True)
        colors = ['lightcoral', 'lightblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        axes[i].set_title(metric_name, fontsize=10)
        axes[i].tick_params(axis='x', rotation=15)
    
    fig.suptitle(
        "Metricas de Segmentacao por Classe",
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout()
    save_figure(fig, "exp2_metricas_boxplot.png", subdir="experimento2")
    
    # Imprimir analise de textura
    print("\n[ANALISE DE TEXTURA E HETEROGENEIDADE]")
    for cls in sorted(texture_analysis.keys()):
        cls_name = class_name_map.get(cls, cls)
        tex_vars = [t["tissue_variance"] for t in texture_analysis[cls]]
        bg_vars = [t["background_variance"] for t in texture_analysis[cls]]
        ratios = [t["variance_ratio"] for t in texture_analysis[cls]]
        print(f"  {cls_name}:")
        print(f"    Variancia tecidual: {np.mean(tex_vars):.2f}")
        print(f"    Variancia fundo: {np.mean(bg_vars):.2f}")
        print(f"    Razao de variancia: {np.mean(ratios):.2f}")
    
    print("\n[EXPERIMENTO 2 CONCLUIDO]")
    return segmentation_results, all_metrics, texture_analysis
