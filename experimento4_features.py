"""
=============================================================================
experimento4_features.py - Experimento 4: Extracao e Analise de Caracteristicas
=============================================================================

Procedimentos:
    a) Extrair pelo menos 4 descritores de Haralick (GLCM)
       - Parametros GLCM: distancias e angulos justificados
    b) Extrair assinatura fractal multiescala (box-counting)
       - Regressao linear no espaco log-log

Justificativa dos parametros GLCM:
    - Distancias [1, 3, 5]: Capturam dependencias texturais em multiplas
      escalas. d=1 captura textura fina (adjacencia direta), d=3 e d=5
      capturam padroes de maior alcance espacial.
    - Angulos [0, 45, 90, 135]: Analise isotropica que captura
      texturas independentes de orientacao.
    - Niveis de quantizacao (32): Reduz ruido de quantizacao mantendo
      resolucao suficiente para discriminar padroes texturais.

Justificativa do box-counting:
    - Escalas [2, 4, 8, 16, 32, 64]: Cobrem faixa de resolucao que vai
      de detalhes celulares finos ate estruturas teciduais maiores.
    - A dimensao fractal quantifica a complexidade estrutural.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import measure
from scipy import stats as scipy_stats
import config
from utils import save_figure


# =============================================================================
# HARALICK FEATURES (GLCM)
# =============================================================================

def compute_glcm(image_gray, distances, angles, levels=32):
    """
    Calcula Matriz de Co-ocorrencia de Niveis de Cinza (GLCM).
    """
    if image_gray.max() > levels - 1:
        image_quantized = (image_gray / image_gray.max() * (levels - 1)).astype(np.uint8)
    else:
        image_quantized = image_gray.astype(np.uint8)
    
    image_quantized = np.clip(image_quantized, 0, levels - 1)
    
    glcm = graycomatrix(
        image_quantized,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )
    
    return glcm


def extract_haralick_features(image_gray, distances=None, angles=None,
                               levels=None, feature_names=None):
    """
    Extrai descritores de Haralick da GLCM.
    """
    if distances is None:
        distances = config.GLCM_DISTANCES
    if angles is None:
        angles = config.GLCM_ANGLES
    if levels is None:
        levels = config.GLCM_LEVELS
    if feature_names is None:
        feature_names = config.HARALICK_FEATURES
    
    glcm = compute_glcm(image_gray, distances, angles, levels)
    
    features = {}
    features_per_angle = {}
    
    for prop_name in feature_names:
        values = graycoprops(glcm, prop_name)
        features[prop_name] = np.mean(values)
        features_per_angle[prop_name] = np.mean(values, axis=0)
    
    return features, features_per_angle


def extract_haralick_from_masked(image_gray, binary_mask, **kwargs):
    """
    Extrai descritores de Haralick apenas das regioes segmentadas.
    """
    masked = image_gray.copy()
    masked[binary_mask == 0] = 0
    
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        feature_names = kwargs.get('feature_names', config.HARALICK_FEATURES)
        return {name: 0.0 for name in feature_names}, {}
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    cropped = masked[rmin:rmax+1, cmin:cmax+1]
    cropped_mask = binary_mask[rmin:rmax+1, cmin:cmax+1]
    cropped_region = cropped.copy()
    cropped_region[cropped_mask == 0] = 0
    
    return extract_haralick_features(cropped_region, **kwargs)


# =============================================================================
# DIMENSAO FRACTAL (BOX-COUNTING)
# =============================================================================

def box_counting(binary_image, scales=None):
    """
    Calcula box-counting para estimar dimensao fractal.
    """
    if scales is None:
        scales = config.FRACTAL_SCALES
    
    binary = binary_image.astype(bool)
    counts = []
    valid_scales = []
    
    for scale in scales:
        if scale > min(binary.shape):
            continue
        
        count = 0
        for i in range(0, binary.shape[0], scale):
            for j in range(0, binary.shape[1], scale):
                block = binary[i:i+scale, j:j+scale]
                if np.any(block):
                    count += 1
        
        if count > 0:
            counts.append(count)
            valid_scales.append(scale)
    
    return np.array(valid_scales), np.array(counts, dtype=np.float64)


def compute_fractal_dimension(binary_image, scales=None, min_scales=None):
    """
    Calcula dimensao fractal via regressao linear no espaco log-log.
    """
    if min_scales is None:
        min_scales = config.FRACTAL_MIN_SCALES
    
    scales_used, counts = box_counting(binary_image, scales)
    
    if len(scales_used) < min_scales:
        return 0.0, 0.0, scales_used, counts, None
    
    log_scales = np.log(1.0 / scales_used)
    log_counts = np.log(counts)
    
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        log_scales, log_counts
    )
    
    fractal_dim = slope
    r_squared = r_value ** 2
    fit_values = np.exp(intercept) * (scales_used ** slope)
    
    return fractal_dim, r_squared, scales_used, counts, fit_values


def compute_multiscale_fractal_signature(binary_image, scales=None):
    """
    Calcula assinatura fractal multiescala completa.
    """
    if scales is None:
        scales = config.FRACTAL_SCALES
    
    scales_used, counts = box_counting(binary_image, scales)
    
    local_dimensions = []
    for i in range(len(scales_used) - 1):
        if counts[i] > 0 and counts[i+1] > 0:
            d_local = np.log(counts[i] / counts[i+1]) / np.log(scales_used[i+1] / scales_used[i])
            local_dimensions.append(d_local)
        else:
            local_dimensions.append(0)
    
    fractal_dim, r_squared, _, _, _ = compute_fractal_dimension(binary_image, scales)
    
    signature = {
        "scales": scales_used,
        "counts": counts,
        "local_dimensions": np.array(local_dimensions),
        "global_dimension": fractal_dim,
        "r_squared": r_squared,
    }
    
    return signature


# =============================================================================
# EXTRACAO COMBINADA
# =============================================================================

def extract_all_features(image_gray, binary_mask, image_rgb=None):
    """
    Extrai todas as caracteristicas de uma imagem.
    """
    feature_dict = {}
    
    # Haralick features
    haralick_features, _ = extract_haralick_from_masked(image_gray, binary_mask)
    for name, value in haralick_features.items():
        feature_dict[f"haralick_{name}"] = value
    
    # Features de cor
    if image_rgb is not None:
        mask_bool = binary_mask.astype(bool)
        for ch_idx, ch_name in enumerate(["R", "G", "B"]):
            channel = image_rgb[:, :, ch_idx]
            masked_channel = channel[mask_bool]
            
            if len(masked_channel) > 0:
                feature_dict[f"color_{ch_name}_mean"] = np.mean(masked_channel)
                feature_dict[f"color_{ch_name}_std"] = np.std(masked_channel)
            else:
                feature_dict[f"color_{ch_name}_mean"] = 0
                feature_dict[f"color_{ch_name}_std"] = 0
    
    # Dimensao fractal
    fractal_dim, r_squared, _, _, _ = compute_fractal_dimension(binary_mask)
    feature_dict["fractal_dimension"] = fractal_dim
    feature_dict["fractal_r_squared"] = r_squared
    
    # Assinatura fractal multiescala
    signature = compute_multiscale_fractal_signature(binary_mask)
    for s, c in zip(signature["scales"], signature["counts"]):
        feature_dict[f"fractal_boxcount_scale{s}"] = c
    for i, d in enumerate(signature["local_dimensions"]):
        feature_dict[f"fractal_local_dim_{i}"] = d
    
    # Features geometricas
    labeled = measure.label(binary_mask)
    regions = measure.regionprops(labeled)
    
    if len(regions) > 0:
        areas = [r.area for r in regions]
        feature_dict["geom_n_regions"] = len(regions)
        feature_dict["geom_mean_area"] = np.mean(areas)
        feature_dict["geom_std_area"] = np.std(areas)
        feature_dict["geom_max_area"] = np.max(areas)
        
        solidities = [r.solidity for r in regions if r.area > 10]
        feature_dict["geom_mean_solidity"] = np.mean(solidities) if solidities else 0
        
        eccentricities = [r.eccentricity for r in regions if r.area > 10]
        feature_dict["geom_mean_eccentricity"] = np.mean(eccentricities) if eccentricities else 0
    else:
        feature_dict["geom_n_regions"] = 0
        feature_dict["geom_mean_area"] = 0
        feature_dict["geom_std_area"] = 0
        feature_dict["geom_max_area"] = 0
        feature_dict["geom_mean_solidity"] = 0
        feature_dict["geom_mean_eccentricity"] = 0
    
    feature_names = list(feature_dict.keys())
    feature_vector = np.array([feature_dict[name] for name in feature_names])
    
    return feature_vector, feature_names


def run_experiment_4(images_rgb, selected_channels, refined_results, class_name_map):
    """
    Executa o Experimento 4 completo.
    
    Parametros:
        images_rgb: dict {classe: [lista de imagens RGB]}
        selected_channels: dict {classe: [lista de canais HSV]}
        refined_results: resultados do Experimento 3
        class_name_map: dict {chave_classe: nome_legivel}
    
    Retorna:
        all_features: matriz (n_samples, n_features)
        all_labels: vetor (n_samples,)
        feature_names: lista com nomes das features
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 4 - Extracao e Analise de Caracteristicas")
    print("=" * 70)
    
    print(f"\nParametros GLCM:")
    print(f"  Distancias: {config.GLCM_DISTANCES}")
    print(f"  Angulos: {[f'{a:.4f} rad ({np.degrees(a):.0f} graus)' for a in config.GLCM_ANGLES]}")
    print(f"  Niveis de quantizacao: {config.GLCM_LEVELS}")
    print(f"  Descritores: {config.HARALICK_FEATURES}")
    
    print(f"\nParametros Fractais:")
    print(f"  Escalas (box-counting): {config.FRACTAL_SCALES}")
    
    all_features_list = []
    all_labels = []
    feature_names = None
    
    for cls in sorted(images_rgb.keys()):
        cls_name = class_name_map.get(cls, cls)
        print(f"\n--- Classe: {cls_name} ---")
        
        for i in range(len(images_rgb[cls])):
            img_rgb = images_rgb[cls][i]
            channel = selected_channels[cls][i]
            mask = refined_results[cls][i]["mask_after"]
            
            feature_vec, fnames = extract_all_features(channel, mask, img_rgb)
            
            if feature_names is None:
                feature_names = fnames
            
            all_features_list.append(feature_vec)
            all_labels.append(cls)
    
    all_features = np.array(all_features_list)
    all_labels = np.array(all_labels)
    
    print(f"\nTotal de amostras: {len(all_labels)}")
    print(f"Total de features: {len(feature_names)}")
    
    # Tratar valores invalidos
    nan_count = np.sum(np.isnan(all_features))
    inf_count = np.sum(np.isinf(all_features))
    if nan_count > 0 or inf_count > 0:
        print(f"\n[AVISO] Valores invalidos: {nan_count} NaN, {inf_count} Inf")
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Visualizacoes
    print("\n[GERACAO DE FIGURAS]")
    
    # Assinatura fractal
    for cls in sorted(refined_results.keys()):
        cls_name = class_name_map.get(cls, cls)
        n_show = min(3, len(refined_results[cls]))
        for i in range(n_show):
            mask = refined_results[cls][i]["mask_after"]
            signature = compute_multiscale_fractal_signature(mask)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].loglog(
                signature["scales"], signature["counts"],
                'bo-', markersize=8, label='Box-counting'
            )
            
            if len(signature["scales"]) >= config.FRACTAL_MIN_SCALES:
                log_scales = np.log(1.0 / signature["scales"])
                log_counts = np.log(signature["counts"])
                slope, intercept, r_value, _, _ = scipy_stats.linregress(
                    log_scales, log_counts
                )
                fit_vals = np.exp(intercept) * (signature["scales"] ** slope)
                axes[0].loglog(
                    signature["scales"], fit_vals,
                    'r--', linewidth=2,
                    label=f'Regressao (D={slope:.3f}, R2={r_value**2:.3f})'
                )
            
            axes[0].set_xlabel("Escala (epsilon)", fontsize=12)
            axes[0].set_ylabel("N(epsilon)", fontsize=12)
            axes[0].set_title("Box-Counting Log-Log", fontsize=13, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)
            
            if len(signature["local_dimensions"]) > 0:
                x_pos = range(len(signature["local_dimensions"]))
                axes[1].bar(x_pos, signature["local_dimensions"], color='steelblue')
                axes[1].set_xlabel("Par de Escalas", fontsize=12)
                axes[1].set_ylabel("Dimensao Local", fontsize=12)
                axes[1].set_title("Dimensao Fractal Multiescala", fontsize=13, fontweight='bold')
                axes[1].set_xticks(x_pos)
                axes[1].set_xticklabels(
                    [f"{signature['scales'][i]}-{signature['scales'][i+1]}"
                     for i in range(len(signature["local_dimensions"]))],
                    fontsize=9
                )
            
            fig.suptitle(
                f"Assinatura Fractal - {cls_name} (Imagem {i+1})",
                fontsize=14, fontweight='bold'
            )
            fig.tight_layout()
            save_figure(
                fig,
                f"exp4_fractal_{cls}_img{i+1}.png",
                subdir="experimento4"
            )
    
    # GLCM heatmap
    for cls in sorted(selected_channels.keys()):
        cls_name = class_name_map.get(cls, cls)
        channel = selected_channels[cls][0]
        mask = refined_results[cls][0]["mask_after"]
        
        glcm = compute_glcm(
            channel * mask,
            config.GLCM_DISTANCES,
            config.GLCM_ANGLES,
            config.GLCM_LEVELS
        )
        
        glcm_mean = np.mean(glcm[:, :, :, :], axis=(2, 3))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(np.log1p(glcm_mean), cmap='hot', interpolation='nearest')
        ax.set_xlabel("Nivel de Cinza j", fontsize=12)
        ax.set_ylabel("Nivel de Cinza i", fontsize=12)
        ax.set_title(
            f"GLCM (log) - {cls_name}",
            fontsize=14, fontweight='bold'
        )
        plt.colorbar(im, ax=ax, label="log(1 + frequencia)")
        fig.tight_layout()
        save_figure(
            fig,
            f"exp4_glcm_{cls}.png",
            subdir="experimento4"
        )
    
    # Boxplot dimensoes fractais
    fractal_dims = {cls: [] for cls in sorted(refined_results.keys())}
    for cls in sorted(refined_results.keys()):
        for i in range(len(refined_results[cls])):
            mask = refined_results[cls][i]["mask_after"]
            fd, _, _, _, _ = compute_fractal_dimension(mask)
            fractal_dims[cls].append(fd)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [fractal_dims[cls] for cls in sorted(fractal_dims.keys())]
    labels = [class_name_map.get(cls, cls) for cls in sorted(fractal_dims.keys())]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ['lightcoral', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Dimensao Fractal", fontsize=12)
    ax.set_title("Dimensao Fractal por Classe", fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, "exp4_fractal_boxplot.png", subdir="experimento4")
    
    print("\n[EXPERIMENTO 4 CONCLUIDO]")
    return all_features, all_labels, feature_names
