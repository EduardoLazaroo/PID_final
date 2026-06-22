"""
=============================================================================
experimento4_features.py - Experimento 4: Extracao e Analise de Caracteristicas
=============================================================================

Procedimentos:
    a) Extrair pelo menos 4 descritores de Haralick (GLCM)
    b) Extrair assinatura fractal multiescala (box-counting)
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import measure
from scipy import stats as scipy_stats
import config
from utils import save_figure


# Numero maximo de escalas fractais (para padronizar tamanho do vetor)
MAX_FRACTAL_SCALES = len(config.FRACTAL_SCALES)
MAX_LOCAL_DIMS = MAX_FRACTAL_SCALES - 1


def compute_glcm(image_gray, distances, angles, levels=32):
    """Calcula Matriz de Co-ocorrencia de Niveis de Cinza (GLCM)."""
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
    """Extrai descritores de Haralick da GLCM."""
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
    for prop_name in feature_names:
        values = graycoprops(glcm, prop_name)
        features[prop_name] = np.mean(values)
    
    return features


def extract_haralick_from_masked(image_gray, binary_mask, **kwargs):
    """Extrai descritores de Haralick apenas das regioes segmentadas."""
    masked = image_gray.copy()
    masked[binary_mask == 0] = 0
    
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        feature_names = kwargs.get('feature_names', config.HARALICK_FEATURES)
        return {name: 0.0 for name in feature_names}
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = masked[rmin:rmax+1, cmin:cmax+1]
    
    return extract_haralick_features(cropped, **kwargs)


def box_counting(binary_image, scales=None):
    """Calcula box-counting para estimar dimensao fractal."""
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
                if np.any(binary[i:i+scale, j:j+scale]):
                    count += 1
        if count > 0:
            counts.append(count)
            valid_scales.append(scale)
    
    return np.array(valid_scales), np.array(counts, dtype=np.float64)


def compute_fractal_dimension(binary_image, scales=None, min_scales=None):
    """Calcula dimensao fractal via regressao linear log-log."""
    if min_scales is None:
        min_scales = config.FRACTAL_MIN_SCALES
    
    scales_used, counts = box_counting(binary_image, scales)
    
    if len(scales_used) < min_scales:
        return 0.0, 0.0
    
    log_scales = np.log(1.0 / scales_used)
    log_counts = np.log(counts)
    
    slope, intercept, r_value, _, _ = scipy_stats.linregress(log_scales, log_counts)
    return slope, r_value ** 2


def extract_all_features(image_gray, binary_mask, image_rgb=None):
    """
    Extrai todas as caracteristicas de uma imagem.
    Retorna vetor de tamanho fixo.
    """
    features = []
    names = []
    
    # 1. Haralick features (6)
    haralick = extract_haralick_from_masked(image_gray, binary_mask)
    for name in config.HARALICK_FEATURES:
        features.append(haralick.get(name, 0.0))
        names.append(f"haralick_{name}")
    
    # 2. Features de cor RGB (6)
    mask_bool = binary_mask.astype(bool)
    for ch_idx, ch_name in enumerate(["R", "G", "B"]):
        if image_rgb is not None and np.any(mask_bool):
            channel = image_rgb[:, :, ch_idx]
            masked_ch = channel[mask_bool]
            features.append(np.mean(masked_ch))
            features.append(np.std(masked_ch))
        else:
            features.extend([0.0, 0.0])
        names.append(f"color_{ch_name}_mean")
        names.append(f"color_{ch_name}_std")
    
    # 3. Dimensao fractal (2)
    fd, r2 = compute_fractal_dimension(binary_mask)
    features.extend([fd, r2])
    names.extend(["fractal_dimension", "fractal_r_squared"])
    
    # 4. Box-counting em escalas fixas (MAX_FRACTAL_SCALES)
    scales_used, counts = box_counting(binary_mask)
    for s in config.FRACTAL_SCALES:
        if s in scales_used:
            idx = np.where(scales_used == s)[0][0]
            features.append(counts[idx])
        else:
            features.append(0.0)
        names.append(f"fractal_boxcount_scale{s}")
    
    # 5. Dimensoes locais (MAX_LOCAL_DIMS)
    local_dims = []
    for i in range(len(scales_used) - 1):
        if counts[i] > 0 and counts[i+1] > 0:
            d_local = np.log(counts[i] / counts[i+1]) / np.log(scales_used[i+1] / scales_used[i])
            local_dims.append(d_local)
        else:
            local_dims.append(0.0)
    
    for i in range(MAX_LOCAL_DIMS):
        if i < len(local_dims):
            features.append(local_dims[i])
        else:
            features.append(0.0)
        names.append(f"fractal_local_dim_{i}")
    
    # 6. Features geometricas (5)
    labeled = measure.label(binary_mask)
    regions = measure.regionprops(labeled)
    
    if len(regions) > 0:
        areas = [r.area for r in regions]
        features.extend([
            len(regions),
            np.mean(areas),
            np.std(areas),
            np.max(areas),
            np.mean([r.solidity for r in regions if r.area > 10]) if any(r.area > 10 for r in regions) else 0,
        ])
    else:
        features.extend([0.0] * 5)
    names.extend(["geom_n_regions", "geom_mean_area", "geom_std_area",
                  "geom_max_area", "geom_mean_solidity"])
    
    return np.array(features), names


def run_experiment_4(images_rgb, selected_channels, refined_results, class_name_map):
    """Executa o Experimento 4 completo."""
    print("\n" + "=" * 70)
    print("EXPERIMENTO 4 - Extracao e Analise de Caracteristicas")
    print("=" * 70)
    
    print(f"\nParametros GLCM:")
    print(f"  Distancias: {config.GLCM_DISTANCES}")
    print(f"  Angulos: {[f'{np.degrees(a):.0f} graus' for a in config.GLCM_ANGLES]}")
    print(f"  Niveis: {config.GLCM_LEVELS}")
    print(f"  Descritores: {config.HARALICK_FEATURES}")
    print(f"\nParametros Fractais: escalas={config.FRACTAL_SCALES}")
    
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
    if np.any(np.isnan(all_features)) or np.any(np.isinf(all_features)):
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Visualizacoes
    print("\n[GERACAO DE FIGURAS]")
    
    # Assinatura fractal
    for cls in sorted(refined_results.keys()):
        cls_name = class_name_map.get(cls, cls)
        for i in range(min(2, len(refined_results[cls]))):
            mask = refined_results[cls][i]["mask_after"]
            scales_used, counts = box_counting(mask)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.loglog(scales_used, counts, 'bo-', markersize=8, label='Box-counting')
            
            if len(scales_used) >= config.FRACTAL_MIN_SCALES:
                fd, r2 = compute_fractal_dimension(mask)
                log_s = np.log(1.0 / scales_used)
                log_c = np.log(counts)
                slope, intercept, _, _, _ = scipy_stats.linregress(log_s, log_c)
                fit = np.exp(intercept) * (scales_used ** slope)
                ax.loglog(scales_used, fit, 'r--', linewidth=2,
                         label=f'Regressao (D={slope:.3f}, R2={r2:.3f})')
            
            ax.set_xlabel("Escala", fontsize=12)
            ax.set_ylabel("N(escala)", fontsize=12)
            ax.set_title(f"Assinatura Fractal - {cls_name}", fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            save_figure(fig, f"exp4_fractal_{cls}_img{i+1}.png", subdir="experimento4")
    
    # GLCM heatmap
    for cls in sorted(selected_channels.keys()):
        cls_name = class_name_map.get(cls, cls)
        channel = selected_channels[cls][0]
        mask = refined_results[cls][0]["mask_after"]
        
        glcm = compute_glcm(channel * mask, config.GLCM_DISTANCES,
                           config.GLCM_ANGLES, config.GLCM_LEVELS)
        glcm_mean = np.mean(glcm[:, :, :, :], axis=(2, 3))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(np.log1p(glcm_mean), cmap='hot', interpolation='nearest')
        ax.set_xlabel("Nivel j", fontsize=12)
        ax.set_ylabel("Nivel i", fontsize=12)
        ax.set_title(f"GLCM (log) - {cls_name}", fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        save_figure(fig, f"exp4_glcm_{cls}.png", subdir="experimento4")
    
    # Boxplot dimensoes fractais
    fractal_dims = {cls: [] for cls in sorted(refined_results.keys())}
    for cls in sorted(refined_results.keys()):
        for i in range(len(refined_results[cls])):
            mask = refined_results[cls][i]["mask_after"]
            fd, _ = compute_fractal_dimension(mask)
            fractal_dims[cls].append(fd)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [fractal_dims[cls] for cls in sorted(fractal_dims.keys())]
    labels = [class_name_map.get(cls, cls) for cls in sorted(fractal_dims.keys())]
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightcoral', 'lightblue']):
        patch.set_facecolor(color)
    ax.set_ylabel("Dimensao Fractal", fontsize=12)
    ax.set_title("Dimensao Fractal por Classe", fontsize=14, fontweight='bold')
    fig.tight_layout()
    save_figure(fig, "exp4_fractal_boxplot.png", subdir="experimento4")
    
    # Salvar dados para uso posterior (permite executar exp5+ sem refazer exp1-4)
    import os
    features_dir = config.FEATURES_DIR
    np.save(os.path.join(features_dir, "exp4_all_features.npy"), all_features)
    np.save(os.path.join(features_dir, "exp4_all_labels.npy"), all_labels)
    # Salvar nomes das features como texto
    with open(os.path.join(features_dir, "exp4_feature_names.txt"), "w") as f:
        for name in feature_names:
            f.write(name + "\n")
    print(f"  [DADOS SALVOS] {features_dir}")
    print(f"    - exp4_all_features.npy ({all_features.shape})")
    print(f"    - exp4_all_labels.npy ({all_labels.shape})")
    print(f"    - exp4_feature_names.txt ({len(feature_names)} features)")
    
    print("\n[EXPERIMENTO 4 CONCLUIDO]")
    return all_features, all_labels, feature_names
