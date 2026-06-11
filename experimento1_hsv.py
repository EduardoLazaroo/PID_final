"""
=============================================================================
experimento1_hsv.py - Experimento 1: Representacao HSV e Analise dos Canais
=============================================================================

Procedimentos:
    a) Converter imagens RGB para HSV
    b) Observar separadamente os canais H, S e V
    c) Selecionar um canal para processamento posterior
    d) Justificar criterios de escolha

Justificativa da escolha do canal:
    Em imagens histopatologicas coradas com Hematoxilina e Eosina (H&E),
    a saturacao (S) do modelo HSV e particularmente informativa porque:
    
    1. O canal H (Hue) captura a matiz, mas e sensivel a variacoes de
       iluminacao e calibracao do scanner.
    2. O canal V (Value) captura luminancia, mas confunde intensidade
       com informacao cromatica.
    3. O canal S (Saturation) mede a "pureza" da cor: regioes com tecido
       corado apresentam alta saturacao (rosa/roxo), enquanto regioes de
       fundo (lamina branca) apresentam saturacao proxima de zero.
    
    Portanto, o canal S oferece a melhor separacao entre tecido e fundo,
    sendo ideal para segmentacao subsequente.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import config
from utils import save_figure, plot_channel_analysis


def rgb_to_hsv(image_rgb):
    """
    Converte imagem RGB para HSV usando OpenCV.
    
    Parametros:
        image_rgb: array numpy (H, W, 3) em RGB, valores 0-255
    
    Retorna:
        hsv: array numpy (H, W, 3) em HSV
             H: 0-179 (OpenCV escala), S: 0-255, V: 0-255
        h, s, v: canais individuais como arrays 2D
    """
    # OpenCV espera BGR, entao convertemos RGB -> BGR -> HSV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    h = hsv[:, :, 0].astype(np.float64)
    s = hsv[:, :, 1].astype(np.float64)
    v = hsv[:, :, 2].astype(np.float64)
    
    return hsv, h, s, v


def analyze_channel_statistics(h, s, v):
    """
    Calcula estatisticas descritivas de cada canal HSV.
    
    Retorna:
        stats: dict com estatisticas por canal
    """
    channels = {"H": h, "S": s, "V": v}
    stats = {}
    
    for name, ch in channels.items():
        stats[name] = {
            "min": np.min(ch),
            "max": np.max(ch),
            "mean": np.mean(ch),
            "std": np.std(ch),
            "median": np.median(ch),
            "q25": np.percentile(ch, 25),
            "q75": np.percentile(ch, 75),
        }
    
    return stats


def select_channel(h, s, v, method="auto"):
    """
    Seleciona o canal HSV mais adequado para processamento.
    
    Parametros:
        h, s, v: canais HSV
        method: "auto" (baseado em criterio) ou nome do canal ("H", "S", "V")
    
    Retorna:
        selected: array 2D do canal selecionado
        channel_name: nome do canal selecionado
        justification: texto de justificativa
    """
    if method == "auto":
        # Criterio: coeficiente de variacao (CV = std/mean)
        # O canal com maior CV tende a ter melhor separabilidade
        channels = {"H": h, "S": s, "V": v}
        cv_values = {}
        
        for name, ch in channels.items():
            mean_val = np.mean(ch)
            if mean_val > 0:
                cv_values[name] = np.std(ch) / mean_val
            else:
                cv_values[name] = 0
        
        # Para imagens H&E, o canal S e geralmente o mais discriminativo
        # Usamos o CV como metrica, mas damos preferencia ao S se proximo
        selected_name = max(cv_values, key=cv_values.get)
        
        # Se S tem CV proximo do maximo (dentro de 20%), preferir S
        max_cv = cv_values[selected_name]
        if cv_values["S"] >= 0.8 * max_cv:
            selected_name = "S"
        
        channel_name = selected_name
        selected = channels[channel_name]
        
    elif method in ("H", "S", "V"):
        channel_name = method
        channels = {"H": h, "S": s, "V": v}
        selected = channels[channel_name]
    else:
        raise ValueError(f"Metodo de selecao invalido: {method}")
    
    justification = (
        f"Canal '{channel_name}' selecionado.\n"
        f"Justificativa: Em imagens histopatologicas H&E, o canal de "
        f"Saturacao (S) distingue eficientemente regioes coradas (tecido) "
        f"de fundo (lamina), pois tecido apresenta alta saturacao enquanto "
        f"o fundo branco tem saturacao proxima de zero."
    )
    
    return selected, channel_name, justification


def run_experiment_1(sample_images, class_name_map):
    """
    Executa o Experimento 1 completo.
    
    Parametros:
        sample_images: dict {classe: [lista de imagens RGB]}
        class_name_map: dict {chave_classe: nome_legivel}
    
    Retorna:
        selected_channels: dict {classe: [lista de canais selecionados]}
        all_stats: estatisticas dos canais
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTO 1 - Representacao HSV e Analise dos Canais")
    print("=" * 70)
    
    selected_channels = {}
    all_stats = {}
    
    for cls in sorted(sample_images.keys()):
        cls_name = class_name_map.get(cls, cls)
        print(f"\n--- Classe: {cls_name} ---")
        selected_channels[cls] = []
        cls_stats = {"H": [], "S": [], "V": []}
        
        for img_rgb in sample_images[cls]:
            # a) Converter RGB para HSV
            hsv, h, s, v = rgb_to_hsv(img_rgb)
            
            # b) Analisar estatisticas dos canais
            stats = analyze_channel_statistics(h, s, v)
            for ch_name in ["H", "S", "V"]:
                cls_stats[ch_name].append(stats[ch_name])
            
            # c) Selecionar canal
            selected, ch_name, justification = select_channel(
                h, s, v, method=config.HSV_CHANNEL_SELECTED
            )
            selected_channels[cls].append(selected)
        
        # Agregar estatisticas da classe
        all_stats[cls] = {}
        for ch_name in ["H", "S", "V"]:
            means = [s["mean"] for s in cls_stats[ch_name]]
            stds = [s["std"] for s in cls_stats[ch_name]]
            all_stats[cls][ch_name] = {
                "mean_of_means": np.mean(means),
                "mean_of_stds": np.mean(stds),
            }
            print(f"  Canal {ch_name}: media={np.mean(means):.2f}, "
                  f"desvio={np.mean(stds):.2f}")
        
        print(f"\n  Canal selecionado: {config.HSV_CHANNEL_SELECTED}")
    
    # Visualizacoes
    print("\n[GERACAO DE FIGURAS]")
    
    # Figura 1: Analise dos canais HSV para uma imagem exemplo de cada classe
    for cls in sorted(sample_images.keys()):
        img_rgb = sample_images[cls][0]
        _, h, s, v = rgb_to_hsv(img_rgb)
        
        fig = plot_channel_analysis(
            [h, s, v],
            ["H (Matiz)", "S (Saturacao)", "V (Valor)"],
            selected_idx="HSV".index(config.HSV_CHANNEL_SELECTED)
        )
        save_figure(
            fig,
            f"exp1_canais_hsv_{cls}.png",
            subdir="experimento1"
        )
    
    # Figura 2: Histogramas dos canais para cada classe
    for cls in sorted(sample_images.keys()):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        n_samples = min(10, len(sample_images[cls]))
        
        for i, ch_name in enumerate(["H", "S", "V"]):
            for j in range(n_samples):
                _, h, s, v = rgb_to_hsv(sample_images[cls][j])
                channels = {"H": h, "S": s, "V": v}
                axes[i].hist(
                    channels[ch_name].ravel(), bins=50,
                    alpha=0.3, density=True
                )
            axes[i].set_title(f"Canal {ch_name}", fontsize=12)
            axes[i].set_xlabel("Valor")
            axes[i].set_ylabel("Densidade")
        
        cls_name = class_name_map.get(cls, cls)
        fig.suptitle(
            f"Histogramas dos Canais HSV - {cls_name}",
            fontsize=14, fontweight='bold'
        )
        fig.tight_layout()
        save_figure(
            fig,
            f"exp1_histogramas_hsv_{cls}.png",
            subdir="experimento1"
        )
    
    # Figura 3: Canal selecionado para amostras de cada classe
    for cls in sorted(sample_images.keys()):
        n_show = min(4, len(sample_images[cls]))
        images_show = [sample_images[cls][i] for i in range(n_show)]
        channels_show = [selected_channels[cls][i] for i in range(n_show)]
        
        fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
        for i in range(n_show):
            axes[0, i].imshow(images_show[i])
            axes[0, i].set_title("Original RGB", fontsize=9)
            axes[0, i].axis('off')
            axes[1, i].imshow(channels_show[i], cmap='gray')
            axes[1, i].set_title(f"Canal {config.HSV_CHANNEL_SELECTED}", fontsize=9)
            axes[1, i].axis('off')
        
        cls_name = class_name_map.get(cls, cls)
        fig.suptitle(
            f"Canal Selecionado ({config.HSV_CHANNEL_SELECTED}) - {cls_name}",
            fontsize=14, fontweight='bold'
        )
        fig.tight_layout()
        save_figure(
            fig,
            f"exp1_canal_selecionado_{cls}.png",
            subdir="experimento1"
        )
    
    print("\n[EXPERIMENTO 1 CONCLUIDO]")
    return selected_channels, all_stats
