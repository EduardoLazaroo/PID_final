# Apresentacao profissional

## Identidade e valor

PID_final e um pipeline academico de classificacao de imagens histopatologicas. Organizado em experimentos, cobre espaco HSV, segmentacao Multi-Otsu e maxima entropia, morfologia, descritores de textura/cor/geometria, selecao RELIEF e classificacao.

O trabalho e direcionado a estudantes e pesquisadores de PDI, visao computacional e analise de imagens histopatologicas.

## Texto para publicacao

> Este projeto organiza um fluxo completo de Processamento Digital de Imagens aplicado a imagens histopatologicas. Em vez de tratar a classificacao como uma etapa isolada, o pipeline explicita segmentacao, morfologia, descritores de textura, cor e geometria, selecao de atributos e avaliacao de classificadores. O desafio foi estruturar os experimentos de forma modular, permitindo executar etapas ou classes especificas e comparar seus efeitos. Os resultados incluidos devem ser lidos junto ao recorte do dataset LC25000, que precisa ser disponibilizado separadamente para reproducao.

## Evidencias visuais recomendadas

1. Sequencia HSV, mascara e operacoes morfologicas (`evidencias/exp2_segmentacao_colon_aca_img1.png`, `evidencias/exp3_pipeline_morfologico_colon_aca.png`).
2. Scores e selecao RELIEF (`evidencias/exp5_relief_scores.png`).
3. Matriz de confusao e comparativo dos classificadores (`evidencias/exp6_comparacao_classificadores.png`, `evidencias/exp6_matriz_confusao.png`).
4. Curvas ROC e resumo integrado dos resultados (`evidencias/exp6_curvas_roc.png`, `evidencias/discussao_final_resumo.png`).
5. Relatorio Executivo Word diagramado (`PID_final_Relatorio_Publicacao.docx`).

## Prompt para imagem complementar

```text
Diagrama academico de processamento de imagem histopatologica: lamina microscopica passa por conversao HSV, segmentacao, morfologia, extracao de textura e cor, selecao RELIEF e classificacao; incluir matriz de confusao estilizada, paleta clinica azul e magenta, fundo claro, sem diagnosticos e sem logotipos.
```
