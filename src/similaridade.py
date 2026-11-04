import os
import cv2
import numpy as np
from src.processamento import processar_imagem_caminho

def calcular_similaridade_bordas(bordas1, bordas2):
    # Calcula a similaridade baseada diretamente nas bordas
    bordas1 = cv2.resize(bordas1, (300, 300))
    bordas2 = cv2.resize(bordas2, (300, 300))
    diferenca = cv2.absdiff(bordas1, bordas2)
    erro = np.mean(diferenca)
    return float(1 - (erro / 255))

def buscar_imagens_semelhantes(bordas_consulta, pasta_dataset="dataset", limite=5):
    # Pesquisa as imagens utilizando o método tradicional com OpenCV
    resultados = []
    extensoes_validas = (".jpg", ".jpeg", ".png")
    for categoria in os.listdir(pasta_dataset):
        caminho_categoria = os.path.join(pasta_dataset, categoria)
        if not os.path.isdir(caminho_categoria):
            continue
        for nome_arquivo in os.listdir(caminho_categoria):
            if not nome_arquivo.lower().endswith(extensoes_validas):
                continue
            caminho_imagem = os.path.join(caminho_categoria, nome_arquivo)
            bordas_dataset = processar_imagem_caminho(caminho_imagem)
            if bordas_dataset is None:
                continue
            similaridade = calcular_similaridade_bordas(bordas_consulta, bordas_dataset)
            resultados.append({"caminho": caminho_imagem, "categoria": categoria, "similaridade": similaridade})
    resultados.sort(key=lambda resultado: resultado["similaridade"], reverse=True)
    return resultados[:limite]