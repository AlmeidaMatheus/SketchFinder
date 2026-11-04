import cv2
import numpy as np
from PIL import Image

def carregar_imagem(arquivo):
    # Converte o arquivo enviado pelo Streamlit para uma matriz NumPy
    imagem_pil = Image.open(arquivo).convert("RGB")
    return np.array(imagem_pil)

def converter_para_cinza(imagem):
    # Converte uma imagem RGB para escala de cinza
    return cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

def detectar_bordas(imagem_cinza):
    # Detecta bordas usando o algoritmo Canny
    return cv2.Canny(imagem_cinza, 100, 200)

def redimensionar_imagem(imagem, largura=300, altura=300):
    # Redimensiona todas as imagens para o mesmo tamanho
    return cv2.resize(imagem, (largura, altura))

def processar_imagem(arquivo):
    # Executa o processamento utilizado na visualização e comparação tradicional
    original = carregar_imagem(arquivo)
    redimensionada = redimensionar_imagem(original)
    cinza = converter_para_cinza(redimensionada)
    bordas = detectar_bordas(cinza)
    return original, redimensionada, cinza, bordas

def processar_imagem_caminho(caminho):
    # Processa uma imagem existente no dataset
    imagem = cv2.imread(caminho)
    if imagem is None:
        return None
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    redimensionada = redimensionar_imagem(imagem)
    cinza = converter_para_cinza(redimensionada)
    return detectar_bordas(cinza)