import os
import pickle
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity

MODELO_CLIP = "openai/clip-vit-base-patch32"
ARQUIVO_EMBEDDINGS_CLIP = "embeddings_clip.pkl"
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

modelo_clip = CLIPModel.from_pretrained(MODELO_CLIP).to(dispositivo)
processador_clip = CLIPProcessor.from_pretrained(MODELO_CLIP)
modelo_clip.eval()

def normalizar_embedding(embedding):
    # Normaliza o vetor para facilitar a comparação por cosseno
    norma = np.linalg.norm(embedding)
    if norma == 0:
        return embedding
    return embedding / norma

def extrair_tensor_embedding(resultado):
    # Extrai o Tensor independentemente do formato retornado pelo Transformers
    if isinstance(resultado, torch.Tensor):
        return resultado
    if hasattr(resultado, "pooler_output") and resultado.pooler_output is not None:
        return resultado.pooler_output
    if hasattr(resultado, "last_hidden_state") and resultado.last_hidden_state is not None:
        return resultado.last_hidden_state[:, 0]
    raise TypeError(f"Formato de embedding não suportado: {type(resultado)}")

def gerar_embedding_imagem_clip(imagem):
    # Transforma uma imagem em um embedding CLIP
    if not isinstance(imagem, Image.Image):
        imagem = Image.fromarray(imagem)
    imagem = imagem.convert("RGB")
    entradas = processador_clip(images=imagem, return_tensors="pt")
    entradas = {chave: valor.to(dispositivo) for chave, valor in entradas.items()}
    with torch.no_grad():
        resultado = modelo_clip.get_image_features(**entradas)
    embedding = extrair_tensor_embedding(resultado)
    embedding = embedding.squeeze().cpu().numpy()
    return normalizar_embedding(embedding)

def gerar_embedding_texto_clip(texto):
    # Transforma um texto em um embedding CLIP
    entradas = processador_clip(text=[texto], return_tensors="pt", padding=True, truncation=True)
    entradas = {chave: valor.to(dispositivo) for chave, valor in entradas.items()}
    with torch.no_grad():
        resultado = modelo_clip.get_text_features(**entradas)
    embedding = extrair_tensor_embedding(resultado)
    embedding = embedding.squeeze().cpu().numpy()
    return normalizar_embedding(embedding)

def calcular_similaridade_clip(embedding1, embedding2):
    # Calcula a similaridade de cosseno entre dois embeddings
    return float(cosine_similarity([embedding1], [embedding2])[0][0])

def gerar_indice_clip(pasta_dataset="dataset"):
    # Gera e salva os embeddings CLIP de todas as imagens do dataset
    indice = []
    extensoes_validas = (".jpg", ".jpeg", ".png")
    if not os.path.exists(pasta_dataset):
        return 0
    for categoria in os.listdir(pasta_dataset):
        caminho_categoria = os.path.join(pasta_dataset, categoria)
        if not os.path.isdir(caminho_categoria):
            continue
        for nome_arquivo in os.listdir(caminho_categoria):
            if not nome_arquivo.lower().endswith(extensoes_validas):
                continue
            caminho_imagem = os.path.join(caminho_categoria, nome_arquivo)
            try:
                imagem = Image.open(caminho_imagem).convert("RGB")
                embedding = gerar_embedding_imagem_clip(imagem)
                indice.append({"caminho": caminho_imagem, "categoria": categoria, "embedding": embedding})
            except Exception as erro:
                print(f"Erro ao processar {caminho_imagem} com CLIP: {erro}")
    with open(ARQUIVO_EMBEDDINGS_CLIP, "wb") as arquivo:
        pickle.dump(indice, arquivo)
    return len(indice)

def carregar_indice_clip():
    # Carrega o índice CLIP salvo anteriormente
    if not os.path.exists(ARQUIVO_EMBEDDINGS_CLIP):
        return None
    with open(ARQUIVO_EMBEDDINGS_CLIP, "rb") as arquivo:
        return pickle.load(arquivo)

def buscar_por_embedding_clip(embedding_consulta, limite=5):
    # Compara o embedding da consulta com as imagens indexadas
    indice = carregar_indice_clip()
    if indice is None:
        return []
    resultados = []
    for item in indice:
        similaridade = calcular_similaridade_clip(embedding_consulta, item["embedding"])
        resultados.append({"caminho": item["caminho"], "categoria": item["categoria"], "similaridade": similaridade})
    resultados.sort(key=lambda resultado: resultado["similaridade"], reverse=True)
    return resultados[:limite]

def buscar_imagens_clip(imagem_consulta, limite=5):
    # Pesquisa utilizando uma imagem ou desenho
    embedding = gerar_embedding_imagem_clip(imagem_consulta)
    return buscar_por_embedding_clip(embedding, limite)

def buscar_texto_clip(texto, limite=5):
    # Pesquisa utilizando uma descrição textual
    embedding = gerar_embedding_texto_clip(texto)
    return buscar_por_embedding_clip(embedding, limite)
