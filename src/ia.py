import os
import pickle
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics.pairwise import cosine_similarity

ARQUIVO_EMBEDDINGS = "embeddings.pkl"

# Carrega a ResNet-18 pré-treinada
pesos = ResNet18_Weights.DEFAULT
transformacao = pesos.transforms()
modelo = resnet18(weights=pesos)
modelo.fc = torch.nn.Identity()
modelo.eval()

def gerar_embedding(imagem):
    # Transforma uma imagem em um vetor de 512 características
    if not isinstance(imagem, Image.Image):
        imagem = Image.fromarray(imagem)
    imagem = imagem.convert("RGB")
    tensor = transformacao(imagem).unsqueeze(0)
    with torch.no_grad():
        embedding = modelo(tensor)
    return embedding.squeeze().numpy()

def calcular_similaridade_ia(embedding1, embedding2):
    # Calcula a similaridade de cosseno entre dois embeddings
    return float(cosine_similarity([embedding1], [embedding2])[0][0])

def gerar_indice_embeddings(pasta_dataset="dataset"):
    # Gera os embeddings de todas as imagens e salva o índice
    indice = []
    extensoes_validas = (".jpg", ".jpeg", ".png")
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
                embedding = gerar_embedding(imagem)
                indice.append({"caminho": caminho_imagem, "categoria": categoria, "embedding": embedding})
            except Exception as erro:
                print(f"Erro ao processar {caminho_imagem}: {erro}")
    with open(ARQUIVO_EMBEDDINGS, "wb") as arquivo:
        pickle.dump(indice, arquivo)
    return len(indice)

def carregar_indice_embeddings():
    # Carrega o índice previamente gerado
    if not os.path.exists(ARQUIVO_EMBEDDINGS):
        return None
    with open(ARQUIVO_EMBEDDINGS, "rb") as arquivo:
        return pickle.load(arquivo)

def buscar_imagens_ia(imagem_consulta, limite=5):
    # Pesquisa imagens utilizando os embeddings previamente indexados
    indice = carregar_indice_embeddings()
    if indice is None:
        return []
    embedding_consulta = gerar_embedding(imagem_consulta)
    resultados = []
    for item in indice:
        similaridade = calcular_similaridade_ia(embedding_consulta, item["embedding"])
        resultados.append({"caminho": item["caminho"], "categoria": item["categoria"], "similaridade": similaridade})
    resultados.sort(key=lambda resultado: resultado["similaridade"], reverse=True)
    return resultados[:limite]