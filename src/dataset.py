import os
import shutil
from PIL import Image
from torchvision.datasets import CIFAR100

PASTA_DATASET = "dataset"
PASTA_CACHE = ".dataset_cache"
ARQUIVO_RESNET = "embeddings.pkl"
ARQUIVO_CLIP = "embeddings_clip.pkl"

CATEGORIAS_DISPONIVEIS = [
    "bicycle",
    "bus",
    "motorcycle",
    "pickup_truck",
    "train",
    "house",
    "lion",
    "rocket"
]

def excluir_arquivo(caminho):
    # Exclui um arquivo caso exista
    if os.path.exists(caminho):
        os.remove(caminho)

def limpar_indices():
    # Remove os índices gerados anteriormente
    excluir_arquivo(ARQUIVO_RESNET)
    excluir_arquivo(ARQUIVO_CLIP)

def excluir_dataset():
    # Exclui dataset, cache e índices
    if os.path.exists(PASTA_DATASET):
        shutil.rmtree(PASTA_DATASET)
    if os.path.exists(PASTA_CACHE):
        shutil.rmtree(PASTA_CACHE)
    limpar_indices()
    os.makedirs(PASTA_DATASET, exist_ok=True)

def contar_imagens_dataset():
    # Conta as imagens existentes no dataset
    total = 0
    if not os.path.exists(PASTA_DATASET):
        return total
    for raiz, pastas, arquivos in os.walk(PASTA_DATASET):
        total += sum(1 for arquivo in arquivos if arquivo.lower().endswith((".jpg", ".jpeg", ".png")))
    return total

def contar_por_categoria():
    # Conta as imagens de cada categoria
    quantidades = {}
    if not os.path.exists(PASTA_DATASET):
        return quantidades
    for categoria in os.listdir(PASTA_DATASET):
        caminho = os.path.join(PASTA_DATASET, categoria)
        if not os.path.isdir(caminho):
            continue
        quantidades[categoria] = sum(1 for arquivo in os.listdir(caminho) if arquivo.lower().endswith((".jpg", ".jpeg", ".png")))
    return quantidades

def preparar_dataset(categorias, quantidade_por_categoria=30):
    # Carrega o CIFAR-100 e seleciona somente as categorias escolhidas
    if len(categorias) == 0:
        return 0

    dataset_cifar = CIFAR100(root=PASTA_CACHE, train=True, download=True)
    classes = dataset_cifar.classes

    categorias_invalidas = [categoria for categoria in categorias if categoria not in classes]
    if categorias_invalidas:
        raise ValueError(f"Categorias inexistentes no CIFAR-100: {', '.join(categorias_invalidas)}")

    if os.path.exists(PASTA_DATASET):
        shutil.rmtree(PASTA_DATASET)
    os.makedirs(PASTA_DATASET, exist_ok=True)
    limpar_indices()

    indices_categorias = {classes.index(categoria): categoria for categoria in categorias}
    contadores = {categoria: 0 for categoria in categorias}

    for imagem, indice_classe in dataset_cifar:
        if indice_classe not in indices_categorias:
            continue

        categoria = indices_categorias[indice_classe]

        if contadores[categoria] >= quantidade_por_categoria:
            continue

        pasta_categoria = os.path.join(PASTA_DATASET, categoria)
        os.makedirs(pasta_categoria, exist_ok=True)

        imagem = imagem.convert("RGB")
        imagem = imagem.resize((224, 224), Image.Resampling.LANCZOS)
        numero = contadores[categoria] + 1
        caminho = os.path.join(pasta_categoria, f"{categoria}_{numero:03d}.jpg")
        imagem.save(caminho, "JPEG", quality=90)
        contadores[categoria] += 1

        if all(valor >= quantidade_por_categoria for valor in contadores.values()):
            break

    if os.path.exists(PASTA_CACHE):
        shutil.rmtree(PASTA_CACHE)

    return sum(contadores.values())