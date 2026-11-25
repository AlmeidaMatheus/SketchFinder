import hashlib
import re
import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torchvision.models import resnet18, ResNet18_Weights
from transformers import CLIPModel, CLIPProcessor

MODELO_CLIP = "openai/clip-vit-base-patch32"

st.set_page_config(page_title="AI Draw Battle", layout="wide")

@st.cache_resource
def carregar_modelos():
    # Carrega os modelos apenas uma vez
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pesos = ResNet18_Weights.DEFAULT
    resnet = resnet18(weights=pesos).to(device).eval()
    clip = CLIPModel.from_pretrained(MODELO_CLIP).to(device).eval()
    processador = CLIPProcessor.from_pretrained(MODELO_CLIP)
    return device, resnet, pesos.transforms(), list(pesos.meta["categories"]), clip, processador

def normalizar(texto):
    return re.sub(r"\s+", " ", texto.lower().strip().replace("_", " "))

def acertou(resposta, palpite):
    resposta, palpite = normalizar(resposta), normalizar(palpite)
    return resposta == palpite or resposta in [normalizar(x) for x in re.split(r"[,;/]", palpite)]

def pil(imagem):
    return Image.fromarray(imagem).convert("RGB")

def classificar_resnet(imagem):
    # Classifica entre as classes originais do ImageNet
    device, modelo, transformacao, categorias, _, _ = carregar_modelos()
    tensor = transformacao(pil(imagem)).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(modelo(tensor), dim=1)[0]

    indice = probs.argmax().item()
    return categorias[indice], probs[indice].item()

def probabilidade_resnet(imagem, resposta):
    # Retorna quanto a ResNet atribuiu à resposta correta
    device, modelo, transformacao, categorias, _, _ = carregar_modelos()
    indice = next((i for i, categoria in enumerate(categorias) if acertou(resposta, categoria)), None)

    if indice is None:
        return None

    tensor = transformacao(pil(imagem)).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(modelo(tensor), dim=1)[0]

    return probs[indice].item()

def preparar_clip(resposta):
    # Prepara os rótulos textuais usados pelo CLIP
    device, _, _, categorias, modelo, processador = carregar_modelos()
    candidatos = [normalizar(resposta)]

    for categoria in categorias:
        categoria = normalizar(categoria)
        if categoria not in candidatos:
            candidatos.append(categoria)

    textos = [f"a simple black and white drawing of a {x}" for x in candidatos]
    entradas = processador(text=textos, return_tensors="pt", padding=True, truncation=True)
    entradas = {k: v.to(device) for k, v in entradas.items()}

    with torch.no_grad():
        embeddings = modelo.get_text_features(**entradas)

    if not isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.pooler_output

    return candidatos, embeddings / embeddings.norm(dim=-1, keepdim=True)

def embedding_clip(imagem):
    # Gera o embedding visual do desenho
    device, _, _, _, modelo, processador = carregar_modelos()
    entradas = processador(images=pil(imagem), return_tensors="pt")
    entradas = {k: v.to(device) for k, v in entradas.items()}

    with torch.no_grad():
        embedding = modelo.get_image_features(**entradas)

    if not isinstance(embedding, torch.Tensor):
        embedding = embedding.pooler_output

    return embedding / embedding.norm(dim=-1, keepdim=True)

def classificar_clip(imagem):
    # Retorna o rótulo mais semelhante
    similaridades = embedding_clip(imagem) @ st.session_state.embeddings_clip.T
    indice = similaridades.argmax(dim=1).item()
    return st.session_state.candidatos_clip[indice], similaridades[0, indice].item()

def similaridade_resposta_clip(imagem):
    # A resposta correta sempre é o primeiro candidato
    similaridade = embedding_clip(imagem) @ st.session_state.embeddings_clip[0].unsqueeze(1)
    return similaridade[0, 0].item()

def iniciar_estado():
    valores = {
        "ativo": False,
        "encerrado": False,
        "vencedor": None,
        "resposta": "",
        "historico": [],
        "ultimo_hash": None,
        "desenho": None,
        "candidatos_clip": [],
        "embeddings_clip": None,
        "score_resposta_clip": None,
        "score_resposta_resnet": None,
        "canvas_id": 0
    }

    for chave, valor in valores.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor

def iniciar(resposta):
    candidatos, embeddings = preparar_clip(resposta)
    st.session_state.ativo = True
    st.session_state.encerrado = False
    st.session_state.vencedor = None
    st.session_state.resposta = normalizar(resposta)
    st.session_state.historico = []
    st.session_state.ultimo_hash = None
    st.session_state.desenho = None
    st.session_state.candidatos_clip = candidatos
    st.session_state.embeddings_clip = embeddings
    st.session_state.score_resposta_clip = None
    st.session_state.score_resposta_resnet = None
    st.session_state.canvas_id += 1

def resetar():
    st.session_state.ativo = False
    st.session_state.encerrado = False
    st.session_state.vencedor = None
    st.session_state.resposta = ""
    st.session_state.historico = []
    st.session_state.ultimo_hash = None
    st.session_state.desenho = None
    st.session_state.candidatos_clip = []
    st.session_state.embeddings_clip = None
    st.session_state.score_resposta_clip = None
    st.session_state.score_resposta_resnet = None
    st.session_state.canvas_id += 1

def analisar(imagem):
    # Cada alteração do desenho gera um palpite por modelo
    palpite_clip, score_clip = classificar_clip(imagem)
    palpite_resnet, score_resnet = classificar_resnet(imagem)

    st.session_state.historico.append({
        "clip": palpite_clip,
        "clip_score": score_clip,
        "resnet": palpite_resnet,
        "resnet_score": score_resnet
    })

    resposta = st.session_state.resposta
    clip_acertou = acertou(resposta, palpite_clip)
    resnet_acertou = acertou(resposta, palpite_resnet)

    if clip_acertou and resnet_acertou:
        st.session_state.vencedor = "Empate"
    elif clip_acertou:
        st.session_state.vencedor = "CLIP"
    elif resnet_acertou:
        st.session_state.vencedor = "ResNet-18"

    if st.session_state.vencedor:
        encerrar()

def encerrar():
    # Revela a resposta e calcula os scores finais
    if st.session_state.desenho is not None:
        st.session_state.score_resposta_clip = similaridade_resposta_clip(st.session_state.desenho)
        st.session_state.score_resposta_resnet = probabilidade_resnet(st.session_state.desenho, st.session_state.resposta)

    st.session_state.encerrado = True

def mostrar_atual():
    if not st.session_state.historico:
        return

    ultimo = st.session_state.historico[-1]
    coluna1, coluna2 = st.columns(2)

    with coluna1:
        st.metric("CLIP", ultimo["clip"], f"{ultimo['clip_score'] * 100:.2f}%")

    with coluna2:
        st.metric("ResNet-18", ultimo["resnet"], f"{ultimo['resnet_score'] * 100:.2f}%")

def mostrar_historico():
    if not st.session_state.historico:
        return

    st.subheader("Historico de palpites")

    for i, item in reversed(list(enumerate(st.session_state.historico, 1))):
        c1, c2, c3 = st.columns([0.4, 2, 2])
        c1.write(f"**{i}**")
        c2.write(f"**CLIP:** {item['clip']} ({item['clip_score'] * 100:.2f}%)")
        c3.write(f"**ResNet-18:** {item['resnet']} ({item['resnet_score'] * 100:.2f}%)")

def mostrar_resultado():
    if st.session_state.vencedor:
        st.success(f"{st.session_state.vencedor} venceu.")
    else:
        st.warning("Rodada encerrada sem vencedor.")

    st.write(f"Resposta correta: **{st.session_state.resposta}**")
    mostrar_atual()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("CLIP")
        score = st.session_state.score_resposta_clip
        if score is not None:
            st.write(f"Similaridade com a resposta: **{score * 100:.2f}%**")
        st.caption("Similaridade entre o desenho e o texto, nao e probabilidade.")

    with c2:
        st.subheader("ResNet-18")
        score = st.session_state.score_resposta_resnet
        if score is None:
            st.write("A resposta nao existe entre as classes da ResNet-18.")
        else:
            st.write(f"Probabilidade da resposta: **{score * 100:.2f}%**")

    mostrar_historico()

iniciar_estado()

st.write("Desenhe um objeto e veja qual modelo consegue descobrir primeiro.")

if not st.session_state.ativo:
    resposta = st.text_input(
        "O que sera desenhado?",
        placeholder="Exemplo: house, bicycle, dragon"
    )

    st.caption("Use preferencialmente ingles. A resposta ficara escondida durante a rodada.")

    if st.button("Iniciar", use_container_width=True):
        if not resposta.strip():
            st.warning("Informe o que sera desenhado.")
        else:
            with st.spinner("Preparando modelos..."):
                iniciar(resposta)
            st.rerun()

elif st.session_state.encerrado:
    mostrar_resultado()

    if st.button("Jogar novamente", use_container_width=True):
        resetar()
        st.rerun()

else:
    st.info(f"Rodada em andamento. Palpites: {len(st.session_state.historico)}")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Encerrar e revelar resposta", use_container_width=True):
            encerrar()
            st.rerun()

    with c2:
        if st.button("Cancelar rodada", use_container_width=True):
            resetar()
            st.rerun()

    espessura = st.slider("Espessura do pincel:", 1, 20, 5)

    canvas = st_canvas(
        fill_color="#FFFFFF",
        stroke_width=espessura,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=400,
        width=600,
        drawing_mode="freedraw",
        update_streamlit=True,
        return_image_data=True,
        key=f"canvas_{st.session_state.canvas_id}"
    )

    if canvas.image_data is not None:
        desenho = canvas.image_data.astype(np.uint8)[:, :, :3]
        st.session_state.desenho = desenho
        hash_atual = hashlib.md5(desenho.tobytes()).hexdigest()

        if np.mean(desenho) <= 254 and hash_atual != st.session_state.ultimo_hash:
            st.session_state.ultimo_hash = hash_atual

            with st.spinner("Analisando desenho..."):
                analisar(desenho)

            st.rerun()

    mostrar_atual()
    mostrar_historico()