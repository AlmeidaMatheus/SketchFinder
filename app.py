import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from src.processamento import processar_imagem
from src.similaridade import buscar_imagens_semelhantes
from src.ia import buscar_imagens_ia, gerar_indice_embeddings, carregar_indice_embeddings
from src.clip import buscar_imagens_clip, buscar_texto_clip, gerar_indice_clip, carregar_indice_clip
from src.dataset import preparar_dataset, excluir_dataset, contar_imagens_dataset, contar_por_categoria, CATEGORIAS_DISPONIVEIS

st.set_page_config(page_title="SketchFinder", page_icon="🔎", layout="wide")

def exibir_resultados(resultados):
    # Exibe as imagens encontradas em ordem de similaridade
    if len(resultados) == 0:
        st.warning("Nenhuma imagem foi encontrada.")
        return
    st.subheader("Resultados")
    colunas = st.columns(len(resultados))
    for posicao, (coluna, resultado) in enumerate(zip(colunas, resultados), start=1):
        with coluna:
            st.write(f"### {posicao}º")
            st.image(resultado["caminho"], use_container_width=True)
            percentual = resultado["similaridade"] * 100
            st.write(f"**{resultado['categoria']}**")
            st.write(f"Similaridade: {percentual:.2f}%")

def buscar_imagem_por_metodo(metodo, original, bordas):
    # Executa a busca conforme o método selecionado
    if metodo == "CLIP":
        return buscar_imagens_clip(original, limite=5)
    if metodo == "ResNet-18":
        return buscar_imagens_ia(original, limite=5)
    return buscar_imagens_semelhantes(bordas, limite=5)

def verificar_indice(metodo):
    # Verifica se o índice necessário está disponível
    if metodo == "CLIP":
        return carregar_indice_clip() is not None
    if metodo == "ResNet-18":
        return carregar_indice_embeddings() is not None
    return contar_imagens_dataset() > 0

st.title("🔎 SketchFinder")
st.write("Encontre imagens semelhantes utilizando uma imagem, um desenho ou uma descrição.")
st.divider()

indice_resnet = carregar_indice_embeddings()
indice_clip = carregar_indice_clip()
total_imagens = contar_imagens_dataset()

st.sidebar.title("Dataset")

if total_imagens == 0:
    st.sidebar.warning("Nenhuma imagem no dataset.")
else:
    st.sidebar.success(f"{total_imagens} imagens disponíveis.")

with st.sidebar.expander("📥 Preparar dataset", expanded=total_imagens == 0):
    categorias = st.multiselect("Categorias:", CATEGORIAS_DISPONIVEIS, default=CATEGORIAS_DISPONIVEIS)
    quantidade = st.number_input("Imagens por categoria:", min_value=5, max_value=100, value=30, step=5)
    total_estimado = len(categorias) * quantidade
    st.write(f"Total: **{total_estimado} imagens**")

    if st.button("📥 Preparar", use_container_width=True):
        if len(categorias) == 0:
            st.warning("Selecione pelo menos uma categoria.")
        else:
            with st.spinner("Baixando e preparando o dataset..."):
                total = preparar_dataset(categorias, quantidade)
            st.success(f"{total} imagens preparadas.")
            st.info("Agora gere os índices CLIP e ResNet-18.")
            st.rerun()

if indice_resnet is None:
    st.sidebar.warning("ResNet-18: não indexado.")
else:
    st.sidebar.success(f"ResNet-18: {len(indice_resnet)} imagens.")

if indice_clip is None:
    st.sidebar.warning("CLIP: não indexado.")
else:
    st.sidebar.success(f"CLIP: {len(indice_clip)} imagens.")

if st.sidebar.button("🔄 Gerar/Regerar índices", use_container_width=True):
    if contar_imagens_dataset() == 0:
        st.sidebar.warning("Prepare o dataset primeiro.")
    else:
        with st.spinner("Gerando índice ResNet-18..."):
            quantidade_resnet = gerar_indice_embeddings()
        with st.spinner("Gerando índice CLIP..."):
            quantidade_clip = gerar_indice_clip()
        st.sidebar.success(f"ResNet-18: {quantidade_resnet} imagens.")
        st.sidebar.success(f"CLIP: {quantidade_clip} imagens.")
        st.rerun()

with st.sidebar.expander("📊 Conteúdo do dataset"):
    quantidades = contar_por_categoria()
    if len(quantidades) == 0:
        st.write("Dataset vazio.")
    else:
        for categoria, total in quantidades.items():
            st.write(f"**{categoria}:** {total}")

st.sidebar.divider()

confirmar_exclusao = st.sidebar.checkbox("Confirmar exclusão do dataset")

if st.sidebar.button("🗑️ Excluir dataset", use_container_width=True):
    if not confirmar_exclusao:
        st.sidebar.warning("Marque a confirmação antes de excluir.")
    else:
        excluir_dataset()
        st.sidebar.success("Dataset e índices excluídos.")
        st.rerun()

aba_imagem, aba_desenho, aba_texto = st.tabs(["📷 Buscar por imagem", "🎨 Buscar por desenho", "📝 Buscar por texto"])

with aba_imagem:
    st.subheader("Buscar por imagem")
    st.write("Envie uma imagem para encontrar os itens mais semelhantes no dataset.")
    arquivo = st.file_uploader("Selecione uma imagem", type=["jpg", "jpeg", "png"], key="upload_imagem")

    if arquivo is not None:
        original, redimensionada, cinza, bordas = processar_imagem(arquivo)
        coluna_imagem, coluna_configuracao = st.columns([1, 2])

        with coluna_imagem:
            st.image(original, caption="Imagem selecionada", use_container_width=True)

        with coluna_configuracao:
            st.write("### Configurações da busca")
            metodo_imagem = st.selectbox("Método de busca:", ["CLIP", "ResNet-18", "OpenCV - Bordas"], key="metodo_imagem")

            if st.button("🔎 Buscar imagens semelhantes", key="buscar_imagem"):
                if not verificar_indice(metodo_imagem):
                    st.warning(f"O método {metodo_imagem} ainda não está disponível.")
                else:
                    with st.spinner("Analisando imagem..."):
                        resultados = buscar_imagem_por_metodo(metodo_imagem, original, bordas)
                    exibir_resultados(resultados)

with aba_desenho:
    st.subheader("Faça seu desenho")
    st.write("Desenhe o objeto que deseja pesquisar utilizando linhas pretas.")
    espessura = st.slider("Espessura do pincel:", min_value=1, max_value=20, value=5)
    metodo_desenho = st.selectbox("Método de busca:", ["CLIP", "ResNet-18"], key="metodo_desenho")

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
        key="canvas"
    )

    if canvas.image_data is not None:
        desenho = canvas.image_data.astype(np.uint8)
        desenho_rgb = desenho[:, :, :3]

        if st.button("🔎 Buscar pelo desenho", key="buscar_desenho"):
            if np.mean(desenho_rgb) > 254:
                st.warning("Faça um desenho antes de realizar a busca.")
            elif not verificar_indice(metodo_desenho):
                st.warning(f"O índice do método {metodo_desenho} ainda não foi gerado.")
            else:
                with st.spinner("Analisando seu desenho..."):
                    if metodo_desenho == "CLIP":
                        resultados = buscar_imagens_clip(desenho_rgb, limite=5)
                    else:
                        resultados = buscar_imagens_ia(desenho_rgb, limite=5)
                exibir_resultados(resultados)

with aba_texto:
    st.subheader("Buscar por texto")
    st.write("Descreva o que deseja encontrar no dataset.")
    texto = st.text_input("Descrição:", placeholder="Exemplo: a photo of a bicycle")

    if st.button("🔎 Buscar pela descrição", key="buscar_texto"):
        if texto.strip() == "":
            st.warning("Digite uma descrição antes de realizar a busca.")
        elif carregar_indice_clip() is None:
            st.warning("O índice do CLIP ainda não foi gerado.")
        else:
            with st.spinner("Interpretando a descrição..."):
                resultados = buscar_texto_clip(texto.strip(), limite=5)
            exibir_resultados(resultados)