import streamlit as st
from sentiment_automaton import SentimentAutomaton
import time

st.set_page_config(
    page_title="Análise de Sentimento - Feedback de Alunos",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

sa = SentimentAutomaton()

# Sidebar com explicação detalhada
with st.sidebar:
    st.header("Sobre a Análise")
    st.write(
        """
        Este app analisa o sentimento dos feedbacks fornecidos por alunos nas aulas online.
        Ajuda professores e a própria plataforma a entenderem melhor a percepção dos estudantes, 
        identificando pontos fortes e áreas que precisam de melhorias.
        """
    )
    st.markdown("---")
    st.write("Repositório GitHub:")
    st.markdown(
        "[Acesse o repositório aqui](https://github.com/bombinideh/automato)"
    )
    st.markdown("---")
    st.write("Equipe:")
    st.write("Deborah Bombini, Gabriel Rabelo, Guilherme de Azevedo, João Gabriel Ortiz e Pedro Telli")
    st.markdown("---")

st.markdown(
    "<h1 style='text-align:center; color:#4B8BBE; margin-bottom: 0;'>Análise de Sentimento de Feedbacks</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center; color:#555; font-size:18px; margin-top:5px;'>Digite um feedback de aluno para analisar o sentimento expressado.</p>",
    unsafe_allow_html=True,
)

frase = st.text_area(
    label="Digite o feedback do aluno aqui:",
    key="input_frase",
    height=120,
    max_chars=300,
    placeholder="Ex: 'A aula foi ótima, mas o professor é ruim.'",
)

btn_style = """
<style>
    div.stButton > button {
        position: relative;
        font-size: 1.875rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: all 250ms;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    div.stButton > button {
        background-color: #f0f0f0;
        color: #242424;
        border-radius: 0.5rem;
        box-shadow: 
            inset 0 1px 0 0 #f4f4f4,
            0 1px 0 0 #efefef,
            0 2px 0 0 #ececec,
            0 4px 0 0 #e0e0e0,
            0 5px 0 0 #dedede,
            0 6px 0 0 #dcdcdc,
            0 7px 0 0 #cacaca,
            0 7px 8px 0 #cecece;
        font-weight: bold;
        padding: 12px 28px;
        font-size: 16px;
        width: 180px;
        margin: auto;
        display: block;
    }
    .stButton > button:hover {
        color: #242424;
        transform: translateY(4px);
        box-shadow: 
            inset 0 1px 0 0 #f4f4f4,
            0 1px 0 0 #efefef,
            0 1px 0 0 #ececec,
            0 2px 0 0 #e0e0e0,
            0 2px 0 0 #dedede,
            0 3px 0 0 #dcdcdc,
            0 4px 0 0 #cacaca,
            0 4px 6px 0 #cecece;
    }
</style>
"""
st.markdown(btn_style, unsafe_allow_html=True)

analisar = st.button("Analisar Feedback", key="btn_analisar")

if analisar:
    if frase.strip():
        with st.spinner("Analisando o feedback..."):
            time.sleep(1)
            sent = sa.analyze(frase)
        if not sent['valid']:
            st.error(f"❌ Erro na análise: {sent['error']}")
        else:
            col1, col2 = st.columns([3, 4])
            with col1:
                st.markdown(
                    f"""
                    <div style='border: 2px solid #4B8BBE; border-radius: 10px; padding: 20px; background-color: white; box-shadow: 2px 2px 10px rgba(75,139,190,0.2); margin-top: 20px;'>
                        <p style='font-size: 22px; font-weight: 600; color: #1F3557; margin-bottom: 5px;'>📊 Sentimento Detectado: <b>{sent['mood']}</b></p>
                        <p style='font-family: monospace; font-size: 16px; color: #555;'>Caminho: <code>{sent['path']}</code></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                sa.draw_matplotlib(path=sent['path'], fname='automato.png')
                st.image('automato.png', caption='Autômato de Sentimento', use_container_width=True, clamp=True)
    else:
        st.warning("⚠️ Por favor, insira um feedback para análise.")

st.markdown("---")