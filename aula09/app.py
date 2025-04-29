import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import re

# Função para limpeza básica do texto
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^A-Za-zÀ-ÿ0-9\s.,!?]", "", text)  # Remove caracteres especiais
    return text.strip()

# Carrega tradutor apenas uma vez
@st.cache_resource
def get_translator():
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    return tokenizer, model

# Função de tradução
def translate_to_english(text):
    tokenizer, model = get_translator()
    inputs = tokenizer.encode(text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs, max_length=512)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated

# Título do app
st.title("Aplicativo de Análise de Sentimentos com Hugging Face Transformers")

# Área de texto
text = st.text_area("Por favor, escreva sua sentença.")

# Inicializa estado da análise e último texto
if "analyze" not in st.session_state:
    st.session_state.analyze = False
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

# Atualiza estado ao clicar no botão
if st.button("Análise de Sentimentos"):
    st.session_state.analyze = True
    st.session_state.last_text = text

# Executa a análise se o botão foi clicado e há texto
if st.session_state.analyze and st.session_state.last_text.strip():
    text_input = st.session_state.last_text

    try:
        detected_lang = detect(text_input)
        st.write(f"Idioma detectado: {detected_lang}")

        cleaned_text = preprocess_text(text_input)
        st.write(f"Texto pré-processado: {cleaned_text}")

        if detected_lang != "en":
            cleaned_text = translate_to_english(cleaned_text)
            st.write(f"Texto traduzido para análise: {cleaned_text}")

        # Carrega o pipeline de análise de sentimento
        model = pipeline(
            "text-classification",
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            top_k=None
        )

        # Executa a análise
        results = model(cleaned_text)

        st.subheader("Resultado bruto do modelo:")
        st.json(results)

        # Verifica e exibe distribuição de sentimentos
        if results and isinstance(results, list) and all(isinstance(r, dict) for r in results):
            st.subheader("Distribuição de sentimentos (em %):")
            for r in results:
                percentual = round(r["score"] * 100, 2)
                st.write(f"• {r['label'].capitalize()}: {percentual}%")

            st.bar_chart({r["label"].capitalize(): round(r["score"] * 100, 2) for r in results})

            result = max(results, key=lambda x: x.get("score", 0))
            score = result.get("score")
            label = result.get("label")

            if score is not None and label is not None:
                st.success(f"A sentença é {round(score * 100, 2)}% {label}.")
            else:
                st.error("O modelo não retornou 'score' e 'label'.")
        else:
            st.error("O resultado da análise de sentimentos está vazio ou em formato inesperado.")
    except Exception as e:
        st.error(f"Ocorreu um erro durante a análise: {str(e)}")
elif st.session_state.analyze and not st.session_state.last_text.strip():
    st.warning("Por favor, insira um texto para análise.")
