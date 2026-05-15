import streamlit as st
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_translator_model():
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    return tokenizer, model

st.title('Aplikacja opublikowana w środowisku Streamlit prezentująca działanie 2 przykładowych modeli językowych')
st.image('t5transformer.png')
st.header('Każdy z modeli językowych spełnia inną funkcję')

st.subheader("Aktualnie dostępne są 2 opcje:")
st.markdown("Opcją 1 jest wydźwięk sentymentalny tekstu służący użytkownikowi do wprowadzania tekstu, "
            "który następnie jest klasyfikowany przez model, czy wprowadzany tekst jest napisany w kontekście pozytywnym, "
            "czy negatywnym, model taki może służyć firmom zajmującym się sprzedażą produktów do automatycznej "
            "oceny recenzji użytkowników.\n")
st.markdown("Opcją 2 jest przetłumaczenie wprowadzanego przez użytkownika tekstu z języka angielskiego na język niemiecki "
            "przez model językowy od firmy Google."
            "Model ten służy do tłumaczenia tylko tekstu na tekst, wynika tak też z jego nazwy: "
            "Text-To-Text Transfer Transformer\n")
st.markdown("Pierwsze uruchomienie może chwilę programowi zająć, "
            "gdyż najpierw musi pobrać modele i zainicjować przy pierwszym uruchomieniu.")

option = st.selectbox(
    "Możliwe do wyboru opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie języka angielskiego na niemiecki (eng -> de)"
    ],
    index=None,
    placeholder="Wybierz opcję z listy"
)

if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Analiza wydźwięku")
    if st.button("Analizuj"):
        if text:
            with st.spinner('Trwa analiza wydźwięku'):
                try:
                    classifier = load_sentiment_model()
                    answer = classifier(text)
                    st.success("Analiza zakończona sukcesem")

                    score = answer[0]['score']

                    if answer[0]['label'] == 'POSITIVE':
                        label = 'Pozytywny'
                        delta_val = f"{score:.2%}"
                    else:
                        label = 'Negatywny'
                        delta_val = f"-{score:.2%}"

                    st.metric("Wynik", label, delta_val)

                except Exception as e:
                    st.error(f"Wystąpił błąd podczas analizy: {e}")
        else:
            st.warning("Proszę wprowadzić tekst do oceny przez model.")

elif option == "Tłumaczenie języka angielskiego na niemiecki (eng -> de)":
    text = st.text_area("Wpisz tekst po angielsku do przetłumaczenia:")
    if st.button("Tłumacz"):
        if text:
            with st.spinner('Tłumaczenie w toku'):
                try:
                    tokenizer, model = load_translator_model()
                    input_text = "Translate English to German: " + text

                    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

                    outputs = model.generate(input_ids, max_new_tokens=100)

                    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    st.success("Przetłumaczono!")

                    # Wyświetlenie wyniku w ramce
                    st.info(f"Po niemiecku: {translated_text}")
                except Exception as e:
                    st.error(f"Wystąpił błąd podczas tłumaczenia: {e}")
        else:
            st.warning("Proszę wprowadzić tekst do przetłumaczenia.")
st.divider()
st.subheader("Numer indeksu: s25086")

