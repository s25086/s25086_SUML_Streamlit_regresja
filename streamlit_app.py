import streamlit as st
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer

@st.cache_resource
def load_sentiment_model():
    return pipeline("/sentiment-analysis")

@st.cache_resource
def load_translator_model():
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    return tokenizer, model

st.title('Lab05. Streamlit')
st.image('t5transformer.png')
st.header('Aplikacja tłumacząca tekst z języka angielskiego na język niemiecki')
st.header('Przetwarzanie języka naturalnego')

st.text("Aplikacja aktualnie posiada 2 opcje:")
st.markdown("1 jest wydźwięk sentymentaly tekstu służący użytkownikowi do wprowadzania tekstu, "
            "który następnie jest klasyfikowany przez model, czy wprowadzany tekst jest napisany w kontekście pozytywnym, "
            "czy negatywnym, model taki może służyć firmom zajmującym się sprzedażą produktów do automatycznej "
            "oceny recenzji użytkowników.\n")
st.markdown("2 jest przetłumaczenie wprowadzanego przez użytkownika tekstu z języka angielskiego na język niemiecki "
            "przez model językowy od firmy Google."
            "Model ten służy do tłumaczenia tylko tekstu na tekst, wynika tak też z jego nazwy: "
            "Text-To-Text Transfer Transformer\n")
st.markdown("Pierwsze uruchomienie może chwilę programowi zająć, "
            "gdyż najpierw musi pobrać modele i zainicjować przy pierwszym uruchomieniu.")

option = st.selectbox(
    "Możliwe do wyboru opcje",
    [
        "Wybierz opcję:",
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie języka angielskiego na niemiecki (eng -> de)",
    ],
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

                    # Prezentacja wyniku
                    label = answer[0]['label']
                    score = answer[0]['score']
                    st.metric("Wynik", label, f"{score:.2%}")
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
st.subheader("Numer indeksu: s25086")
st.divider()

st.subheader('Zadanie do wykonania')
st.write('Wykorzystaj Huggin Face do stworzenia swojej własnej aplikacji tłumaczącej tekst z języka angielskiego na język niemiecki. Zmodyfikuj powyższy kod dodając do niego kolejną opcję, tj. tłumaczenie tekstu. Informacje potrzebne do zmodyfikowania kodu znajdziesz na stronie Huggin Face - https://huggingface.co/docs/transformers/index')
st.write('🐞 Dodaj właściwy tytuł do swojej aplikacji, może jakieś grafiki?')
st.write('🐞 Dodaj krótką instrukcję i napisz do czego służy aplikacja')
st.write('🐞 Wpłyń na user experience, dodaj informacje o ładowaniu, sukcesie, błędzie, itd.')
st.write('🐞 Na końcu umieść swój numer indeksu')
st.write('🐞 Stwórz nowe repozytorium na GitHub, dodaj do niego swoją aplikację, plik z wymaganiami (requirements.txt)')
st.write('🐞 Udostępnij stworzoną przez siebie aplikację (https://share.streamlit.io) a link prześlij do prowadzącego')
