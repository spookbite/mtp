import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import scipy

#st.set_page_config(layout="wide")
st.set_page_config(page_title='MTP_19HS20057', layout='wide')

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_label_score(payload):
    inputs = tokenizer(payload, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = {k: v for k, v in zip(model.config.id2label.values(
    ), scipy.special.softmax(logits.numpy().squeeze()))}
    return scores['positive'], scores['negative'], scores['neutral']

# def get_label_score(payload):
#     return 0.8, 0.15, 0.05

st.markdown("<h1 style='text-align: center;'><b>MTP_19HS20057</b></h1>", unsafe_allow_html=True)
st.write("")
st.write("")
st.write("Our model is a pre-trained NLP model to analyze sentiment of financial text. It is built by further training the BERT language model in the finance domain, using a large financial corpus and thereby fine-tuning it for financial sentiment classification. Financial PhraseBank by Malo et al. (2014) is used for fine-tuning.")
st.write("")
st.write("The model will give softmax outputs for three labels: positive, negative or neutral.")
st.write("")
st.write("")
user_input = st.text_input("Please Enter Your Text", "Stocks rallied and the Indian Rupee gained.")
if st.button("Compute"):
    pos, neg, neu = get_label_score(user_input)
    res = {"positive": pos, "negative": neg, "neutral": neu}
    st.code(res)
