import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =============================
# Load Saved Files
# =============================

model = load_model("next_word_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tk = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# =============================
# Function: Top-3 Suggestions
# =============================

def predict_top3_words(text):

    text = text.lower()
    seq = tk.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding='pre')

    pred = model.predict(seq, verbose=0)

    top_indices = np.argsort(pred[0])[-3:]
    top_indices = top_indices[::-1]

    predicted_words = []

    for index in top_indices:
        word = tk.index_word.get(index, "")
        predicted_words.append(word)

    return predicted_words

# =============================
# Streamlit UI
# =============================

st.title("📖 Next Word Prediction using LSTM")

user_input = st.text_input("Type your sentence:")

if st.button("Suggest Next Words"):
    if user_input.strip() != "":
        suggestions = predict_top3_words(user_input)
        st.success("Suggestions:")
        st.write(" | ".join(suggestions))