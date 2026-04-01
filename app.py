import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("Text Generation project by Khushi Kanade")

# Load model (.keras works same as .h5)
model = load_model("TextGenerationModel.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_sequence_length = 6  

def generate_text(seed_text, next_words=10):
    output_text = seed_text

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_length - 1,
            padding='pre'
        )

        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]

        # Convert index → word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        output_text += " " + output_word

    return output_text


# UI
st.title("Text Generator")

seed = st.text_input("Enter seed text")

num_words = st.slider("Words to generate", 1, 20, 10)

if st.button("Generate"):
    if seed.strip() == "":
        st.warning("Please enter some text")
    else:
        result = generate_text(seed, num_words)
        st.success(result)
