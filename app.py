import streamlit as st
import numpy as np
from keras.models import load_model
import random
import json

# Load mappings and model
model = load_model("shakespeare.h5", compile=False)

# Load char_to_idx and idx_to_char from a file or define here
with open("char_to_idx.json") as f:
    char_to_idx = json.load(f)
    char_to_idx = {k: int(v) for k, v in char_to_idx.items()}

with open("idx_to_char.json") as f:
    idx_to_char = json.load(f)
    idx_to_char = {int(k): v for k, v in idx_to_char.items()}



vocab_size = len(char_to_idx)
SEQ_LEN = 60

# Text generation function
def generate_text(seed, length=300, temperature=1.0):
    result = seed
    for _ in range(length):
        input_seq = [char_to_idx.get(c, 0) for c in result[-SEQ_LEN:]]
        input_seq = np.expand_dims(input_seq, axis=0)

        preds = model.predict(input_seq, verbose=0)[0]
        preds = np.log(preds + 1e-8) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))

        next_index = np.random.choice(range(vocab_size), p=preds)
        next_char = idx_to_char[next_index]
        result += next_char
    return result

# Streamlit UI
st.title("🧙 Shakespeare Text Generator")
seed = st.text_input("Enter your seed text:", "to be or not to be that is the")
temperature = st.slider("Creativity (Temperature)", 0.2, 1.5, 0.8)
length = st.slider("Length of generated text", 100, 1000, 300)

if st.button("Generate"):
    seed_text = "to be or not to be that is the"
    generated = generate_text(seed_text, length=length, temperature=temperature)
    st.markdown("### ✍️ Generated Text:")
    st.write(generated)
