from flask import Flask, render_template, request
import numpy as np
import json
from keras.models import load_model

app = Flask(__name__)

# Load model and mappings
model = load_model("shakespeare.h5", compile=False)
SEQ_LEN = 60

with open("char_to_idx.json") as f:
    char_to_idx = json.load(f)
    char_to_idx = {k: int(v) for k, v in char_to_idx.items()}

with open("idx_to_char.json") as f:
    idx_to_char = json.load(f)
    idx_to_char = {int(k): v for k, v in idx_to_char.items()}

vocab_size = len(char_to_idx)

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

@app.route("/", methods=["GET", "POST"])
def index():
    generated = ""
    seed_text = "to be or not to be that is the"
    temperature = 0.8
    length = 300

    if request.method == "POST":
        seed_text = request.form.get("seed", seed_text)
        temperature = float(request.form.get("temperature", 0.8))
        length = int(request.form.get("length", 300))
        generated = generate_text(seed_text, length=length, temperature=temperature)

    return render_template("index.html", generated=generated, seed=seed_text, temperature=temperature, length=length)

if __name__ == "__main__":
    app.run()
