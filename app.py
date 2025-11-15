import io
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import imagehash
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# --------------------------
# EXTRAÇÃO DE FEATURES
# --------------------------

def phash_diff(img1, img2):
    return abs(imagehash.phash(img1) - imagehash.phash(img2))

def avg_color_distance(img1, img2):
    arr1 = np.array(img1).mean(axis=(0, 1))
    arr2 = np.array(img2).mean(axis=(0, 1))
    return np.linalg.norm(arr1 - arr2)

def hist_distance(img1, img2):
    h1 = np.array(img1.histogram())
    h2 = np.array(img2.histogram())
    return np.linalg.norm(h1 - h2)


def extract_features(img1, img2):
    return np.array([
        phash_diff(img1, img2),
        avg_color_distance(img1, img2),
        hist_distance(img1, img2)
    ]).reshape(1, -1)

# --------------------------
# TREINAMENTO DO NAIVE BAYES
# --------------------------

# Dados artificiais para treinar (padrões comuns)
X_train = np.array([
    [0, 0, 0],        # iguais
    [2, 10, 5000],    # muito parecidas
    [5, 30, 20000],   # parecidas
    [10, 60, 50000],  # diferentes
    [20, 120, 100000],
    [30, 200, 200000]
])

y_train = np.array([
    2, 1, 1, 0, 0, 0   # 2 = idêntica, 1 = parecida, 0 = diferente
])

model = GaussianNB()
model.fit(X_train, y_train)

# --------------------------
# HTML
# --------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Comparação Inteligente de Imagens (Naive Bayes)</title>
    <style>
        body { font-family: Arial; margin: 40px; }
    </style>
</head>
<body>
    <h2>Comparação Inteligente de Imagens (Naive Bayes)</h2>
    <form action="/compare" method="post" enctype="multipart/form-data">
        <p>Imagem 1:</p>
        <input type="file" name="img1" required><br><br>

        <p>Imagem 2:</p>
        <input type="file" name="img2" required><br><br>

        <button type="submit">Comparar</button>
    </form>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

# --------------------------
# COMPARAÇÃO
# --------------------------

@app.route("/compare", methods=["POST"])
def compare():
    file1 = request.files["img1"].read()
    file2 = request.files["img2"].read()

    img1 = Image.open(io.BytesIO(file1)).convert("RGB")
    img2 = Image.open(io.BytesIO(file2)).convert("RGB")

    feats = extract_features(img1, img2)
    pred = int(model.predict(feats)[0])

    if pred == 2:
        res = "Imagens Idênticas"
    elif pred == 1:
        res = "Imagens Parecidas"
    else:
        res = "Imagens Diferentes"

    return jsonify({
        "classification": res,
        "phash_diff": int(phash_diff(img1, img2)),
        "avg_color_distance": float(avg_color_distance(img1, img2)),
        "hist_distance": float(hist_distance(img1, img2))
    })


if __name__ == "__main__":
    app.run(debug=True)
