Comparação Inteligente de Imagens usando Hash Perceptual e Naive Bayes



O Naive Bayes não entende imagens diretamente — mas ele entende features numéricas.

Então vamos extrair features simples, por exemplo:

✔ 1. Diferença do Hash Perceptual (pHash)
✔ 2. Distância da média de cor (RGB)
✔ 3. Distância dos histogramas de cor

Depois treinamos um Naive Bayes para classificar:

0 → imagens diferentes

1 → imagens parecidas

2 → imagens idênticas



//////////////////////////////////////////////////////////////////////

📦 Instalar dependências

No terminal:

pip install flask pillow imagehash tensorflow keras numpy

▶️ Como rodar
python app.py


Acesse no navegador:

http://127.0.0.1:5000/


Faça upload das duas imagens → recebe JSON com o resultado.

/////////////////////////////////////////////////////////////////////
