# gera_face_embedding_mcig.py
import os, pickle, numpy as np, face_recognition
from pathlib import Path

PASTA_FOTOS   = Path("Fotos")
ARQ_SAIDA_PKL = Path("face_embedding.pkl")

def gerar_embedding_medio(pasta: Path):
    embeddings = []
    for img in pasta.glob("*.[jp][pn]g"):
        faces = face_recognition.face_encodings(
            face_recognition.load_image_file(img))
        if faces:
            embeddings.append(faces[0])
            print(f"✅  {img.name}")
        else:
            print(f"⚠️  Nenhum rosto em {img.name}")
    if not embeddings:
        raise RuntimeError("Nenhum rosto válido encontrado.")
    return np.mean(embeddings, axis=0)

if __name__ == "__main__":
    vetor = gerar_embedding_medio(PASTA_FOTOS)
    with ARQ_SAIDA_PKL.open("wb") as f:
        pickle.dump(vetor, f)
    print(f"✅  Embedding salvo em {ARQ_SAIDA_PKL}")
