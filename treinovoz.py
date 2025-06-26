# gera_voice_embedding_mcig.py
import os, pickle, numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

PASTA_WAVS    = Path("Audios")
ARQ_SAIDA_PKL = Path("voice_embedding.pkl")

def gerar_embedding_medio_voz(pasta: Path):
    enc = VoiceEncoder()
    embs = []
    for wav in pasta.glob("*.wav"):
        try:
            embs.append(enc.embed_utterance(preprocess_wav(wav)))
            print(f"✅  {wav.name}")
        except Exception as e:
            print(f"⚠️  {wav.name} → {e}")
    if not embs:
        raise RuntimeError("Nenhum áudio válido encontrado.")
    return np.mean(embs, axis=0)

if __name__ == "__main__":
    vetor = gerar_embedding_medio_voz(PASTA_WAVS)
    with ARQ_SAIDA_PKL.open("wb") as f:
        pickle.dump(vetor, f)
    print(f"✅  Embedding salvo em {ARQ_SAIDA_PKL}")
