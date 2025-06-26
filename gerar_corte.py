#!/usr/bin/env python
# gerar_corte.py  –  Voz do MC IG → clipes de 15-60 s + pré-buffer 5 s
# Requisitos: resemblyzer webrtcvad numpy tqdm torch ffmpeg

from pathlib import Path
import subprocess, json, pickle, numpy as np, webrtcvad, torch
from tqdm import tqdm
from resemblyzer import VoiceEncoder, preprocess_wav

# ---------- caminhos ----------
DL          = Path("downloads")
WAV_FILE    = DL / "video.wav"
MP4_FILE    = DL / "video.mp4"
EMB_PICKLE  = Path("voice_embedding.pkl")
OUT_DIR     = Path("cortes")
OUT_DIR.mkdir(exist_ok=True)

# ---------- hiperparâmetros ----------
WIN_DUR   = 1.5     # s – janela do encoder
HOP_DUR   = 0.375   # s – passo (25 % overlap)
SIM_THR   = 0.60    # cosseno mínimo → MC IG
MERGE_GAP = 1.0     # s – une segmentos se gap ≤ 1 s
PRE_PAD   = 5.0     # s – **novo**: pega 5 s antes do ponto inicial
POST_PAD  = 0.25    # s – pequena folga no fim
MIN_LEN   = 15.0    # s – corte mínimo
MAX_LEN   = 60.0    # s – corte máximo

# ---------- util ----------
def stream_embeddings(wav: np.ndarray, sr: int, enc: VoiceEncoder):
    win = int(WIN_DUR * sr)
    hop = int(HOP_DUR * sr)
    for start in range(0, len(wav) - win + 1, hop):
        chunk = wav[start : start + win]
        emb = enc.embed_utterance(chunk)
        yield (emb / (np.linalg.norm(emb) + 1e-8), start / sr)

# ---------- 1. embeddings ----------
print("🔊  Extraindo embeddings…")
enc = VoiceEncoder()
wav = preprocess_wav(WAV_FILE)
sr  = 16000

embs, splits_s = zip(*tqdm(stream_embeddings(wav, sr, enc), desc="Embeddings"))
embs = np.vstack(embs)

# ---------- 2. similaridade ----------
target = pickle.load(open(EMB_PICKLE, "rb"))
target /= np.linalg.norm(target) + 1e-8
sim = embs @ target

# ---------- 3. VAD ----------
vad = webrtcvad.Vad(2)
frame = int(0.03 * sr)
speech = [
    vad.is_speech((wav[i:i+frame] * 32768).astype(np.int16).tobytes(), sr)
    for i in range(0, len(wav), frame)
]
speech_mask = np.repeat(speech, int(0.03 / HOP_DUR))[: len(sim)]
sim[~speech_mask] = 0.0

# ---------- 4. segmentos ----------
pos = np.where(sim >= SIM_THR)[0]
segments = []
if pos.size:
    start = pos[0]
    for i, j in zip(pos, pos[1:]):
        if splits_s[j] - splits_s[i] > MERGE_GAP:
            segments.append((start, i))
            start = j
    segments.append((start, pos[-1]))

# ---------- 5. cortes com PRE_PAD ----------
clips = []
for i0, i1 in segments:
    t0 = max(0, splits_s[i0] - PRE_PAD)
    t1 = splits_s[i1] + WIN_DUR + POST_PAD
    dur = t1 - t0
    if dur < MIN_LEN:
        continue
    while dur > MAX_LEN:
        clips.append((t0, t0 + MAX_LEN))
        t0 += MAX_LEN
        dur -= MAX_LEN
    clips.append((t0, t1))

if not clips:
    raise SystemExit("⚠️  Nenhum corte elegível encontrado.")

print(f"🎬  {len(clips)} cortes candidatos encontrados")

# ---------- 6. exporta ----------
for n, (ini, fim) in enumerate(clips, 1):
    out = OUT_DIR / f"corte_{n:02d}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{ini:.3f}", "-to", f"{fim:.3f}",
         "-i", str(MP4_FILE), "-c", "copy", str(out)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )

print("✅  Cortes salvos em", OUT_DIR.resolve())
print("🗒️  Lista (s):", json.dumps([(round(a,2), round(b,2)) for a,b in clips], indent=2))

# ---------- 7. limpa originais ----------
for fp in (WAV_FILE, MP4_FILE):
    if fp.exists():
        fp.unlink()
print("🧹  Arquivos originais deletados de", DL)
