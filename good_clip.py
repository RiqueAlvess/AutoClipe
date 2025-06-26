#!/usr/bin/env python
# good_clip.py  –  Mantém apenas os cortes aprovados
# Requer: sentence-transformers, faster-whisper, spacy, soundfile, tqdm

from pathlib import Path
import json
from tqdm import tqdm
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer, util
import spacy

# ───────── CONFIG ─────────
CORTES_DIR   = Path("cortes")
KEYWORDS_TXT = Path("keywords.txt")
MODEL_EMB    = "paraphrase-multilingual-MiniLM-L12-v2"
SIM_THR      = 0.45          # coseno ≥ 0.45 ⇢ hit
MIN_RATIO    = 0.80          # ≥ 80 % das frases precisam ter hit
WHISPER_SIZE = "small"
OUT_JSON     = Path("cortes_semanticos.json")

# Deletar reprovados ou mover aprovados?
APAGAR_REPROVADO = True                 # False = apenas loga
MOVER_APROVADO   = False                # True = cria cortes_ok/ e move
# ──────────────────────────

# 1) keywords
keywords = [l.strip().lower() for l in KEYWORDS_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]

# 2) modelos
embedder = SentenceTransformer(MODEL_EMB)
try:
    nlp = spacy.load("pt_core_news_sm", disable=["ner", "tagger"])
except OSError:
    nlp = spacy.blank("pt")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")
whisper = WhisperModel(WHISPER_SIZE, device="cpu")

kw_emb = embedder.encode(keywords, normalize_embeddings=True)

def split_frases(txt: str):
    return [" ".join(t.text for t in doc).strip()
            for doc in nlp(txt).sents if doc.text.strip()]

def analisar(mp4: Path):
    segs, info = whisper.transcribe(str(mp4), language="pt", vad_filter=True)
    texto = " ".join(s.text for s in segs).lower()
    frases = split_frases(texto)
    if not frases:
        return 0.0
    f_emb = embedder.encode(frases, normalize_embeddings=True)
    sims  = util.cos_sim(f_emb, kw_emb).max(dim=1).values.cpu().numpy()
    return (sims >= SIM_THR).mean()     # razão de frases com hit

aprovados = []
dest_ok = CORTES_DIR / "cortes_ok"
if MOVER_APROVADO:
    dest_ok.mkdir(exist_ok=True)

for clip in tqdm(sorted(CORTES_DIR.glob("*.mp4"))):
    ratio = analisar(clip)
    if ratio >= MIN_RATIO:
        aprovados.append(clip.name)
        if MOVER_APROVADO:
            clip.rename(dest_ok / clip.name)
    else:
        if APAGAR_REPROVADO:
            clip.unlink()               # remove do disco

OUT_JSON.write_text(json.dumps(aprovados, indent=2, ensure_ascii=False))
print(f"\n✅ {len(aprovados)} clipes aprovados.  Lista → {OUT_JSON}")
