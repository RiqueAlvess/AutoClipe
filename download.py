# download.py
from yt_dlp import YoutubeDL
from pathlib import Path
import subprocess

DESTINO = Path("downloads")
DESTINO.mkdir(parents=True, exist_ok=True)

def baixar_video_e_audio(url: str, destino: Path = DESTINO) -> tuple[Path, Path]:
    mp4_path = _gerar_nome_unico(destino, "video", ".mp4")
    wav_path = mp4_path.with_suffix(".wav")

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "merge_output_format": "mp4",
        "outtmpl": str(mp4_path),
        "quiet": True,
        "noplaylist": False,
        "restrictfilenames": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    cmd = ["ffmpeg", "-y", "-i", str(mp4_path), "-ac", "1", "-ar", "16000", str(wav_path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    return mp4_path, wav_path

def _gerar_nome_unico(pasta: Path, base: str, ext: str) -> Path:
    candidate = pasta / f"{base}{ext}"
    count = 1
    while candidate.exists():
        candidate = pasta / f"{base}_{count}{ext}"
        count += 1
    return candidate
