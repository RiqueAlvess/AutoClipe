import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import os
import sys
from download import baixar_video_e_audio  # Importa direto

class AutoClipeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AutoClipe – Cortes com IA")
        self.geometry("500x430")
        self.configure(bg="#f2f2f2")
        self.resizable(False, False)

        self._estilizar_widgets()
        self._criar_interface()

    def _estilizar_widgets(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 10, "bold"),
                        foreground="black", background="#87CEFA", padding=6)
        style.map("TButton", background=[("active", "#00BFFF")])
        style.configure("TEntry", padding=6)
        style.configure("TLabel", background="#f2f2f2", font=("Segoe UI", 10, "bold"))

    def _criar_interface(self):
        label_url = ttk.Label(self, text="URL DO VÍDEO:")
        label_url.pack(pady=(20, 5))

        self.entry_url = ttk.Entry(self, width=50)
        self.entry_url.pack(pady=5)

        self.btn_gerar = ttk.Button(self, text="GERAR CORTES", command=self._iniciar_pipeline_thread)
        self.btn_gerar.pack(pady=10)

        separador = ttk.Separator(self, orient='horizontal')
        separador.pack(fill='x', pady=15)

        frame_botoes = tk.Frame(self, bg="#f2f2f2")
        frame_botoes.pack(pady=10)

        ttk.Button(frame_botoes, text="ABRIR CORTES", command=lambda: self._abrir_diretorio("cortes")).grid(row=0, column=0, padx=10, pady=5)
        ttk.Button(frame_botoes, text="ABRIR ÁUDIOS", command=lambda: self._abrir_diretorio("Audios")).grid(row=0, column=1, padx=10, pady=5)
        ttk.Button(frame_botoes, text="EDITAR CONTEXTO", command=lambda: self._abrir_arquivo("keywords.txt")).grid(row=0, column=2, padx=10, pady=5)

        self.btn_treinar = ttk.Button(self, text="TREINAR I.A", command=lambda: self._executar_script("treinovoz.py"))
        self.btn_treinar.pack(pady=20)

        self.status = ttk.Label(self, text="", anchor="center", font=("Segoe UI", 9, "italic"))
        self.status.pack(pady=5)

    def _iniciar_pipeline_thread(self):
        threading.Thread(target=self._executar_pipeline, daemon=True).start()

    def _executar_pipeline(self):
        url = self.entry_url.get().strip()
        if not url:
            self._mostrar_erro("Por favor, insira uma URL.")
            return

        try:
            self._atualizar_status("🔄 Baixando vídeo e áudio...")
            mp4, wav = baixar_video_e_audio(url)

            self._atualizar_status("✂️ Gerando cortes...")
            subprocess.run([sys.executable, "gerar_corte.py"], check=True)

            self._atualizar_status("📊 Aplicando filtros finais...")
            subprocess.run([sys.executable, "good_clip.py"], check=True)

            self._atualizar_status("✅ Finalizado com sucesso!")
        except Exception as e:
            self._mostrar_erro(f"Erro: {e}")

    def _atualizar_status(self, msg):
        self.status.config(text=msg)

    def _mostrar_erro(self, msg):
        self.status.config(text="❌ Ocorreu um erro.")
        messagebox.showerror("Erro", msg)

    def _abrir_diretorio(self, pasta):
        if not os.path.isdir(pasta):
            self._mostrar_erro(f"Pasta não encontrada: {pasta}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(pasta)
            elif sys.platform.startswith("darwin"):
                subprocess.call(["open", pasta])
            else:
                subprocess.call(["xdg-open", pasta])
        except Exception as e:
            self._mostrar_erro(str(e))

    def _abrir_arquivo(self, caminho):
        if not os.path.isfile(caminho):
            self._mostrar_erro(f"Arquivo não encontrado: {caminho}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(caminho)
            elif sys.platform.startswith("darwin"):
                subprocess.call(["open", caminho])
            else:
                subprocess.call(["xdg-open", caminho])
        except Exception as e:
            self._mostrar_erro(str(e))

    def _executar_script(self, script):
        if not os.path.isfile(script):
            self._mostrar_erro(f"Script não encontrado: {script}")
            return
        try:
            subprocess.Popen([sys.executable, script])
        except Exception as e:
            self._mostrar_erro(str(e))

if __name__ == "__main__":
    app = AutoClipeApp()
    app.mainloop()
