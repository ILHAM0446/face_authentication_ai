import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from pathlib import Path
import sys
import os

root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))

def open_register():
    script = root_path / "interfaces" / "registre_interface.py"
    if script.exists():
        import subprocess
        subprocess.Popen([sys.executable, str(script)])
    else:
        messagebox.showerror("Erreur", f"Fichier non trouvé: {script}")

def open_login():
    script = root_path / "interfaces" / "login_interface.py"
    if script.exists():
        import subprocess
        subprocess.Popen([sys.executable, str(script)])
    else:
        messagebox.showerror("Erreur", f"Fichier non trouvé: {script}")

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700

root = tk.Tk()
root.title("🔐 Face Authentication AI")
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
root.configure(bg="#324A5F")

style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 13, "bold"), padding=10)
style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), background="#324A5F", foreground="white")

ttk.Label(root, text="Système d'Authentification Faciale", style="Title.TLabel").pack(pady=80)

reg_btn = ttk.Button(root, text="➕ Register", command=open_register, width=50)
reg_btn.pack(pady=10)
reg_btn.configure(style="TButton")

log_btn = ttk.Button(root, text="🔑 Login", command=open_login, width=50)
log_btn.pack(pady=10)
log_btn.configure(style="TButton")

quit_btn = tk.Button(root, text="❌ Quitter", font=("Segoe UI", 13, "bold"),
                     bg="#D9534F", fg="white", width=30, command=root.quit)
quit_btn.pack(pady=20)

root.mainloop()
