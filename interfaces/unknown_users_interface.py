import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path
import os

def show_unknown_users_screen():
    root = tk.Toplevel()
    root.title("Utilisateurs inconnus")
    root.geometry("900x600")
    root.configure(bg="#f4f4f8")

    main_frame = tk.Frame(root, bg="#f4f4f8")
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)

    top_frame = tk.Frame(main_frame, bg="#f4f4f8")
    top_frame.pack(fill="x", pady=(0, 10))

    title = tk.Label(top_frame, text="Tentatives d'accès non reconnues", font=("Segoe UI", 22, "bold"), bg="#f4f4f8")
    title.pack(side="left", expand=True)

    btn_close = tk.Button(top_frame, text="Fermer", command=root.destroy, font=("Segoe UI", 11, "bold"), bg="#D32F2F", fg="white", padx=15, pady=5)
    btn_close.pack(side="right")

    content_frame = tk.Frame(main_frame, bg="#f4f4f8")
    content_frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(content_frame, bg="#f4f4f8", highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True)

    scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")

    canvas.configure(yscrollcommand=scrollbar.set)
    container = tk.Frame(canvas, bg="#f4f4f8")
    canvas.create_window((0, 0), window=container, anchor="nw")

    folder = Path("unknown_users")
    if not folder.exists():
        folder.mkdir(exist_ok=True)
    
    files = [f for f in folder.glob("*.*") if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    images = []

    if not files:
        no_data = tk.Label(container, text="Aucun visage non reconnu", font=("Segoe UI", 16), bg="#f4f4f8", fg="#999")
        no_data.pack(pady=50)
    else:
        for f in files:
            try:
                img = Image.open(f)
                img = img.resize((180, 180))
                img_tk = ImageTk.PhotoImage(img)
                images.append(img_tk)
                card = tk.Frame(container, bg="white", bd=2, relief="ridge")
                card.pack(pady=10)
                tk.Label(card, image=img_tk, bg="white").pack()
                tk.Label(card, text=f.name, bg="white", font=("Segoe UI", 12)).pack(pady=5)
            except Exception as e:
                print(f"Erreur lors du chargement de {f}: {e}")

    container.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

    root.mainloop()
