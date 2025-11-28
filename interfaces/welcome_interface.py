# interfaces/welcome_interface.py
import tkinter as tk
from tkinter import ttk

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700


def show_welcome_screen(username, parent=None):
    """Affiche une fenêtre de bienvenue stylée et grande."""
    win = tk.Toplevel(parent) if parent else tk.Tk()
    win.title("Bienvenue")
    win.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    win.configure(bg="#ffffff")

    # Style simple
    title = tk.Label(win, text=f"Bienvenue {username} \u2728", font=("Segoe UI", 30, "bold"), bg="#ffffff", fg="#2E7D32")
    title.pack(pady=60)

    sub = tk.Label(win, text="Vous êtes connecté avec succès !", font=("Segoe UI", 18), bg="#ffffff")
    sub.pack(pady=10)

    # Image emoji large
    emoji = tk.Label(win, text="😊", font=("Segoe UI", 80), bg="#ffffff")
    emoji.pack(pady=20)

    def close_welcome():
        try:
            win.destroy()
        except Exception:
            pass
        if parent:
            try:
                parent.deiconify()
            except Exception:
                pass

    btn = tk.Button(win, text="Fermer", command=close_welcome, font=("Segoe UI", 14, "bold"), bg="#1976D2", fg="white", padx=20, pady=10)
    btn.pack(pady=40)

    # Rendre la fenêtre modale
    try:
        win.transient(parent)
        win.grab_set()
    except Exception:
        pass

    return win
