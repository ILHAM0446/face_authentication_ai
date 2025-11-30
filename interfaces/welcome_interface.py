import tkinter as tk
from tkinter import ttk

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700


def show_welcome_screen(username, parent=None):
    from unknown_users_interface import show_unknown_users_screen
    
    win = tk.Toplevel(parent) if parent else tk.Tk()
    win.title("Bienvenue")
    win.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    win.configure(bg="#ffffff")
    title = tk.Label(win, text=f"Bienvenue {username} \u2728", font=("Segoe UI", 30, "bold"), bg="#ffffff", fg="#2E7D32")
    title.pack(pady=60)

    sub = tk.Label(win, text="Vous êtes connecté avec succès !", font=("Segoe UI", 18), bg="#ffffff")
    sub.pack(pady=10)
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
    btn.pack(pady=20)

    btn_unknown = tk.Button(win, text="👤 Voir utilisateurs inconnus qui essayent d'entrer", command=lambda: show_unknown_users_screen(win), font=("Segoe UI", 12, "bold"), bg="#F57C00", fg="white", padx=15, pady=8)
    btn_unknown.pack(pady=10)
    try:
        win.transient(parent)
        win.grab_set()
    except Exception:
        pass

    return win
