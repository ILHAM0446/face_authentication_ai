# interface/login_interface.py
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from models.face_encoder import FaceEncoder
from models.face_detector import FaceDetector
from utils.preprocessing import crop_face
from database.database_manager import DatabaseManager

encoder = FaceEncoder()
detector = FaceDetector(detector_type="haar")
db = DatabaseManager()

output_dir = root / "captured_faces"
output_dir.mkdir(exist_ok=True)


def recognize_user():
    """Ouvre la caméra, capture un visage, génère un embedding et compare avec Supabase"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Erreur", "Impossible d'ouvrir la caméra")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        faces, _ = detector.detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, "Appuyez sur 'c' pour capturer", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Login - Reconnaissance Faciale", frame)
        key = cv2.waitKey(1) & 0xFF

        # ------------- CAPTURE -------------
        if key == ord("c"):

            if len(faces) == 0:
                messagebox.showwarning("Attention", "Aucun visage détecté !")
                continue

            x, y, w, h = faces[0]
            face_img = crop_face(frame, (x, y, w, h), margin=10)

            img_path = str(output_dir / "login_face.jpg")
            cv2.imwrite(img_path, face_img)

            # --- Génère embedding de l'utilisateur actuel ---
            emb = encoder.encode_face(img_path, user_id=None)
            if emb is None:
                messagebox.showerror("Erreur", "Impossible de lire le visage.")
                break

            # --- Récupère EMBEDDINGS + USER_ID en BD ---
            rows = db.get_all_embeddings()
            if not rows:
                messagebox.showerror("Erreur", "Aucun utilisateur enregistré.")
                break

            best_score = 9999
            best_user_id = None

            # --- Comparaison de distances ---
            for user in rows:
                try:
                    db_emb = np.array(user["embedding"])
                    uid = user["user_id"]

                    dist = np.linalg.norm(emb - db_emb)
                    if dist < best_score:
                        best_score = dist
                        best_user_id = uid
                except Exception as e:
                    print("[ERREUR]", e)
                    continue

            # --------- RECONNAISSANCE ----------
            if best_user_id and best_score < 0.45:

                # Récupération du username depuis la table users
                user_info = db.get_user_by_id(best_user_id)

                if user_info and "name" in user_info:
                    username = user_info["name"]
                else:
                    username = "Utilisateur inconnu"

                messagebox.showinfo("Succès",
                                    f"👤 Utilisateur reconnu : {username}\nDistance = {round(best_score, 3)}")
            else:
                messagebox.showerror("Accès Refusé", "Utilisateur non reconnu.")
            break

        elif key == 27:  # ESC -> Quitter
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- INTERFACE TK --------------------
root_tk = tk.Tk()
root_tk.title("🔑 Login - Face Authentication")
root_tk.geometry("400x300")
root_tk.configure(bg="#f0f0f5")

style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=10)
style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), background="#f0f0f5")

ttk.Label(root_tk, text="Login via Reconnaissance Faciale", style="Title.TLabel").pack(pady=20)

ttk.Button(root_tk, text="🔍 Lancer la Reconnaissance",
           command=recognize_user).pack(pady=30)

tk.Button(root_tk, text="❌ Quitter", bg="#D9534F", fg="white",
          font=("Segoe UI", 13, "bold"), command=root_tk.quit).pack(pady=20)

root_tk.mainloop()
