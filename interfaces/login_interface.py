# interface/login_interface.py
import sys
from pathlib import Path
import cv2
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # Pour style moderne


# --- Ajout du path racine pour importer models/ ---
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from models.face_encoder import FaceEncoder
from models.face_detector import FaceDetector
from utils.preprocessing import crop_face  # Si tu as utils/preprocessing.py
from database.database_manager import DatabaseManager

encoder = FaceEncoder()
detector = FaceDetector(detector_type="haar")
db_manager = DatabaseManager()

# --- Répertoire de sauvegarde ---
output_dir = root / "captured_faces"
output_dir.mkdir(exist_ok=True)
face_count = len(list(output_dir.glob("face_*"))) + 1

# --- Fonction capture caméra ---
def open_camera_and_capture(username=None):
    global face_count
    
    if not username or username.strip() == "":
        messagebox.showerror("Erreur", "Veuillez entrer votre nom")
        return
    
    # Créer l'utilisateur dans Supabase
    user_id = db_manager.create_user(username.strip())
    if not user_id:
        messagebox.showerror("Erreur", "Impossible de créer l'utilisateur")
        return
    
    messagebox.showinfo("Succès", f"Utilisateur créé : {username}")
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Erreur", "Impossible d'ouvrir la caméra")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Détection des visages
        faces, _ = detector.detect_faces(frame)

        # Dessin des cadres
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Visages detectes: {len(faces)}  | 'c'=capture, ESC=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.imshow("Authentification Caméra", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if len(faces) == 0:
                messagebox.showwarning("Attention", "Aucun visage détecté !")
            else:
                # On sauvegarde le visage encadré
                for (x, y, w, h) in faces:
                    face_img = crop_face(frame, (x, y, w, h), margin=10)
                    if face_img is not None:
                        file_path = output_dir / f"face_{face_count}.jpg"
                        cv2.imwrite(str(file_path), face_img)
                        messagebox.showinfo("Capture", f"Image sauvegardée → {file_path}")

                        # --- Encoder le visage et enregistrer dans Supabase ---
                        embedding = encoder.encode_face(str(file_path), user_id=user_id)
                        if embedding is not None:
                            messagebox.showinfo("Succès", "Embedding généré et enregistré dans la base de données !")
                        face_count += 1
            break

        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- FENÊTRE TKINTER STYLÉE ----------
root_tk = tk.Tk()
root_tk.title("🔐 Face Authentication AI")
root_tk.geometry("420x420")
root_tk.configure(bg="#f0f0f5")

# Style général
style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=10)
style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), background="#f0f0f5")
style.configure("Normal.TLabel", font=("Segoe UI", 11), background="#f0f0f5")

# Titre
ttk.Label(root_tk, text="FACE SECURITY", style="Title.TLabel").pack(pady=15)

# Message
ttk.Label(root_tk, text="Entrez votre nom et lancez la capture", style="Normal.TLabel").pack(pady=5)

# Champ d'entrée du nom d'utilisateur
username_label = ttk.Label(root_tk, text="Nom d'utilisateur :", style="Normal.TLabel")
username_label.pack(pady=5)
username_entry = ttk.Entry(root_tk, width=30, font=("Segoe UI", 11))
username_entry.pack(pady=5, padx=20)

# Bouton capture avec fonction wrapper
def start_capture():
    username = username_entry.get()
    open_camera_and_capture(username)

ttk.Button(root_tk, text="📸 Lancer la Capture", command=start_capture).pack(pady=25)

# Affichage du status
status_label = ttk.Label(root_tk, text="En attente…", style="Normal.TLabel")
status_label.pack(pady=10)

# Footer
ttk.Label(root_tk, text="© 2025 - Secure AI Systems", style="Normal.TLabel").pack(side="bottom", pady=10)

root_tk.mainloop()
