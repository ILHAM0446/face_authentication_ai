import sys
import cv2
import re
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from collections import Counter

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))
sys.path.append(str(root / "interfaces"))

from welcome_interface import show_welcome_screen
from models.face_encoder import FaceEncoder
from models.face_detector import FaceDetector
from models.age_gender_model import AgeGenderPredictor
from utils.preprocessing import crop_face
from database.database_manager import DatabaseManager

encoder = FaceEncoder()
detector = FaceDetector(detector_type="haar")
db = DatabaseManager()

output_dir = root / "captured_faces"
output_dir.mkdir(exist_ok=True)

unknown_dir = root / "unknown_users"
unknown_dir.mkdir(exist_ok=True)


def save_unknown_face(face_img):
    try:
        existing = list(unknown_dir.glob("face_*.jpg"))
        max_idx = 0
        for p in existing:
            m = re.search(r"face_(\d+)", p.name)
            if m:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
        
        next_idx = max_idx + 1
        output_path = unknown_dir / f"face_{next_idx}.jpg"
        
        success = cv2.imwrite(str(output_path), face_img)
        if success:
            print(f"💾 Visage non reconnu sauvegardé → {output_path}")
            return str(output_path)
        else:
            print(f"⚠️ Erreur lors de la sauvegarde du visage non reconnu")
            return None
    except Exception as e:
        print(f"❌ Erreur save_unknown_face: {e}")
        return None


def recognize_user():
    # Initialiser le prédicteur d'âge et genre
    try:
        age_gender_predictor = AgeGenderPredictor(model_path=str(root / "models" / "age_gender_model_final_complete.keras"))
    except Exception as e:
        print(f"⚠️ Impossible de charger le modèle d'âge/genre : {e}")
        age_gender_predictor = None
    
    # Buffers pour stabiliser les prédictions
    age_buffer = []
    gender_buffer = []
    BUFFER_SIZE = 10
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Erreur", "Impossible d'ouvrir la caméra")
        return

    camera_active = True
    frame_count = 0
    prediction_finalized = False
    final_prediction = None
    no_face_count = 0
    
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            continue

        faces, _ = detector.detect_faces(frame)
        
        # Gestion de la réinitialisation si aucun visage n'est détecté
        if len(faces) == 0:
            no_face_count += 1
            if no_face_count > 20:  # Environ 1-2 secondes sans visage
                if prediction_finalized:
                    print("🔄 Réinitialisation de la prédiction (plus de visage détecté)")
                prediction_finalized = False
                final_prediction = None
                age_buffer = []
                gender_buffer = []
                no_face_count = 0
        else:
            no_face_count = 0

        # Si la prédiction n'est pas encore finalisée et qu'on détecte un visage
        if not prediction_finalized and len(faces) > 0:
            # Prendre le premier visage détecté
            x, y, w, h = faces[0]
            
            # Prédiction d'âge et genre
            if age_gender_predictor and age_gender_predictor.model is not None:
                try:
                    face_img = crop_face(frame, (x, y, w, h), margin_pct=0.4)
                    if face_img is not None:
                        age, gender, _ = age_gender_predictor.predict(face_img)
                        
                        if age is not None and gender is not None:
                            # Ajouter au buffer
                            age_buffer.append(age)
                            gender_buffer.append(gender)
                            
                            # Si on a atteint 5 échantillons, on fige le résultat
                            if len(age_buffer) >= 5:
                                # Calculer la moyenne pour l'âge
                                avg_age = int(sum(age_buffer) / len(age_buffer))
                                
                                # Pour le genre, prendre le plus fréquent
                                avg_gender = Counter(gender_buffer).most_common(1)[0][0]
                                
                                # Figer la prédiction
                                final_prediction = (avg_age, avg_gender)
                                prediction_finalized = True
                                print(f"🔒 Prédiction finalisée : {avg_age} ans, {avg_gender}")
                                
                except Exception as e:
                    print(f"⚠️ Erreur lors de la prédiction : {e}")
        
        # Afficher les rectangles et les prédictions pour tous les visages
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Afficher la prédiction finale si disponible
            if final_prediction is not None:
                avg_age, avg_gender = final_prediction
                text = f"{avg_age} ans, {avg_gender}"
                cv2.putText(frame, text, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Sinon afficher "Analyse..." si on est en cours
            elif not prediction_finalized:
                 cv2.putText(frame, "Analyse...", (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, "Appuyez sur 'c' pour capturer", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Login - Reconnaissance Faciale", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):

            if len(faces) == 0:
                messagebox.showwarning("Attention", "Aucun visage détecté !")
                continue

            x, y, w, h = faces[0]
            face_img = crop_face(frame, (x, y, w, h), margin=10)
            
            if face_img is None:
                messagebox.showerror("Erreur", "Impossible de découper le visage.")
                continue

            img_path = str(output_dir / "face_1.jpg")
            try:
                written = cv2.imwrite(img_path, face_img)
            except Exception as e:
                written = False
                print(f"Erreur lors de l'écriture du fichier : {e}")

            if not written:
                messagebox.showerror("Erreur", f"Impossible de sauvegarder l'image : {img_path}")
                print(f" cv2.imwrite a échoué pour {img_path}")
                continue

            print(f" Visage capturé et sauvegardé → {img_path}")

            emb = encoder.encode_face(img_path, user_id=None)
            if emb is None:
                messagebox.showerror("Erreur", "Impossible de lire le visage.")
                camera_active = False
                break

            rows = db.get_all_embeddings()
            if not rows:
                messagebox.showerror("Erreur", "Aucun utilisateur enregistré.")
                camera_active = False
                break

            best_score = 9999
            best_user_id = None

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

            if best_user_id and best_score < 0.45:

                user_info = db.get_user_by_id(best_user_id)

                if user_info and "name" in user_info:
                    username = user_info["name"]
                else:
                    username = "Utilisateur inconnu"
                show_welcome_screen(username)

            else:
                save_unknown_face(face_img)
                messagebox.showerror("Accès Refusé", "Utilisateur non reconnu")
            camera_active = False
            break

        elif key == 27: 
            camera_active = False
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Permet à OpenCV de traiter la fermeture


root_tk = tk.Tk()
root_tk.title("🔑 Login - Face Authentication")
root_tk.geometry("700x600")
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