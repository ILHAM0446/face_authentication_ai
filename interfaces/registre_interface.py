import sys
from pathlib import Path
import cv2
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from collections import Counter


root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from models.face_encoder import FaceEncoder
from models.face_detector import FaceDetector
from models.age_gender_model import AgeGenderPredictor
from utils.preprocessing import crop_face
from database.database_manager import DatabaseManager

encoder = FaceEncoder()
detector = FaceDetector(detector_type="haar")
db_manager = DatabaseManager()

output_dir = root / "captured_faces"
output_dir.mkdir(exist_ok=True)
face_count = len(list(output_dir.glob("face_*"))) + 1

def open_camera_and_capture(username=None):
    global face_count
    
    if not username or username.strip() == "":
        messagebox.showerror("Erreur", "Veuillez entrer votre nom")
        return
    
    user_id = db_manager.create_user(username.strip())
    if not user_id:
        messagebox.showerror("Erreur", "Impossible de créer l'utilisateur")
        return
    
    
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

        cv2.putText(frame, f"Visages detectes: {len(faces)}  | 'c'=capture, ESC=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.imshow("Authentification Caméra", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if len(faces) == 0:
                messagebox.showwarning("Attention", "Aucun visage détecté !")
            else:
                for (x, y, w, h) in faces:
                    face_img = crop_face(frame, (x, y, w, h), margin=10)
                    if face_img is not None:
                        file_path = output_dir / "face_1.jpg"
                        try:
                            written = cv2.imwrite(str(file_path), face_img)
                        except Exception as e:
                            written = False
                            print(f"❌ Erreur lors de l'écriture du fichier : {e}")

                        if written:
                            print(f"[DEBUG] Image écrite avec succès: {file_path}")

                            embedding = encoder.encode_face(str(file_path), user_id=user_id)
                            if embedding is not None:
                                print( "Embedding généré et enregistré dans la base de données !")
                        else:
                            messagebox.showerror("Erreur", "La capture n'a pas pu être sauvegardée (cv2.imwrite a renvoyé False)")
                            print("[DEBUG] cv2.imwrite a renvoyé False — vérifiez les permissions du dossier ou le format de l'image")
            camera_active = False
            break

        elif key == 27:
            camera_active = False
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

root_tk = tk.Tk()
root_tk.title("🔐 Face Authentication AI")
root_tk.geometry("700x600")
root_tk.configure(bg="#f0f0f5")

style = ttk.Style()
style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=10)
style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), background="#f0f0f5")
style.configure("Normal.TLabel", font=("Segoe UI", 11), background="#f0f0f5")


ttk.Label(root_tk, text="FACE SECURITY", style="Title.TLabel").pack(pady=15)

ttk.Label(root_tk, text="Entrez votre nom et lancez la capture", style="Normal.TLabel").pack(pady=5)

username_label = ttk.Label(root_tk, text="Nom d'utilisateur :", style="Normal.TLabel")
username_label.pack(pady=5)
username_entry = ttk.Entry(root_tk, width=30, font=("Segoe UI", 11))
username_entry.pack(pady=5, padx=20)

def start_capture():
    username = username_entry.get()
    open_camera_and_capture(username)

ttk.Button(root_tk, text="📸 Lancer la Capture", command=start_capture).pack(pady=25)

status_label = ttk.Label(root_tk, text="En attente…", style="Normal.TLabel")
status_label.pack(pady=10)

ttk.Label(root_tk, text="© 2025 - Secure AI Systems", style="Normal.TLabel").pack(side="bottom", pady=10)

root_tk.mainloop()