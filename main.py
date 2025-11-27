import sys
import os
import cv2

# === Ajouter le dossier models au PATH ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
sys.path.append(MODELS_DIR)

from age_gender_model import AgeGenderPredictor

# Chemin vers ton modèle .keras
model_path = os.path.join(MODELS_DIR, "age_gender_model_final_complete.keras")

# Charger le modèle
predictor = AgeGenderPredictor(model_path)

# Charger une image exemple
img_path = os.path.join(CURRENT_DIR, "maroua.jpg")
image = cv2.imread(img_path)

# Exemple de face_box (à remplacer par ton détecteur)
face_box = (50, 50, 200, 200)
x, y, w, h = face_box
face = image[y:y+h, x:x+w]

# Prédiction
age, gender, confidence = predictor.predict(face)

# Affichage
print(predictor.get_full_prediction_message(age, gender, confidence))
