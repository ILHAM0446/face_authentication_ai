import cv2
import numpy as np
import random
from keras.models import load_model

class AgeGenderPredictor:
    def __init__(self, model_path="age_gender_model_final_complete.keras"):
        if model_path:
            try:
                self.model = load_model(model_path, compile=False)
                print("Modèle chargé avec succès !")
            except Exception as e:
                print("Impossible de charger le modèle :", e)
                self.model = None
        else:
            self.model = None

        self.input_shape = (224, 224, 3)
        
        # Seuil de confiance pour inverser la prédiction (ex: 45%)
        self.confidence_threshold = 0.45

    def preprocess_face(self, face_img):
        if face_img is None:
            return None
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def predict(self, face_img):
        if self.model is None:
            raise ValueError("Aucun modèle chargé !")

        face_pixels = self.preprocess_face(face_img)
        if face_pixels is None:
            return None, None, None

        age_pred, gender_pred = self.model.predict(face_pixels, verbose=0)
        age = max(0, min(116, int(age_pred[0][0] * 116)))
        gender_prob = float(gender_pred[0][0])

        gender = "Femme" if gender_prob > 0.5 else "Homme"

        return age, gender, gender_prob