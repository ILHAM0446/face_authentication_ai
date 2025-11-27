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

        # === Condition spéciale : si certitude faible ET prédiction Homme → inverser ===
        if gender_prob < self.confidence_threshold and gender_prob < 0.5:
            gender = "Femme"
            gender_prob = 1 - gender_prob
        else:
            gender = "Femme" if gender_prob > 0.5 else "Homme"

        return age, gender, gender_prob

    def get_funny_age_comment(self, age):
        comments = [
            f"Je dirais que cette personne a environ {age} ans... mais elle a peut-être menti sur sa carte d'identité 😏",
            f"Environ {age} ans, mais avec un sourire de {age-5} ans! ✨",
            f"{age} ans d'expérience dans l'art d'être génial(e)! 🎨",
            f"Je vois {age} bougies sur le gâteau... ou peut-être {age+2}? 🎂",
            f"{age} ans de sagesse accumulée (et de memes regardés) 📚😄",
            f"Mon détecteur d'âge dit {age} ans, plus ou moins quelques années de jeunesse éternelle! 🌟",
        ]
        return random.choice(comments)

    def get_funny_gender_comment(self, gender, confidence):
        if gender == "Femme":
            comments = [
                f"Hmm... ça ressemble fortement à une femme. Confiance: {confidence*100:.0f}% 💄",
                f"Je détecte une femme! Probabilité: {confidence*100:.0f}%. Mon intuition est rarement fausse! 👩✨",
            ]
        else:
            comments = [
                f"Je vois un homme! Confiance: {confidence*100:.0f}% 💪🔥",
                f"C'est clairement un homme, avec {confidence*100:.0f}% de certitude! 🦸‍♂️",
            ]
        return random.choice(comments)

    def get_full_prediction_message(self, age, gender, confidence):
        age_msg = self.get_funny_age_comment(age)
        gender_msg = self.get_funny_gender_comment(gender, confidence)
        full_message = f"🎭 PRÉDICTION DÉTAILLÉE 🎭\n\n"
        full_message += f"📊 ÂGE: {age_msg}\n\n"
        full_message += f"👤 GENRE: {gender_msg}\n\n"
        full_message += f"⚠️ Attention: Ces prédictions sont faites avec humour et peuvent être imprécises!"
        return full_message
