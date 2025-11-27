# models/face_encoder.py
import sys
import cv2
import dlib
import numpy as np
from pathlib import Path

# Ajouter le répertoire racine au path pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database.database_manager import DatabaseManager

class FaceEncoder:
    def __init__(self, model_path="models/dlib_face_recognition_resnet_model_v1.dat"):
        # Détecteur + modèle d'encodage
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.facerec = dlib.face_recognition_model_v1(model_path)
        # Initialiser DatabaseManager pour Supabase
        self.db_manager = DatabaseManager()

    def encode_face(self, img_path, user_id=None):
        """Prend une image et retourne l'embedding. Si user_id est fourni, sauvegarde dans Supabase via DatabaseManager."""
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            print("❌ Aucun visage détecté !")
            return None

        # prend le premier visage détecté
        shape = self.sp(gray, faces[0])
        face_descriptor = self.facerec.compute_face_descriptor(img, shape)
        
        embedding = np.array(face_descriptor)

        # Sauvegarder dans Supabase si user_id est fourni
        if user_id:
            success = self.db_manager.save_face_embedding(user_id, embedding)
            if success:
                print(f"✅ Embedding enregistré dans Supabase pour user_id: {user_id}")
            else:
                print(f"❌ Erreur lors de l'enregistrement de l'embedding")
        
        # Optionnel : sauvegarder aussi localement
        out_dir = Path("embeddings")
        out_dir.mkdir(exist_ok=True)
        file_name = out_dir / (Path(img_path).stem + ".npy")
        np.save(file_name, embedding)

        print(f"✅ Embedding enregistré localement → {file_name}")
        return embedding
