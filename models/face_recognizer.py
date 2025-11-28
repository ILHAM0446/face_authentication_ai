import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.face_encoder import FaceEncoder
from database.database_manager import DatabaseManager
from utils.preprocessing import crop_face
from models.face_detector import FaceDetector

class FaceRecognizer:
    def __init__(self, threshold=0.45, metric="cosine"):
        """
        threshold : seuil utilisé selon la métrique
        metric : 'cosine' ou 'euclidean'
        """
        self.encoder = FaceEncoder()
        self.db = DatabaseManager()
        self.detector = FaceDetector(detector_type="haar")
        self.threshold = threshold
        self.metric = metric

        self.user_embeddings = None
        self.user_prototypes = None

    def _l2_normalize(self, v):
        v = np.array(v, dtype=float)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def compare_embeddings(self, emb1, emb2):
        """Compare deux embeddings selon la métrique choisie.

        - 'cosine' : retourne 1 - cosine_similarity (donc plus petit = plus similaire)
        - 'euclidean' : distance euclidienne classique
        """
        emb1 = np.array(emb1, dtype=float)
        emb2 = np.array(emb2, dtype=float)
        if self.metric == "cosine":
            e1 = self._l2_normalize(emb1)
            e2 = self._l2_normalize(emb2)
            return 1.0 - float(np.dot(e1, e2))
        else:
            return float(np.linalg.norm(emb1 - emb2))

    def _load_embeddings_from_db(self, force_reload=False):
        """Charge les embeddings depuis la DB et calcule des prototypes par utilisateur."""
        if self.user_embeddings is not None and not force_reload:
            return

        all_embeddings = self.db.get_all_embeddings()
        if not all_embeddings:
            self.user_embeddings = {}
            self.user_prototypes = {}
            return

        user_map = {}
        for row in all_embeddings:
            uid = row["user_id"]
            emb = np.array(row["embedding"], dtype=float)
            user_map.setdefault(uid, []).append(emb)

        prototypes = {}
        for uid, embs in user_map.items():
            mat = np.vstack(embs)
            mean = np.mean(mat, axis=0)
            prototypes[uid] = self._l2_normalize(mean) if self.metric == "cosine" else mean

        self.user_embeddings = user_map
        self.user_prototypes = prototypes

    def recognize(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            print("❌ Impossible de charger l'image capturée.")
            return None

        faces, _ = self.detector.detect_faces(img)
        if len(faces) == 0:
            print("❌ Aucun visage détecté pour la reconnaissance.")
            return None

        (x, y, w, h) = faces[0]
        face_img = crop_face(img, (x, y, w, h), margin=10)

        if face_img is None:
            print("❌ Impossible de découper le visage.")
            return None

        temp_path = Path("captured_faces/face_rec_temp.jpg")
        cv2.imwrite(str(temp_path), face_img)

        embedding = self.encoder.encode_face(str(temp_path))
        if embedding is None:
            print("❌ Embedding non généré.")
            return None

        self._load_embeddings_from_db()
        if not self.user_prototypes:
            print("⚠️ Aucun embedding enregistré dans la base.")
            return None

        query_emb = self._l2_normalize(embedding) if self.metric == "cosine" else embedding

        best_user = None
        best_score = float("inf")

        for uid, proto in self.user_prototypes.items():
            d_proto = self.compare_embeddings(query_emb, proto)

            embs = self.user_embeddings.get(uid, [])
            d_min = float("inf")
            for db_emb in embs:
                db_emb_cmp = self._l2_normalize(db_emb) if self.metric == "cosine" else db_emb
                d = self.compare_embeddings(query_emb, db_emb_cmp)
                if d < d_min:
                    d_min = d

            d_use = min(d_proto, d_min)

            if d_use < best_score:
                best_score = d_use
                best_user = uid

        print(f"→ Meilleure distance trouvée : {best_score} (metric={self.metric})")

        if best_score < self.threshold:
            print(f"✅ Visage reconnu ! Utilisateur = {best_user}")
            return best_user

        print("❌ Aucun match fiable.")
        return None
