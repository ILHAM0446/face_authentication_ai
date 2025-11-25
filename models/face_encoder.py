"""
models/face_encoder.py

Responsabilités :
- Charger le modèle d'encodage facial dlib (dlib_face_recognition_resnet_model_v1.dat)
- Fournir get_embedding(face_image) -> np.ndarray (128-d)
- Fournir des utilitaires de comparaison : euclidean_distance, cosine_similarity, is_match, compare_embeddings

Hypothèses :
- `face_image` est une image numpy (H,W,3) représentant le visage déjà recadré.
- L'image peut être au format BGR (OpenCV) ou RGB ; la fonction tentera de détecter et convertir.

Dépendances : dlib, numpy, cv2

Auteur : Membre 2 (suggestion de template)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import dlib
import cv2

# --- Configuration par défaut ---
DEFAULT_DLIB_MODEL = "models/dlib_face_recognition_resnet_model_v1.dat"
# Seuil par défaut pour une correspondance (peut être ajusté après validation)
DEFAULT_EUCLIDEAN_THRESHOLD = 0.6


class FaceEncoder:
    """Classe encapsulant le modèle d'encodage facial dlib.

    Exemple d'utilisation :
        encoder = FaceEncoder(model_path)
        emb = encoder.get_embedding(face_image)
    """

    def __init__(self, model_path: str = DEFAULT_DLIB_MODEL):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle introuvable : {model_path}. Téléchargez le fichier et placez-le ici.")

        # Charge le modèle d'encodage (ResNet dlib)
        self._fr_model = dlib.face_recognition_model_v1(model_path)

        # dlib nécessite des landmarks pour l'alignement si on applique compute_face_descriptor
        # Cependant, on suppose que l'image donnée est déjà un visage recadré et raisonnablement aligné.
        # Si vous voulez une meilleure robustesse, fournissez un shape_predictor et appliquez la transformation.
        self._shape_predictor: Optional[dlib.shape_predictor] = None

    def set_shape_predictor(self, predictor_path: str):
        """(Optionnel) Charger un shape_predictor pour extraire les landmarks si vous souhaitez
        appliquer un alignement plus précis avant l'encodage.
        """
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"shape_predictor introuvable : {predictor_path}")
        self._shape_predictor = dlib.shape_predictor(predictor_path)

    def _ensure_rgb(self, img: np.ndarray) -> np.ndarray:
        """Retourne une image RGB 8-bit à partir d'une image fournie (BGR ou RGB).

        Détection heuristique simple : si la moyenne du canal 0 est > moyenne canal 2, on suppose BGR.
        (OpenCV utilise BGR par défaut.)
        """
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("face_image doit être une image couleur (H,W,3)")

        img = img.astype(np.uint8)
        # Heuristique pour détecter BGR vs RGB
        if img[..., 0].mean() > img[..., 2].mean():
            # Probablement BGR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img.copy()
        return img_rgb

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Calcule et renvoie l'embedding 128-d pour une image de visage recadrée.

        Args:
            face_image: numpy array (H,W,3) représentant le visage déjà recadré.
        Returns:
            embedding: np.ndarray de forme (128,) dtype=float32
        """
        img_rgb = self._ensure_rgb(face_image)

        # Convertir en image dlib
        img_dlib = dlib.load_rgb_image("temp_conversion_not_used.jpg") if False else None
        # Méthode alternative : construire un objet dlib.image_from_numpy
        # dlib accepte directement les numpy arrays en tant que images pour compute_face_descriptor

        # Si nous avons un shape_predictor, on calcule dlib.full_object_detection
        if self._shape_predictor is not None:
            # Détection simple : dlib.rectangle couvrant l'image entière (on suppose visage recadré)
            h, w = img_rgb.shape[:2]
            rect = dlib.rectangle(left=0, top=0, right=w - 1, bottom=h - 1)
            shape = self._shape_predictor(img_rgb, rect)
            vec = self._fr_model.compute_face_descriptor(img_rgb, shape)
        else:
            # Sans shape predictor, dlib propose une surcharge qui prend directement l'image
            # Cependant compute_face_descriptor requiert généralement un shape. Une technique
            # courante pour des images déjà recadrées est d'appeler la version qui accepte
            # un full_object_detection. Si on n'a pas de shape, on peut approximer des landmarks
            # en plaçant des points standards — mais ici on utilisera une astuce :
            # convertir l'image en dlib array et appeler une méthode alternative.

            # Créons un rectangle couvrant toute l'image et utilisons la méthode par défaut.
            h, w = img_rgb.shape[:2]
            rect = dlib.rectangle(left=0, top=0, right=w - 1, bottom=h - 1)

            # Si compute_face_descriptor refuse sans shape, on peut utiliser une approximation
            # simple : obtenir une shape_predictor par défaut si disponible (non chargé ici).
            # Pour garantir robustesse, on va approximer des landmarks centraux — ce n'est pas
            # optimal pour la production mais fonctionne si le visage est frontal et centré.

            # Tentative d'obtenir un shape si dlib a un predicteur interne (rare) - sinon on
            # convertit l'image en matrice et applique la méthode compute_face_descriptor en
            # fournissant un `None` pour shape n'est pas supporté. Donc on calcule un shape "artefact".

            # Générer un shape factice aligné au centre :
            shape = _approximate_shape_from_rect(rect)
            vec = self._fr_model.compute_face_descriptor(img_rgb, shape)

        emb = np.array(vec, dtype=np.float32)
        # Normaliser l'embedding (optionnel mais souvent utile pour cosine)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb


def _approximate_shape_from_rect(rect: dlib.rectangle) -> dlib.full_object_detection:
    """Crée une full_object_detection factice à partir d'un rectangle.

    Utilisé uniquement quand aucun shape_predictor n'est fourni. Cette approximation crée
    quelques points repères aux coins et centre du rectangle. Ce n'est pas précis mais
    permet d'obtenir un embedding lorsque l'image est un visage centré.
    """
    # dlib.full_object_detection attend un objet shape contenant 68 points si utilisé avec le modèle 68
    # Construire une structure compatible : nous allons créer 5 points standards (ce n'est pas 68)
    # Malheureusement dlib.full_object_detection est un type c++ compliqué à construire manuellement.
    # Pour contournement simple, utilisons la fonction dlib.rectangle_as_rectangle et puis un petit wrapper.
    # NOTE: Construire un vrai full_object_detection en Python est difficile sans shape_predictor.
    # Ici nous allons lever une erreur explicite pour inciter l'utilisateur à charger un shape_predictor.
    raise RuntimeError(
        "Aucun shape_predictor chargé — pour des résultats fiables, appelez set_shape_predictor(predictor_path) "
        "avec 'shape_predictor_68_face_landmarks.dat' avant d'appeler get_embedding()."
    )


# --- Fonctions utilitaires ---

def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Distance Euclidienne entre deux embeddings.

    Plus la distance est petite, plus les embeddings (et donc visages) sont proches.
    """
    emb1 = np.asarray(emb1, dtype=np.float32)
    emb2 = np.asarray(emb2, dtype=np.float32)
    return float(np.linalg.norm(emb1 - emb2))


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Similarité cosinus entre deux embeddings (1 = identique, -1 = opposé).
    Les embeddings doivent idéalement être normalisés.
    """
    emb1 = np.asarray(emb1, dtype=np.float32)
    emb2 = np.asarray(emb2, dtype=np.float32)
    denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2)) + 1e-10
    return float(np.dot(emb1, emb2) / denom)


def is_match(emb1: np.ndarray, emb2: np.ndarray, threshold: float = DEFAULT_EUCLIDEAN_THRESHOLD) -> bool:
    """Retourne True si la distance euclidienne est inférieure au seuil.

    On utilise la distance euclidienne par défaut, car c'est l'approche la plus répandue avec dlib.
    """
    return euclidean_distance(emb1, emb2) <= threshold


def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> dict:
    """Compare deux embeddings et renvoie un dict contenant distance et similarité.

    Exemple de retour : {'euclidean': 0.45, 'cosine': 0.98, 'match': True}
    """
    euc = euclidean_distance(emb1, emb2)
    cos = cosine_similarity(emb1, emb2)
    match = euc <= DEFAULT_EUCLIDEAN_THRESHOLD
    return {"euclidean": euc, "cosine": cos, "match": match}


# --- Petit démonstrateur (ne s'exécute que si lancé directement) ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tester face_encoder sur une image de visage recadrée")
    parser.add_argument("--model", default=DEFAULT_DLIB_MODEL, help="Chemin vers dlib_face_recognition_resnet_model_v1.dat")
    parser.add_argument("--shape", default=None, help="(Optionnel) chemin vers shape_predictor_68_face_landmarks.dat")
    parser.add_argument("img", help="Image de visage recadrée (jpg/png)")
    args = parser.parse_args()

    encoder = FaceEncoder(args.model)
    if args.shape:
        encoder.set_shape_predictor(args.shape)

    img = cv2.imread(args.img)
    if img is None:
        raise SystemExit(f"Impossible de lire l'image {args.img}")

    emb = encoder.get_embedding(img)
    print("Embedding (norme, dim):", np.linalg.norm(emb), emb.shape)
