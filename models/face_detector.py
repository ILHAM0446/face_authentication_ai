import cv2
import dlib
import os
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

class FaceDetector:
    def __init__(self, detector_type="haar", landmark_path=None):
        """
        detector_type : "haar" ou "dlib"
        landmark_path : chemin vers shape_predictor_68_face_landmarks.dat (optionnel)
        """
        self.detector_type = detector_type
        
        if detector_type == "haar":
            local_haar = Path(__file__).resolve().parent / "haarcascade_frontalface_default.xml"
            tried = []

            if local_haar.exists():
                tried.append(str(local_haar))
                self.detector = cv2.CascadeClassifier(str(local_haar))
            else:
                self.detector = None

            if self.detector is None or self.detector.empty():
                cv_detector = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
                tried.append(str(cv_detector))
                self.detector = cv2.CascadeClassifier(str(cv_detector))

            if self.detector.empty():
                raise ValueError(f"Impossible de charger haarcascade.\nChemins testés : {tried}")

        elif detector_type == "dlib":
            self.detector = dlib.get_frontal_face_detector()

        else:
            raise ValueError("detector_type doit être 'haar' ou 'dlib'.")

        self.predictor = None
        if landmark_path and os.path.exists(landmark_path):
            self.predictor = dlib.shape_predictor(landmark_path)

    def detect_faces(self, image):
        """
        Détecte les visages dans une image.
        Retour : (faces, landmarks)
            faces : [(x, y, w, h)]
            landmarks : [shape_68_points]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = []
        landmarks = []

        if self.detector_type == "haar":
            detected = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            for (x, y, w, h) in detected:
                faces.append((x, y, w, h))

        else:
            detected = self.detector(gray, 1)
            for d in detected:
                faces.append((d.left(), d.top(), d.width(), d.height()))

        if self.predictor:
            for (x, y, w, h) in faces:
                rect = dlib.rectangle(x, y, x + w, y + h)
                shape = self.predictor(gray, rect)
                landmarks.append(shape)

        return faces, landmarks