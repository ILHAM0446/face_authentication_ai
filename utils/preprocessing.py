import cv2
import numpy as np

def resize_image(image, size=(160, 160)):
    """Redimensionner l’image."""
    return cv2.resize(image, size)

def normalize_image(image):
    """Normalisation entre 0 et 1."""
    return image.astype("float32") / 255.0

def convert_to_rgb(image):
    """Convertir BGR → RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def crop_face(image, box, margin=10):
    """
    Découpe un visage avec une petite marge.
    box = (x, y, w, h)
    """
    x, y, w, h = box

    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, image.shape[1])
    y2 = min(y + h + margin, image.shape[0])

    return image[y1:y2, x1:x2]
