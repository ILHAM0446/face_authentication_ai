import sys
import os
import cv2
from pathlib import Path
import time
import platform
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')


# Ajouter le dossier racine au path Python
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from models.face_detector import FaceDetector
from utils.preprocessing import crop_face


def open_camera():
    """Essaye tous les backends Windows et retourne la caméra opérationnelle"""

    backends = [
        cv2.CAP_DSHOW,     # Le plus fiable sur Windows
        cv2.CAP_MSMF,      # Parfois instable
        cv2.CAP_VFW,       # Ancien backend
        cv2.CAP_ANY        # Fallback
    ]

    for backend in backends:
        print(f"\n=== Essai du backend {backend} ===")

        cap = cv2.VideoCapture(0, backend)

        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la caméra avec ce backend")
            continue

        # Test de lecture
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ Caméra ouverte mais frame noire / vide")
            cap.release()
            continue

        print("✅ Backend fonctionnel !")
        return cap

    print("❌ Aucun backend n'a pu ouvrir correctement la caméra.")
    return None


def main():
    print("🔍 Ouverture de la caméra...")

    cap = open_camera()

    if cap is None:
        return

    # Réglages caméra
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    detector = FaceDetector(detector_type="haar")
    
    # --- FIXED: chemin correct vers captured_faces dans face_authentication_ai ---
    output_dir = root / "captured_faces"
    output_dir.mkdir(exist_ok=True)
    
    # Initialise face_count au dernier fichier existant + 1
    existing = list(output_dir.glob("face_*"))
    max_idx = 0
    for p in existing:
        m = re.search(r"face_(\d+)", p.name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    face_count = max_idx + 1

    print("\n🎥 Caméra démarrée")
    print("➡ 'c' pour capturer un visage")
    print("➡ ESC pour quitter\n")

    try:
        while True:

            ret, frame = cap.read()

            if not ret or frame is None:
                print("⚠️ Frame vide reçue — la caméra ne répond plus.")
                time.sleep(0.05)
                continue

            # Détection des visages
            faces, _ = detector.detect_faces(frame)

            # Dessin rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Visages: {len(faces)}  |  'c'=capture  'ESC'=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            cv2.imshow("Camera Face Detection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("👋 Fermeture...")
                break

            if key == ord("c"):
                if len(faces) == 0:
                    print("⚠️ Aucun visage détecté")
                else:
                    for (x, y, w, h) in faces:
                        face_img = crop_face(frame, (x, y, w, h), margin=10)
                        if face_img is not None:
                            file_path = output_dir / f"face_{face_count}.jpg"
                            cv2.imwrite(str(file_path), face_img)
                            print(f"📸 Visage sauvegardé → {file_path}")
                            face_count += 1

    except KeyboardInterrupt:
        print("\n⏸️ Interruption utilisateur")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✔ Caméra fermée. Fin du programme.")


if __name__ == "__main__":
    main()

