# Système de Reconnaissance Faciale

Ce document présente l'ensemble du fonctionnement du **projet de reconnaissance faciale**, incluant :

* les bibliothèques utilisées,
* les modèles IA,
* le pipeline complet d'inscription et de login,
* les interfaces,
* ainsi qu’un résumé global.

---

# 1. Introduction du Projet

Notre projet est un système complet de **reconnaissance faciale** permettant :

* l’**inscription** d’un utilisateur via son visage,
* la **connexion automatique** par reconnaissance faciale,
* la **prédiction de l’âge et du genre**,
* la gestion des **utilisateurs inconnus**,
* le stockage des **embeddings faciaux dans Supabase**.

Objectif :
Identifier automatiquement un utilisateur à partir de son visage en utilisant des modèles avancés d’IA.

---

# 2. Bibliothèques Utilisées

## Vision par ordinateur

* **OpenCV (cv2)** : lecture d’images, dessin, conversions.
* **dlib** : détection faciale, landmarks 68 points et embeddings 128D.

##  Manipulation des données

* **NumPy** : vecteurs, calculs, embeddings.
* **os / pathlib** : gestion des fichiers et chemins.

## Deep Learning

* **TensorFlow / Keras** :

  * chargement du modèle âge/genre,
  * entraînement et fine-tuning,
  * métriques personnalisées,
  * data augmentation.

## Base de données

* **Supabase** : stockage des utilisateurs, embeddings, inconnus.

---

# 3. Modèles Utilisés

## 3.1 Modèle de Reconnaissance Faciale (Dlib)

Dans `face_encoder.py` :

* `shape_predictor_68_face_landmarks.dat`
* `dlib_face_recognition_resnet_model_v1.dat`

Fonctions :

1. Détection du visage
2. Extraction des landmarks
3. Génération d’un **embedding 128 dimensions**

---

## 3.2 Modèle Âge & Genre (TensorFlow / Keras)

Dans `age_gender_model.py` :

* Input : image 224×224
* Output :

  * `age_output` → âge (régression)
  * `gender_output` → probabilités (H/F)

### 🔹 Phase 1 – Warm-up

* MobileNetV2 gelé
* Entraînement de la tête du réseau uniquement

### 🔹 Phase 2 – Fine-Tuning

* Dégel des **40 dernières couches**
* Faible learning rate
* Affinage des performances

Métriques :

* F1-score
* Précision / Rappel
* MAE pour l’âge

---

# 4. Fonctionnement du Système

# 4.1 Inscription (Register)

### ✔️ Détection du visage

Caméra ouverte → rectangle vert → prédiction âge/genre.

### ✔️ Prédictions stabilisées

On capture **5 images successives** :

* Âge final = moyenne
* Genre final = classe majoritaire

### ✔️ Capture

L’utilisateur appuie sur **C**.

### ✔️ Stockage

1. Encodage en embedding 128D
2. Envoi dans Supabase via `DatabaseManager`
3. Création du compte

---

# 4.2 Connexion (Login)

1. Détection + prédiction âge/genre
2. Capture
3. Génération d’un embedding
4. Récupération des embeddings stockés
5. Calcul de la distance (euclidienne ou cosine)

### ✔️ Si distance < seuil :

Utilisateur reconnu → accès accordé

### ✔️ Sinon :

Utilisateur inconnu → image enregistrée dans `unknown_users`

---

# 5. Interfaces du Projet

### 1️⃣ `login_interface.py`

Interface de connexion par reconnaissance faciale.

### 2️⃣ `registre_interface.py`

Interface d'inscription.

### 3️⃣ `camera_interface.py`

Affichage caméra + rectangles + captures.

### 4️⃣ `unknown_users_interface.py`

Liste des visages inconnus détectés.

### 5️⃣ `welcome_interface.py`

Page d’accueil après authentification réussie.

