# 🤖 Face Authentication AI

## 🎯 Description
Ce projet a pour objectif de développer un **système d’authentification faciale intelligent** capable de :
- Détecter le visage d’un utilisateur via caméra.
- Estimer **l’âge** et le **genre** du visage détecté.
- Vérifier si le visage correspond à un utilisateur enregistré.
- Accorder ou refuser l’accès selon la correspondance.
- Enregistrer les nouveaux visages non reconnus dans une base de données MySQL.

Projet réalisé en **Python** par une équipe de 4 membres dans le cadre d’un projet d’intelligence artificielle.

---

## 🧩 Fonctionnalités principales
- 📷 Détection faciale en temps réel (OpenCV)  
- 🧠 Encodage et reconnaissance de visages (Face Recognition)  
- 👤 Prédiction d’âge et de genre (DeepFace / modèle pré-entraîné)  
- 🗄️ Gestion et stockage des utilisateurs dans MySQL  
- 🖥️ Interface caméra et panneau administrateur  

---

## 🧱 Structure du projet

face_authentication_ai/
│
├── main.py # Point d'entrée principal - (Chef de projet)
│
├── models/ # Modèles IA
│ ├── face_detector.py # Détection du visage - (Membre 1)
│ ├── face_encoder.py # Encodage et comparaison de visages - (Membre 2)
│ └── age_gender_model.py # Prédiction de l'âge et du genre - (Membre 4)
│
├── core/ # Logique métier
│ ├── authentication_system.py # Reconnaissance et gestion d'accès - (Membre 3)
│ └── user_manager.py # Gestion des utilisateurs - (Membre 2)
│
├── database/ # Gestion MySQL
│ └── database_manager.py # Connexion, création tables, enregistrement - (Membre 3)
│
├── interface/ # Interfaces utilisateur
│ ├── camera_interface.py # Flux caméra + capture du visage - (Membre 1)
│ └── admin_interface.py # Affichage des connexions et infos - (Membre 4)
│
├── data/ # Données locales
│ ├── users/ # Images des utilisateurs connus
│ └── unknown/ # Visages inconnus détectés
│
├── utils/ # Fonctions utilitaires
│ └── helpers.py # Logs, formatage, etc. - (Tous)
│
├── requirements.txt # Dépendances Python
├── README.md # Documentation du projet
└── .gitignore # Fichiers à ignorer

yaml
Copier le code

---

## 👥 Répartition des membres et responsabilités

| Membre | Rôle principal | Responsabilités |
|--------|----------------|----------------|
| **Membre 1** | Détection faciale | Implémentation de la détection avec OpenCV + interface caméra |
| **Membre 2** | Encodage & gestion utilisateurs | Génération et comparaison d’encodages + interface gestion utilisateurs |
| **Membre 3** | Reconnaissance & Base de données | Authentification, gestion MySQL, intégration globale du système |
| **Membre 4** | Âge & Genre + Interface admin | Modèle prédictif âge/genre et affichage dans panneau admin |
| **Chef de projet** | Coordination | Supervision, intégration finale, vérification des modules |

---

## 🧠 Bibliothèques principales

| Type | Bibliothèques |
|------|----------------|
| Vision | `opencv-python` |
| Reconnaissance | `face_recognition` |
| Prédiction âge/genre | `deepface` ou `cvlib` |
| Base de données | `mysql-connector-python` |
| Traitement | `numpy`, `pandas` |
| Interface | `tkinter` ou `streamlit` |

---

## ⚙️ Installation

1. **Cloner le projet :**
   ```bash
   git clone https://github.com/votre-compte/face_authentication_ai.git
   cd face_authentication_ai