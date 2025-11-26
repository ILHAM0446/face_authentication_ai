# database/database_manager.py
"""
DatabaseManager pour face_security_db (MySQL).
- Utilise mysql-connector-python
- Stocke embeddings en JSON (chaîne)
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np
import mysql.connector
from mysql.connector import Error

# IMPORTANT : ne pas mettre les credentials en dur dans le code en production.
# Crée un fichier config.py avec DB_HOST, DB_USER, DB_PASSWORD, DB_NAME ou utilise variables d'environnement.
try:
    from database.config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
except Exception:
    # Valeurs par défaut (à remplacer)
    DB_HOST = "localhost"
    DB_USER = "root"
    DB_PASSWORD = ""
    DB_NAME = "face_security_db"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def serialize_encoding(encoding: np.ndarray) -> str:
    """Convertit un numpy array en JSON string pour stockage."""
    return json.dumps(encoding.tolist())


def deserialize_encoding(text: str) -> np.ndarray:
    """Convertit une JSON string en numpy array."""
    try:
        arr = json.loads(text)
        return np.array(arr, dtype=float)
    except Exception as e:
        logger.exception("Erreur lors de la désérialisation de l'encodage: %s", e)
        return np.array([])


class DatabaseManager:
    def __init__(self,
                 host: str = DB_HOST,
                 user: str = DB_USER,
                 password: str = DB_PASSWORD,
                 database: str = DB_NAME):
        self.config = dict(host=host, user=user, password=password, database=database, auth_plugin='mysql_native_password')
        self.conn = None
        self._connect()

    def _connect(self):
        try:
            self.conn = mysql.connector.connect(**self.config)
            if self.conn.is_connected():
                logger.info("Connecté à MySQL sur %s (base: %s)", self.config['host'], self.config['database'])
        except Error as e:
            logger.exception("Erreur connexion MySQL: %s", e)
            raise

    def close(self):
        if self.conn and self.conn.is_connected():
            self.conn.close()
            logger.info("Connexion MySQL fermée")

    # ---------- Table creation helpers ----------
    def create_tables(self):
        """Créer tables users et logs si elles n'existent pas."""
        create_users = """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(150) NOT NULL,
            encoding TEXT NOT NULL,
            date_added DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_access DATETIME DEFAULT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        create_logs = """
        CREATE TABLE IF NOT EXISTS logs (
            log_id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NULL,
            success TINYINT(1) NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            info TEXT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        cursor = self.conn.cursor()
        cursor.execute(create_users)
        cursor.execute(create_logs)
        self.conn.commit()
        cursor.close()
        logger.info("Tables créées (users, logs) si inexistantes.")

    # ---------- CRUD Users ----------
    def add_user(self, name: str, encoding: np.ndarray) -> int:
        """Ajoute un utilisateur et renvoie son id."""
        enc_str = serialize_encoding(encoding)
        sql = "INSERT INTO users (name, encoding) VALUES (%s, %s)"
        cursor = self.conn.cursor()
        cursor.execute(sql, (name, enc_str))
        self.conn.commit()
        user_id = cursor.lastrowid
        cursor.close()
        logger.info("Utilisateur ajouté id=%s name=%s", user_id, name)
        return user_id

    def update_user_encoding(self, user_id: int, encoding: np.ndarray) -> None:
        enc_str = serialize_encoding(encoding)
        sql = "UPDATE users SET encoding=%s WHERE id=%s"
        cursor = self.conn.cursor()
        cursor.execute(sql, (enc_str, user_id))
        self.conn.commit()
        cursor.close()
        logger.info("Encodage utilisateur mis à jour id=%s", user_id)

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        sql = "SELECT id, name, encoding, date_added, last_access FROM users WHERE id=%s"
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute(sql, (user_id,))
        row = cursor.fetchone()
        cursor.close()
        if row:
            row['encoding'] = deserialize_encoding(row['encoding'])
        return row

    def get_user_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        sql = "SELECT id, name, encoding, date_added, last_access FROM users WHERE name=%s"
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute(sql, (name,))
        row = cursor.fetchone()
        cursor.close()
        if row:
            row['encoding'] = deserialize_encoding(row['encoding'])
        return row

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Retourne la liste des utilisateurs avec encodages (numpy arrays)."""
        sql = "SELECT id, name, encoding, date_added, last_access FROM users"
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        users = []
        for r in rows:
            r['encoding'] = deserialize_encoding(r['encoding'])
            users.append(r)
        return users

    def delete_user(self, user_id: int) -> None:
        sql = "DELETE FROM users WHERE id=%s"
        cursor = self.conn.cursor()
        cursor.execute(sql, (user_id,))
        self.conn.commit()
        cursor.close()
        logger.info("Utilisateur supprimé id=%s", user_id)

    def update_last_access(self, user_id: int) -> None:
        sql = "UPDATE users SET last_access=%s WHERE id=%s"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = self.conn.cursor()
        cursor.execute(sql, (now, user_id))
        self.conn.commit()
        cursor.close()
        logger.info("Mise à jour last_access pour user id=%s", user_id)

    # ---------- Logs ----------
    def record_log(self, user_id: Optional[int], success: bool, info: Optional[str] = None) -> int:
        """Enregistre une tentative d'accès."""
        sql = "INSERT INTO logs (user_id, success, info) VALUES (%s, %s, %s)"
        cursor = self.conn.cursor()
        cursor.execute(sql, (user_id, int(success), info))
        self.conn.commit()
        log_id = cursor.lastrowid
        cursor.close()
        logger.info("Log enregistré id=%s user_id=%s success=%s", log_id, user_id, success)
        return log_id

    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        sql = "SELECT l.log_id, l.user_id, u.name as user_name, l.success, l.timestamp, l.info " \
              "FROM logs l LEFT JOIN users u ON l.user_id = u.id ORDER BY l.timestamp DESC LIMIT %s"
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute(sql, (limit,))
        rows = cursor.fetchall()
        cursor.close()
        return rows


# ---------- exemple d'utilisation ----------
if __name__ == "__main__":
    # usage de test rapide (remplacer credentails dans config.py)
    db = DatabaseManager()
    db.create_tables()

    # créer un embedding factice
    fake_embedding = np.random.rand(128)
    uid = db.add_user("Test_User", fake_embedding)

    # lire tous
    all_users = db.get_all_users()
    print("Users count:", len(all_users))
    print("First user:", all_users[0]['name'])

    # enregister log
    db.record_log(uid, True, info="Test login success")

    # fermer
    db.close()
