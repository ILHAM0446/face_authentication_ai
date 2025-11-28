import os
import json
import numpy as np
from dotenv import load_dotenv
try:
    from supabase import create_client, Client
    _HAS_SUPABASE = True
except Exception:
    _HAS_SUPABASE = False

load_dotenv()

import requests


class DatabaseManager:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        anon_key = os.getenv("SUPABASE_KEY")

        key = service_key if service_key else anon_key

        if not url or not key:
            raise ValueError("❌ SUPABASE_URL ou SUPABASE_KEY/SUPABASE_SERVICE_ROLE_KEY manquant dans .env")

        self.url = url.rstrip("/")
        self.key = key

        if _HAS_SUPABASE:
            self.supabase: Client = create_client(self.url, self.key)
        else:
            self.supabase = None

        self._headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }


    def create_user(self, name: str):
        """Créer un utilisateur. Retourne l'id (uuid) ou None."""
        try:
            if _HAS_SUPABASE and self.supabase is not None:
                response = self.supabase.table("users").insert({"name": name}).execute()
                if response.data:
                    return response.data[0]["id"]
                return None
            resp = requests.post(
                f"{self.url}/rest/v1/users",
                headers=self._headers,
                json={"name": name},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("id")
            return None
        except Exception as e:
            print(f"❌ Erreur création user : {e}")
            return None

    def get_all_users(self):
        try:
            if _HAS_SUPABASE and self.supabase is not None:
                response = self.supabase.table("users").select("*").execute()
                return response.data if response.data else []
            resp = requests.get(f"{self.url}/rest/v1/users", headers=self._headers, timeout=10)
            resp.raise_for_status()
            return resp.json() if resp.json() else []
        except Exception as e:
            print("❌ Erreur get_all_users:", e)
            return []

    def get_user_by_id(self, user_id: str):
        """Récupère un utilisateur par son ID"""
        try:
            if _HAS_SUPABASE and self.supabase is not None:
                response = self.supabase.table("users").select("*").eq("id", user_id).execute()
                if response.data:
                    return response.data[0]
                return None
            resp = requests.get(
                f"{self.url}/rest/v1/users?id=eq.{user_id}",
                headers=self._headers,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            return None
        except Exception as e:
            print(f"❌ Erreur get_user_by_id: {e}")
            return None

    def delete_user(self, user_id: str):
        try:
            if _HAS_SUPABASE and self.supabase is not None:
                self.supabase.table("users").delete().eq("id", user_id).execute()
                return True
            resp = requests.delete(
                f"{self.url}/rest/v1/users?id=eq.{user_id}", headers=self._headers, timeout=10
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ Erreur suppression user : {e}")
            return False


    def save_face_embedding(self, user_id: str, embedding: np.ndarray):
        """Enregistrer embedding en JSON dans Supabase"""
        try:
            embedding_json = embedding.tolist()

            if _HAS_SUPABASE and self.supabase is not None:
                response = self.supabase.table("face_embeddings").insert({
                    "user_id": user_id,
                    "embedding": embedding_json,
                }).execute()
                return response.data is not None

            resp = requests.post(
                f"{self.url}/rest/v1/face_embeddings",
                headers=self._headers,
                json={"user_id": user_id, "embedding": embedding_json},
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ Erreur insertion embedding : {e}")
            return False

    def get_all_embeddings(self):
        """Récupère tous les embeddings + noms"""
        try:
            if _HAS_SUPABASE and self.supabase is not None:
                response = self.supabase.table("face_embeddings").select(
                    "id, user_id, embedding, users(name)"
                ).execute()

                results = []
                for item in response.data:
                    arr = np.array(item["embedding"])
                    results.append({
                        "user_id": item["user_id"],
                        "name": item["users"]["name"],
                        "embedding": arr,
                    })
                return results

            params = {"select": "id,user_id,embedding,users(name)"}
            resp = requests.get(
                f"{self.url}/rest/v1/face_embeddings",
                headers=self._headers,
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data:
                arr = np.array(item.get("embedding", []))
                results.append({
                    "user_id": item.get("user_id"),
                    "name": item.get("users", {}).get("name") if item.get("users") else None,
                    "embedding": arr,
                })
            return results
        except Exception as e:
            print(f"❌ Erreur récupération embeddings : {e}")
            return []
