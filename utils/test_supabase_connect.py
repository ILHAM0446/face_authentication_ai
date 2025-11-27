import socket
import os
import sys
import traceback
import requests
# Ajouter le dossier racine au sys.path pour importer le package local `database`
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database.database_manager import DatabaseManager

HOST = os.getenv("SUPABASE_URL", "").replace("https://", "").replace("http://", "")

print("SUPABASE_URL (env):", os.getenv("SUPABASE_URL"))
print("HOST to resolve:", HOST)

try:
    print("\n--- DNS resolution (socket.getaddrinfo) ---")
    infos = socket.getaddrinfo(HOST, 443)
    for info in infos:
        print(info)
except Exception as e:
    print("DNS resolution failed:")
    traceback.print_exc()

try:
    print("\n--- HTTP GET root (requests) ---")
    url = os.getenv("SUPABASE_URL") or f"https://{HOST}"
    r = requests.get(url, timeout=10)
    print("status_code:", r.status_code)
    print("headers:\n", r.headers)
except Exception as e:
    print("HTTP GET failed:")
    traceback.print_exc()

try:
    print("\n--- Test DatabaseManager.create_user ---")
    db = DatabaseManager()
    uid = db.create_user("test_connect_user")
    print("create_user returned:", uid)
except Exception as e:
    print("DatabaseManager test failed:")
    traceback.print_exc()

print("\n--- Env proxy variables ---")
print("HTTP_PROXY:", os.getenv("HTTP_PROXY"))
print("HTTPS_PROXY:", os.getenv("HTTPS_PROXY"))
print("NO_PROXY:", os.getenv("NO_PROXY"))
