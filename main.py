# test_db.py
import numpy as np
from database.database_manager import DatabaseManager

def main():
    db = DatabaseManager()
    db.create_tables()

    # Embedding factice (128 dims)
    emb = np.linspace(0.01, 0.128, 128)
    uid = db.add_user("alice_test", emb)
    print("Ajouté user id:", uid)

    users = db.get_all_users()
    print("Total users:", len(users))
    for u in users:
        print("User:", u['id'], u['name'], "encoding_len:", len(u['encoding']))

    # Log
    db.record_log(uid, True, "Connexion test automatique")
    logs = db.get_logs(limit=10)
    for log in logs:
        print(log)

    db.close()

if __name__ == "__main__":
    main()