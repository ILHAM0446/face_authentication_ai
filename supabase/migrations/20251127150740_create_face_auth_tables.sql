-- TABLE DES UTILISATEURS
CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  created_at timestamptz DEFAULT now()
);

-- TABLE DES EMBEDDINGS
CREATE TABLE IF NOT EXISTS face_embeddings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  embedding jsonb NOT NULL,
  created_at timestamptz DEFAULT now()
);

-- Activer la sécurité RLS
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE face_embeddings ENABLE ROW LEVEL SECURITY;


-- ===============================
-- POLICIES : USERS
-- ===============================

CREATE POLICY "Users can read all users"
  ON users FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Users can insert users"
  ON users FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Users can delete users"
  ON users FOR DELETE
  TO authenticated
  USING (true);


-- ===============================
-- POLICIES : FACE EMBEDDINGS
-- ===============================

CREATE POLICY "Users can read embeddings"
  ON face_embeddings FOR SELECT
  TO authenticated
  USING (true);

CREATE POLICY "Users can insert embeddings"
  ON face_embeddings FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Users can delete embeddings"
  ON face_embeddings FOR DELETE
  TO authenticated
  USING (true);

-- Index pour accélérer les recherches
CREATE INDEX IF NOT EXISTS idx_face_embeddings_user_id
  ON face_embeddings(user_id);
