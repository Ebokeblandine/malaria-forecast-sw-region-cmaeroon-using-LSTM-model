import sqlite3
import hashlib

conn = sqlite3.connect("malaria_forecasts.db")

# ── Step 1: Save existing users ───────────────────────────────
existing = conn.execute("""
    SELECT username, full_name, role, district,
           password_hash, is_active, is_approved
    FROM users
""").fetchall()
print(f"Found {len(existing)} existing users to preserve")

# ── Step 2: Drop old table ────────────────────────────────────
conn.execute("DROP TABLE IF EXISTS users")
print("Dropped old users table")

# ── Step 3: Create clean table with correct columns ───────────
conn.execute("""
    CREATE TABLE users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT    NOT NULL UNIQUE,
        full_name     TEXT    NOT NULL,
        role          TEXT    NOT NULL DEFAULT 'user',
        district      TEXT    DEFAULT 'All Districts',
        password_hash TEXT    NOT NULL,
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        is_active     INTEGER DEFAULT 1,
        is_approved   INTEGER DEFAULT 1
    )
""")
print("Created clean users table")

# ── Step 4: Re-insert existing users ─────────────────────────
admin_hash = hashlib.sha256('admin123'.encode()).hexdigest()

# Always ensure admin exists
conn.execute("""
    INSERT INTO users (username, full_name, role, district,
                       password_hash, is_active, is_approved)
    VALUES ('admin', 'System Administrator', 'admin',
            'All Districts', ?, 1, 1)
""", (admin_hash,))
print("Admin account restored (admin / admin123)")

# Re-insert other users (skip admin since we just added it)
restored = 0
for row in existing:
    username, full_name, role, district, pw_hash, is_active, is_approved = row
    if username == 'admin':
        continue
    if not pw_hash:
        pw_hash = hashlib.sha256('changeme123'.encode()).hexdigest()
    try:
        conn.execute("""
            INSERT INTO users (username, full_name, role, district,
                               password_hash, is_active, is_approved)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (username.strip().lower(), full_name, role,
              district or 'All Districts', pw_hash,
              is_active, is_approved))
        restored += 1
        print(f"Restored: {username}")
    except Exception as e:
        print(f"Skipped {username}: {e}")

conn.commit()

# ── Step 5: Verify ────────────────────────────────────────────
print(f"\nAll users in clean table:")
for row in conn.execute(
    "SELECT id, username, full_name, role, is_active, is_approved FROM users"
).fetchall():
    print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]} | "
          f"active={row[4]} | approved={row[5]}")

conn.close()
print("\nDone — registration will now work correctly.")
print("Login: admin / admin123")
