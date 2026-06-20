import sqlite3
import hashlib

conn = sqlite3.connect("malaria_forecasts.db")

# Copy old password column into password_hash for all existing users
conn.execute("""
    UPDATE users 
    SET password_hash = password 
    WHERE password_hash = '' OR password_hash IS NULL
""")
print("Copied old passwords to password_hash column")

# Make sure admin account has correct hash for 'admin123'
admin_hash = hashlib.sha256('admin123'.encode()).hexdigest()
conn.execute("""
    UPDATE users 
    SET password_hash = ?,
        full_name = 'System Administrator',
        role = 'admin',
        district = 'All Districts',
        is_active = 1,
        is_approved = 1
    WHERE username = 'admin'
""", (admin_hash,))
print("Admin account updated")

# Check if admin exists, create if not
count = conn.execute("SELECT COUNT(*) FROM users WHERE username='admin'").fetchone()[0]
if count == 0:
    conn.execute("""
        INSERT INTO users (username, password_hash, full_name, role, district, is_active, is_approved)
        VALUES ('admin', ?, 'System Administrator', 'admin', 'All Districts', 1, 1)
    """, (admin_hash,))
    print("Admin account created fresh")

conn.commit()

# Show all users
print("\nAll users in database:")
for row in conn.execute("SELECT id, username, full_name, role, is_active, is_approved FROM users").fetchall():
    print(f"  id={row[0]} | {row[1]} | {row[2]} | {row[3]} | active={row[4]} | approved={row[5]}")

conn.close()
print("\nDone — you can now log in with admin / admin123")
