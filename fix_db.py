import sqlite3

conn = sqlite3.connect("malaria_forecasts.db")

fixes = [
    "ALTER TABLE users ADD COLUMN district TEXT DEFAULT 'All Districts'",
    "ALTER TABLE users ADD COLUMN is_approved INTEGER DEFAULT 1",
    "ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1",
    "ALTER TABLE users ADD COLUMN full_name TEXT DEFAULT 'User'",
    "ALTER TABLE users ADD COLUMN password_hash TEXT DEFAULT ''",
]

for sql in fixes:
    try:
        conn.execute(sql)
        col = sql.split("COLUMN")[1].strip().split()[0]
        print(f"Added column: {col}")
    except Exception as e:
        col = sql.split("COLUMN")[1].strip().split()[0]
        print(f"Already exists: {col}")

conn.commit()

# Show final table structure
print("\nFinal users table columns:")
for row in conn.execute("PRAGMA table_info(users)").fetchall():
    print(f"  {row[1]} ({row[2]})")

conn.close()
print("\nDone — database fixed.")
