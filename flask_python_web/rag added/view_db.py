import sqlite3

conn = sqlite3.connect("chat.db")
c = conn.cursor()

print("\n📌 Mesajlar (messages tablosu):")
try:
    for row in c.execute("SELECT * FROM messages"):
        print(row)
except sqlite3.OperationalError:
    print("messages tablosu bulunamadı.")

print("\n⚙️ Ayarlar (settings tablosu):")
try:
    for row in c.execute("SELECT * FROM settings"):
        print(row)
except sqlite3.OperationalError:
    print("settings tablosu bulunamadı.")

conn.close()
