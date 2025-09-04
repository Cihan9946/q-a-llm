import sqlite3

def init_db():
    with sqlite3.connect("chat.db") as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            session_id TEXT,
            user_msg TEXT,
            bot_msg TEXT
        )
        """)
        conn.commit()

def save_message(session_id, user_msg, bot_msg):
    with sqlite3.connect("chat.db") as conn:
        c = conn.cursor()
        c.execute("INSERT INTO messages (session_id, user_msg, bot_msg) VALUES (?, ?, ?)",
                  (session_id, user_msg, bot_msg))
        conn.commit()

def get_history(session_id):
    with sqlite3.connect("chat.db") as conn:
        c = conn.cursor()
        c.execute("SELECT user_msg, bot_msg FROM messages WHERE session_id = ?", (session_id,))
        rows = c.fetchall()
        return [{"user": row[0], "bot": row[1]} for row in rows]

def get_all_sessions():
    with sqlite3.connect("chat.db") as conn:
        c = conn.cursor()
        c.execute("SELECT DISTINCT session_id FROM messages")
        sessions = c.fetchall()
        result = {}
        for sid in sessions:
            result[sid[0]] = get_history(sid[0])
        return result

def reset_session(session_id):
    with sqlite3.connect("chat.db") as conn:
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
        
def init_settings():
    with sqlite3.connect("chat.db") as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """)
        conn.commit()

def save_selected_model(model_name):
    with sqlite3.connect("chat.db") as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO settings (key, value)
            VALUES ('selected_model', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, (model_name,))
        conn.commit()

def get_selected_model():
    with sqlite3.connect("chat.db") as conn:
        c = conn.cursor()
        c.execute("SELECT value FROM settings WHERE key = 'selected_model'")
        result = c.fetchone()
        return result[0] if result else "turkcell-custom"  # fallback
    
def init_settings():
    with sqlite3.connect("chat.db") as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """)
        conn.commit()    
        