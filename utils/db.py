import psycopg2
import psycopg2.extras

# ─── Database Configuration ───
DB_CONFIG = {
    "host": "localhost",
    "database": "cropguard_db",
    "user": "cropguard_user",
    "password": "cropguard123",
    "port": 5432
}

def get_connection():
    """Get a PostgreSQL database connection."""
    return psycopg2.connect(**DB_CONFIG)


# ─── User Functions ───

def create_user(username: str, email: str, password_hash: str) -> bool:
    """Insert a new user. Returns True on success, False if username/email exists."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
            (username, email, password_hash)
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except psycopg2.errors.UniqueViolation:
        return False
    except Exception as e:
        print(f"[DB ERROR] create_user failed: {type(e).__name__}: {e}")
        return False


def get_user_by_username(username: str) -> dict | None:
    """Fetch user record by username. Returns dict or None."""
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"Error fetching user: {e}")
        return None


def get_user_by_email(email: str) -> dict | None:
    """Fetch user record by email."""
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"Error fetching user by email: {e}")
        return None


# ─── Scan History Functions ───

def save_scan(user_id: int, disease_name: str, confidence: float, severity: str) -> bool:
    """Save a scan result to the database."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO scan_history (user_id, disease_name, confidence, severity)
               VALUES (%s, %s, %s, %s)""",
            (user_id, disease_name, confidence, severity)
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving scan: {e}")
        return False


def get_scan_history(user_id: int) -> list:
    """Get the scan history for a user, most recent first."""
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            """SELECT disease_name, confidence, severity, scanned_at
               FROM scan_history WHERE user_id = %s
               ORDER BY scanned_at DESC LIMIT 20""",
            (user_id,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"Error fetching scan history: {e}")
        return []


def get_disease_frequency(user_id: int) -> list:
    """Get disease name + count for bar chart, ordered by frequency."""
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            """SELECT disease_name, COUNT(*) as count
               FROM scan_history WHERE user_id = %s
               GROUP BY disease_name ORDER BY count DESC LIMIT 10""",
            (user_id,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"Error fetching disease frequency: {e}")
        return []


def get_daily_scan_counts(user_id: int) -> list:
    """Get scan counts per day for the last 30 days."""
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            """SELECT DATE(scanned_at) as scan_date, COUNT(*) as count
               FROM scan_history WHERE user_id = %s
                 AND scanned_at >= NOW() - INTERVAL '30 days'
               GROUP BY scan_date ORDER BY scan_date""",
            (user_id,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"Error fetching daily scans: {e}")
        return []


def get_severity_breakdown(user_id: int) -> list:
    """Get count per severity level."""
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            """SELECT severity, COUNT(*) as count
               FROM scan_history WHERE user_id = %s
               GROUP BY severity ORDER BY count DESC""",
            (user_id,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"Error fetching severity breakdown: {e}")
        return []
