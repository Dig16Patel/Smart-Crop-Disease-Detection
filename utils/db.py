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

# ─── Alert Functions ───

def setup_alert_columns():
    """Ensure users table has alert preference columns."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS alerts_enabled BOOLEAN DEFAULT FALSE;")
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS alert_city VARCHAR(100);")
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        pass

def update_user_alerts(user_id: int, enabled: bool, city: str = None):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET alerts_enabled = %s, alert_city = %s WHERE id = %s",
            (enabled, city, user_id)
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating alerts: {e}")
        return False

def get_alert_users():
    """Get all users who opted into alerts."""
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT id, username, email, alert_city FROM users WHERE alerts_enabled = TRUE AND alert_city IS NOT NULL")
        users = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(u) for u in users]
    except Exception as e:
        print(f"Error fetching alert users: {e}")
        return []

# ─── Forum Functions ───

def create_forum_tables():
    """Create tables for the community forum if they don't exist."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS forum_posts (
                id SERIAL PRIMARY KEY,
                user_name VARCHAR(100) NOT NULL,
                disease_name VARCHAR(100),
                severity VARCHAR(50),
                question TEXT NOT NULL,
                image_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS forum_comments (
                id SERIAL PRIMARY KEY,
                post_id INTEGER REFERENCES forum_posts(id) ON DELETE CASCADE,
                user_name VARCHAR(100) NOT NULL,
                comment TEXT NOT NULL,
                is_expert BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error creating forum tables: {e}")

def add_forum_post(user_name, disease_name, severity, question, image_data):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO forum_posts (user_name, disease_name, severity, question, image_data) VALUES (%s, %s, %s, %s, %s)",
            (user_name, disease_name, severity, question, image_data)
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error adding forum post: {e}")
        return False

def get_forum_posts():
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM forum_posts ORDER BY created_at DESC")
        posts = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(p) for p in posts]
    except Exception as e:
        print(f"Error fetching forum posts: {e}")
        return []

def add_forum_comment(post_id, user_name, comment, is_expert=False):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO forum_comments (post_id, user_name, comment, is_expert) VALUES (%s, %s, %s, %s)",
            (post_id, user_name, comment, is_expert)
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error adding forum comment: {e}")
        return False

def get_forum_comments(post_id):
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM forum_comments WHERE post_id = %s ORDER BY created_at ASC", (post_id,))
        comments = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(c) for c in comments]
    except Exception as e:
        print(f"Error fetching forum comments: {e}")
        return []


# ─── Scan History Functions ───

def setup_location_columns():
    """Ensure latitude and longitude columns exist in scan_history."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("ALTER TABLE scan_history ADD COLUMN IF NOT EXISTS latitude FLOAT;")
        cur.execute("ALTER TABLE scan_history ADD COLUMN IF NOT EXISTS longitude FLOAT;")
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        # Ignore permission errors if the columns were already added manually by the owner
        pass

def save_scan(user_id: int, disease_name: str, confidence: float, severity: str, latitude: float = None, longitude: float = None) -> bool:
    """Save a scan result to the database."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO scan_history (user_id, disease_name, confidence, severity, latitude, longitude)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (user_id, disease_name, confidence, severity, latitude, longitude)
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving scan: {e}")
        return False

def delete_scan(scan_id: int, user_id: int) -> bool:
    """Delete a scan by ID, only if it belongs to the given user."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM scan_history WHERE id = %s AND user_id = %s",
            (scan_id, user_id)
        )
        deleted = cur.rowcount > 0
        conn.commit()
        cur.close()
        conn.close()
        return deleted
    except Exception as e:
        print(f"Error deleting scan: {e}")
        return False


def get_all_scans_with_location():
    """Get all scans that have location data for the map."""
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            """SELECT disease_name, severity, latitude, longitude, scanned_at
               FROM scan_history 
               WHERE latitude IS NOT NULL AND longitude IS NOT NULL
               ORDER BY scanned_at DESC LIMIT 500"""
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"Error fetching map data: {e}")
        return []


def get_scan_history(user_id: int) -> list:
    """Get the scan history for a user, most recent first."""
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            """SELECT id, disease_name, confidence, severity, scanned_at
               FROM scan_history WHERE user_id = %s
               ORDER BY scanned_at DESC LIMIT 50""",
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
