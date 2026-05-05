import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Ensure .env is loaded (if not already done globally)
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")

def send_alert_email(to_email: str, username: str, city: str, risk_level: str, condition: str, message: str) -> bool:
    """Send a weather risk alert email."""
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        print("[WARNING] Email credentials not set in .env. Simulating email instead.")
        print(f"--- MOCK EMAIL ---")
        print(f"To: {to_email}")
        print(f"Subject: ⚠️ CropGuard Alert - {risk_level} Disease Risk in {city}")
        print(f"Body: {username}, be warned: weather is {condition}. {message}")
        print("------------------")
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg['From'] = SMTP_USERNAME
        msg['To'] = to_email
        msg['Subject'] = f"⚠️ CropGuard Alert: {risk_level} Disease Risk in {city}"
        
        html_body = f"""
        <html>
          <body style="font-family: Arial, sans-serif; color: #333;">
            <div style="max-width: 500px; margin: auto; border: 1px solid #ddd; border-radius: 10px; overflow: hidden;">
                <div style="background-color: {'#ef4444' if risk_level == 'High' else '#f59e0b'}; padding: 20px; color: white; text-align: center;">
                    <h2>🌿 CropGuard AI Alert</h2>
                </div>
                <div style="padding: 20px;">
                    <p>Hi <b>{username}</b>,</p>
                    <p>We are tracking weather changes in <b>{city}</b>.</p>
                    <div style="border-left: 4px solid {'#ef4444' if risk_level == 'High' else '#f59e0b'}; padding-left: 10px; margin: 20px 0; background: #f9f9f9; padding: 10px;">
                        <h4 style="margin: 0 0 5px 0;">{risk_level} Risk Detected! ⚠️</h4>
                        <p style="margin: 0;"><b>Current condition:</b> {condition}</p>
                        <p style="margin: 10px 0 0 0;"><i>{message}</i></p>
                    </div>
                    <p>To view the outbreak map and get proactive treatment recommendations, log into your dashboard.</p>
                    <p>Stay safe,<br/><b>The CropGuard Team</b></p>
                </div>
            </div>
          </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, "html"))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
            
        print(f"Email successfully sent to {to_email}")
        return True
    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")
        return False
