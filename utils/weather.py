import os
import requests

# Load .env file manually (no extra package needed)
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def get_weather(city: str) -> dict | None:
    """Fetch current weather for a city. Returns dict or None on failure."""
    try:
        resp = requests.get(BASE_URL, params={
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }, timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        print(f"Weather API error: {e}")
        return None


def assess_disease_risk(temp: float, humidity: float, condition: str) -> dict:
    """
    Return a disease risk level and message based on weather conditions.
    High humidity + moderate temp = high fungal disease risk.
    """
    condition_lower = condition.lower()

    if humidity >= 80 and 18 <= temp <= 30:
        return {
            "level": "High",
            "color": "#ef4444",
            "bg": "#fef2f2",
            "border": "#fecaca",
            "icon": "fa-triangle-exclamation",
            "message": "High humidity & warm temperatures — ideal conditions for fungal diseases (blight, mildew). Inspect crops immediately."
        }
    elif humidity >= 65 or condition_lower in ("rain", "drizzle", "thunderstorm"):
        return {
            "level": "Moderate",
            "color": "#f97316",
            "bg": "#fff7ed",
            "border": "#fed7aa",
            "icon": "fa-circle-exclamation",
            "message": "Moderate risk — wet or humid conditions may promote leaf spot and rust diseases. Monitor closely."
        }
    elif temp >= 35:
        return {
            "level": "Low",
            "color": "#eab308",
            "bg": "#fefce8",
            "border": "#fde68a",
            "icon": "fa-circle-info",
            "message": "High heat may cause heat stress on crops. Ensure adequate irrigation."
        }
    else:
        return {
            "level": "Low",
            "color": "#10b981",
            "bg": "#ecfdf5",
            "border": "#bbf7d0",
            "icon": "fa-circle-check",
            "message": "Current weather conditions have low disease risk. Continue regular monitoring."
        }


def weather_icon_emoji(condition: str) -> str:
    """Map OpenWeatherMap condition to an emoji."""
    c = condition.lower()
    if "clear" in c:      return "☀️"
    if "cloud" in c:      return "⛅"
    if "rain" in c:       return "🌧️"
    if "drizzle" in c:    return "🌦️"
    if "thunder" in c:    return "⛈️"
    if "snow" in c:       return "❄️"
    if "mist" in c or "fog" in c: return "🌫️"
    return "🌡️"
