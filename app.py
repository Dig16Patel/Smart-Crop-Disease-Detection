import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from huggingface_hub import hf_hub_download
from utils.preprocess import preprocess_image
from utils.recommendations import get_recommendation

# ─── HuggingFace Config ───
HF_REPO_ID  = "23it085/plant_disease_detector"
HF_FILENAME = "plant_disease_stable.keras"
HF_TOKEN    = os.environ.get("HF_TOKEN")
LOCAL_MODEL = "models/plant_disease_stable.keras"

# Exact class order from training (matches HuggingFace EfficientNetB0 model)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
from utils.db import (get_user_by_username, create_user, save_scan, get_scan_history, 
                      get_disease_frequency, get_daily_scan_counts, get_severity_breakdown,
                      create_forum_tables, add_forum_post, get_forum_posts, add_forum_comment, get_forum_comments,
                      setup_location_columns, get_all_scans_with_location,
                      setup_alert_columns, update_user_alerts, get_alert_users,
                      delete_scan)
from utils.auth import hash_password, verify_password, validate_registration
from utils.report import generate_report_pdf
from utils.weather import get_weather, assess_disease_risk, weather_icon_emoji
from utils.email_service import send_alert_email
from utils.llm_chatbot import get_agronomist_response

# ─── Page Config ───
st.set_page_config(
    page_title="CropGuard AI — Smart Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Session State Init ───
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "login"  # "login" or "register"

if "current_scan_result" not in st.session_state:
    st.session_state.current_scan_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "scan_analyzed" not in st.session_state:
    st.session_state.scan_analyzed = False

if "current_scan_result" not in st.session_state:
    st.session_state.current_scan_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "scan_analyzed" not in st.session_state:
    st.session_state.scan_analyzed = False
    
# Variables for AI Chatbot Context
if "current_scan_result" not in st.session_state:
    st.session_state.current_scan_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Load Model (from HuggingFace, cached locally) ───
@st.cache_resource
def load_model():
    """Downloads model from HuggingFace Space on first run, then caches locally."""
    if not os.path.exists(LOCAL_MODEL):
        with st.spinner("⬇️ Downloading model from HuggingFace (first run only)..."):
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                token=HF_TOKEN,
                repo_type="space",
                local_dir="models"
            )
    return tf.keras.models.load_model(LOCAL_MODEL)

# Initialize forum tables
create_forum_tables()
setup_location_columns()
setup_alert_columns()


# ══════════════════════════════════════════════════════
#  AUTH PAGES CSS + FUNCTIONS
# ══════════════════════════════════════════════════════
AUTH_CSS = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    h1, h2, h3, h4, h5, h6, p, a, li, button, label, input, textarea,
    .stMarkdown, .stButton, .stTextInput, .stSelectbox {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    /* Dark gradient background */
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%) !important; }

    /* Hide sidebar */
    section[data-testid="stSidebar"] { display: none !important; }

    /* ── Card: wrap the middle column in a visible glass box ── */
    div[data-testid="column"]:nth-child(2) > div[data-testid="stVerticalBlockBorderWrapper"] > div,
    div[data-testid="column"]:nth-child(2) > div > div[data-testid="stVerticalBlock"] {
        background: rgba(20, 30, 55, 0.82) !important;
        border: 1.5px solid rgba(99, 102, 241, 0.55) !important;
        border-top: 3px solid #6366f1 !important;
        border-radius: 22px !important;
        padding: 36px 36px 28px 36px !important;
        backdrop-filter: blur(28px) !important;
        box-shadow:
            0 0 0 1px rgba(99,102,241,0.12),
            0 8px 32px rgba(99,102,241,0.18),
            0 24px 64px rgba(0,0,0,0.55) !important;
    }

    /* Input fields */
    .stTextInput input {
        background: rgba(15,23,42,0.55) !important;
        border: 1.5px solid rgba(148,163,184,0.18) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
        padding: 12px 16px !important;
        font-size: 0.9rem !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stTextInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.18) !important;
    }
    .stTextInput label {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 13px 20px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 16px rgba(99,102,241,0.35) !important;
        letter-spacing: 0.2px !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(99,102,241,0.55) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* Auth helper elements */
    .auth-logo { text-align: center; margin-bottom: 20px; }
    .auth-logo-icon {
        width: 68px; height: 68px; border-radius: 18px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 1.8rem;
        box-shadow: 0 10px 30px rgba(99,102,241,0.45);
        margin-bottom: 12px;
    }
    .auth-logo h1 { color:#f8fafc !important; font-size:1.7rem !important; font-weight:800 !important; margin:0 0 3px 0 !important; }
    .auth-logo p  { color:#64748b !important; font-size:0.82rem !important; margin:0 !important; }
    .auth-title   { font-size:1.25rem !important; font-weight:800 !important; color:#f1f5f9 !important; margin:0 0 4px 0 !important; }
    .auth-sub     { font-size:0.8rem  !important; color:#64748b  !important; margin:0 0 18px 0 !important; }
    .auth-card-header { border-bottom:1px solid rgba(148,163,184,0.1); padding-bottom:16px; margin-bottom:18px; }
    .auth-divider {
        display:flex; align-items:center; gap:12px; margin:16px 0;
    }
    .auth-divider hr  { flex:1; border:none; border-top:1px solid rgba(148,163,184,0.12); margin:0; }
    .auth-divider span { font-size:0.72rem; color:#475569; font-weight:600; letter-spacing:0.8px; }
    .auth-error {
        background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.3);
        border-radius:10px; padding:11px 14px; color:#fca5a5; font-size:0.82rem; margin-bottom:12px;
    }
    .auth-success {
        background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3);
        border-radius:10px; padding:11px 14px; color:#6ee7b7; font-size:0.82rem; margin-bottom:12px;
    }
</style>
"""

def show_login_page():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        # Logo above card
        st.markdown("""
        <div class="auth-logo">
            <div class="auth-logo-icon">🌿</div>
            <h1>CropGuard AI</h1>
            <p>Intelligent Crop Health Monitor</p>
        </div>
        """, unsafe_allow_html=True)

        # Card title inside the column (visually part of the glass card)
        st.markdown("""
        <div class="auth-card-header">
            <p class="auth-title">Welcome back 👋</p>
            <p class="auth-sub">Sign in to continue to CropGuard AI</p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter your username", key="login_username")
        password = st.text_input("Password", placeholder="Enter your password", type="password", key="login_password")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        if st.button("Sign In →", key="login_btn", use_container_width=True):
            if not username or not password:
                st.markdown('<div class="auth-error"><i class="fa-solid fa-circle-exclamation"></i> Please fill in all fields.</div>', unsafe_allow_html=True)
            else:
                try:
                    user = get_user_by_username(username)
                except Exception as db_err:
                    st.markdown(f'<div class="auth-error"><i class="fa-solid fa-server"></i> Database connection failed: {db_err}<br><small>Make sure PostgreSQL is running.</small></div>', unsafe_allow_html=True)
                    user = None

                if user is None:
                    # Check if DB is reachable at all to give a better message
                    try:
                        from utils.db import get_connection
                        conn = get_connection()
                        conn.close()
                        # DB is up but user not found
                        st.markdown('<div class="auth-error"><i class="fa-solid fa-triangle-exclamation"></i> Username not found. Please check your username or <b>register</b> first.</div>', unsafe_allow_html=True)
                    except Exception as conn_err:
                        st.markdown(f'<div class="auth-error"><i class="fa-solid fa-server"></i> Cannot connect to database.<br><small>{conn_err}</small></div>', unsafe_allow_html=True)
                elif verify_password(password, user["password_hash"]):
                    st.session_state.logged_in = True
                    st.session_state.user = {"id": user["id"], "username": user["username"], "email": user["email"]}
                    st.rerun()
                else:
                    st.markdown('<div class="auth-error"><i class="fa-solid fa-triangle-exclamation"></i> Incorrect password. Please try again.</div>', unsafe_allow_html=True)

        st.markdown('<div class="auth-divider"><hr><span>NEW HERE?</span><hr></div>', unsafe_allow_html=True)

        if st.button("Create an Account", key="goto_register", use_container_width=True):
            st.session_state.auth_page = "register"
            st.rerun()


def show_register_page():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("""
        <div class="auth-logo">
            <div class="auth-logo-icon">🌿</div>
            <h1>CropGuard AI</h1>
            <p>Create your free account</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="auth-card-header">
            <p class="auth-title">Create Account</p>
            <p class="auth-sub">Join CropGuard AI to save your scan history</p>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Choose a username", key="reg_username")
        email    = st.text_input("Email", placeholder="Enter your email", key="reg_email")
        password = st.text_input("Password", placeholder="Create a password (min 6 chars)", type="password", key="reg_password")
        confirm  = st.text_input("Confirm Password", placeholder="Repeat your password", type="password", key="reg_confirm")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        if st.button("Create Account →", key="register_btn", use_container_width=True):
            errors = validate_registration(username, email, password, confirm)
            if errors:
                for err in errors:
                    st.markdown(f'<div class="auth-error"><i class="fa-solid fa-circle-exclamation"></i> {err}</div>', unsafe_allow_html=True)
            else:
                pw_hash = hash_password(password)
                success = create_user(username, email, pw_hash)
                if success:
                    st.markdown('<div class="auth-success"><i class="fa-solid fa-circle-check"></i> Account created! Please sign in.</div>', unsafe_allow_html=True)
                    st.session_state.auth_page = "login"
                    st.rerun()
                else:
                    st.markdown('<div class="auth-error"><i class="fa-solid fa-circle-exclamation"></i> Username or email already exists.</div>', unsafe_allow_html=True)

        st.markdown('<div class="auth-divider"><hr><span>ALREADY HAVE AN ACCOUNT?</span><hr></div>', unsafe_allow_html=True)

        if st.button("← Back to Sign In", key="goto_login", use_container_width=True):
            st.session_state.auth_page = "login"
            st.rerun()


# ── Auth Gate: show login/register if not logged in ──
if not st.session_state.logged_in:
    if st.session_state.auth_page == "register":
        show_register_page()
    else:
        show_login_page()
    st.stop()

# ══════════════════════════════════════════════════════
#  CUSTOM CSS — Full Premium Theme
# ══════════════════════════════════════════════════════
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* ═══ GLOBAL ═══ */
    /* Target only text elements to avoid breaking Streamlit icons */
    h1, h2, h3, h4, h5, h6, p, a, li, button, label, input, textarea,
    .stMarkdown, .stButton, .stTextInput, .stSelectbox {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .stApp {
        background: #f0f2f5;
    }
    #MainMenu, footer { visibility: hidden; }

    /* ═══ SIDEBAR ═══ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        box-shadow: 4px 0 24px rgba(0,0,0,0.12);
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 2px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        color: #ffffff !important;
        font-size: 0.95rem !important;
        padding: 8px 18px !important;
        border-radius: 10px !important;
        transition: all 0.25s ease !important;
        cursor: pointer !important;
        border: 1px solid transparent !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
        color: #ffffff !important;
        margin: 0 !important;
        line-height: 1.2 !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(99,102,241,0.15) !important;
        border-color: rgba(99,102,241,0.2) !important;
    }

    /* ═══ BUTTONS ═══ */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 32px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
        letter-spacing: 0.3px !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ═══ HERO ═══ */
    .hero-section {
        position: relative;
        border-radius: 24px;
        overflow: hidden;
        margin-bottom: 40px;
        min-height: 420px;
        display: flex;
        align-items: center;
    }
    .hero-bg {
        position: absolute;
        inset: 0;
        background: url('https://images.unsplash.com/photo-1574943320219-553eb213f72d?w=1400&q=80') center/cover no-repeat;
    }
    .hero-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(15,23,42,0.88) 0%, rgba(30,41,59,0.75) 50%, rgba(15,23,42,0.65) 100%);
    }
    .hero-content {
        position: relative;
        z-index: 2;
        padding: 60px 55px;
        width: 100%;
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(99,102,241,0.2);
        border: 1px solid rgba(99,102,241,0.3);
        color: #a5b4fc;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.15;
        margin-bottom: 6px;
    }
    .hero-title span {
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 16px;
        font-weight: 400;
    }
    .hero-desc {
        font-size: 0.9rem;
        color: #64748b;
        max-width: 520px;
        line-height: 1.7;
        margin-bottom: 32px;
    }
    .hero-stats-row {
        display: flex;
        gap: 32px;
        margin-top: 8px;
    }
    .hero-stat {
        text-align: center;
    }
    .hero-stat-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #a5b4fc;
    }
    .hero-stat-lbl {
        font-size: 0.68rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }

    /* ═══ FEATURE CARDS ═══ */
    .f-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 32px 24px;
        text-align: center;
        border: 1px solid #e5e7eb;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: default;
    }
    .f-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.08);
        border-color: #c7d2fe;
    }
    .f-icon {
        width: 64px;
        height: 64px;
        border-radius: 16px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 18px;
        color: #fff;
    }
    .f-icon.indigo { background: linear-gradient(135deg, #6366f1, #4f46e5); }
    .f-icon.emerald { background: linear-gradient(135deg, #10b981, #059669); }
    .f-icon.amber { background: linear-gradient(135deg, #f59e0b, #d97706); }
    .f-card h4 {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0 0 8px 0;
    }
    .f-card p {
        font-size: 0.85rem;
        color: #64748b;
        line-height: 1.6;
        margin: 0;
    }

    /* ═══ STEPS ═══ */
    .step-wrap {
        text-align: center;
        padding: 10px;
    }
    .step-num {
        width: 44px; height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #fff;
        font-weight: 800;
        font-size: 1rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(99,102,241,0.3);
    }
    .step-wrap p {
        font-size: 0.82rem;
        color: #475569;
        line-height: 1.5;
        margin: 0;
    }

    /* ═══ SECTION HEADER ═══ */
    .sec-header {
        text-align: center;
        margin-bottom: 28px;
    }
    .sec-header h3 {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0 0 4px 0;
    }
    .sec-header p {
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0;
    }

    /* ═══ DIVIDER ═══ */
    .sep { border:0; height:1px; background:#e5e7eb; margin:36px 0; }

    /* ═══ PAGE HEADER ═══ */
    .pg-h { font-size:1.6rem; font-weight:800; color:#1e293b; margin-bottom:2px; }
    .pg-sub { font-size:0.88rem; color:#94a3b8; margin-bottom:24px; }
    .col-h { font-size:1rem; font-weight:700; color:#334155; margin-bottom:4px; }
    .col-sub { font-size:0.78rem; color:#94a3b8; margin-bottom:14px; }

    /* ═══ DETECT PAGE BANNER ═══ */
    .detect-banner {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 18px;
        padding: 36px 40px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    .detect-banner::before {
        content: "";
        position: absolute;
        top: -40px; right: -40px;
        width: 180px; height: 180px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(99,102,241,0.2), transparent 70%);
    }
    .detect-banner h2 {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        position: relative;
        z-index: 1;
    }
    .detect-banner p {
        color: #94a3b8;
        font-size: 0.88rem;
        margin: 0;
        position: relative;
        z-index: 1;
    }

    /* ═══ UPLOAD CARD WRAPPER ═══ */
    .upload-wrap {
        background: #ffffff;
        border-radius: 16px;
        padding: 28px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }

    /* ═══ RESULT CARDS ═══ */
    .r-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px 22px;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #6366f1;
        margin-bottom: 10px;
        transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
    }
    .r-card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
        transform: translateX(2px);
    }
    .r-card.green { border-left-color: #10b981; }
    .r-card.amber { border-left-color: #f59e0b; }
    .r-card.red   { border-left-color: #ef4444; }
    .r-label {
        font-size: 0.68rem;
        color: #94a3b8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 4px;
    }
    .r-val {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1e293b;
    }
    .r-val-sm {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.65;
    }

    /* ═══ SEVERITY ═══ */
    .sv-none { color: #10b981 !important; }
    .sv-low { color: #eab308 !important; }
    .sv-mod { color: #f97316 !important; }
    .sv-high { color: #ef4444 !important; }

    /* ═══ TREATMENT BOX ═══ */
    .tx-box {
        background: linear-gradient(135deg, #fefce8, #fffbeb);
        border: 1px solid #fde68a;
        border-radius: 16px;
        padding: 24px;
        margin-top: 10px;
    }
    .tx-title {
        font-size: 1rem;
        font-weight: 700;
        color: #92400e;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .tx-item {
        font-size: 0.86rem;
        color: #78350f;
        padding: 9px 0;
        border-bottom: 1px solid rgba(253,230,138,0.5);
        display: flex;
        align-items: flex-start;
        gap: 10px;
        line-height: 1.55;
    }
    .tx-item:last-child { border-bottom: none; }
    .tx-check {
        color: #16a34a;
        font-size: 0.85rem;
        margin-top: 2px;
        flex-shrink: 0;
    }

    /* ═══ EMPTY STATE ═══ */
    .empty-state {
        text-align: center;
        padding: 50px 24px;
        background: #ffffff;
        border-radius: 16px;
        border: 2px dashed #e2e8f0;
    }
    .empty-icon {
        width: 72px; height: 72px;
        border-radius: 50%;
        background: linear-gradient(135deg, #eef2ff, #e0e7ff);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 16px;
    }
    .empty-icon i {
        font-size: 1.6rem;
        color: #6366f1;
    }
    .empty-state h4 {
        font-size: 1rem;
        font-weight: 700;
        color: #334155;
        margin: 0 0 6px 0;
    }
    .empty-state p {
        font-size: 0.82rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.5;
    }

    /* ═══ FOOTER ═══ */
    .app-ft {
        text-align: center;
        padding: 30px 0 10px 0;
        font-size: 0.75rem;
        color: #94a3b8;
    }
    .app-ft a {
        color: #6366f1;
        text-decoration: none;
    }

    /* ═══ FILE UPLOADER ═══ */
    [data-testid="stFileUploader"] section {
        border: 2px dashed #d1d5db !important;
        border-radius: 14px !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #6366f1 !important;
    }

    /* ═══ PROGRESS BAR ═══ */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #10b981, #eab308, #f97316, #ef4444) !important;
        border-radius: 6px !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    # Logo & Brand
    st.markdown("""
    <div style='text-align:center; padding: 28px 0 16px 0;'>
        <div style='
            width:60px; height:60px; border-radius:16px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            display:inline-flex; align-items:center; justify-content:center;
            font-size:1.5rem;
            box-shadow: 0 8px 24px rgba(99,102,241,0.4);
            margin-bottom: 14px;
        '>🌿</div>
        <h3 style='margin:0; font-size:1.2rem; color:#f1f5f9; font-weight:800; letter-spacing:0.3px;'>CropGuard AI</h3>
        <p style='font-size:0.68rem; color:#64748b; margin:4px 0 0 0;'>Intelligent Crop Health Monitor</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(148,163,184,0.08); margin:4px 16px 12px 16px;'>", unsafe_allow_html=True)

    # Navigation Label
    st.markdown("<p style='font-size:0.62rem; color:#475569; font-weight:700; text-transform:uppercase; letter-spacing:1.8px; padding:0 12px; margin-bottom:4px;'><i class='fa-solid fa-compass' style='margin-right:6px; color:#6366f1;'></i>Navigation</p>", unsafe_allow_html=True)

    app_mode = st.radio("Nav", ["🏠  Home", "🔬  Detect Disease", "📜  History", "📊  Dashboard", "💬  Community Forum", "🗺️  Outbreak Map", "📧  Risk Alerts"], label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(148,163,184,0.08); margin:12px 16px;'>", unsafe_allow_html=True)

    # Quick Stats
    st.markdown("<p style='font-size:0.62rem; color:#475569; font-weight:700; text-transform:uppercase; letter-spacing:1.8px; padding:0 12px; margin-bottom:8px;'><i class='fa-solid fa-chart-simple' style='margin-right:6px; color:#10b981;'></i>Quick Stats</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px; padding:0 8px; margin-bottom:8px;'>
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.12); border-radius:10px; padding:10px 8px; text-align:center;'>
            <div style='font-size:1.2rem; font-weight:800; color:#a5b4fc;'>38</div>
            <div style='font-size:0.6rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;'>Classes</div>
        </div>
        <div style='background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.12); border-radius:10px; padding:10px 8px; text-align:center;'>
            <div style='font-size:1.2rem; font-weight:800; color:#6ee7b7;'>92%</div>
            <div style='font-size:0.6rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;'>Accuracy</div>
        </div>
    </div>
    <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px; padding:0 8px; margin-bottom:12px;'>
        <div style='background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.12); border-radius:10px; padding:10px 8px; text-align:center;'>
            <div style='font-size:1.2rem; font-weight:800; color:#fcd34d;'>14</div>
            <div style='font-size:0.6rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;'>Crops</div>
        </div>
        <div style='background:rgba(139,92,246,0.08); border:1px solid rgba(139,92,246,0.12); border-radius:10px; padding:10px 8px; text-align:center;'>
            <div style='font-size:1.2rem; font-weight:800; color:#c4b5fd;'>&lt; 3s</div>
            <div style='font-size:0.6rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;'>Speed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(148,163,184,0.08); margin:4px 16px 12px 16px;'>", unsafe_allow_html=True)

    # Supported Crops
    st.markdown("<p style='font-size:0.62rem; color:#475569; font-weight:700; text-transform:uppercase; letter-spacing:1.8px; padding:0 12px; margin-bottom:8px;'><i class='fa-solid fa-seedling' style='margin-right:6px; color:#f59e0b;'></i>Supported Crops</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:0 8px; max-height:220px; overflow-y:auto;'>
        <div style='display:flex; align-items:center; gap:8px; padding:5px 10px; border-radius:7px; margin-bottom:2px;'><span>🍎</span><span style='font-size:0.76rem; color:#cbd5e1; font-weight:500;'>Apple</span></div>
        <div style='display:flex; align-items:center; gap:8px; padding:5px 10px; border-radius:7px; margin-bottom:2px;'><span>🫐</span><span style='font-size:0.76rem; color:#cbd5e1; font-weight:500;'>Blueberry</span></div>
        <div style='display:flex; align-items:center; gap:8px; padding:5px 10px; border-radius:7px; margin-bottom:2px;'><span>🍒</span><span style='font-size:0.76rem; color:#cbd5e1; font-weight:500;'>Cherry</span></div>
        <div style='display:flex; align-items:center; gap:8px; padding:5px 10px; border-radius:7px; margin-bottom:2px;'><span>🌽</span><span style='font-size:0.76rem; color:#cbd5e1; font-weight:500;'>Corn / Maize</span></div>
        <div style='display:flex; align-items:center; gap:8px; padding:5px 10px; border-radius:7px; margin-bottom:2px;'><span>🍇</span><span style='font-size:0.76rem; color:#cbd5e1; font-weight:500;'>Grape</span></div>
        <div style='display:flex; align-items:center; gap:8px; padding:5px 10px; border-radius:7px; margin-bottom:2px;'><span>🍊</span><span style='font-size:0.76rem; color:#cbd5e1; font-weight:500;'>Orange</span></div>
        <div style='display:flex; align-items:center; gap:8px; padding:5px 10px; border-radius:7px; margin-bottom:2px;'><span>🍑</span><span style='font-size:0.76rem; color:#cbd5e1; font-weight:500;'>Peach</span></div>
        <div style='display:flex; align-items:center; gap:8px; padding:5px 10px; border-radius:7px; margin-bottom:2px;'><span>🫑</span><span style='font-size:0.76rem; color:#cbd5e1; font-weight:500;'>Pepper (Bell)</span></div>
        <div style='display:flex; align-items:center; gap:8px; padding:5px 10px; border-radius:7px; margin-bottom:2px;'><span>🥔</span><span style='font-size:0.76rem; color:#cbd5e1; font-weight:500;'>Potato</span></div>
        <div style='display:flex; align-items:center; gap:8px; padding:5px 10px; border-radius:7px; margin-bottom:2px;'><span>🫙</span><span style='font-size:0.76rem; color:#cbd5e1; font-weight:500;'>Raspberry</span></div>
        <div style='display:flex; align-items:center; gap:8px; padding:5px 10px; border-radius:7px; margin-bottom:2px;'><span>🫘</span><span style='font-size:0.76rem; color:#cbd5e1; font-weight:500;'>Soybean</span></div>
        <div style='display:flex; align-items:center; gap:8px; padding:5px 10px; border-radius:7px; margin-bottom:2px;'><span>🎃</span><span style='font-size:0.76rem; color:#cbd5e1; font-weight:500;'>Squash</span></div>
        <div style='display:flex; align-items:center; gap:8px; padding:5px 10px; border-radius:7px; margin-bottom:2px;'><span>🍓</span><span style='font-size:0.76rem; color:#cbd5e1; font-weight:500;'>Strawberry</span></div>
        <div style='display:flex; align-items:center; gap:8px; padding:5px 10px; border-radius:7px; margin-bottom:2px;'><span>🍅</span><span style='font-size:0.76rem; color:#cbd5e1; font-weight:500;'>Tomato</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(148,163,184,0.08); margin:12px 16px;'>", unsafe_allow_html=True)

    # Pro Tip
    st.markdown("""
    <div style='background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.1); border-radius:12px; padding:14px 16px; margin:0 8px;'>
        <p style='font-size:0.72rem; color:#94a3b8; margin:0; line-height:1.55;'>
            <i class='fa-solid fa-lightbulb' style='color:#fbbf24; margin-right:6px;'></i><strong style='color:#e2e8f0;'>Pro Tip</strong><br>
            Use a clear, well-lit photo of a single leaf for best results.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Version badge
    st.markdown("<div style='position:fixed; bottom:14px; left:14px;'><span style='font-size:0.6rem; color:#475569; background:rgba(148,163,184,0.06); padding:4px 10px; border-radius:6px; border:1px solid rgba(148,163,184,0.08);'><i class='fa-solid fa-code-branch' style='margin-right:4px; color:#6366f1;'></i>v1.0.0</span></div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(148,163,184,0.08); margin:12px 16px;'>", unsafe_allow_html=True)

    # Logged-in user info
    user = st.session_state.user
    st.markdown(f"""
    <div style='background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.12); border-radius:12px; padding:12px 16px; margin:0 8px 8px 8px;'>
        <div style='display:flex; align-items:center; gap:10px;'>
            <div style='width:34px; height:34px; border-radius:50%; background:linear-gradient(135deg,#6366f1,#8b5cf6); display:flex; align-items:center; justify-content:center; font-size:0.9rem;'>
                <i class='fa-solid fa-user' style='color:#fff;'></i>
            </div>
            <div>
                <div style='font-size:0.82rem; font-weight:700; color:#e2e8f0;'>{user['username']}</div>
                <div style='font-size:0.68rem; color:#64748b;'>{user['email']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Logout button with custom red style
    st.markdown("""
    <style>
    div[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #ef4444, #dc2626) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 9px 18px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        margin: 0 8px !important;
        width: calc(100% - 16px) !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 10px rgba(239,68,68,0.3) !important;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        box-shadow: 0 4px 18px rgba(239,68,68,0.5) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("🚪  Sign Out", key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.session_state.auth_page = "login"
        st.rerun()


# ══════════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════════
if app_mode == "🏠  Home":

    # ── Hero Section ──
    st.markdown("""
    <div class="hero-section">
        <div class="hero-bg"></div>
        <div class="hero-overlay"></div>
        <div class="hero-content">
            <div class="hero-tag">
                <i class="fa-solid fa-microchip"></i> AI-POWERED CROP ANALYSIS
            </div>
            <div class="hero-title">
                Protect Your Crops<br>with <span>Artificial Intelligence</span>
            </div>
            <div class="hero-subtitle">Smart Disease Detection & Health Monitoring System</div>
            <div class="hero-desc">
                Upload a photo of any crop leaf and get instant AI-powered disease diagnosis,
                severity assessment, and expert treatment recommendations — all within seconds.
            </div>
            <div class="hero-stats-row">
                <div class="hero-stat">
                    <div class="hero-stat-val">14</div>
                    <div class="hero-stat-lbl">Crop Types</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-val">38</div>
                    <div class="hero-stat-lbl">Classes</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-val">92%</div>
                    <div class="hero-stat-lbl">Accuracy</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-val">&lt; 3s</div>
                    <div class="hero-stat-lbl">Speed</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Features ──
    st.markdown("""
    <div class="sec-header">
        <h3>What We Offer</h3>
        <p>Everything you need to keep your crops healthy</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""
        <div class="f-card">
            <div class="f-icon indigo"><i class="fa-solid fa-camera"></i></div>
            <h4>Instant Scan</h4>
            <p>Simply upload a photo of the affected leaf. Our system accepts JPG, PNG, and JPEG formats.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="f-card">
            <div class="f-icon emerald"><i class="fa-solid fa-brain"></i></div>
            <h4>Deep Learning Analysis</h4>
            <p>A trained CNN model extracts visual features and classifies diseases with high confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="f-card">
            <div class="f-icon amber"><i class="fa-solid fa-prescription-bottle-medical"></i></div>
            <h4>Treatment Plan</h4>
            <p>Get actionable treatment recommendations including pesticides, organic solutions, and preventive care.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)

    # ── Weather & Disease Risk Widget ──
    st.markdown("""
    <div class="sec-header">
        <h3>🌤️ Weather & Disease Risk</h3>
        <p>Check current conditions and get crop disease risk alerts for your location</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;
         padding:20px 24px;margin-bottom:16px;'>
        <div style='font-size:0.8rem;font-weight:700;color:#374151;margin-bottom:10px;
             display:flex;align-items:center;gap:8px;'>
            <i class='fa-solid fa-magnifying-glass-location' style='color:#6366f1;'></i>
            Enter your city or town to check live weather conditions
        </div>
    </div>
    """, unsafe_allow_html=True)

    inp_col, btn_col = st.columns([3, 1])
    with inp_col:
        city_input = st.text_input(
            "🏙️  City Name",
            placeholder="e.g.  Mumbai,  Delhi,  Pune,  Nagpur ...",
            key="weather_city",
        )
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔍  Check Weather", use_container_width=True, key="weather_search")

    if city_input and (search_clicked or True):
        weather = get_weather(city_input.strip())
        if weather:
            temp      = weather["main"]["temp"]
            humidity  = weather["main"]["humidity"]
            feels     = weather["main"]["feels_like"]
            wind      = weather["wind"]["speed"]
            condition = weather["weather"][0]["main"]
            desc      = weather["weather"][0]["description"].title()
            emoji     = weather_icon_emoji(condition)
            risk      = assess_disease_risk(temp, humidity, condition)

            wc1, wc2, wc3, wc4 = st.columns(4)
            for col, label, val, icon, color in [
                (wc1, "Temperature",  f"{temp:.1f}°C",  "fa-thermometer-half",  "#6366f1"),
                (wc2, "Humidity",     f"{humidity}%",   "fa-droplet",           "#3b82f6"),
                (wc3, "Condition",    f"{emoji} {desc}","fa-cloud-sun",         "#8b5cf6"),
                (wc4, "Wind Speed",   f"{wind} m/s",    "fa-wind",              "#10b981"),
            ]:
                with col:
                    st.markdown(f"""
                    <div style='background:#fff;border-radius:14px;padding:18px 20px;
                         border:1px solid #e5e7eb;text-align:center;'>
                        <i class='fa-solid {icon}' style='color:{color};font-size:1.3rem;'></i>
                        <div style='font-size:1.2rem;font-weight:800;color:#1e293b;margin:8px 0 2px 0;'>{val}</div>
                        <div style='font-size:0.7rem;color:#94a3b8;font-weight:600;
                             text-transform:uppercase;letter-spacing:1px;'>{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background:{risk["bg"]};border:1px solid {risk["border"]};
                 border-left:4px solid {risk["color"]};border-radius:14px;
                 padding:16px 22px;display:flex;align-items:flex-start;gap:14px;'>
                <i class='fa-solid {risk["icon"]}' style='color:{risk["color"]};font-size:1.3rem;margin-top:2px;'></i>
                <div>
                    <div style='font-size:0.85rem;font-weight:700;color:{risk["color"]};
                         margin-bottom:4px;'>Disease Risk: {risk["level"]}</div>
                    <div style='font-size:0.82rem;color:#475569;line-height:1.5;'>{risk["message"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif weather is None:
            st.markdown("""
            <div style='background:#fef2f2;border:1px solid #fecaca;border-radius:12px;
                 padding:14px 18px;color:#dc2626;font-size:0.85rem;'>
                <i class='fa-solid fa-triangle-exclamation' style='margin-right:8px;'></i>
                City not found or API key not configured. Please check the city name.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)

    # ── How It Works ──
    st.markdown("""
    <div class="sec-header">
        <h3>How It Works</h3>
        <p>Four simple steps to diagnose your crops</p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4, gap="small")
    with s1:
        st.markdown("""
        <div class="step-wrap">
            <div class="step-num">1</div>
            <p><strong>Upload</strong><br>Take a photo & upload the leaf image</p>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="step-wrap">
            <div class="step-num">2</div>
            <p><strong>Preprocess</strong><br>Image is resized & normalized automatically</p>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="step-wrap">
            <div class="step-num">3</div>
            <p><strong>Analyze</strong><br>CNN model extracts features & classifies</p>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown("""
        <div class="step-wrap">
            <div class="step-num">4</div>
            <p><strong>Results</strong><br>Get diagnosis, severity & treatment plan</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="app-ft">
        <p>© 2026 <strong>CropGuard AI</strong> — Built with <i class="fa-solid fa-heart" style="color:#ef4444; font-size:0.7rem;"></i> using Streamlit & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  DETECT DISEASE PAGE
# ══════════════════════════════════════════════════════
elif app_mode == "🔬  Detect Disease":

    # ── Page Banner ──
    st.markdown("""
    <div class="detect-banner">
        <h2><i class="fa-solid fa-microscope" style="margin-right:12px; color:#818cf8;"></i>Disease Recognition</h2>
        <p>Upload a leaf image to get AI-powered disease diagnosis, severity analysis, and treatment recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    # ── Upload Column ──
    with col_upload:
        st.markdown("""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:16px;'>
            <div style='width:36px; height:36px; border-radius:10px; background:linear-gradient(135deg,#6366f1,#818cf8); display:flex; align-items:center; justify-content:center;'>
                <i class='fa-solid fa-cloud-arrow-up' style='color:#fff; font-size:0.85rem;'></i>
            </div>
            <div>
                <div style='font-size:0.95rem; font-weight:700; color:#1e293b;'>Upload Leaf Image</div>
                <div style='font-size:0.72rem; color:#94a3b8;'>Drag & drop or browse — JPG, PNG, JPEG</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.markdown("<div class='upload-wrap'>", unsafe_allow_html=True)
            st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("📍 Location Data (for Outbreak Map)", expanded=False):
                st.markdown("<p style='font-size:0.8rem; color:#64748b;'>Help other farmers by sharing this scan's location on the Outbreak Map.</p>", unsafe_allow_html=True)
                share_location = st.checkbox("Share my scan on the Live Map", value=True)

                # City name search → accurate coordinates via OpenStreetMap (no IP guessing)
                city_search = st.text_input(
                    "🏙️ Enter your city/village",
                    placeholder="e.g. Nadiad, Gujarat",
                    key="loc_city_search"
                )
                detected_lat, detected_lon = 22.6916, 72.8634  # Default: Nadiad, Gujarat

                if city_search:
                    try:
                        import requests
                        resp = requests.get(
                            "https://nominatim.openstreetmap.org/search",
                            params={"q": city_search, "format": "json", "limit": 1},
                            headers={"User-Agent": "CropGuardAI/1.0"},
                            timeout=4
                        ).json()
                        if resp:
                            detected_lat = float(resp[0]["lat"])
                            detected_lon = float(resp[0]["lon"])
                            st.success(f"✅ Found: **{resp[0]['display_name'][:65]}**")
                        else:
                            st.warning("City not found. Check spelling or enter lat/lon manually.")
                    except Exception:
                        st.warning("Could not connect to location service. Enter coordinates manually.")

                map_lat = st.number_input("Latitude",  value=detected_lat, format="%.4f", key="loc_lat")
                map_lon = st.number_input("Longitude", value=detected_lon, format="%.4f", key="loc_lon")
                st.caption(f"📌 Will save at: `{map_lat:.4f}, {map_lon:.4f}`")

            predict_clicked = st.button("🔍  Analyze Disease", use_container_width=True)
        else:
            predict_clicked = False
            share_location = False
            map_lat, map_lon = None, None
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon"><i class="fa-solid fa-leaf"></i></div>
                <h4>No Image Uploaded</h4>
                <p>Choose or drag a leaf photo above to start analysis</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Results Column ──
    with col_result:
        st.markdown("""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:16px;'>
            <div style='width:36px; height:36px; border-radius:10px; background:linear-gradient(135deg,#10b981,#34d399); display:flex; align-items:center; justify-content:center;'>
                <i class='fa-solid fa-chart-column' style='color:#fff; font-size:0.85rem;'></i>
            </div>
            <div>
                <div style='font-size:0.95rem; font-weight:700; color:#1e293b;'>Analysis Results</div>
                <div style='font-size:0.72rem; color:#94a3b8;'>Disease diagnosis & treatment info</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if uploaded_file is None:
            st.session_state.scan_analyzed = False
            st.session_state.current_scan_result = None
            st.session_state.chat_history = []
        elif predict_clicked:
            st.session_state.scan_analyzed = True

        if uploaded_file is not None and st.session_state.scan_analyzed:
            if predict_clicked or st.session_state.current_scan_result is None:
                with st.spinner("🧠 AI is analyzing the image..."):
                    try:
                        processed_image = preprocess_image(image)
                        model = load_model()

                        predictions = model.predict(processed_image)
                        idx = int(np.argmax(predictions))
                        name = CLASS_NAMES[idx]
                        conf = float(predictions[0][idx]) * 100
                        info = get_recommendation(name)

                        severity = info["severity"]
                        sev_cls, sev_score = "sv-none", 0
                        sev_card = "green"
                        if severity == "Low":
                            sev_cls, sev_score, sev_card = "sv-low", 25, "amber"
                        elif severity == "Moderate":
                            sev_cls, sev_score, sev_card = "sv-mod", 50, "amber"
                        elif severity == "High":
                            sev_cls, sev_score, sev_card = "sv-high", 85, "red"

                        display_name = name.replace("___", " — ").replace("_", " ")

                        # Auto-save scan to database
                        save_scan(
                            user_id=st.session_state.user["id"],
                            disease_name=display_name,
                            confidence=round(conf, 2),
                            severity=severity,
                            latitude=map_lat if share_location else None,
                            longitude=map_lon if share_location else None
                        )
                        
                        st.session_state.current_scan_result = {
                            "name": name, "display_name": display_name, "conf": conf,
                            "info": info, "severity": severity, "sev_cls": sev_cls,
                            "sev_score": sev_score, "sev_card": sev_card
                        }
                        st.session_state.chat_history = []
                    except Exception as e:
                        st.error(f"⚠️ Error classifying: {e}")
                        st.session_state.scan_analyzed = False
                        
            if st.session_state.current_scan_result is not None:
                res = st.session_state.current_scan_result
                name = res["name"]
                display_name = res["display_name"]
                conf = res["conf"]
                info = res["info"]
                severity = res["severity"]
                sev_cls = res["sev_cls"]
                sev_score = res["sev_score"]
                sev_card = res["sev_card"]
                try:
                    # Success indicator
                    st.markdown("""
                    <div style='background:linear-gradient(135deg,#ecfdf5,#f0fdf4); border:1px solid #bbf7d0; border-radius:12px; padding:14px 18px; margin-bottom:12px; display:flex; align-items:center; gap:10px;'>
                        <i class='fa-solid fa-circle-check' style='color:#16a34a; font-size:1.1rem;'></i>
                        <span style='font-size:0.85rem; color:#166534; font-weight:600;'>Analysis completed successfully — saved to history</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Disease Name
                    st.markdown(f"""
                    <div class="r-card">
                        <div class="r-label"><i class="fa-solid fa-virus" style="margin-right:4px; color:#6366f1;"></i>Detected Disease</div>
                        <div class="r-val">{display_name}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Confidence & Severity
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown(f"""
                        <div class="r-card">
                            <div class="r-label"><i class="fa-solid fa-bullseye" style="margin-right:4px; color:#6366f1;"></i>Confidence</div>
                            <div class="r-val" style="color:#6366f1;">{conf:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with rc2:
                        st.markdown(f"""
                        <div class="r-card {sev_card}">
                            <div class="r-label"><i class="fa-solid fa-gauge-high" style="margin-right:4px;"></i>Severity</div>
                            <div class="r-val {sev_cls}">{severity}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.progress(sev_score)

                    # Description
                    st.markdown(f"""
                    <div class="r-card green">
                        <div class="r-label"><i class="fa-solid fa-circle-info" style="margin-right:4px; color:#10b981;"></i>About This Disease</div>
                        <div class="r-val-sm">{info['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Treatment
                    items = "".join(
                        f'<div class="tx-item"><span class="tx-check"><i class="fa-solid fa-circle-check"></i></span>{t}</div>'
                        for t in info["treatment"]
                    )
                    st.markdown(f"""
                    <div class="tx-box">
                        <div class="tx-title"><i class="fa-solid fa-kit-medical"></i> Treatment Recommendations</div>
                        {items}
                    </div>
                    """, unsafe_allow_html=True)

                    # 🛒 Recommended Products / Store Links
                    if "commerce_links" in info and info["commerce_links"]:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("#### 🛒 Suggested Treatments")
                        st.markdown("<p style='font-size:0.85rem; color:#64748b; margin-top:-8px;'>Purchase recommended treatments online to protect your crops instantly.</p>", unsafe_allow_html=True)
                        
                        cols = st.columns(len(info["commerce_links"]))
                        for idx, link in enumerate(info["commerce_links"]):
                            with cols[idx]:
                                st.markdown(f"""
                                <a href="{link['url']}" target="_blank" style="text-decoration:none;">
                                    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:16px; text-align:center; transition:0.2s; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom:10px;" onmouseover="this.style.transform='translateY(-2px)'; this.style.borderColor='#6366f1';" onmouseout="this.style.transform='translateY(0)'; this.style.borderColor='#e2e8f0';">
                                        <i class="fa-solid {link.get('icon', 'fa-box')}" style="font-size:2rem; color:#6366f1; margin-bottom:12px;"></i>
                                        <h5 style="color:#1e293b; font-size:0.95rem; margin:0 0 8px 0; font-weight:700;">{link['name']}</h5>
                                        <span style="background:#f1f5f9; color:#475569; font-size:0.75rem; padding:4px 10px; border-radius:20px; font-weight:600;"><i class="fa-brands fa-amazon" style="margin-right:4px; color:#f59e0b;"></i> Buy Now</span>
                                    </div>
                                </a>
                                """, unsafe_allow_html=True)


                    # ── Download PDF Report ──
                    st.markdown("<br>", unsafe_allow_html=True)
                    try:
                        pdf_bytes = generate_report_pdf(
                            username=st.session_state.user["username"],
                            disease_name=display_name,
                            confidence=conf,
                            severity=severity,
                            description=info.get("description", ""),
                            treatments=info.get("treatment", []),
                        )
                        fname = f"CropGuard_Report_{display_name.replace(' ', '_')}.pdf"
                        st.download_button(
                            label="📥  Download PDF Report",
                            data=pdf_bytes,
                            file_name=fname,
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as pdf_err:
                        st.warning(f"Could not generate PDF: {pdf_err}")
                        
                    # 🤖 AI Agronomist Chatbot
                    st.markdown("<hr style='border-color:rgba(148,163,184,0.2); margin:24px 0;'>", unsafe_allow_html=True)
                    st.markdown(f"#### 🤖 AI Agronomist Chat")
                    st.markdown("<p style='font-size:0.85rem; color:#64748b;'>Ask follow-up questions about this crop.</p>", unsafe_allow_html=True)
                    
                    chat_container = st.container()
                    
                    # Display existing history
                    with chat_container:
                        for msg in st.session_state.chat_history:
                            with st.chat_message(msg["role"]):
                                st.write(msg["content"])
                                
                    # Get user input
                    user_msg = st.chat_input(f"Ask about {display_name} treatment...")
                    if user_msg:
                        with chat_container:
                            with st.chat_message("user"):
                                st.write(user_msg)
                            st.session_state.chat_history.append({"role": "user", "content": user_msg})
                            
                            with st.chat_message("assistant"):
                                with st.spinner("Thinking..."):
                                    agronomist_reply = get_agronomist_response(display_name, info, user_msg, st.session_state.chat_history[:-1])
                                    st.write(agronomist_reply)
                            st.session_state.chat_history.append({"role": "assistant", "content": agronomist_reply})

                except Exception as render_err:
                    st.error(f"⚠️ Error rendering results: {render_err}")
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon"><i class="fa-solid fa-chart-pie"></i></div>
                <h4>Awaiting Analysis</h4>
                <p>Upload an image and click "Analyze Disease" to see results here</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="app-ft"><p>© 2026 <strong>CropGuard AI</strong></p></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  HISTORY PAGE
# ══════════════════════════════════════════════════════
elif app_mode == "\U0001f4dc  History":

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1e293b,#0f172a); border-radius:20px;
         padding:32px 36px; margin-bottom:28px; position:relative; overflow:hidden;'>
        <div style='position:absolute;top:-40px;right:-40px;width:200px;height:200px;
             border-radius:50%;background:radial-gradient(circle,rgba(99,102,241,0.18),transparent 70%);'></div>
        <p style='font-size:0.7rem;color:#6366f1;font-weight:700;letter-spacing:2px;
             text-transform:uppercase;margin:0 0 8px 0;'>YOUR RECORDS</p>
        <h2 style='margin:0 0 6px 0;color:#f1f5f9;font-size:1.8rem;font-weight:800;'>Scan History</h2>
        <p style='margin:0;color:#64748b;font-size:0.9rem;'>All your past crop disease analyses in one place.</p>
    </div>
    """, unsafe_allow_html=True)

    scans = get_scan_history(st.session_state.user["id"])

    if not scans:
        st.markdown("""
        <div style='text-align:center; padding:60px 24px; background:#fff;
             border-radius:18px; border:2px dashed #e2e8f0;'>
            <div style='width:72px;height:72px;border-radius:50%;background:#f1f5f9;
                 display:flex;align-items:center;justify-content:center;
                 margin:0 auto 18px auto;font-size:1.8rem;'>\U0001f4dc</div>
            <h4 style='color:#1e293b;margin:0 0 8px 0;'>No Scans Yet</h4>
            <p style='color:#94a3b8;font-size:0.85rem;margin:0;'>
                Go to <strong>Detect Disease</strong> and analyze your first crop image!
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        total     = len(scans)
        high_risk = sum(1 for s in scans if s["severity"] == "High")
        avg_conf  = sum(s["confidence"] for s in scans) / total

        c1, c2, c3 = st.columns(3)
        stat_data = [
            (c1, "Total Scans",    str(total),          "fa-microscope",           "#6366f1"),
            (c2, "High Risk",      str(high_risk),      "fa-triangle-exclamation", "#ef4444"),
            (c3, "Avg Confidence", f"{avg_conf:.1f}%",  "fa-bullseye",             "#10b981"),
        ]
        for col, label, val, icon, color in stat_data:
            with col:
                st.markdown(f"""
                <div style='background:#fff;border-radius:16px;padding:22px 24px;
                     border:1px solid #e5e7eb;text-align:center;'>
                    <i class='fa-solid {icon}' style='color:{color};font-size:1.4rem;'></i>
                    <div style='font-size:1.6rem;font-weight:800;color:#1e293b;margin:8px 0 2px 0;'>{val}</div>
                    <div style='font-size:0.75rem;color:#94a3b8;font-weight:600;
                         text-transform:uppercase;letter-spacing:1px;'>{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        sev_colors = {"None": "#10b981", "Low": "#eab308", "Moderate": "#f97316", "High": "#ef4444"}
        sev_bg     = {"None": "#ecfdf5", "Low": "#fefce8", "Moderate": "#fff7ed",  "High": "#fef2f2"}

        for i, scan in enumerate(scans):
            sev   = scan.get("severity", "None")
            color = sev_colors.get(sev, "#94a3b8")
            bg    = sev_bg.get(sev, "#f8fafc")
            ts    = scan["scanned_at"].strftime("%d %b %Y, %I:%M %p") if scan.get("scanned_at") else "N/A"
            scan_id = scan.get("id")

            card_col, del_col = st.columns([9, 1])
            with card_col:
                st.markdown(f"""
                <div style='background:#fff;border:1px solid #e5e7eb;border-left:4px solid {color};
                     border-radius:14px;padding:18px 22px;margin-bottom:2px;
                     display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;'>
                    <div style='flex:1;min-width:180px;'>
                        <div style='font-size:0.95rem;font-weight:700;color:#1e293b;margin-bottom:4px;'>
                            <i class='fa-solid fa-leaf' style='color:{color};margin-right:6px;'></i>
                            {scan['disease_name']}
                        </div>
                        <div style='font-size:0.75rem;color:#94a3b8;'>
                            <i class='fa-regular fa-clock' style='margin-right:4px;'></i>{ts}
                        </div>
                    </div>
                    <div style='display:flex;gap:12px;align-items:center;'>
                        <div style='text-align:center;'>
                            <div style='font-size:0.65rem;color:#94a3b8;font-weight:600;
                                 text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;'>Confidence</div>
                            <div style='font-size:1rem;font-weight:700;color:#6366f1;'>{scan['confidence']:.1f}%</div>
                        </div>
                        <div style='background:{bg};color:{color};font-size:0.72rem;font-weight:700;
                             padding:5px 14px;border-radius:20px;border:1px solid {color};
                             text-transform:uppercase;letter-spacing:0.5px;'>{sev}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with del_col:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                if scan_id and st.button("🗑️", key=f"del_{scan_id}_{i}", help="Delete this scan"):
                    if delete_scan(scan_id, st.session_state.user["id"]):
                        st.success("Deleted.")
                        st.rerun()
                    else:
                        st.error("Could not delete.")
            st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="app-ft"><p>\u00a9 2026 <strong>CropGuard AI</strong></p></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  DASHBOARD PAGE
# ══════════════════════════════════════════════════════
elif app_mode == "\U0001f4ca  Dashboard":
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1e293b,#0f172a); border-radius:20px;
         padding:32px 36px; margin-bottom:28px; position:relative; overflow:hidden;'>
        <div style='position:absolute;top:-40px;right:-40px;width:200px;height:200px;
             border-radius:50%;background:radial-gradient(circle,rgba(16,185,129,0.18),transparent 70%);'></div>
        <p style='font-size:0.7rem;color:#10b981;font-weight:700;letter-spacing:2px;
             text-transform:uppercase;margin:0 0 8px 0;'>ANALYTICS</p>
        <h2 style='margin:0 0 6px 0;color:#f1f5f9;font-size:1.8rem;font-weight:800;'>Dashboard</h2>
        <p style='margin:0;color:#64748b;font-size:0.9rem;'>Insights from your scan data at a glance.</p>
    </div>
    """, unsafe_allow_html=True)

    uid = st.session_state.user["id"]
    scans     = get_scan_history(uid)
    freq_data = get_disease_frequency(uid)
    daily_data = get_daily_scan_counts(uid)
    sev_data  = get_severity_breakdown(uid)

    if not scans:
        st.markdown("""
        <div style='text-align:center;padding:60px 24px;background:#fff;
             border-radius:18px;border:2px dashed #e2e8f0;'>
            <div style='font-size:2.5rem;margin-bottom:16px;'>📊</div>
            <h4 style='color:#1e293b;margin:0 0 8px 0;'>No Data Yet</h4>
            <p style='color:#94a3b8;font-size:0.85rem;margin:0;'>
                Run some scans first to see analytics here!
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        total     = len(scans)
        high_risk = sum(1 for s in scans if s["severity"] == "High")
        avg_conf  = sum(s["confidence"] for s in scans) / total
        unique_d  = len(set(s["disease_name"] for s in scans))

        # ── Summary KPI cards ──
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, icon, color in [
            (c1, "Total Scans",       str(total),          "fa-microscope",           "#6366f1"),
            (c2, "Unique Diseases",   str(unique_d),        "fa-dna",                  "#8b5cf6"),
            (c3, "High Risk Scans",   str(high_risk),       "fa-triangle-exclamation", "#ef4444"),
            (c4, "Avg Confidence",    f"{avg_conf:.1f}%",   "fa-bullseye",             "#10b981"),
        ]:
            with col:
                st.markdown(f"""
                <div style='background:#fff;border-radius:16px;padding:20px 22px;
                     border:1px solid #e5e7eb;text-align:center;
                     border-top:3px solid {color};'>
                    <i class='fa-solid {icon}' style='color:{color};font-size:1.3rem;'></i>
                    <div style='font-size:1.5rem;font-weight:800;color:#1e293b;margin:8px 0 2px 0;'>{val}</div>
                    <div style='font-size:0.72rem;color:#94a3b8;font-weight:600;
                         text-transform:uppercase;letter-spacing:1px;'>{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 1: Disease Frequency Bar + Severity Donut ──
        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown("#### 🦠 Most Common Diseases")
            if freq_data:
                diseases = [d["disease_name"] for d in freq_data]
                counts   = [d["count"] for d in freq_data]
                fig_bar = go.Figure(go.Bar(
                    x=counts, y=diseases, orientation="h",
                    marker=dict(
                        color=counts,
                        colorscale=[[0,"#c7d2fe"],[1,"#6366f1"]],
                        showscale=False
                    ),
                    text=counts, textposition="outside",
                    hovertemplate="%{y}: %{x} scans<extra></extra>"
                ))
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=20, t=10, b=0),
                    height=320,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, autorange="reversed",
                               tickfont=dict(size=11, color="#475569")),
                    font=dict(family="Plus Jakarta Sans")
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with col_r:
            st.markdown("#### ⚠️ Severity Breakdown")
            if sev_data:
                labels = [s["severity"] for s in sev_data]
                values = [s["count"] for s in sev_data]
                clr_map = {"None":"#10b981","Low":"#eab308","Moderate":"#f97316","High":"#ef4444"}
                colors  = [clr_map.get(l, "#94a3b8") for l in labels]
                fig_pie = go.Figure(go.Pie(
                    labels=labels, values=values,
                    hole=0.55,
                    marker=dict(colors=colors, line=dict(color="#fff", width=2)),
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value} scans<extra></extra>"
                ))
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=320,
                    legend=dict(orientation="v", font=dict(size=11)),
                    font=dict(family="Plus Jakarta Sans")
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # ── Row 2: Daily Scan Timeline ──
        st.markdown("#### 📅 Scan Activity (Last 30 Days)")
        if daily_data:
            dates  = [str(d["scan_date"]) for d in daily_data]
            dcounts= [d["count"] for d in daily_data]
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=dates, y=dcounts, mode="lines+markers",
                line=dict(color="#6366f1", width=2.5, shape="spline"),
                marker=dict(color="#6366f1", size=7),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.08)",
                hovertemplate="%{x}: %{y} scans<extra></extra>"
            ))
            fig_line.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=10, b=0),
                height=260,
                xaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8", size=10)),
                yaxis=dict(showgrid=True, gridcolor="rgba(226,232,240,0.5)",
                           tickfont=dict(color="#94a3b8", size=10), zeroline=False),
                font=dict(family="Plus Jakarta Sans")
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Run more scans over multiple days to see the activity timeline.")

    st.markdown('<div class="app-ft"><p>© 2026 <strong>CropGuard AI</strong></p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  COMMUNITY FORUM PAGE
# ══════════════════════════════════════════════════════
elif app_mode == "💬  Community Forum":
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0f172a,#1e293b); border-radius:20px;
         padding:32px 36px; margin-bottom:28px; position:relative; overflow:hidden; border: 1px solid rgba(99,102,241,0.2);'>
        <div style='position:absolute;top:-40px;right:-40px;width:200px;height:200px;
             border-radius:50%;background:radial-gradient(circle,rgba(99,102,241,0.18),transparent 70%);'></div>
        <p style='font-size:0.7rem;color:#8b5cf6;font-weight:700;letter-spacing:2px;
             text-transform:uppercase;margin:0 0 8px 0;'>EXPERT Q&A</p>
        <h2 style='margin:0 0 6px 0;color:#f1f5f9;font-size:1.8rem;font-weight:800;'>Community Forum</h2>
        <p style='margin:0;color:#94a3b8;font-size:0.9rem;'>Connect, share anonymized scans, and get expert advice.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("➕ Create a New Post", expanded=False):
        with st.form("new_post_form", clear_on_submit=True):
            st.markdown("#### Ask the Community")
            post_disease = st.text_input("Disease or Crop Name (Optional)", placeholder="e.g., Tomato Early Blight")
            post_severity = st.selectbox("Observed Severity", ["Select", "None", "Low", "Moderate", "High"])
            post_question = st.text_area("Your Question *", placeholder="What should I do about these spots?", height=100)
            post_image = st.file_uploader("Upload Image (Optional)", type=["jpg", "png", "jpeg"])
            
            submit_post = st.form_submit_button("Post Question")
            if submit_post:
                if not post_question.strip():
                    st.error("Please ask a question.")
                else:
                    import base64
                    img_data = None
                    if post_image is not None:
                        img_data = base64.b64encode(post_image.read()).decode("utf-8")
                    
                    if add_forum_post(st.session_state.user["username"], post_disease, 
                                      post_severity if post_severity != "Select" else None, 
                                      post_question, img_data):
                        st.success("Post created successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to create post.")

    st.markdown("<hr style='border: 1px solid #1e293b; margin: 30px 0;'>", unsafe_allow_html=True)
    
    posts = get_forum_posts()
    if not posts:
        st.info("No posts yet. Be the first to ask a question!")
    else:
        for post in posts:
            with st.container():
                st.markdown(f"""
                <div style='background:#f8fafc; border-radius:12px; padding:20px; border:1px solid #e2e8f0; margin-bottom:12px;'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:10px;'>
                        <strong><i class='fa-solid fa-user' style='color:#6366f1; margin-right:8px;'></i> {post['user_name']}</strong>
                        <span style='color:#94a3b8; font-size:0.8rem;'>{post['created_at'].strftime('%b %d, %Y %H:%M') if post['created_at'] else ''}</span>
                    </div>
                    {f"<div style='font-size:0.85rem; color:#475569; margin-bottom:10px;'><span style='background:#e0e7ff; color:#3730a3; padding:2px 8px; border-radius:10px;'>{post['disease_name']}</span> <span style='background:#fef3c7; color:#b45309; padding:2px 8px; border-radius:10px;'>Severity: {post['severity']}</span></div>" if post.get('disease_name') or post.get('severity') else ""}
                    <p style='color:#1e293b; font-size:1.05rem; margin-top:5px;'>{post['question']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if post.get("image_data"):
                    st.markdown(f'<img src="data:image/jpeg;base64,{post["image_data"]}" style="max-width:300px; border-radius:10px; margin-bottom:15px; border:1px solid #cbd5e1;">', unsafe_allow_html=True)

                comments = get_forum_comments(post['id'])
                if comments:
                    for comment in comments:
                        expert_badge = "<span style='background:#10b981; color:white; padding:2px 6px; border-radius:4px; font-size:0.7rem; margin-left:8px; font-weight:bold;'><i class='fa-solid fa-circle-check'></i> Expert</span>" if comment.get('is_expert') else ""
                        st.markdown(f"""
                        <div style='background:#ffffff; border-radius:8px; padding:12px 16px; margin:4px 0 4px 20px; border-left:3px solid {"#10b981" if comment.get('is_expert') else "#94a3b8"}; border-top:1px solid #f1f5f9; border-right:1px solid #f1f5f9; border-bottom:1px solid #f1f5f9;'>
                            <div style='font-size:0.85rem; color:#64748b; margin-bottom:6px;'>
                                <strong>{comment['user_name']}</strong> {expert_badge}
                                <span style='float:right; font-size:0.75rem;'>{comment['created_at'].strftime('%b %d, %H:%M') if comment['created_at'] else ''}</span>
                            </div>
                            <p style='color:#334155; margin:0; font-size:0.95rem;'>{comment['comment']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Add comment section
                with st.expander(f"💬 Reply to {post['user_name']}", expanded=False):
                    with st.form(f"comment_form_{post['id']}", clear_on_submit=True):
                        new_comment = st.text_area("Write a reply...", height=80, key=f"text_{post['id']}")
                        
                        # Just a simple expert checkbox for demo purposes. 
                        # In production this would be linked to the user's role in the DB.
                        is_expert_reply = st.checkbox("Post as Expert", value=False)
                        
                        if st.form_submit_button("Submit Reply"):
                            if new_comment.strip():
                                if add_forum_comment(post['id'], st.session_state.user["username"], new_comment, is_expert_reply):
                                    st.success("Reply added!")
                                    st.rerun()
                                else:
                                    st.error("Failed to add reply.")
                            else:
                               st.error("Comment cannot be empty.")
                
                st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown('<div class="app-ft"><p>© 2026 <strong>CropGuard AI</strong></p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  OUTBREAK MAP PAGE
# ══════════════════════════════════════════════════════
elif app_mode == "🗺️  Outbreak Map":
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown("""
    <div style='background:linear-gradient(135deg,#052e16,#065f46); border-radius:20px;
         padding:32px 36px; margin-bottom:28px; position:relative; overflow:hidden; border: 1px solid rgba(16,185,129,0.2);'>
        <div style='position:absolute;top:-40px;right:-40px;width:200px;height:200px;
             border-radius:50%;background:radial-gradient(circle,rgba(16,185,129,0.18),transparent 70%);'></div>
        <p style='font-size:0.7rem;color:#10b981;font-weight:700;letter-spacing:2px;
             text-transform:uppercase;margin:0 0 8px 0;'>LIVE TRACKING</p>
        <h2 style='margin:0 0 6px 0;color:#f1f5f9;font-size:1.8rem;font-weight:800;'>Disease Outbreak Map</h2>
        <p style='margin:0;color:#a7f3d0;font-size:0.9rem;'>
            Monitor crop disease spread across regions. Enable location sharing when scanning to contribute to this map.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Demo seed data (India regions) shown when no real DB data ──
    DEMO_DATA = [
        {"disease_name": "Tomato -- Late blight",       "severity": "High",     "latitude": 18.52, "longitude": 73.85,  "region": "Pune, MH"},
        {"disease_name": "Tomato -- Early blight",      "severity": "Moderate", "latitude": 19.07, "longitude": 72.87,  "region": "Mumbai, MH"},
        {"disease_name": "Potato -- Late blight",       "severity": "High",     "latitude": 28.61, "longitude": 77.20,  "region": "Delhi"},
        {"disease_name": "Corn -- Northern Leaf Blight","severity": "Moderate", "latitude": 22.57, "longitude": 88.36,  "region": "Kolkata, WB"},
        {"disease_name": "Tomato -- Bacterial spot",    "severity": "Moderate", "latitude": 13.08, "longitude": 80.27,  "region": "Chennai, TN"},
        {"disease_name": "Grape -- Black rot",          "severity": "High",     "latitude": 17.38, "longitude": 78.49,  "region": "Hyderabad, TS"},
        {"disease_name": "Apple -- Apple scab",         "severity": "Low",      "latitude": 34.08, "longitude": 74.79,  "region": "Srinagar, JK"},
        {"disease_name": "Potato -- Early blight",      "severity": "Moderate", "latitude": 26.85, "longitude": 80.94,  "region": "Lucknow, UP"},
        {"disease_name": "Tomato -- Leaf Mold",         "severity": "Low",      "latitude": 23.02, "longitude": 72.57,  "region": "Ahmedabad, GJ"},
        {"disease_name": "Tomato -- healthy",           "severity": "None",     "latitude": 12.97, "longitude": 77.59,  "region": "Bengaluru, KA"},
        {"disease_name": "Corn -- Common rust",         "severity": "Moderate", "latitude": 21.25, "longitude": 81.63,  "region": "Raipur, CG"},
        {"disease_name": "Apple -- Cedar apple rust",   "severity": "Moderate", "latitude": 31.10, "longitude": 77.17,  "region": "Shimla, HP"},
        {"disease_name": "Tomato -- Septoria leaf spot","severity": "High",     "latitude": 25.59, "longitude": 85.13,  "region": "Patna, BR"},
        {"disease_name": "Grape -- Esca",               "severity": "High",     "latitude": 16.70, "longitude": 74.24,  "region": "Kolhapur, MH"},
        {"disease_name": "Tomato -- Late blight",       "severity": "High",     "latitude": 27.10, "longitude": 78.01,  "region": "Agra, UP"},
        {"disease_name": "Potato -- healthy",           "severity": "None",     "latitude": 30.73, "longitude": 76.78,  "region": "Chandigarh"},
        {"disease_name": "Tomato -- Target Spot",       "severity": "Moderate", "latitude": 21.14, "longitude": 79.08,  "region": "Nagpur, MH"},
        {"disease_name": "Corn -- Gray leaf spot",      "severity": "Low",      "latitude": 26.44, "longitude": 74.63,  "region": "Ajmer, RJ"},
        {"disease_name": "Strawberry -- Leaf scorch",   "severity": "Low",      "latitude": 28.02, "longitude": 73.31,  "region": "Bikaner, RJ"},
        {"disease_name": "Orange -- Citrus greening",   "severity": "High",     "latitude": 15.49, "longitude": 73.82,  "region": "Goa"},
    ]

    # Fetch real DB data; fall back to demo if empty
    raw_map_data = get_all_scans_with_location()
    using_demo   = len(raw_map_data) == 0

    # ── Debug: show what's in the DB ──
    with st.expander("🔍 Debug: Check DB Location Data", expanded=False):
        if not raw_map_data:
            st.warning("No scans with location found in the database. Make sure you checked '✅ Share my scan on the Live Map' before analyzing.")
        else:
            st.success(f"✅ Found **{len(raw_map_data)}** scan(s) with location data in the database:")
            debug_df = pd.DataFrame(raw_map_data)
            st.dataframe(debug_df[["disease_name","severity","latitude","longitude","scanned_at"]], use_container_width=True, hide_index=True)


    if using_demo:
        map_data = DEMO_DATA
        st.info("📍 No real scan location data yet — showing demo data. Enable **'Share my scan on the Live Map'** when running a scan to contribute!")
    else:
        map_data = []
        for r in raw_map_data:
            map_data.append({
                "disease_name": r.get("disease_name", "Unknown"),
                "severity":     r.get("severity", "Unknown"),
                "latitude":     r.get("latitude"),
                "longitude":    r.get("longitude"),
                "region":       "User Reported"
            })

    df = pd.DataFrame(map_data)

    # ── Summary Stat Cards ──
    total_reports  = len(df)
    high_count     = len(df[df["severity"] == "High"])
    top_disease    = df["disease_name"].value_counts().idxmax() if not df.empty else "N/A"
    top_disease_lbl = top_disease.replace("___", " - ").replace("_", " ")

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, val, label, color in [
        (c1, "fa-location-dot",   str(total_reports),           "Total Reports",    "#6366f1"),
        (c2, "fa-triangle-exclamation", str(high_count),        "High Severity",    "#ef4444"),
        (c3, "fa-seedling",       str(df["disease_name"].nunique()), "Disease Types","#10b981"),
        (c4, "fa-map-pin",        str(df["region"].nunique() if "region" in df.columns else "-"), "Regions", "#f59e0b"),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:#fff; border-radius:14px; padding:18px 20px;
                 border:1px solid #e5e7eb; border-top:3px solid {color}; text-align:center;'>
                <i class='fa-solid {icon}' style='color:{color}; font-size:1.3rem;'></i>
                <div style='font-size:1.6rem; font-weight:800; color:#1e293b; margin:8px 0 2px 0;'>{val}</div>
                <div style='font-size:0.7rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:1px;'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filters ──
    fc1, fc2, fc3 = st.columns([1.5, 1.5, 1])
    with fc1:
        sev_opts = ["High", "Moderate", "Low", "None"]
        sev_filter = st.multiselect("🔴 Filter by Severity", sev_opts, default=sev_opts)
    with fc2:
        diseases = sorted(df["disease_name"].unique().tolist())
        dis_filter = st.multiselect("🌿 Filter by Disease", diseases, default=diseases)
    with fc3:
        map_style = st.selectbox("🗺️ Map Style", ["open-street-map", "carto-positron", "carto-darkmatter"])

    filtered_df = df[df["severity"].isin(sev_filter) & df["disease_name"].isin(dis_filter)]

    st.markdown("<br>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No data matches your current filters.")
    else:
        # ── Plotly Map (no Mapbox token needed) ──
        color_map = {
            "High":     "#ef4444",
            "Moderate": "#f97316",
            "Low":      "#eab308",
            "None":     "#10b981",
            "Unknown":  "#64748b"
        }
        size_map = {"High": 20, "Moderate": 14, "Low": 10, "None": 8, "Unknown": 8}

        filtered_df = filtered_df.copy()
        filtered_df["color"]       = filtered_df["severity"].map(lambda x: color_map.get(x, "#64748b"))
        filtered_df["marker_size"] = filtered_df["severity"].map(lambda x: size_map.get(x, 8))
        filtered_df["label"]       = filtered_df["disease_name"].str.replace("___", " - ").str.replace("_", " ")

        fig = px.scatter_mapbox(
            filtered_df,
            lat="latitude",
            lon="longitude",
            color="severity",
            size="marker_size",
            size_max=22,
            hover_name="label",
            hover_data={"severity": True, "region": True, "latitude": False,
                        "longitude": False, "marker_size": False, "color": False},
            color_discrete_map=color_map,
            zoom=4,
            center={"lat": filtered_df["latitude"].median(), "lon": filtered_df["longitude"].median()},
            mapbox_style=map_style,
            height=520,
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                title="Severity",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#e5e7eb",
                borderwidth=1,
                font=dict(size=12)
            ),
            mapbox=dict(zoom=4)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Legend
        st.markdown("""
        <div style="display:flex; justify-content:center; gap:24px; margin-top:8px; flex-wrap:wrap;">
            <div style="display:flex;align-items:center;gap:6px;font-size:0.82rem;color:#475569;">
                <span style="width:14px;height:14px;border-radius:50%;background:#ef4444;display:inline-block;"></span> High Severity</div>
            <div style="display:flex;align-items:center;gap:6px;font-size:0.82rem;color:#475569;">
                <span style="width:14px;height:14px;border-radius:50%;background:#f97316;display:inline-block;"></span> Moderate</div>
            <div style="display:flex;align-items:center;gap:6px;font-size:0.82rem;color:#475569;">
                <span style="width:14px;height:14px;border-radius:50%;background:#eab308;display:inline-block;"></span> Low</div>
            <div style="display:flex;align-items:center;gap:6px;font-size:0.82rem;color:#475569;">
                <span style="width:14px;height:14px;border-radius:50%;background:#10b981;display:inline-block;"></span> Healthy</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Bottom Row: Hotspot Table + Severity Chart ──
        tb1, tb2 = st.columns([1.6, 1])

        with tb1:
            st.markdown("#### 🔥 Disease Hotspots")
            hotspot = (
                filtered_df.groupby(["disease_name", "region", "severity"])
                .size().reset_index(name="Reports")
                .sort_values("Reports", ascending=False)
                .head(10)
            )
            hotspot["Disease"] = hotspot["disease_name"].str.replace("___", " - ").str.replace("_", " ")
            hotspot["Severity Badge"] = hotspot["severity"].map({
                "High":     "🔴 High",
                "Moderate": "🟠 Moderate",
                "Low":      "🟡 Low",
                "None":     "🟢 Healthy",
            }).fillna("⚪ Unknown")
            st.dataframe(
                hotspot[["Disease", "region", "Severity Badge", "Reports"]].rename(columns={"region": "Region"}),
                use_container_width=True, hide_index=True, height=320
            )

        with tb2:
            st.markdown("#### 📊 Severity Breakdown")
            sev_counts = filtered_df["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Count"]
            sev_colors_list = [color_map.get(s, "#64748b") for s in sev_counts["Severity"]]
            fig_sev = go.Figure(go.Bar(
                x=sev_counts["Count"],
                y=sev_counts["Severity"],
                orientation="h",
                marker_color=sev_colors_list,
                text=sev_counts["Count"],
                textposition="outside"
            ))
            fig_sev.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=30, t=10, b=0),
                height=320,
                showlegend=False,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(tickfont=dict(size=13, color="#334155")),
                font=dict(family="Plus Jakarta Sans")
            )
            st.plotly_chart(fig_sev, use_container_width=True)

    st.markdown('<div class="app-ft"><p>© 2026 <strong>CropGuard AI</strong></p></div>', unsafe_allow_html=True)



# ══════════════════════════════════════════════════════
#  EMAIL ALERTS SETTINGS
# ══════════════════════════════════════════════════════
elif app_mode == "📧  Risk Alerts":
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1e1b4b,#064e3b); border-radius:20px;
         padding:32px 36px; margin-bottom:28px; position:relative; overflow:hidden; border: 1px solid rgba(16,185,129,0.2);'>
        <p style='font-size:0.7rem;color:#10b981;font-weight:700;letter-spacing:2px;
             text-transform:uppercase;margin:0 0 8px 0;'>PROACTIVE PREVENTION</p>
        <h2 style='margin:0 0 6px 0;color:#f1f5f9;font-size:1.8rem;font-weight:800;'>Automated Weather & Disease Alerts</h2>
        <p style='margin:0;color:#a7f3d0;font-size:0.9rem;'>Get notified via email before highly infectious diseases like Late Blight strike your area due to high humidity and rain.</p>
    </div>
    """, unsafe_allow_html=True)

    uid = st.session_state.user["id"]
    current_usr = get_user_by_username(st.session_state.user["username"])
    
    is_enabled = current_usr.get("alerts_enabled", False) if current_usr else False
    city_val = current_usr.get("alert_city", "") if current_usr and current_usr.get("alert_city") else ""

    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("### ⚙️ Your Alert Preferences")
        with st.form("alert_settings_form"):
            enable_alerts = st.toggle("Receive Weather Risk Emails", value=is_enabled)
            target_city = st.text_input("Farm / Target City", value=city_val, placeholder="e.g., London")
            
            save_button = st.form_submit_button("Save Preferences")
            
            if save_button:
                if enable_alerts and not target_city.strip():
                    st.error("Please enter a city to enable alerts.")
                else:
                    if update_user_alerts(uid, enable_alerts, target_city.strip()):
                        st.success("Alert preferences updated successfully!")
                        # Update session user details too to avoid stale data
                        st.session_state.user["alerts_enabled"] = enable_alerts
                        st.session_state.user["alert_city"] = target_city.strip()
                        st.rerun()
                    else:
                        st.error("There was an error saving your preferences.")

    with c2:
        st.markdown("### 🧪 Simulate Daily Alert Job")
        st.markdown("""
        <p style='font-size:0.85rem; color:#64748b;'>
        In a production environment, this occurs automatically every morning via a CRON job. 
        Click the button below to manually trigged the script that checks all users' cities via the Weather API and sends out risk emails.
        </p>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Run Alert Automation Now", type="primary", use_container_width=True):
            subscribers = get_alert_users()
            if not subscribers:
                st.info("No users are currently subscribed to alerts.")
            else:
                sent_count = 0
                with st.spinner("Checking conditions and emailing users..."):
                    for sub in subscribers:
                        city = sub['alert_city']
                        weather = get_weather(city)
                        if weather:
                            main_cond = weather['weather'][0]['main']
                            t = weather['main']['temp']
                            h = weather['main']['humidity']
                            
                            risk = assess_disease_risk(t, h, main_cond)
                            
                            if risk['level'] in ["High", "Moderate"]:
                                success = send_alert_email(
                                    to_email=sub['email'], 
                                    username=sub['username'], 
                                    city=city, 
                                    risk_level=risk['level'], 
                                    condition=main_cond, 
                                    message=risk['message']
                                )
                                if success:
                                    sent_count += 1
                                    
                st.success(f"Job completed. Sent {sent_count} alert(s) to users in high/moderate risk zones. Check terminal log for Mock Emails!")
