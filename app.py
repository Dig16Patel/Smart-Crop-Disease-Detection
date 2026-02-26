import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
from utils.preprocess import preprocess_image
from utils.recommendations import get_recommendation
from utils.db import get_user_by_username, create_user, save_scan, get_scan_history, get_disease_frequency, get_daily_scan_counts, get_severity_breakdown
from utils.auth import hash_password, verify_password, validate_registration
from utils.report import generate_report_pdf
from utils.weather import get_weather, assess_disease_risk, weather_icon_emoji

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

# ─── Load Model ───
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/crop_disease_model.h5")

def load_class_indices():
    with open("models/class_indices.json", "r") as f:
        return json.load(f)


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
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); }

    /* Auth card */
    .auth-wrap {
        max-width: 440px;
        margin: 0 auto;
        padding: 40px 0;
    }
    .auth-logo {
        text-align: center;
        margin-bottom: 32px;
    }
    .auth-logo-icon {
        width: 68px; height: 68px;
        border-radius: 18px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 1.8rem;
        box-shadow: 0 10px 30px rgba(99,102,241,0.4);
        margin-bottom: 16px;
    }
    .auth-logo h1 {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        margin: 0 0 4px 0 !important;
    }
    .auth-logo p {
        color: #64748b !important;
        font-size: 0.85rem !important;
        margin: 0 !important;
    }
    .auth-card {
        background: rgba(30,41,59,0.7);
        border: 1px solid rgba(148,163,184,0.1);
        border-radius: 20px;
        padding: 36px 36px;
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .auth-title {
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        color: #f1f5f9 !important;
        margin: 0 0 4px 0 !important;
    }
    .auth-sub {
        font-size: 0.82rem !important;
        color: #64748b !important;
        margin: 0 0 24px 0 !important;
    }

    /* Input styling */
    .stTextInput input {
        background: rgba(15,23,42,0.6) !important;
        border: 1px solid rgba(148,163,184,0.15) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
        padding: 12px 16px !important;
        font-size: 0.9rem !important;
    }
    .stTextInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    }
    .stTextInput label {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }

    /* Primary button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 20px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
    }

    /* Divider */
    .auth-divider {
        display: flex; align-items: center; gap: 12px;
        margin: 20px 0;
    }
    .auth-divider hr {
        flex: 1; border: none;
        border-top: 1px solid rgba(148,163,184,0.12);
        margin: 0;
    }
    .auth-divider span {
        font-size: 0.75rem; color: #475569;
    }

    /* Switch link */
    .auth-switch {
        text-align: center;
        margin-top: 20px;
        font-size: 0.82rem;
        color: #64748b;
    }
    .auth-error {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 10px;
        padding: 12px 16px;
        color: #fca5a5;
        font-size: 0.82rem;
        margin-bottom: 16px;
    }
    .auth-success {
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 10px;
        padding: 12px 16px;
        color: #6ee7b7;
        font-size: 0.82rem;
        margin-bottom: 16px;
    }
    /* Center buttons on auth pages */
    .auth-btn-wrap { text-align: center; margin-top: 6px; }
    .auth-btn-wrap .stButton > button {
        display: inline-block !important;
        width: 100% !important;
    }
    /* Auth card inner header */
    .auth-card-header {
        border-bottom: 1px solid rgba(148,163,184,0.1);
        padding-bottom: 18px;
        margin-bottom: 20px;
    }
    /* Hide sidebar on auth pages */
    section[data-testid="stSidebar"] { display: none !important; }
</style>
"""

def show_login_page():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        # Logo
        st.markdown("""
        <div class="auth-logo">
            <div class="auth-logo-icon">🌿</div>
            <h1>CropGuard AI</h1>
            <p>Intelligent Crop Health Monitor</p>
        </div>
        """, unsafe_allow_html=True)

        # Card header (title + subtitle as pure HTML — no raw open divs)
        st.markdown("""
        <div style='background:rgba(30,41,59,0.7); border:1px solid rgba(148,163,184,0.1);
             border-radius:20px; padding:32px 32px 16px 32px;
             backdrop-filter:blur(20px); box-shadow:0 20px 60px rgba(0,0,0,0.3);'>
            <div class="auth-card-header">
                <p class="auth-title">Welcome back 👋</p>
                <p class="auth-sub">Sign in to continue to CropGuard AI</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter your username", key="login_username")
        password = st.text_input("Password", placeholder="Enter your password", type="password", key="login_password")

        if st.button("Sign In →", key="login_btn", use_container_width=True):
            if not username or not password:
                st.markdown('<div class="auth-error"><i class="fa-solid fa-circle-exclamation"></i> Please fill in all fields.</div>', unsafe_allow_html=True)
            else:
                user = get_user_by_username(username)
                if user and verify_password(password, user["password_hash"]):
                    st.session_state.logged_in = True
                    st.session_state.user = {"id": user["id"], "username": user["username"], "email": user["email"]}
                    st.rerun()
                else:
                    st.markdown('<div class="auth-error"><i class="fa-solid fa-triangle-exclamation"></i> Invalid username or password.</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="auth-divider"><hr><span>NEW HERE?</span><hr></div>
        """, unsafe_allow_html=True)

        if st.button("Create an Account", key="goto_register", use_container_width=True):
            st.session_state.auth_page = "register"
            st.rerun()


def show_register_page():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div class="auth-logo">
            <div class="auth-logo-icon">🌿</div>
            <h1>CropGuard AI</h1>
            <p>Create your free account</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background:rgba(30,41,59,0.7); border:1px solid rgba(148,163,184,0.1);
             border-radius:20px; padding:32px 32px 16px 32px;
             backdrop-filter:blur(20px); box-shadow:0 20px 60px rgba(0,0,0,0.3);'>
            <div class="auth-card-header">
                <p class="auth-title">Create Account</p>
                <p class="auth-sub">Join CropGuard AI to save your scan history</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Choose a username", key="reg_username")
        email    = st.text_input("Email", placeholder="Enter your email", key="reg_email")
        password = st.text_input("Password", placeholder="Create a password (min 6 chars)", type="password", key="reg_password")
        confirm  = st.text_input("Confirm Password", placeholder="Repeat your password", type="password", key="reg_confirm")

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

        st.markdown("""
        <div class="auth-divider"><hr><span>ALREADY HAVE AN ACCOUNT?</span><hr></div>
        """, unsafe_allow_html=True)

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

    app_mode = st.radio("Nav", ["🏠  Home", "🔬  Detect Disease", "📜  History", "📊  Dashboard"], label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(148,163,184,0.08); margin:12px 16px;'>", unsafe_allow_html=True)

    # Quick Stats
    st.markdown("<p style='font-size:0.62rem; color:#475569; font-weight:700; text-transform:uppercase; letter-spacing:1.8px; padding:0 12px; margin-bottom:8px;'><i class='fa-solid fa-chart-simple' style='margin-right:6px; color:#10b981;'></i>Quick Stats</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px; padding:0 8px; margin-bottom:12px;'>
        <div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.12); border-radius:10px; padding:12px 8px; text-align:center;'>
            <div style='font-size:1.2rem; font-weight:800; color:#a5b4fc;'>6+</div>
            <div style='font-size:0.6rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;'>Diseases</div>
        </div>
        <div style='background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.12); border-radius:10px; padding:12px 8px; text-align:center;'>
            <div style='font-size:1.2rem; font-weight:800; color:#6ee7b7;'>92%</div>
            <div style='font-size:0.6rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;'>Accuracy</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(148,163,184,0.08); margin:4px 16px 12px 16px;'>", unsafe_allow_html=True)

    # Supported Crops
    st.markdown("<p style='font-size:0.62rem; color:#475569; font-weight:700; text-transform:uppercase; letter-spacing:1.8px; padding:0 12px; margin-bottom:8px;'><i class='fa-solid fa-seedling' style='margin-right:6px; color:#f59e0b;'></i>Supported Crops</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding:0 8px;'>
        <div style='display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:8px; margin-bottom:4px; transition:background 0.2s;'>
            <span style='font-size:1rem;'>🍅</span>
            <span style='font-size:0.78rem; color:#cbd5e1; font-weight:500;'>Tomato</span>
        </div>
        <div style='display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:8px; margin-bottom:4px;'>
            <span style='font-size:1rem;'>🌽</span>
            <span style='font-size:0.78rem; color:#cbd5e1; font-weight:500;'>Corn / Maize</span>
        </div>
        <div style='display:flex; align-items:center; gap:10px; padding:8px 12px; border-radius:8px; margin-bottom:4px;'>
            <span style='font-size:1rem;'>🥔</span>
            <span style='font-size:0.78rem; color:#cbd5e1; font-weight:500;'>Potato</span>
        </div>
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
                    <div class="hero-stat-val">3+</div>
                    <div class="hero-stat-lbl">Crop Types</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-val">6+</div>
                    <div class="hero-stat-lbl">Diseases</div>
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
            predict_clicked = st.button("🔍  Analyze Disease", use_container_width=True)
        else:
            predict_clicked = False
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

        if uploaded_file is not None and predict_clicked:
            with st.spinner("🧠 AI is analyzing the image..."):
                try:
                    processed_image = preprocess_image(image)
                    model = load_model()
                    class_indices = load_class_indices()
                    class_names = {v: k for k, v in class_indices.items()}

                    predictions = model.predict(processed_image)
                    idx = np.argmax(predictions)
                    name = class_names[idx]
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

                    display_name = name.replace("_", " ")

                    # Auto-save scan to database
                    save_scan(
                        user_id=st.session_state.user["id"],
                        disease_name=display_name,
                        confidence=round(conf, 2),
                        severity=severity
                    )

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

                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
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

        for scan in scans:
            sev   = scan.get("severity", "None")
            color = sev_colors.get(sev, "#94a3b8")
            bg    = sev_bg.get(sev, "#f8fafc")
            ts    = scan["scanned_at"].strftime("%d %b %Y, %I:%M %p") if scan.get("scanned_at") else "N/A"
            st.markdown(f"""
            <div style='background:#fff;border:1px solid #e5e7eb;border-left:4px solid {color};
                 border-radius:14px;padding:18px 22px;margin-bottom:10px;
                 display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;'>
                <div style='flex:1;min-width:180px;'>
                    <div style='font-size:0.95rem;font-weight:700;color:#1e293b;margin-bottom:4px;'>
                        <i class='fa-solid fa-leaf' style='color:{color};margin-right:6px;'></i>
                        {scan["disease_name"]}
                    </div>
                    <div style='font-size:0.75rem;color:#94a3b8;'>
                        <i class='fa-regular fa-clock' style='margin-right:4px;'></i>{ts}
                    </div>
                </div>
                <div style='display:flex;gap:12px;align-items:center;'>
                    <div style='text-align:center;'>
                        <div style='font-size:0.65rem;color:#94a3b8;font-weight:600;
                             text-transform:uppercase;letter-spacing:1px;margin-bottom:2px;'>Confidence</div>
                        <div style='font-size:1rem;font-weight:700;color:#6366f1;'>{scan["confidence"]:.1f}%</div>
                    </div>
                    <div style='background:{bg};color:{color};font-size:0.72rem;font-weight:700;
                         padding:5px 14px;border-radius:20px;border:1px solid {color};
                         text-transform:uppercase;letter-spacing:0.5px;'>{sev}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

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

    st.markdown('<div class="app-ft"><p>\u00a9 2026 <strong>CropGuard AI</strong></p></div>', unsafe_allow_html=True)
