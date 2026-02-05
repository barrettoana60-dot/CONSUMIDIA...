import streamlit as st
import datetime
import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from collections import Counter
from pathlib import Path
from PIL import Image
import uuid

# ======================================================
# CONFIG BÁSICA
# ======================================================

st.set_page_config(
    page_title="PQR – Social Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STATE_FILE = "pqr_state.json"
AVATAR_DIR = Path("avatars")
AVATAR_DIR.mkdir(exist_ok=True)

# ======================================================
# CSS – EXATAMENTE COMO O ANVIL (CINZA CLARO, CARDS BRANCOS)
# ======================================================

CSS = """
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

.stApp {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    background-color: #f0f2f5;
    color: #333333;
}

[data-testid="stSidebar"] { display: none; }

.main-container {
    display: flex;
    gap: 20px;
    padding: 20px;
    max-width: 1600px;
    margin: 0 auto;
}

/* SIDEBAR */
.sidebar {
    width: 280px;
    background: #ffffff;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    height: fit-content;
    position: sticky;
    top: 20px;
}

.sidebar-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 30px;
}

.sidebar-avatar {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    font-size: 18px;
    overflow: hidden;
}

.sidebar-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.sidebar-user-info {
    display: flex;
    flex-direction: column;
}

.sidebar-user-name {
    font-size: 14px;
    font-weight: 600;
    color: #333;
}

.sidebar-user-role {
    font-size: 12px;
    color: #999;
}

.sidebar-menu {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.sidebar-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    color: #666;
    transition: all 0.2s;
    user-select: none;
}

.sidebar-item:hover {
    background-color: #f5f5f5;
    color: #333;
}

.sidebar-item-active {
    background-color: #e8eaf6;
    color: #667eea;
    font-weight: 600;
}

.sidebar-icon {
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}

.sidebar-label {
    flex: 1;
}

/* CONTEÚDO PRINCIPAL */
.content-wrapper {
    display: flex;
    gap: 20px;
    flex: 1;
}

/* PROFILE CARD */
.profile-card {
    width: 320px;
    background: #ffffff;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    height: fit-content;
    position: sticky;
    top: 20px;
}

.profile-header {
    text-align: center;
    margin-bottom: 20px;
}

.profile-avatar-big {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    margin: 0 auto 16px;
    background: #e0e0e0;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
}

.profile-avatar-big img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.profile-name {
    font-size: 18px;
    font-weight: 600;
    color: #333;
    margin-bottom: 4px;
}

.profile-role {
    font-size: 13px;
    color: #999;
    margin-bottom: 16px;
}

.profile-follow-btn {
    width: 100%;
    padding: 10px;
    border: none;
    border-radius: 8px;
    background-color: #b3d9ff;
    color: #0066cc;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}

.profile-follow-btn:hover {
    background-color: #99ccff;
}

.profile-divider {
    height: 1px;
    background-color: #e8e8e8;
    margin: 16px 0;
}

.profile-bio {
    font-size: 13px;
    color: #666;
    line-height: 1.5;
    margin-bottom: 16px;
}

.profile-stats {
    display: flex;
    justify-content: space-around;
    padding: 12px 0;
    border-top: 1px solid #e8e8e8;
    border-bottom: 1px solid #e8e8e8;
    margin-bottom: 16px;
}

.profile-stat {
    text-align: center;
}

.profile-stat-value {
    font-size: 16px;
    font-weight: 600;
    color: #333;
}

.profile-stat-label {
    font-size: 11px;
    color: #999;
    margin-top: 4px;
}

.profile-interests {
    font-size: 12px;
    color: #666;
}

.profile-interests-label {
    font-weight: 600;
    margin-bottom: 8px;
}

.profile-interest-tag {
    display: inline-block;
    background-color: #f0f0f0;
    padding: 4px 8px;
    border-radius: 4px;
    margin-right: 4px;
    margin-bottom: 4px;
    font-size: 11px;
}

/* FEED CARD */
.feed-card {
    flex: 1;
    background: #ffffff;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.feed-header {
    font-size: 16px;
    font-weight: 600;
    color: #333;
    margin-bottom: 8px;
}

.feed-subheader {
    font-size: 12px;
    color: #999;
    margin-bottom: 20px;
}

.feed-item {
    display: flex;
    gap: 12px;
    padding: 16px 0;
    border-bottom: 1px solid #e8e8e8;
}

.feed-item:last-child {
    border-bottom: none;
}

.feed-item-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: #e0e0e0;
    flex-shrink: 0;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
}

.feed-item-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.feed-item-content {
    flex: 1;
}

.feed-item-name {
    font-size: 13px;
    font-weight: 600;
    color: #333;
}

.feed-item-action {
    font-size: 13px;
    color: #666;
}

.feed-item-time {
    font-size: 12px;
    color: #999;
    margin-top: 4px;
}

.feed-item-text {
    font-size: 13px;
    color: #666;
    margin-top: 8px;
    line-height: 1.4;
    font-style: italic;
}

.feed-item-heart {
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 16px;
    flex-shrink: 0;
}

/* FORM INPUTS */
.form-group {
    margin-bottom: 16px;
}

.form-label {
    display: block;
    font-size: 13px;
    font-weight: 600;
    color: #333;
    margin-bottom: 6px;
}

.form-input {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid #ddd;
    border-radius: 6px;
    font-size: 13px;
    font-family: inherit;
}

.form-input:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.form-textarea {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid #ddd;
    border-radius: 6px;
    font-size: 13px;
    font-family: inherit;
    resize: vertical;
}

.form-textarea:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.btn-primary {
    padding: 10px 20px;
    border: none;
    border-radius: 6px;
    background-color: #667eea;
    color: white;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-primary:hover {
    background-color: #5568d3;
}

.btn-secondary {
    padding: 10px 20px;
    border: 1px solid #ddd;
    border-radius: 6px;
    background-color: #f5f5f5;
    color: #333;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-secondary:hover {
    background-color: #eeeeee;
}

/* AUTH SCREEN */
.auth-container {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.auth-card {
    background: white;
    border-radius: 12px;
    padding: 40px;
    width: 100%;
    max-width: 400px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
}

.auth-title {
    font-size: 24px;
    font-weight: 700;
    color: #333;
    margin-bottom: 8px;
}

.auth-subtitle {
    font-size: 13px;
    color: #999;
    margin-bottom: 24px;
}

.auth-tabs {
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
    border-bottom: 1px solid #e8e8e8;
}

.auth-tab {
    padding: 12px 0;
    border: none;
    background: none;
    color: #999;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}

.auth-tab-active {
    color: #667eea;
    border-bottom-color: #667eea;
}

.auth-form {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.auth-input {
    padding: 10px 12px;
    border: 1px solid #ddd;
    border-radius: 6px;
    font-size: 13px;
    font-family: inherit;
}

.auth-input:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.auth-btn {
    padding: 12px;
    border: none;
    border-radius: 6px;
    background-color: #667eea;
    color: white;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}

.auth-btn:hover {
    background-color: #5568d3;
}

.auth-link {
    text-align: center;
    font-size: 12px;
    color: #999;
}

.auth-link a {
    color: #667eea;
    text-decoration: none;
    cursor: pointer;
}

.auth-link a:hover {
    text-decoration: underline;
}

/* NOTIFICAÇÕES */
.notification-success {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 13px;
}

.notification-error {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 13px;
}

.notification-info {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 13px;
}

/* MODAL / EXPANDER */
.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}

.modal-content {
    background: white;
    border-radius: 12px;
    padding: 32px;
    width: 100%;
    max-width: 500px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}

.modal-title {
    font-size: 18px;
    font-weight: 600;
    color: #333;
    margin-bottom: 16px;
}

.modal-close {
    position: absolute;
    top: 16px;
    right: 16px;
    background: none;
    border: none;
    font-size: 24px;
    cursor: pointer;
    color: #999;
}

/* RESPONSIVE */
@media (max-width: 1200px) {
    .main-container {
        flex-direction: column;
    }
    .sidebar {
        width: 100%;
        position: static;
    }
    .content-wrapper {
        flex-direction: column;
    }
    .profile-card {
        width: 100%;
        position: static;
    }
    .feed-card {
        width: 100%;
    }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ======================================================
# DATACLASSES
# ======================================================

@dataclass
class User:
    id: str
    name: str
    email: str
    password: str
    role: str = "Pesquisador"
    bio: str = ""
    avatar_path: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    followers: List[str] = field(default_factory=list)
    following: List[str] = field(default_factory=list)

@dataclass
class Post:
    id: str
    author_id: str
    content: str
    tags: List[str] = field(default_factory=list)
    likes: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    shares: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

@dataclass
class Comment:
    id: str
    post_id: str
    author_id: str
    text: str
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

@dataclass
class TimelineStep:
    id: str
    owner_id: str
    title: str
    description: str
    status: str = "Planejado"
    due_date: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))

@dataclass
class ResearchDoc:
    id: str
    owner_id: str
    title: str
    category: str
    content: str
    url: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))

@dataclass
class MindNode:
    id: str
    text: str
    children: List['MindNode'] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class Slide:
    id: str
    owner_id: str
    title: str
    objective: str
    content: str
    theme: str = "default"
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))

@dataclass
class ChatMessage:
    id: str
    from_id: str
    text: str
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

@dataclass
class Notification:
    id: str
    user_id: str
    type: str
    from_user_id: str
    action: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

@dataclass
class Activity:
    id: str
    user_id: str
    type: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

# ======================================================
# ESTADO GLOBAL
# ======================================================

def init_session_state():
    if "users" not in st.session_state:
        st.session_state.users = []
    if "posts" not in st.session_state:
        st.session_state.posts = []
    if "comments" not in st.session_state:
        st.session_state.comments = []
    if "timeline" not in st.session_state:
        st.session_state.timeline = []
    if "docs" not in st.session_state:
        st.session_state.docs = []
    if "mindmap" not in st.session_state:
        st.session_state.mindmap = None
    if "slides" not in st.session_state:
        st.session_state.slides = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "notifications" not in st.session_state:
        st.session_state.notifications = []
    if "activities" not in st.session_state:
        st.session_state.activities = []
    if "current_user_id" not in st.session_state:
        st.session_state.current_user_id = None
    if "current_view" not in st.session_state:
        st.session_state.current_view = "Pages"
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

init_session_state()

# ======================================================
# PERSISTÊNCIA
# ======================================================

def save_state_to_file():
    data = {
        "users": [asdict(u) for u in st.session_state.users],
        "posts": [asdict(p) for p in st.session_state.posts],
        "comments": [asdict(c) for c in st.session_state.comments],
        "timeline": [asdict(t) for t in st.session_state.timeline],
        "docs": [asdict(d) for d in st.session_state.docs],
        "slides": [asdict(s) for s in st.session_state.slides],
        "chat_messages": [asdict(m) for m in st.session_state.chat_messages],
        "notifications": [asdict(n) for n in st.session_state.notifications],
        "activities": [asdict(a) for a in st.session_state.activities],
    }
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_state_from_file():
    if not os.path.exists(STATE_FILE):
        return

    with open(STATE_FILE, "r") as f:
        data = json.load(f)

    st.session_state.users = [User(**u) for u in data.get("users", [])]
    st.session_state.posts = [Post(**p) for p in data.get("posts", [])]
    st.session_state.comments = [Comment(**c) for c in data.get("comments", [])]
    st.session_state.timeline = [TimelineStep(**t) for t in data.get("timeline", [])]
    st.session_state.docs = [ResearchDoc(**d) for d in data.get("docs", [])]
    st.session_state.slides = [Slide(**s) for s in data.get("slides", [])]
    st.session_state.chat_messages = [ChatMessage(**m) for m in data.get("chat_messages", [])]
    st.session_state.notifications = [Notification(**n) for n in data.get("notifications", [])]
    st.session_state.activities = [Activity(**a) for a in data.get("activities", [])]

load_state_from_file()

# ======================================================
# HELPERS
# ======================================================

def get_current_user() -> Optional[User]:
    if not st.session_state.current_user_id:
        return None
    for u in st.session_state.users:
        if u.id == st.session_state.current_user_id:
            return u
    return None

def avatar_html(user: User, size: int = 40) -> str:
    if user.avatar_path and os.path.exists(user.avatar_path):
        return f'<img src="file://{os.path.abspath(user.avatar_path)}" style="width:{size}px;height:{size}px;border-radius:50%;object-fit:cover;">'
    else:
        initials = "".join([n[0].upper() for n in user.name.split()[:2]])
        return f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:linear-gradient(135deg,#667eea,#764ba2);display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:{size//2}px;">{initials}</div>'

# ======================================================
# TELA DE AUTENTICAÇÃO
# ======================================================

def auth_screen():
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)

    st.markdown('<div class="auth-title">PQR</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Social Research Dashboard</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Entrar", key="tab_login", use_container_width=True):
            st.session_state.auth_mode = "login"
    with col2:
        if st.button("Criar conta", key="tab_signup", use_container_width=True):
            st.session_state.auth_mode = "signup"

    st.markdown("---")

    if st.session_state.auth_mode == "login":
        st.markdown("#### Entrar na sua conta")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Senha", type="password", key="login_password")

        if st.button("Entrar", key="btn_login", use_container_width=True):
            user = next((u for u in st.session_state.users if u.email == email and u.password == password), None)
            if user:
                st.session_state.current_user_id = user.id
                st.success("Bem-vindo!")
                st.experimental_rerun()
            else:
                st.error("Email ou senha incorretos.")

    else:
        st.markdown("#### Criar nova conta")
        name = st.text_input("Nome completo", key="signup_name")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Senha", type="password", key="signup_password")
        password_confirm = st.text_input("Confirmar senha", type="password", key="signup_confirm")

        if st.button("Criar conta", key="btn_signup", use_container_width=True):
            if not name or not email or not password:
                st.error("Preencha todos os campos.")
            elif password != password_confirm:
                st.error("As senhas não conferem.")
            elif any(u.email == email for u in st.session_state.users):
                st.error("Este email já está registrado.")
            else:
                new_user = User(
                    id=str(uuid.uuid4()),
                    name=name,
                    email=email,
                    password=password,
                )
                st.session_state.users.append(new_user)
                save_state_to_file()
                st.session_state.current_user_id = new_user.id
                st.success("Conta criada com sucesso!")
                st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================

def render_sidebar(user: User):
    st.markdown('<div class="sidebar">', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-header">', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-avatar">{avatar_html(user, 44)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-user-info">', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-user-name">{user.name.split()[0]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-user-role">{user.role}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-menu">', unsafe_allow_html=True)

    menu_items = [
        ("📄", "Pages"),
        ("📊", "Analytics"),
        ("✓", "Tasks"),
        ("💰", "Invoice"),
        ("⭐", "Subscribe"),
        ("✉️", "Contact"),
        ("⚙️", "Settings"),
    ]

    for icon, label in menu_items:
        is_active = st.session_state.current_view == label
        active_class = "sidebar-item-active" if is_active else ""
        st.markdown(f'<div class="sidebar-item {active_class}" onclick="alert(\'{label}\')">', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-icon">{icon}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-label">{label}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.session_state.current_view = label
            save_state_to_file()

