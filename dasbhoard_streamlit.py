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
import base64
import io

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
# CSS – EXATAMENTE COMO O ANVIL
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
.content-wrapper {
    display: flex;
    gap: 20px;
    flex: 1;
}
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
    background-color: #e0e0e0;
    margin: 16px 0;
}
.profile-stat {
    display: flex;
    justify-content: space-around;
    margin: 16px 0;
    text-align: center;
}
.profile-stat-item {
    flex: 1;
}
.profile-stat-number {
    font-size: 16px;
    font-weight: 600;
    color: #333;
}
.profile-stat-label {
    font-size: 12px;
    color: #999;
}
.profile-bio {
    font-size: 13px;
    color: #666;
    line-height: 1.5;
    margin: 16px 0;
}
.profile-interests {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 12px;
}
.profile-interest-tag {
    display: inline-block;
    background-color: #f0f0f0;
    color: #666;
    padding: 4px 10px;
    border-radius: 16px;
    font-size: 12px;
}
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
    margin-bottom: 20px;
}
.feed-item {
    display: flex;
    gap: 12px;
    padding: 16px 0;
    border-bottom: 1px solid #f0f0f0;
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
.feed-item-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
}
.feed-item-name {
    font-size: 13px;
    font-weight: 600;
    color: #333;
}
.feed-item-time {
    font-size: 12px;
    color: #999;
}
.feed-item-text {
    font-size: 13px;
    color: #666;
    line-height: 1.4;
}
.feed-item-action {
    font-size: 12px;
    color: #999;
    margin-top: 6px;
}
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
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    width: 100%;
    max-width: 400px;
}
.auth-title {
    font-size: 28px;
    font-weight: 700;
    color: #333;
    text-align: center;
    margin-bottom: 8px;
}
.auth-subtitle {
    font-size: 14px;
    color: #999;
    text-align: center;
    margin-bottom: 24px;
}
.post-card {
    background: #f9f9f9;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
}
.post-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
}
.post-author {
    font-size: 13px;
    font-weight: 600;
    color: #333;
}
.post-time {
    font-size: 12px;
    color: #999;
}
.post-content {
    font-size: 13px;
    color: #666;
    line-height: 1.4;
    margin-bottom: 8px;
}
.post-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.post-tag {
    display: inline-block;
    background-color: #e8eaf6;
    color: #667eea;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
}
.post-actions {
    display: flex;
    gap: 12px;
    margin-top: 8px;
    font-size: 12px;
}
.post-action {
    cursor: pointer;
    color: #999;
}
.post-action:hover {
    color: #667eea;
}
.avatar-upload-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 20px;
}
.avatar-upload-btn {
    margin-top: 10px;
    background: #667eea;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    cursor: pointer;
    font-size: 12px;
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
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@dataclass
class Comment:
    id: str
    post_id: str
    author_id: str
    text: str
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@dataclass
class TimelineStep:
    id: str
    owner_id: str
    title: str
    description: str
    status: str = "Não iniciado"
    due_date: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))

@dataclass
class ResearchDoc:
    id: str
    owner_id: str
    title: str
    category: str
    content: str
    link: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))

@dataclass
class MindNode:
    id: str
    owner_id: str
    title: str
    description: str
    tags: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

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
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@dataclass
class Notification:
    id: str
    user_id: str
    type: str
    from_user_id: str
    text: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ======================================================
# INICIALIZAÇÃO DE SESSION STATE
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
    if "mindnodes" not in st.session_state:
        st.session_state.mindnodes = []
    if "slides" not in st.session_state:
        st.session_state.slides = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "notifications" not in st.session_state:
        st.session_state.notifications = []
    if "current_user_id" not in st.session_state:
        st.session_state.current_user_id = None
    if "current_view" not in st.session_state:
        st.session_state.current_view = "Pages"
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"
    if "avatar_upload" not in st.session_state:
        st.session_state.avatar_upload = None

init_session_state()

# ======================================================
# FUNÇÕES DE ESTADO
# ======================================================
def save_state_to_file():
    data = {
        "users": [asdict(u) for u in st.session_state.users],
        "posts": [asdict(p) for p in st.session_state.posts],
        "comments": [asdict(c) for c in st.session_state.comments],
        "timeline": [asdict(t) for t in st.session_state.timeline],
        "docs": [asdict(d) for d in st.session_state.docs],
        "mindnodes": [asdict(m) for m in st.session_state.mindnodes],
        "slides": [asdict(s) for s in st.session_state.slides],
        "chat_messages": [asdict(m) for m in st.session_state.chat_messages],
        "notifications": [asdict(n) for n in st.session_state.notifications],
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
    st.session_state.mindnodes = [MindNode(**m) for m in data.get("mindnodes", [])]
    st.session_state.slides = [Slide(**s) for s in data.get("slides", [])]
    st.session_state.chat_messages = [ChatMessage(**m) for m in data.get("chat_messages", [])]
    st.session_state.notifications = [Notification(**n) for n in data.get("notifications", [])]

load_state_from_file()

# ======================================================
# FUNÇÕES AUXILIARES
# ======================================================
def get_current_user() -> Optional[User]:
    if not st.session_state.current_user_id:
        return None
    return next((u for u in st.session_state.users if u.id == st.session_state.current_user_id), None)

def avatar_html(user: User, size: int = 44) -> str:
    if user.avatar_path and os.path.exists(user.avatar_path):
        return f'<img src="file://{os.path.abspath(user.avatar_path)}" style="width:{size}px;height:{size}px;border-radius:50%;object-fit:cover;">'
    initials = "".join([n[0].upper() for n in user.name.split()[:2]])
    return f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:linear-gradient(135deg,#667eea,#764ba2);display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:{size//2}px;">{initials}</div>'

def save_avatar(user_id: str, image_bytes: bytes) -> str:
    avatar_path = AVATAR_DIR / f"{user_id}.png"
    with open(avatar_path, "wb") as f:
        f.write(image_bytes)
    return str(avatar_path)

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
        if st.button(f"{icon} {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.current_view = label
            save_state_to_file()
            st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# PROFILE CARD
# ======================================================
def render_profile_card(user: User):
    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.markdown('<div class="profile-header">', unsafe_allow_html=True)

    # Avatar upload section
    st.markdown('<div class="avatar-upload-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="profile-avatar-big">{avatar_html(user, 120)}</div>', unsafe_allow_html=True)

    # File uploader for avatar
    uploaded_file = st.file_uploader("Escolha uma imagem", type=["png", "jpg", "jpeg"], key="avatar_upload")
    if uploaded_file is not None:
        # Save the uploaded file
        avatar_path = save_avatar(user.id, uploaded_file.getvalue())
        user.avatar_path = avatar_path

        # Update user in session state
        for i, u in enumerate(st.session_state.users):
            if u.id == user.id:
                st.session_state.users[i] = user
                break

        save_state_to_file()
        st.success("Avatar atualizado!")
        st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)  # Close avatar-upload-container

    st.markdown(f'<div class="profile-name">{user.name}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="profile-role">{user.role}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Follow", key="btn_follow", use_container_width=True):
        st.success("Você está seguindo este usuário!")

    st.markdown('<div class="profile-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-stat">', unsafe_allow_html=True)
    st.markdown('<div class="profile-stat-item"><div class="profile-stat-number">42</div><div class="profile-stat-label">Posts</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-stat-item"><div class="profile-stat-number">128</div><div class="profile-stat-label">Seguidores</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-stat-item"><div class="profile-stat-number">89</div><div class="profile-stat-label">Seguindo</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="profile-bio">{user.bio or "Sem bio ainda"}</div>', unsafe_allow_html=True)

    if user.interests:
        st.markdown('<div class="profile-interests">', unsafe_allow_html=True)
        for interest in user.interests:
            st.markdown(f'<span class="profile-interest-tag">{interest}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# FEED CARD
# ======================================================
def render_feed_card():
    st.markdown('<div class="feed-card">', unsafe_allow_html=True)
    st.markdown('<div class="feed-header">Atividades recentes</div>', unsafe_allow_html=True)
    # Feed de exemplo
    feed_items = [
        {"name": "João Silva", "action": "curtiu seu post", "time": "há 2 horas"},
        {"name": "Maria Santos", "action": "comentou em seu post", "time": "há 4 horas"},
        {"name": "Pedro Costa", "action": "começou a seguir você", "time": "há 1 dia"},
        {"name": "Ana Oliveira", "action": "compartilhou seu post", "time": "há 2 dias"},
        {"name": "Carlos Mendes", "action": "salvou seu post", "time": "há 3 dias"},
    ]
    for item in feed_items:
        st.markdown('<div class="feed-item">', unsafe_allow_html=True)
        st.markdown(f'<div class="feed-item-avatar" style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;font-weight:bold;">{item["name"][0]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="feed-item-content">', unsafe_allow_html=True)
        st.markdown(f'<div class="feed-item-header"><div class="feed-item-name">{item["name"]}</div><div class="feed-item-time">{item["time"]}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="feed-item-action">{item["action"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# VIEWS
# ======================================================
def view_pages():
    user = get_current_user()
    st.markdown('<div class="feed-card">', unsafe_allow_html=True)
    st.markdown("### Pages")
    st.write("Aqui você pode criar e gerenciar suas páginas de pesquisa.")
    st.markdown("#### Criar nova página")
    page_title = st.text_input("Título da página", key="page_title")
    page_content = st.text_area("Conteúdo", key="page_content", height=150)
    if st.button("Criar página", key="btn_create_page"):
        if page_title and page_content:
            st.success("Página criada com sucesso!")
        else:
            st.error("Preencha todos os campos.")
    st.markdown('</div>', unsafe_allow_html=True)

def view_analytics():
    st.markdown('<div class="feed-card">', unsafe_allow_html=True)
    st.markdown("### Analytics")
    st.write("Estatísticas e análises da sua pesquisa.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Posts", 42)
    col2.metric("Engajamento", "89%")
    col3.metric("Seguidores", 128)
    st.markdown('</div>', unsafe_allow_html=True)

def view_tasks():
    user = get_current_user()
    st.markdown('<div class="feed-card">', unsafe_allow_html=True)
    st.markdown("### Tasks")
    st.markdown("#### Criar nova tarefa")
    task_title = st.text_input("Título da tarefa", key="task_title")
    task_desc = st.text_area("Descrição", key="task_desc", height=100)
    task_status = st.selectbox("Status", ["Não iniciado", "Em progresso", "Concluído"], key="task_status")
    task_date = st.date_input("Data limite", key="task_date")
    if st.button("Criar tarefa", key="btn_create_task"):
        if task_title:
            new_task = TimelineStep(
                id=str(uuid.uuid4()),
                owner_id=user.id,
                title=task_title,
                description=task_desc,
                status=task_status,
                due_date=str(task_date),
            )
            st.session_state.timeline.append(new_task)
            save_state_to_file()
            st.success("Tarefa criada!")
            st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def view_invoice():
    st.markdown('<div class="feed-card">', unsafe_allow_html=True)
    st.markdown("### Invoice")
    st.write("Gerenciamento de faturas e pagamentos.")
    st.markdown('</div>', unsafe_allow_html=True)

def view_subscribe():
    st.markdown('<div class="feed-card">', unsafe_allow_html=True)
    st.markdown("### Subscribe")
    st.write("Gerenciamento de assinaturas.")
    st.markdown('</div>', unsafe_allow_html=True)

def view_contact():
    st.markdown('<div class="feed-card">', unsafe_allow_html=True)
    st.markdown("### Contact")
    st.write("Contatos e suporte.")
    st.markdown('</div>', unsafe_allow_html=True)

def view_settings():
    st.markdown('<div class="feed-card">', unsafe_allow_html=True)
    st.markdown("### Settings")
    st.write("Configurações da conta.")
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# MAIN APP
# ======================================================
def main():
    current_user = get_current_user()

    if not current_user:
        auth_screen()
    else:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)

        # Sidebar
        with st.container():
            render_sidebar(current_user)

        # Main Content
        st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

        # Left Column (Profile Card)
        with st.container():
            render_profile_card(current_user)

        # Right Column (Content)
        with st.container():
            if st.session_state.current_view == "Pages":
                view_pages()
            elif st.session_state.current_view == "Analytics":
                view_analytics()
            elif st.session_state.current_view == "Tasks":
                view_tasks()
            elif st.session_state.current_view == "Invoice":
                view_invoice()
            elif st.session_state.current_view == "Subscribe":
                view_subscribe()
            elif st.session_state.current_view == "Contact":
                view_contact()
            elif st.session_state.current_view == "Settings":
                view_settings()
            else:
                render_feed_card()

        st.markdown('</div>', unsafe_allow_html=True)  # Close content-wrapper
        st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

if __name__ == "__main__":
    main()
