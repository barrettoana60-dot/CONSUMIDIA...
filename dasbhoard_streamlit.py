import streamlit as st
import datetime
import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from collections import Counter
import re
import uuid
from pathlib import Path
from io import BytesIO
from PIL import Image

# ======================================================
# CONFIG BÁSICA
# ======================================================

st.set_page_config(
    page_title="PQR – Rede de Pesquisa Qualitativa",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STATE_FILE = "pqr_state.json"
AVATAR_DIR = Path("avatars")
AVATAR_DIR.mkdir(exist_ok=True)

# ======================================================
# CSS – TEMA AZUL ESCURO MODERNO + LIQUID GLASS
# ======================================================

MODERN_CSS = """
<style>
:root {
    --pqr-primary: #3b82f6;      /* azul */
    --pqr-primary-soft: rgba(59, 130, 246, 0.18);
    --pqr-accent: #22c55e;       /* verde */
    --pqr-bg: #020617;           /* quase preto azulado */
    --pqr-bg-card: rgba(15,23,42,0.92);
    --pqr-border-soft: rgba(148,163,184,0.35);
    --pqr-text-main: #e5e7eb;
    --pqr-text-soft: #9ca3af;
}

/* esconder sidebar */
[data-testid="stSidebar"] {
    display: none;
}

/* fundo geral */
.stApp {
    background:
      radial-gradient(circle at top left, #0b1120, #020617 50%, #020617),
      #020617;
    color: var(--pqr-text-main);
    font-family: system-ui,-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",sans-serif;
}

.block-container {
    padding-top: 0.5rem;
    padding-bottom: 0.8rem;
    max-width: 1150px;
}

/* header */
.pqr-header {
    position: sticky;
    top: 0;
    z-index: 999;
    padding: 10px 0 12px;
    background: linear-gradient(to bottom, rgba(15,23,42,0.95), rgba(15,23,42,0.9));
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(148,163,184,0.4);
}

/* linha com logo, nav e perfil */
.pqr-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
}

/* logo */
.pqr-logo-line {
    display: flex;
    align-items: center;
    gap: 10px;
}
.pqr-logo-avatar {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    background: conic-gradient(from 200deg, #3b82f6, #22c55e, #a855f7, #3b82f6);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #0b1120;
    font-weight: 800;
    font-size: 0.9rem;
    box-shadow: 0 0 18px rgba(59,130,246,0.7);
}
.pqr-title-text {
    display: flex;
    flex-direction: column;
}
.pqr-title-main {
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}
.pqr-title-sub {
    font-size: 0.78rem;
    color: var(--pqr-text-soft);
}

/* user pill header */
.user-pill-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 10px;
    border-radius: 999px;
    background: radial-gradient(circle at top left, rgba(15,23,42,0.95), rgba(15,23,42,0.9));
    border: 1px solid rgba(148,163,184,0.4);
    box-shadow: 0 8px 24px rgba(15,23,42,0.85);
}
.user-pill-avatar {
    width: 32px;
    height: 32px;
    border-radius: 999px;
    background: linear-gradient(135deg, #1d4ed8, #22c55e);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #020617;
    font-weight: 700;
    overflow: hidden;
}
.user-pill-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 999px;
}

/* sininho */
.pqr-bell {
    margin-left: 10px;
    width: 28px;
    height: 28px;
    border-radius: 999px;
    background: radial-gradient(circle at top, rgba(148,163,184,0.3), rgba(15,23,42,1));
    display:flex;
    align-items:center;
    justify-content:center;
    border: 1px solid rgba(148,163,184,0.6);
    cursor:pointer;
    position: relative;
}
.pqr-bell span {
    font-size: 0.9rem;
}
.pqr-bell-dot {
    position:absolute;
    top:3px;
    right:4px;
    width: 7px;
    height: 7px;
    border-radius:999px;
    background:#f97316;
}

/* navegação */
.pqr-nav {
    margin-top: 10px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.pqr-nav-item {
    padding: 5px 12px;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.4);
    font-size: 0.78rem;
    color: var(--pqr-text-soft);
    background: radial-gradient(circle at top left, rgba(15,23,42,0.9), rgba(15,23,42,0.96));
    cursor: pointer;
    user-select: none;
}
.pqr-nav-item-active {
    background: linear-gradient(135deg, #3b82f6, #22c55e);
    color: #020617;
    border-color: rgba(59,130,246,0.9);
}

/* cartões glass */
.glass-main {
    margin-top: 16px;
    background: var(--pqr-bg-card);
    border-radius: 18px;
    border: 1px solid var(--pqr-border-soft);
    box-shadow: 0 22px 60px rgba(0,0,0,0.75);
    padding: 18px 22px;
    backdrop-filter: blur(26px);
}

/* seções internas */
.glass-section {
    background: rgba(15,23,42,0.9);
    border-radius: 14px;
    border: 1px solid rgba(148,163,184,0.35);
    padding: 12px 14px;
}

/* posts */
.post-card {
    background: radial-gradient(circle at top left, rgba(15,23,42,1), rgba(15,23,42,0.98));
    border-radius: 14px;
    border: 1px solid rgba(148,163,184,0.35);
    padding: 10px 12px;
    margin-bottom: 10px;
}
.post-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
    font-size: 0.84rem;
}
.post-meta {
    font-size: 0.75rem;
    color: var(--pqr-text-soft);
}
.post-body {
    font-size: 0.88rem;
    margin: 6px 0 8px;
}
.post-actions {
    display: flex;
    gap: 16px;
    font-size: 0.78rem;
    color: var(--pqr-text-soft);
}

/* ícones de ação */
.post-action-pill {
    display:inline-flex;
    align-items:center;
    gap:4px;
    padding:3px 9px;
    border-radius:999px;
    border:1px solid rgba(148,163,184,0.45);
    background:rgba(15,23,42,0.9);
}

/* chat bubble */
.chat-bubble {
    padding: 7px 9px;
    border-radius: 10px;
    margin-bottom: 6px;
    font-size: 0.84rem;
    background: rgba(15,23,42,0.96);
    border: 1px solid rgba(148,163,184,0.5);
}
.chat-meta {
    font-size: 0.70rem;
    color: var(--pqr-text-soft);
    margin-bottom: 2px;
}

/* mapa mental resumo */
.mind-node {
    font-size: 0.84rem;
    margin: 2px 0;
}
.mind-node-label {
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(30,64,175,0.7);
}
.mind-node-selected {
    background: var(--pqr-primary-soft);
    color: var(--pqr-primary);
    border: 1px solid rgba(37,99,235,0.8);
}

/* badge */
.pqr-badge {
    display:inline-block;
    padding:3px 10px;
    border-radius:999px;
    font-size:0.72rem;
    letter-spacing:0.08em;
    text-transform:uppercase;
    background:var(--pqr-primary-soft);
    border:1px solid rgba(37,99,235,0.6);
    color:var(--pqr-primary);
}

/* inputs */
textarea, input, select {
    border-radius: 9px !important;
    background:#020617 !important;
    color:var(--pqr-text-main) !important;
    border:1px solid rgba(148,163,184,0.5) !important;
}

/* botões glass */
.stButton > button {
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.6);
    background: radial-gradient(circle at top left, rgba(248,250,252,0.1), rgba(15,23,42,0.98));
    color: var(--pqr-text-main);
    font-size: 0.84rem;
    padding: 0.3rem 0.9rem;
    transition: all 0.15s ease-out;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 26px rgba(0,0,0,0.8);
    border-color: rgba(59,130,246,0.9);
}

/* radio (caso apareça) */
[data-baseweb="radio"] > div {
    gap: 4px;
}
</style>
"""

st.markdown(MODERN_CSS, unsafe_allow_html=True)

# ======================================================
# MODELOS
# ======================================================

@dataclass
class User:
    id: str
    name: str
    email: str
    type: str
    password: str
    interests: List[str] = field(default_factory=list)
    avatar_path: Optional[str] = None   # caminho local para PNG/JPG

@dataclass
class Post:
    id: str
    user_id: str
    text: str
    created_at: str
    likes: int = 0
    liked_by: List[str] = field(default_factory=list)   # user_ids
    saved_by: List[str] = field(default_factory=list)   # user_ids
    comments: List[Dict[str, str]] = field(default_factory=list)  # {user_name, text, time}
    shared_count: int = 0

@dataclass
class ChatMessage:
    id: str
    from_id: str
    to_id: Optional[str]  # None => canal geral
    text: str
    time: str

# conexões de interesse simples
@dataclass
class Connection:
    user_id: str
    other_id: str
    score: float

# ======================================================
# PERSISTÊNCIA
# ======================================================

def default_state_dict() -> Dict[str, Any]:
    return {
        "users": [],
        "current_user_id": None,
        "posts": [],
        "chat_messages": [],
        "notifications": [],  # texto simples
        "current_view": "Feed",
    }

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return default_state_dict()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = default_state_dict()
        base.update(data)
        return base
    except Exception:
        return default_state_dict()

def save_state():
    data = {
        "users": [asdict(u) for u in st.session_state.users],
        "current_user_id": st.session_state.current_user_id,
        "posts": [asdict(p) for p in st.session_state.posts],
        "chat_messages": [asdict(c) for c in st.session_state.chat_messages],
        "notifications": st.session_state.notifications,
        "current_view": st.session_state.get("current_view", "Feed"),
    }
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Não foi possível salvar o estado: {e}")

# ======================================================
# SESSION INIT
# ======================================================

def init_state():
    if "initialized" in st.session_state:
        return
    persisted = load_state()
    st.session_state.users = [User(**u) for u in persisted["users"]]
    st.session_state.current_user_id = persisted["current_user_id"]
    st.session_state.posts = [Post(**p) for p in persisted["posts"]]
    st.session_state.chat_messages = [ChatMessage(**c) for c in persisted["chat_messages"]]
    st.session_state.notifications = persisted.get("notifications", [])
    st.session_state.current_view = persisted.get("current_view", "Feed")
    st.session_state.initialized = True

init_state()

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

def map_type_label(t: str) -> str:
    return {
        "ic": "Iniciação Científica",
        "extensao": "Extensão",
        "doutorando": "Doutorando",
        "voluntario": "Voluntário",
        "prodig": "PRODIG",
        "mentoria": "Mentoria",
    }.get(t, "Bolsista")

def parse_interests(s: str) -> List[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]

def calc_interest_connections(current: User, users: List[User]) -> List[Connection]:
    res: List[Connection] = []
    set_i = set(current.interests)
    if not set_i:
        return []
    for u in users:
        if u.id == current.id:
            continue
        inter = set_i.intersection(set(u.interests))
        if not inter:
            continue
        score = len(inter) / max(len(set_i), 1)
        res.append(Connection(user_id=current.id, other_id=u.id, score=score))
    res.sort(key=lambda c: c.score, reverse=True)
    return res

def avatar_html(user: User, size: int = 32) -> str:
    if user.avatar_path and os.path.exists(user.avatar_path):
        return f'<img src="{user.avatar_path}" style="width:{size}px;height:{size}px;border-radius:999px;object-fit:cover;">'
    ini = user.name[:1].upper() if user.name else "U"
    return ini

# ======================================================
# AUTH
# ======================================================

def auth_screen():
    _, col, _ = st.columns([1, 2.4, 1])
    with col:
        st.markdown('<div class="glass-main">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="pqr-logo-line">
                <div class="pqr-logo-avatar">P</div>
                <div class="pqr-title-text">
                    <div class="pqr-title-main">PQR</div>
                    <div class="pqr-title-sub">Rede qualitativa com cara de rede social</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        tabs = st.tabs(["Entrar", "Criar conta"])

        with tabs[0]:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Senha", type="password", key="login_pw")
            if st.button("Entrar"):
                user = next((u for u in st.session_state.users if u.email == email), None)
                if not user or user.password != password:
                    st.error("Credenciais inválidas.")
                else:
                    st.session_state.current_user_id = user.id
                    save_state()
                    st.experimental_rerun()

        with tabs[1]:
            name = st.text_input("Nome completo", key="reg_name")
            email_c = st.text_input("Email institucional", key="reg_email")
            type_label = st.selectbox(
                "Tipo de vínculo",
                [
                    "Selecione…",
                    "Iniciação Científica",
                    "Extensão",
                    "Doutorando",
                    "Voluntário",
                    "PRODIG",
                    "Mentoria",
                ],
                key="reg_type",
            )
            pw = st.text_input("Senha (mín. 6 caracteres)", type="password", key="reg_pw")
            interests_str = st.text_input(
                "Interesses (separados por vírgula)",
                placeholder="ex.: inclusão digital, saúde mental, aprendizagem ativa",
                key="reg_interests",
            )
            avatar_file = st.file_uploader(
                "Foto de perfil (PNG/JPG)",
                type=["png", "jpg", "jpeg"],
                key="reg_avatar",
            )

            if st.button("Criar conta"):
                if not name.strip() or not email_c.strip():
                    st.warning("Nome e email são obrigatórios.")
                elif len(pw) < 6:
                    st.warning("Senha muito curta.")
                elif type_label == "Selecione…":
                    st.warning("Selecione o tipo de vínculo.")
                elif any(u.email == email_c for u in st.session_state.users):
                    st.error("Já existe usuário com este email.")
                else:
                    t_map = {
                        "Iniciação Científica": "ic",
                        "Extensão": "extensao",
                        "Doutorando": "doutorando",
                        "Voluntário": "voluntario",
                        "PRODIG": "prodig",
                        "Mentoria": "mentoria",
                    }
                    t = t_map.get(type_label, "ic")
                    avatar_path = None
                    if avatar_file is not None:
                        img = Image.open(avatar_file).convert("RGB")
                        uid = str(uuid.uuid4())
                        avatar_path = str(AVATAR_DIR / f"{uid}.jpg")
                        img.save(avatar_path, format="JPEG", quality=90)

                    new_user = User(
                        id=str(uuid.uuid4()),
                        name=name.strip(),
                        email=email_c.strip(),
                        type=t,
                        password=pw,
                        interests=parse_interests(interests_str),
                        avatar_path=avatar_path,
                    )
                    st.session_state.users.append(new_user)
                    st.session_state.current_user_id = new_user.id
                    save_state()
                    st.success("Conta criada.")
                    st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# HEADER + NAV + NOTIFICAÇÕES
# ======================================================

VIEWS = ["Feed", "Análises", "Conexões", "Chat", "Configurações"]

def render_header(user: User):
    if "current_view" not in st.session_state:
        st.session_state.current_view = "Feed"

    st.markdown('<div class="pqr-header">', unsafe_allow_html=True)
    colL, colR = st.columns([3, 2])

    with colL:
        st.markdown(
            """
            <div class="pqr-header-row">
                <div class="pqr-logo-line">
                    <div class="pqr-logo-avatar">P</div>
                    <div class="pqr-title-text">
                        <div class="pqr-title-main">PQR</div>
                        <div class="pqr-title-sub">sua pesquisa, como rede social</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with colR:
        av_html = avatar_html(user)
        notif_count = len(st.session_state.notifications)
        dot_html = '<div class="pqr-bell-dot"></div>' if notif_count > 0 else ""
        st.markdown(
            f"""
            <div class="pqr-header-row" style="justify-content:flex-end;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <div class="pqr-bell">
                        <span>🔔</span>
                        {dot_html}
                    </div>
                    <div class="user-pill-header">
                        <div class="user-pill-avatar">{av_html}</div>
                        <div style="font-size:0.78rem;">
                            {user.name.split(" ")[0]}<br/>
                            <span style="color:#9ca3af;">{map_type_label(user.type)}</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ao clicar no sininho, mostramos notificações
        if notif_count > 0:
            with st.expander(f"Notificações ({notif_count})", expanded=False):
                for n in reversed(st.session_state.notifications[-20:]):
                    st.write("- " + n)

    # nav
    st.markdown('<div class="pqr-nav">', unsafe_allow_html=True)
    nav_cols = st.columns(len(VIEWS))
    for i, (vname, col) in enumerate(zip(VIEWS, nav_cols)):
        with col:
            active = (st.session_state.current_view == vname)
            cls = "pqr-nav-item-active" if active else "pqr-nav-item"
            if st.button(vname, key=f"nav_{i}"):
                st.session_state.current_view = vname
                save_state()
                st.experimental_rerun()
            # div oculto só pra CSS
            st.markdown(
                f'<div class="{cls}" style="display:none;">{vname}</div>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# VIEW: FEED (curtir, salvar, comentar, compartilhar)
# ======================================================

def view_feed():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Feed da pesquisa")

    user = get_current_user()
    if not user:
        st.warning("Entre na sua conta para ver e publicar no feed.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with st.expander("Criar novo post", expanded=True):
        text = st.text_area("Compartilhe um avanço, dúvida ou insight…", key="new_post")
        if st.button("Publicar"):
            if not text.strip():
                st.warning("Escreva algo antes de publicar.")
            else:
                p = Post(
                    id=str(uuid.uuid4()),
                    user_id=user.id,
                    text=text.strip(),
                    created_at=datetime.datetime.now().isoformat(),
                )
                st.session_state.posts.insert(0, p)
                save_state()
                st.success("Post publicado.")

    st.write("---")
    if not st.session_state.posts:
        st.info("Ainda não há posts. Publique o primeiro.")
    else:
        for p in st.session_state.posts:
            author = next((u for u in st.session_state.users if u.id == p.user_id), None)
            if not author:
                continue
            av_html = avatar_html(author)
            try:
                dt = datetime.datetime.fromisoformat(p.created_at)
                ts = dt.strftime("%d/%m %H:%M")
            except Exception:
                ts = p.created_at

            st.markdown(
                f"""
                <div class="post-card">
                    <div class="post-header">
                        <div class="user-pill-avatar">{av_html}</div>
                        <div>
                            <strong>{author.name}</strong><br/>
                            <span class="post-meta">{ts}</span>
                        </div>
                    </div>
                    <div class="post-body">{p.text}</div>
                """,
                unsafe_allow_html=True,
            )

            # ações
            col_like, col_save, col_comment, col_share = st.columns([0.7,0.7,1.1,0.9])
            with col_like:
                liked = user.id in p.liked_by
                label = "💙 Curtido" if liked else "🤍 Curtir"
                if st.button(label, key=f"like_{p.id}"):
                    if liked:
                        p.liked_by.remove(user.id)
                        p.likes = max(0, p.likes - 1)
                    else:
                        p.liked_by.append(user.id)
                        p.likes += 1
                        # notificação simples pro autor
                        if author.id != user.id:
                            st.session_state.notifications.append(
                                f"{user.name} curtiu seu post."
                            )
                    save_state()
                    st.experimental_rerun()
            with col_save:
                saved = user.id in p.saved_by
                label = "📎 Salvo" if saved else "📥 Salvar"
                if st.button(label, key=f"save_{p.id}"):
                    if saved:
                        p.saved_by.remove(user.id)
                    else:
                        p.saved_by.append(user.id)
                    save_state()
                    st.experimental_rerun()
            with col_comment:
                if st.button("💬 Comentar", key=f"comment_btn_{p.id}"):
                    st.session_state[f"show_comments_{p.id}"] = not st.session_state.get(
                        f"show_comments_{p.id}", False
                    )
            with col_share:
                if st.button("🔁 Compartilhar", key=f"share_{p.id}"):
                    p.shared_count += 1
                    save_state()
                    st.success("Post marcado como compartilhado.")

            # contagem
            st.markdown(
                f"""
                <div class="post-actions">
                    <span>👍 {p.likes}</span>
                    <span>💾 {len(p.saved_by)} salvos</span>
                    <span>🔁 {p.shared_count} compartilhamentos</span>
                    <span>💬 {len(p.comments)} comentários</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # área de comentários
            if st.session_state.get(f"show_comments_{p.id}", False):
                st.write("— Comentários —")
                for c in p.comments[-10:]:
                    st.markdown(
                        f"**{c['user_name']}** – {c['time']}  \n{c['text']}"
                    )
                c_text = st.text_input(
                    "Seu comentário", key=f"comment_input_{p.id}", label_visibility="collapsed"
                )
                if st.button("Enviar comentário", key=f"send_comment_{p.id}"):
                    if c_text.strip():
                        now = datetime.datetime.now().strftime("%d/%m %H:%M")
                        p.comments.append(
                            {
                                "user_name": user.name.split(" ")[0] or user.name,
                                "text": c_text.strip(),
                                "time": now,
                            }
                        )
                        if author.id != user.id:
                            st.session_state.notifications.append(
                                f"{user.name} comentou em seu post."
                            )
                        save_state()
                        st.experimental_rerun()
                    else:
                        st.warning("Escreva algo no comentário.")

            st.markdown("</div>", unsafe_allow_html=True)  # fecha post-card
            st.write("")

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: ANÁLISES – GRÁFICOS SIMPLES
# ======================================================

def view_analytics():
    import pandas as pd

    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Análises da sua atividade")

    user = get_current_user()
    if not user:
        st.warning("Entre na sua conta para ver análises.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # posts por dia (gerais e do usuário)
    if not st.session_state.posts:
        st.info("Ainda não há posts suficientes para gráficos.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    rows = []
    for p in st.session_state.posts:
        d = p.created_at[:10]
        rows.append(
            {
                "data": d,
                "autor": p.user_id,
                "likes": p.likes,
            }
        )
    df = pd.DataFrame(rows)

    st.subheader("Volume de posts")
    col1, col2 = st.columns(2)
    with col1:
        df_all = df.groupby("data").size().reset_index(name="posts")
        st.line_chart(df_all.set_index("data"))
        st.caption("Postagens totais por dia.")
    with col2:
        df_me = df[df["autor"] == user.id].groupby("data").size().reset_index(name="meus_posts")
        if not df_me.empty:
            st.line_chart(df_me.set_index("data"))
            st.caption("Seus posts por dia.")
        else:
            st.info("Você ainda não publicou nada.")

    st.write("---")
    st.subheader("Distribuição de likes")
    df_likes = df.copy()
    df_likes["likes"] = df_likes["likes"].astype(int)
    by_author = df_likes.groupby("autor")["likes"].sum().reset_index()
    if not by_author.empty:
        by_author["nome"] = by_author["autor"].apply(
            lambda uid: next((u.name.split(" ")[0] for u in st.session_state.users if u.id == uid), "Outro")
        )
        st.bar_chart(by_author.set_index("nome")["likes"])
        st.caption("Total de curtidas recebidas por pessoa.")
    else:
        st.info("Ainda não há curtidas suficientes para análise.")

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: CONEXÕES (INTERESSES EM COMUM)
# ======================================================

def view_connections():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Conexões de interesse")

    user = get_current_user()
    if not user:
        st.warning("Entre na sua conta para ver conexões.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # permitir atualizar interesses
    with st.expander("Seus interesses"):
        intr_str = ", ".join(user.interests)
        new_intr = st.text_input(
            "Interesses (separados por vírgula)",
            value=intr_str,
        )
        if st.button("Atualizar interesses"):
            user.interests = parse_interests(new_intr)
            save_state()
            st.success("Interesses atualizados.")

    conns = calc_interest_connections(user, st.session_state.users)
    if not conns:
        st.info("Ainda não encontrei conexões fortes de interesse. Adicione interesses e convide mais pessoas.")
    else:
        st.write("Pessoas com maior afinidade temática:")
        for c in conns[:10]:
            other = next(u for u in st.session_state.users if u.id == c.other_id)
            inter = set(user.interests).intersection(set(other.interests))
            av_html = avatar_html(other, size=28)
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                    <div class="user-pill-avatar" style="width:28px;height:28px;">{av_html}</div>
                    <div style="font-size:0.82rem;">
                        <strong>{other.name}</strong><br/>
                        <span style="color:#9ca3af;">Afinidade: {c.score:.0%} · Interesses comuns: {', '.join(inter)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: CHAT (mínimo, geral)
# ======================================================

def view_chat():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Chat geral")

    user = get_current_user()
    if not user:
        st.warning("Entre para conversar.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # mensagens gerais (to_id=None)
    msgs = [m for m in st.session_state.chat_messages if m.to_id is None]
    for m in msgs[-50:]:
        author = next((u for u in st.session_state.users if u.id == m.from_id), None)
        name = author.name.split(" ")[0] if author else "Usuário"
        st.markdown(
            f"""
            <div class="chat-bubble">
                <div class="chat-meta">{name} – {m.time}</div>
                {m.text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("---")
    txt = st.text_input("Sua mensagem", key="chat_input")
    if st.button("Enviar"):
        if txt.strip():
            now = datetime.datetime.now().strftime("%d/%m %H:%M")
            st.session_state.chat_messages.append(
                ChatMessage(
                    id=str(uuid.uuid4()),
                    from_id=user.id,
                    to_id=None,
                    text=txt.strip(),
                    time=now,
                )
            )
            save_state()
            st.experimental_rerun()
        else:
            st.warning("Escreva algo.")

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: CONFIGURAÇÕES
# ======================================================

def view_settings():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Configurações & perfil")

    user = get_current_user()
    if not user:
        st.warning("Entre na sua conta.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    av_html = avatar_html(user, size=48)
    st.markdown(
        f"""
        <div class="user-pill-header" style="margin-bottom:10px;">
            <div class="user-pill-avatar" style="width:48px;height:48px;">{av_html}</div>
            <div>
                <strong>{user.name}</strong><br/>
                <span style="font-size:0.78rem;color:#9ca3af;">{user.email} – {map_type_label(user.type)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Atualizar foto de perfil"):
        avatar_file = st.file_uploader(
            "Nova foto (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
            key="new_avatar",
        )
        if st.button("Salvar foto"):
            if avatar_file is not None:
                img = Image.open(avatar_file).convert("RGB")
                uid = str(uuid.uuid4())
                avatar_path = AVATAR_DIR / f"{uid}.jpg"
                img.save(avatar_path, format="JPEG", quality=90)
                user.avatar_path = str(avatar_path)
                save_state()
                st.success("Foto atualizada.")
                st.experimental_rerun()
            else:
                st.warning("Selecione um arquivo.")

    st.write("---")
    st.subheader("Sessão")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Salvar dados agora"):
            save_state()
            st.success("Dados salvos.")
    with col2:
        if st.button("Sair da conta"):
            st.session_state.current_user_id = None
            save_state()
            st.success("Sessão encerrada.")
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# MAIN
# ======================================================

def main():
    user = get_current_user()
    if not user:
        auth_screen()
        return

    render_header(user)

    view = st.session_state.current_view
    if view == "Feed":
        view_feed()
    elif view == "Análises":
        view_analytics()
    elif view == "Conexões":
        view_connections()
    elif view == "Chat":
        view_chat()
    elif view == "Configurações":
        view_settings()
    else:
        view_feed()

if __name__ == "__main__":
    main()
