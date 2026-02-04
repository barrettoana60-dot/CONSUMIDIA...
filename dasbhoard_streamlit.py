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
# CSS – AZUL ESCURO MODERNO + LIQUID GLASS
# ======================================================

MODERN_CSS = """
<style>
:root {
    --pqr-primary: #3b82f6;
    --pqr-primary-soft: rgba(59, 130, 246, 0.18);
    --pqr-accent: #22c55e;
    --pqr-bg: #020617;
    --pqr-bg-card: rgba(15,23,42,0.92);
    --pqr-border-soft: rgba(148,163,184,0.35);
    --pqr-text-main: #e5e7eb;
    --pqr-text-soft: #9ca3af;
}

[data-testid="stSidebar"] { display: none; }

.stApp {
    background:
      radial-gradient(circle at top left, #0b1120, #020617 55%, #020617),
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

.pqr-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
}

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
.pqr-title-text { display: flex; flex-direction: column; }
.pqr-title-main {
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
}
.pqr-title-sub { font-size: 0.78rem; color: var(--pqr-text-soft); }

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
    display:flex;align-items:center;justify-content:center;
    color:#020617;font-weight:700;overflow:hidden;
}
.user-pill-avatar img {
    width:100%;height:100%;object-fit:cover;border-radius:999px;
}

/* bell */
.pqr-bell {
    margin-left: 10px;
    width: 28px; height: 28px;
    border-radius: 999px;
    background: radial-gradient(circle at top, rgba(148,163,184,0.3), rgba(15,23,42,1));
    display:flex;align-items:center;justify-content:center;
    border: 1px solid rgba(148,163,184,0.6);
    cursor:pointer; position: relative;
}
.pqr-bell span { font-size: 0.9rem; }
.pqr-bell-dot {
    position:absolute;top:3px;right:4px;
    width:7px;height:7px;border-radius:999px;background:#f97316;
}

/* nav */
.pqr-nav { margin-top: 10px; display:flex;gap:8px;flex-wrap:wrap; }
.pqr-nav-item {
    padding: 5px 12px;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.4);
    font-size: 0.78rem;
    color: var(--pqr-text-soft);
    background: radial-gradient(circle at top left, rgba(15,23,42,0.9), rgba(15,23,42,0.96));
    cursor:pointer;user-select:none;
}
.pqr-nav-item-active {
    background: linear-gradient(135deg, #3b82f6, #22c55e);
    color: #020617;
    border-color: rgba(59,130,246,0.9);
}

/* cards */
.glass-main {
    margin-top: 16px;
    background: var(--pqr-bg-card);
    border-radius: 18px;
    border: 1px solid var(--pqr-border-soft);
    box-shadow: 0 22px 60px rgba(0,0,0,0.75);
    padding: 18px 22px;
    backdrop-filter: blur(26px);
}
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
.post-header { display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:0.84rem; }
.post-meta { font-size:0.75rem;color:var(--pqr-text-soft); }
.post-body { font-size:0.88rem;margin:6px 0 8px; }
.post-actions { display:flex;gap:16px;font-size:0.78rem;color:var(--pqr-text-soft); }

/* chat */
.chat-bubble {
    padding:7px 9px;border-radius:10px;margin-bottom:6px;
    font-size:0.84rem;background:rgba(15,23,42,0.96);
    border:1px solid rgba(148,163,184,0.5);
}
.chat-meta { font-size:0.70rem;color:var(--pqr-text-soft);margin-bottom:2px; }

/* badge */
.pqr-badge {
    display:inline-block;padding:3px 10px;border-radius:999px;
    font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;
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

/* botões */
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
    avatar_path: Optional[str] = None

@dataclass
class Post:
    id: str
    user_id: str
    text: str
    created_at: str
    likes: int = 0
    liked_by: List[str] = field(default_factory=list)
    saved_by: List[str] = field(default_factory=list)
    comments: List[Dict[str, str]] = field(default_factory=list)
    shared_count: int = 0

@dataclass
class ChatMessage:
    id: str
    from_id: str
    to_id: Optional[str]
    text: str
    time: str

# ======================================================
# PERSISTÊNCIA
# ======================================================

def default_state_dict() -> Dict[str, Any]:
    return {
        "users": [],
        "current_user_id": None,
        "posts": [],
        "chat_messages": [],
        "notifications": [],
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
        "users": [asdict(u) for u in st.session_state.get("users", [])],
        "current_user_id": st.session_state.get("current_user_id"),
        "posts": [asdict(p) for p in st.session_state.get("posts", [])],
        "chat_messages": [asdict(c) for c in st.session_state.get("chat_messages", [])],
        "notifications": st.session_state.get("notifications", []),
        "current_view": st.session_state.get("current_view", "Feed"),
    }
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Não foi possível salvar o estado: {e}")

# ======================================================
# SESSION INIT SEM ERRO
# ======================================================

if "initialized" not in st.session_state:
    persisted = load_state()
    st.session_state.users = [User(**u) for u in persisted["users"]]
    st.session_state.current_user_id = persisted.get("current_user_id")
    st.session_state.posts = [Post(**p) for p in persisted["posts"]]
    st.session_state.chat_messages = [ChatMessage(**c) for c in persisted["chat_messages"]]
    st.session_state.notifications = persisted.get("notifications", [])
    st.session_state.current_view = persisted.get("current_view", "Feed")
    st.session_state.initialized = True

# ======================================================
# HELPERS
# ======================================================

def get_current_user() -> Optional[User]:
    user_id = st.session_state.get("current_user_id")
    if not user_id:
        return None
    for u in st.session_state.get("users", []):
        if u.id == user_id:
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

        # ENTRAR
        with tabs[0]:
            email = st.text_input("E‑mail")
            pwd = st.text_input("Senha", type="password")
            if st.button("Entrar"):
                user = next((u for u in st.session_state.users if u.email == email and u.password == pwd), None)
                if user:
                    st.session_state.current_user_id = user.id
                    save_state()
                    st.experimental_rerun()
                else:
                    st.error("E‑mail ou senha incorretos.")

        # CRIAR CONTA
        with tabs[1]:
            name = st.text_input("Nome completo")
            email2 = st.text_input("E‑mail para cadastro")
            pwd2 = st.text_input("Defina uma senha", type="password")
            tipo = st.selectbox(
                "Tipo",
                [
                    ("ic", "Iniciação Científica"),
                    ("extensao", "Extensão"),
                    ("doutorando", "Doutorando"),
                    ("voluntario", "Voluntário"),
                    ("prodig", "PRODIG"),
                    ("mentoria", "Mentoria"),
                ],
                format_func=lambda x: x[1],
            )
            intr = st.text_input("Interesses (separados por vírgula)")

            if st.button("Criar conta"):
                if not name.strip() or not email2.strip() or not pwd2.strip():
                    st.warning("Preencha nome, e‑mail e senha.")
                elif any(u.email == email2 for u in st.session_state.users):
                    st.error("Já existe usuário com este e‑mail.")
                else:
                    u = User(
                        id=str(uuid.uuid4()),
                        name=name.strip(),
                        email=email2.strip(),
                        password=pwd2.strip(),
                        type=tipo[0],
                        interests=parse_interests(intr),
                        avatar_path=None,
                    )
                    st.session_state.users.append(u)
                    st.session_state.current_user_id = u.id
                    save_state()
                    st.success("Conta criada. Você já está logada(o).")
                    st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# HEADER + NAV
# ======================================================

NAV_VIEWS = ["Feed", "Análises", "Conexões", "Chat", "Configurações"]

def render_header(user: User):
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
                        <div class="pqr-title-sub">sua rede de pesquisa qualitativa</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with colR:
        av_html = avatar_html(user)
        notif_count = len(st.session_state.get("notifications", []))
        dot = '<div class="pqr-bell-dot"></div>' if notif_count > 0 else ""
        st.markdown(
            f"""
            <div class="pqr-header-row" style="justify-content:flex-end;">
                <div class="user-pill-header">
                    <div class="user-pill-avatar">{av_html}</div>
                    <div style="font-size:0.78rem;">
                        {user.name.split(" ")[0]}<br/>
                        <span style="color:#9ca3af;">{map_type_label(user.type)}</span>
                    </div>
                    <div class="pqr-bell">{dot}<span>🔔</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # nav
    st.markdown('<div class="pqr-nav">', unsafe_allow_html=True)
    nav_cols = st.columns(len(NAV_VIEWS))
    for i, (view_name, col) in enumerate(zip(NAV_VIEWS, nav_cols)):
        with col:
            active = (st.session_state.get("current_view", "Feed") == view_name)
            cls = "pqr-nav-item-active" if active else "pqr-nav-item"
            if st.button(view_name, key=f"nav_btn_{i}"):
                st.session_state.current_view = view_name
                save_state()
                st.experimental_rerun()
            st.markdown(
                f'<div class="{cls}" style="display:none;">{view_name}</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# VIEW: FEED
# ======================================================

def view_feed():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Feed de pesquisa")

    user = get_current_user()
    if not user:
        st.warning("Entre para ver e postar no feed.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # criar post
    with st.form("new_post"):
        st.write("Compartilhe um avanço, dúvida ou insight:")
        text = st.text_area("Escreva algo", key="post_text", label_visibility="collapsed")
        ok = st.form_submit_button("Publicar")
        if ok and text.strip():
            p = Post(
                id=str(uuid.uuid4()),
                user_id=user.id,
                text=text.strip(),
                created_at=datetime.datetime.now().isoformat(),
            )
            st.session_state.posts.insert(0, p)
            st.session_state.notifications.append("Seu post foi publicado.")
            save_state()
            st.experimental_rerun()

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

            if st.session_state.get(f"show_comments_{p.id}", False):
                st.write("— Comentários —")
                for c in p.comments[-10:]:
                    st.markdown(f"**{c['user_name']}** – {c['time']}  \n{c['text']}")
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

            st.markdown("</div>", unsafe_allow_html=True)
            st.write("")

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: ANÁLISES
# ======================================================

def view_analytics():
    import pandas as pd

    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Análises da sua atividade")

    user = get_current_user()
    if not user:
        st.warning("Entre para ver análises.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not st.session_state.posts:
        st.info("Ainda não há posts suficientes para gráficos.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    rows = []
    for p in st.session_state.posts:
        d = p.created_at[:10]
        rows.append({"data": d, "autor": p.user_id, "likes": p.likes})
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
# VIEW: CONEXÕES
# ======================================================

def view_connections():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Conexões de interesse")

    user = get_current_user()
    if not user:
        st.warning("Entre na sua conta.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with st.expander("Seus interesses"):
        intr_str = ", ".join(user.interests)
        new_intr = st.text_input("Interesses (separados por vírgula)", value=intr_str)
        if st.button("Atualizar interesses"):
            user.interests = parse_interests(new_intr)
            save_state()
            st.success("Interesses atualizados.")

    set_i = set(user.interests)
    if not set_i:
        st.info("Adicione interesses para ver conexões.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    conns = []
    for other in st.session_state.users:
        if other.id == user.id:
            continue
        inter = set_i.intersection(set(other.interests))
        if not inter:
            continue
        score = len(inter) / max(len(set_i), 1)
        conns.append((other, score, inter))
    conns.sort(key=lambda x: x[1], reverse=True)

    if not conns:
        st.info("Por enquanto ninguém compartilha interesses. Convide colegas.")
    else:
        st.write("Pessoas com maior afinidade temática:")
        for other, score, inter in conns[:10]:
            av_html = avatar_html(other, size=28)
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                    <div class="user-pill-avatar" style="width:28px;height:28px;">{av_html}</div>
                    <div style="font-size:0.82rem;">
                        <strong>{other.name}</strong><br/>
                        <span style="color:#9ca3af;">
                            Afinidade: {score:.0%} · Interesses comuns: {', '.join(inter)}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: CHAT
# ======================================================

def view_chat():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Chat geral")

    user = get_current_user()
    if not user:
        st.warning("Entre para conversar.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

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
                <span style="font-size:0.78rem;color:#9ca3af;">
                    {user.email} – {map_type_label(user.type)}
                </span>
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

    view = st.session_state.get("current_view", "Feed")
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
