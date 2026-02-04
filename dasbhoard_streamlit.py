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

CSS = """
<style>
:root {
    --pqr-primary: #3b82f6;
    --pqr-primary-soft: rgba(59, 130, 246, 0.16);
    --pqr-accent: #22c55e;
    --pqr-bg: #020617;
    --pqr-bg-card: rgba(15,23,42,0.94);
    --pqr-border-soft: rgba(148,163,184,0.35);
    --pqr-text-main: #e5e7eb;
    --pqr-text-soft: #9ca3af;
}

/* esconder sidebar */
[data-testid="stSidebar"] { display: none; }

/* fundo geral */
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
    display:flex;align-items:center;gap:10px;
}
.pqr-logo-avatar {
    width: 36px;height:36px;border-radius:12px;
    background: conic-gradient(from 200deg,#3b82f6,#22c55e,#a855f7,#3b82f6);
    display:flex;align-items:center;justify-content:center;
    color:#020617;font-weight:800;font-size:0.9rem;
    box-shadow:0 0 18px rgba(59,130,246,0.7);
}
.pqr-title-text { display:flex;flex-direction:column; }
.pqr-title-main {
    font-size:1.2rem;font-weight:700;
    letter-spacing:0.10em;text-transform:uppercase;
}
.pqr-title-sub { font-size:0.78rem;color:var(--pqr-text-soft); }

/* user pill header */
.user-pill-header {
    display:flex;align-items:center;gap:10px;
    padding:4px 10px;
    border-radius:999px;
    background: radial-gradient(circle at top left, rgba(15,23,42,0.95), rgba(15,23,42,0.9));
    border:1px solid rgba(148,163,184,0.4);
    box-shadow:0 8px 24px rgba(15,23,42,0.85);
}
.user-pill-avatar {
    width:32px;height:32px;border-radius:999px;
    background:linear-gradient(135deg,#1d4ed8,#22c55e);
    display:flex;align-items:center;justify-content:center;
    color:#020617;font-weight:700;overflow:hidden;
}
.user-pill-avatar img{
    width:100%;height:100%;object-fit:cover;border-radius:999px;
}

/* bell notificações */
.pqr-bell{
    margin-left:10px;width:28px;height:28px;border-radius:999px;
    background:radial-gradient(circle at top,rgba(148,163,184,0.3),rgba(15,23,42,1));
    display:flex;align-items:center;justify-content:center;
    border:1px solid rgba(148,163,184,0.6);
    cursor:pointer;position:relative;
}
.pqr-bell span{font-size:0.9rem;}
.pqr-bell-dot{
    position:absolute;top:3px;right:4px;
    width:7px;height:7px;border-radius:999px;background:#f97316;
}

/* nav pills */
.pqr-nav{
    margin-top:10px;display:flex;flex-wrap:wrap;gap:8px;
}
.pqr-nav-pill{
    padding:5px 12px;border-radius:999px;
    border:1px solid rgba(148,163,184,0.5);
    font-size:0.78rem;color:var(--pqr-text-soft);
    background:rgba(15,23,42,0.8);
    cursor:pointer;user-select:none;
}
.pqr-nav-pill-active{
    background:linear-gradient(135deg,#3b82f6,#22c55e);
    color:#0b1120;border-color:rgba(148,163,184,0.8);
}

/* glass main */
.glass-main{
    margin-top:14px;
    background:var(--pqr-bg-card);
    border-radius:18px;
    border:1px solid var(--pqr-border-soft);
    box-shadow:0 20px 45px rgba(15,23,42,0.9);
    padding:18px 22px;
    backdrop-filter:blur(18px);
}
.glass-section{
    background:rgba(15,23,42,0.9);
    border-radius:14px;
    border:1px solid rgba(148,163,184,0.35);
    padding:12px 14px;
}

/* feed posts */
.post-card{
    background:rgba(15,23,42,0.96);
    border-radius:14px;
    border:1px solid rgba(148,163,184,0.4);
    padding:10px 12px;
    margin-bottom:10px;
}
.post-header{display:flex;align-items:center;gap:8px;margin-bottom:4px;font-size:0.83rem;}
.post-meta{font-size:0.75rem;color:var(--pqr-text-soft);}
.post-body{font-size:0.86rem;margin:4px 0 6px;}
.post-tags{font-size:0.75rem;color:#a855f7;}
.post-actions{
    display:flex;gap:12px;font-size:0.78rem;color:var(--pqr-text-soft);margin-top:4px;
}

/* timeline */
.timeline-card{
    border-radius:12px;padding:8px 10px;margin-bottom:6px;
    background:rgba(15,23,42,0.97);
    border:1px solid rgba(148,163,184,0.4);
    font-size:0.8rem;
}
.timeline-card-header{
    display:flex;justify-content:space-between;font-weight:500;margin-bottom:3px;
}
.timeline-card-body{color:var(--pqr-text-soft);font-size:0.78rem;}
.timeline-badge{
    font-size:0.7rem;padding:2px 7px;border-radius:999px;
    background:rgba(37,99,235,0.18);
}

/* chat bubble */
.chat-bubble{
    padding:7px 9px;border-radius:10px;margin-bottom:6px;
    font-size:0.84rem;
    background:rgba(15,23,42,0.96);
    border:1px solid rgba(148,163,184,0.4);
}
.chat-meta{font-size:0.7rem;color:var(--pqr-text-soft);margin-bottom:2px;}

/* liquid buttons */
.liquid-btn{
    display:inline-flex;align-items:center;justify-content:center;
    padding:6px 14px;border-radius:999px;
    border:1px solid rgba(148,163,184,0.6);
    background:
      radial-gradient(circle at top left,rgba(59,130,246,0.4),transparent 55%),
      radial-gradient(circle at bottom right,rgba(15,118,110,0.35),rgba(15,23,42,0.9));
    color:#e5e7eb;font-size:0.8rem;font-weight:500;
    box-shadow:0 10px 30px rgba(15,23,42,0.9);
    cursor:pointer;
}
.liquid-btn:hover{
    filter:brightness(1.1);
}

/* input tweaks */
input, textarea{
    background-color:#020617 !important;
    color:#e5e7eb !important;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ======================================================
# MODELOS DE DADOS
# ======================================================

@dataclass
class User:
    id: str
    name: str
    email: str
    type: str  # "researcher", "client", "participant"
    interests: List[str] = field(default_factory=list)
    avatar_path: Optional[str] = None

@dataclass
class Post:
    id: str
    author_id: str
    text: str
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    likes: List[str] = field(default_factory=list)
    saves: List[str] = field(default_factory=list)
    comments: List[Dict[str, Any]] = field(default_factory=list)  # {"user_id","text","time"}

@dataclass
class TimelineStep:
    id: str
    title: str
    description: str
    status: str  # "planejado","em andamento","concluído"
    owner_id: str
    created_at: str

@dataclass
class ResearchDoc:
    id: str
    title: str
    type: str
    content: str
    created_at: str
    owner_id: str

@dataclass
class MindNode:
    id: str
    label: str
    parent_id: Optional[str]
    note: str = ""
    tags: List[str] = field(default_factory=list)

@dataclass
class ChatMessage:
    id: str
    from_id: str
    to_id: Optional[str]
    text: str
    time: str

# ======================================================
# STATE: LOAD / SAVE
# ======================================================

def default_state_dict() -> Dict[str, Any]:
    return {
        "users": [],
        "current_user_id": None,
        "posts": [],
        "timeline": [],
        "docs": [],
        "mind_nodes": [],
        "chat_messages": [],
        "notifications": [],
        "current_view": "Feed social",
    }

def load_state_from_file():
    if not os.path.exists(STATE_FILE):
        st.session_state.state = default_state_dict()
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        st.session_state.state = default_state_dict()
        return

    state = default_state_dict()
    # users
    raw_users = data.get("users", [])
    users: List[User] = []
    for u in raw_users:
        try:
            users.append(
                User(
                    id=u.get("id", str(uuid.uuid4())),
                    name=u.get("name", "Usuário"),
                    email=u.get("email", ""),
                    type=u.get("type", "researcher"),
                    interests=u.get("interests", []),
                    avatar_path=u.get("avatar_path"),
                )
            )
        except Exception:
            continue
    state["users"] = users

    # posts
    raw_posts = data.get("posts", [])
    posts: List[Post] = []
    for p in raw_posts:
        try:
            posts.append(
                Post(
                    id=p.get("id", str(uuid.uuid4())),
                    author_id=p.get("author_id", ""),
                    text=p.get("text", ""),
                    tags=p.get("tags", []),
                    created_at=p.get("created_at", ""),
                    likes=p.get("likes", []),
                    saves=p.get("saves", []),
                    comments=p.get("comments", []),
                )
            )
        except Exception:
            continue
    state["posts"] = posts

    # timeline
    raw_tl = data.get("timeline", [])
    tl: List[TimelineStep] = []
    for t in raw_tl:
        try:
            tl.append(
                TimelineStep(
                    id=t.get("id", str(uuid.uuid4())),
                    title=t.get("title", ""),
                    description=t.get("description", ""),
                    status=t.get("status", "planejado"),
                    owner_id=t.get("owner_id", ""),
                    created_at=t.get("created_at", ""),
                )
            )
        except Exception:
            continue
    state["timeline"] = tl

    # docs
    raw_docs = data.get("docs", [])
    docs: List[ResearchDoc] = []
    for d in raw_docs:
        try:
            docs.append(
                ResearchDoc(
                    id=d.get("id", str(uuid.uuid4())),
                    title=d.get("title", ""),
                    type=d.get("type", ""),
                    content=d.get("content", ""),
                    created_at=d.get("created_at", ""),
                    owner_id=d.get("owner_id", ""),
                )
            )
        except Exception:
            continue
    state["docs"] = docs

    # mind nodes
    raw_nodes = data.get("mind_nodes", [])
    nodes: List[MindNode] = []
    for n in raw_nodes:
        try:
            nodes.append(
                MindNode(
                    id=n.get("id", str(uuid.uuid4())),
                    label=n.get("label", ""),
                    parent_id=n.get("parent_id"),
                    note=n.get("note", ""),
                    tags=n.get("tags", []),
                )
            )
        except Exception:
            continue
    state["mind_nodes"] = nodes

    # chat
    raw_chat = data.get("chat_messages", [])
    chs: List[ChatMessage] = []
    for c in raw_chat:
        try:
            chs.append(
                ChatMessage(
                    id=c.get("id", str(uuid.uuid4())),
                    from_id=c.get("from_id", ""),
                    to_id=c.get("to_id"),
                    text=c.get("text", ""),
                    time=c.get("time", ""),
                )
            )
        except Exception:
            continue
    state["chat_messages"] = chs

    state["notifications"] = data.get("notifications", [])
    state["current_user_id"] = data.get("current_user_id")
    state["current_view"] = data.get("current_view", "Feed social")

    st.session_state.state = state

def save_state_to_file():
    state = st.session_state.get("state")
    if not state:
        return
    out = {
        "users": [asdict(u) for u in state["users"]],
        "posts": [asdict(p) for p in state["posts"]],
        "timeline": [asdict(t) for t in state["timeline"]],
        "docs": [asdict(d) for d in state["docs"]],
        "mind_nodes": [asdict(n) for n in state["mind_nodes"]],
        "chat_messages": [asdict(c) for c in state["chat_messages"]],
        "notifications": state["notifications"],
        "current_user_id": state["current_user_id"],
        "current_view": state["current_view"],
    }
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# init
if "state" not in st.session_state:
    load_state_from_file()

# short alias
S = st.session_state.state

# ======================================================
# HELPERS
# ======================================================

def map_type_label(t: str) -> str:
    if t == "researcher":
        return "Pesquisador(a)"
    if t == "client":
        return "Cliente"
    if t == "participant":
        return "Participante"
    return t

def get_current_user() -> Optional[User]:
    user_id = S.get("current_user_id")
    if not user_id:
        return None
    for u in S["users"]:
        if isinstance(u, User) and u.id == user_id:
            return u
    return None

def avatar_html(user: User, size: int = 32) -> str:
    if user.avatar_path and os.path.exists(user.avatar_path):
        rel = user.avatar_path
        return f'<img src="{rel}" style="width:{size}px;height:{size}px;border-radius:999px;object-fit:cover;">'
    return user.name[:1].upper()

def ensure_default_mind_root():
    if S["mind_nodes"]:
        return
    root = MindNode(
        id=str(uuid.uuid4()),
        label="Tema central",
        parent_id=None,
        note="Nó principal do mapa da pesquisa",
        tags=["root"],
    )
    S["mind_nodes"].append(root)

# ======================================================
# AUTENTICAÇÃO SIMPLES
# ======================================================

def auth_screen():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Entrar na rede PQR")

    tab_login, tab_cad = st.tabs(["Entrar", "Criar conta"])

    with tab_login:
        email = st.text_input("E-mail")
        if st.button("Entrar", key="btn_login"):
            email = email.strip().lower()
            found = None
            for u in S["users"]:
                if isinstance(u, User) and u.email.lower() == email:
                    found = u
                    break
            if found:
                S["current_user_id"] = found.id
                save_state_to_file()
                st.experimental_rerun()
            else:
                st.error("E-mail não encontrado. Crie uma conta.")

    with tab_cad:
        name = st.text_input("Nome completo")
        email2 = st.text_input("E-mail para cadastro")
        t = st.selectbox(
            "Seu papel na rede",
            ["researcher", "client", "participant"],
            format_func=map_type_label,
        )
        interests_str = st.text_input("Interesses (separados por vírgula)")
        if st.button("Criar conta", key="btn_cad"):
            email2_clean = email2.strip().lower()
            if not name or not email2_clean:
                st.warning("Preencha nome e e-mail.")
            else:
                for u in S["users"]:
                    if isinstance(u, User) and u.email.lower() == email2_clean:
                        st.error("E-mail já cadastrado.")
                        st.stop()
                inter = [i.strip() for i in interests_str.split(",") if i.strip()]
                new_user = User(
                    id=str(uuid.uuid4()),
                    name=name.strip(),
                    email=email2_clean,
                    type=t,
                    interests=inter,
                )
                S["users"].append(new_user)
                S["current_user_id"] = new_user.id
                save_state_to_file()
                st.success("Conta criada. Bem-vinda à rede.")
                st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# HEADER + NAVEGAÇÃO
# ======================================================

VIEWS = [
    "Feed social",
    "Timeline / Etapas",
    "Pasta da pesquisa",
    "Mapa mental",
    "Canvas / Slides",
    "Análise inteligente",
    "Chat",
    "Cadeia de ligação",
    "Configurações",
]

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
        has_notif = len(S["notifications"]) > 0
        dot = '<div class="pqr-bell-dot"></div>' if has_notif else ""
        st.markdown(
            f"""
            <div class="pqr-header-row" style="justify-content:flex-end;">
                <div class="user-pill-header">
                    <div class="user-pill-avatar">{av_html}</div>
                    <div style="font-size:0.78rem;">
                        {user.name.split(" ")[0]}<br/>
                        <span style="color:#9ca3af;">{map_type_label(user.type)}</span>
                    </div>
                    <div class="pqr-bell">
                        <span>🔔</span>
                        {dot}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # nav
    st.markdown('<div class="pqr-nav">', unsafe_allow_html=True)
    nav_cols = st.columns(len(VIEWS))
    for i, (view_name, col) in enumerate(zip(VIEWS, nav_cols)):
        with col:
            is_active = S["current_view"] == view_name
            label = view_name
            if st.button(label, key=f"nav_{i}"):
                S["current_view"] = view_name
                save_state_to_file()
                st.experimental_rerun()
            st.markdown(
                f'<div class="{"pqr-nav-pill-active" if is_active else "pqr-nav-pill"}" style="display:none;">{label}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: FEED SOCIAL
# ======================================================

def view_feed():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Feed social")

    user = get_current_user()
    if not user:
        st.warning("Entre na sua conta para postar e interagir.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # criação de post
    st.markdown("#### Criar novo post")
    text = st.text_area("O que você está pesquisando ou descobrindo?")
    tags_str = st.text_input("Tags (separadas por vírgula)", key="feed_tags")
    if st.button("Publicar", key="btn_post"):
        if text.strip():
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            p = Post(
                id=str(uuid.uuid4()),
                author_id=user.id,
                text=text.strip(),
                tags=tags,
                created_at=datetime.datetime.now().strftime("%d/%m %H:%M"),
            )
            S["posts"].insert(0, p)
            S["notifications"].append(
                {
                    "msg": f"{user.name.split(' ')[0]} publicou um novo post.",
                    "time": datetime.datetime.now().isoformat(),
                }
            )
            save_state_to_file()
            st.success("Post publicado.")
            st.experimental_rerun()
        else:
            st.warning("Escreva algo para postar.")

    st.write("---")
    st.markdown("#### Últimos posts")

    if not S["posts"]:
        st.info("Nenhum post ainda. Comece compartilhando sua pesquisa.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for p in S["posts"]:
        author = next((u for u in S["users"] if u.id == p.author_id), None)
        name = author.name if author else "Usuário"
        av_html = avatar_html(author, size=28) if author else "U"
        tags = ", ".join([f"#{t}" for t in p.tags]) if p.tags else ""
        st.markdown(
            f"""
            <div class="post-card">
                <div class="post-header">
                    <div class="user-pill-avatar" style="width:28px;height:28px;">{av_html}</div>
                    <div>
                        <strong>{name}</strong><br/>
                        <span class="post-meta">{p.created_at}</span>
                    </div>
                </div>
                <div class="post-body">{p.text}</div>
                <div class="post-tags">{tags}</div>
            """,
            unsafe_allow_html=True,
        )

        # ações
        col1, col2, col3, col4 = st.columns(4)
        liked = user.id in p.likes
        saved = user.id in p.saves
        with col1:
            if st.button(
                f"👍 {len(p.likes)}",
                key=f"like_{p.id}",
            ):
                if liked:
                    p.likes.remove(user.id)
                else:
                    p.likes.append(user.id)
                save_state_to_file()
                st.experimental_rerun()
        with col2:
            if st.button(
                f"💾 {len(p.saves)}",
                key=f"save_{p.id}",
            ):
                if saved:
                    p.saves.remove(user.id)
                else:
                    p.saves.append(user.id)
                save_state_to_file()
                st.experimental_rerun()
        with col3:
            if st.button("💬", key=f"comment_{p.id}"):
                S["current_view"] = "Chat"
                S["notifications"].append(
                    {
                        "msg": f"Você abriu o chat a partir do post de {name}.",
                        "time": datetime.datetime.now().isoformat(),
                    }
                )
                save_state_to_file()
                st.experimental_rerun()
        with col4:
            if st.button("🔗", key=f"share_{p.id}"):
                S["notifications"].append(
                    {
                        "msg": f"Você compartilhou um post de {name}.",
                        "time": datetime.datetime.now().isoformat(),
                    }
                )
                save_state_to_file()
                st.success("Post 'compartilhado' (simulado).")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: TIMELINE / ETAPAS
# ======================================================

def view_timeline():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Timeline / Etapas da pesquisa")

    user = get_current_user()
    if not user:
        st.warning("Entre na sua conta.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("#### Nova etapa")
    title = st.text_input("Título da etapa")
    desc = st.text_area("Descrição rápida")
    status = st.selectbox(
        "Status",
        ["planejado", "em andamento", "concluído"],
        format_func=lambda x: x.capitalize(),
    )
    if st.button("Adicionar etapa", key="add_step"):
        if title.strip():
            S["timeline"].append(
                TimelineStep(
                    id=str(uuid.uuid4()),
                    title=title.strip(),
                    description=desc.strip(),
                    status=status,
                    owner_id=user.id,
                    created_at=datetime.datetime.now().strftime("%d/%m %H:%M"),
                )
            )
            save_state_to_file()
            st.success("Etapa adicionada.")
            st.experimental_rerun()
        else:
            st.warning("Dê um título para a etapa.")

    st.write("---")
    st.markdown("#### Suas etapas")
    my_steps = [t for t in S["timeline"] if t.owner_id == user.id]
    if not my_steps:
        st.info("Nenhuma etapa criada ainda.")
    else:
        for t in my_steps:
            st.markdown(
                f"""
                <div class="timeline-card">
                    <div class="timeline-card-header">
                        <span>{t.title}</span>
                        <span class="timeline-badge">{t.status.capitalize()}</span>
                    </div>
                    <div class="timeline-card-body">{t.description}</div>
                    <div class="timeline-card-body">Criado em {t.created_at}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: PASTA DA PESQUISA
# ======================================================

def view_docs():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Pasta da pesquisa")

    user = get_current_user()
    if not user:
        st.warning("Entre na sua conta.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("#### Novo documento")
    title = st.text_input("Título do documento")
    doc_type = st.selectbox("Tipo", ["Notas", "Guia de campo", "Relatório", "Outro"])
    content = st.text_area("Conteúdo")
    if st.button("Salvar documento"):
        if title.strip() and content.strip():
            S["docs"].append(
                ResearchDoc(
                    id=str(uuid.uuid4()),
                    title=title.strip(),
                    type=doc_type,
                    content=content.strip(),
                    created_at=datetime.datetime.now().strftime("%d/%m %H:%M"),
                    owner_id=user.id,
                )
            )
            save_state_to_file()
            st.success("Documento salvo.")
            st.experimental_rerun()
        else:
            st.warning("Título e conteúdo são obrigatórios.")

    st.write("---")
    st.markdown("#### Meus documentos")
    my_docs = [d for d in S["docs"] if d.owner_id == user.id]
    if not my_docs:
        st.info("Nenhum documento ainda.")
    else:
        for d in my_docs:
            with st.expander(f"{d.title} ({d.type}) – {d.created_at}"):
                st.write(d.content)

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: MAPA MENTAL
# ======================================================

def view_mindmap():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Mapa mental da pesquisa")

    user = get_current_user()
    if not user:
        st.warning("Entre na sua conta.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    ensure_default_mind_root()

    # selecionar nó para editar
    nodes = S["mind_nodes"]
    options = {n.label: n.id for n in nodes}
    sel_label = st.selectbox("Selecione um nó para focar", list(options.keys()))
    current_node = next(n for n in nodes if n.id == options[sel_label])

    st.markdown("#### Nó selecionado")
    st.write(f"**{current_node.label}**")
    st.write(current_node.note or "_Sem nota_")
    if current_node.tags:
        st.write("Tags:", ", ".join([f"`{t}`" for t in current_node.tags]))

    st.write("---")
    st.markdown("#### Adicionar nó filho")
    child_label = st.text_input("Título do novo nó")
    child_note = st.text_area("Nota")
    child_tags = st.text_input("Tags (vírgula)")
    if st.button("Adicionar ao mapa"):
        if child_label.strip():
            tags = [t.strip() for t in child_tags.split(",") if t.strip()]
            S["mind_nodes"].append(
                MindNode(
                    id=str(uuid.uuid4()),
                    label=child_label.strip(),
                    parent_id=current_node.id,
                    note=child_note.strip(),
                    tags=tags,
                )
            )
            save_state_to_file()
            st.success("Nó adicionado.")
            st.experimental_rerun()
        else:
            st.warning("Dê um nome ao nó.")

    st.write("---")
    st.markdown("#### Estrutura atual (árvore simples)")

    def render_tree(parent_id: Optional[str], level: int = 0):
        for n in [x for x in nodes if x.parent_id == parent_id]:
            indent = "&nbsp;" * (level * 4)
            st.markdown(
                f"{indent}• **{n.label}**  <span style='color:#9ca3af;font-size:0.75rem;'>"
                + (", ".join([f"`{t}`" for t in n.tags]) if n.tags else "")
                + "</span>",
                unsafe_allow_html=True,
            )
            render_tree(n.id, level + 1)

    render_tree(None)

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: CANVAS / SLIDES (SIMPLIFICADO)
# ======================================================

def view_canvas():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Canvas / Slides da pesquisa")

    st.info(
        "Protótipo simples de organização visual de slides. "
        "Por enquanto, é um espaço de notas por sessão."
    )

    slide_title = st.text_input("Título do slide")
    slide_key = f"slide_{slide_title}" if slide_title else "slide_sem_titulo"
    slide_body = st.text_area("Conteúdo", key=slide_key)

    st.write("Use este espaço como quadro de anotações para seus slides.")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: ANÁLISE INTELIGENTE (RESUMOS BÁSICOS)
# ======================================================

def view_analysis():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Análise inteligente (estatísticas simples)")

    user = get_current_user()
    if not user:
        st.warning("Entre na sua conta.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # contagem de posts, docs e etapas do usuário
    my_posts = [p for p in S["posts"] if p.author_id == user.id]
    my_docs = [d for d in S["docs"] if d.owner_id == user.id]
    my_steps = [t for t in S["timeline"] if t.owner_id == user.id]

    col1, col2, col3 = st.columns(3)
    col1.metric("Posts publicados", len(my_posts))
    col2.metric("Docs na pasta", len(my_docs))
    col3.metric("Etapas no pipeline", len(my_steps))

    st.write("---")
    st.markdown("#### Tags mais usadas nos seus posts")
    tags = []
    for p in my_posts:
        tags.extend(p.tags)
    if tags:
        c = Counter(tags)
        for tag, qtd in c.most_common(10):
            st.write(f"- `{tag}`: {qtd}")
    else:
        st.info("Você ainda não usou tags em seus posts.")

    st.write("---")
    st.markdown("#### Status das suas etapas")
    status_counts = Counter([t.status for t in my_steps])
    if status_counts:
        for s, qtd in status_counts.items():
            st.write(f"- {s.capitalize()}: {qtd}")
    else:
        st.info("Nenhuma etapa criada ainda.")

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

    msgs = [m for m in S["chat_messages"] if m.to_id is None]
    for m in msgs[-50:]:
        author = next((u for u in S["users"] if u.id == m.from_id), None)
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
    msg = st.text_input("Sua mensagem", key="chat_input")
    if st.button("Enviar"):
        if msg.strip():
            cm = ChatMessage(
                id=str(uuid.uuid4()),
                from_id=user.id,
                to_id=None,
                text=msg.strip(),
                time=datetime.datetime.now().strftime("%d/%m %H:%M"),
            )
            S["chat_messages"].append(cm)
            save_state_to_file()
            st.experimental_rerun()
        else:
            st.warning("Escreva algo.")

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: CADEIA DE LIGAÇÃO (CONEXÕES DE INTERESSES)
# ======================================================

def view_network():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Cadeia de ligação por interesses")

    user = get_current_user()
    if not user:
        st.warning("Entre na sua conta.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not S["users"]:
        st.info("Nenhum usuário na rede ainda.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not user.interests:
        st.info("Você ainda não declarou interesses. Vá em Configurações e atualize.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("#### Pessoas com interesses em comum com você")

    def shared_interests(u1: User, u2: User) -> List[str]:
        return list(set(u1.interests) & set(u2.interests))

    for other in S["users"]:
        if other.id == user.id:
            continue
        inter = shared_interests(user, other)
        if inter:
            st.write(
                f"- **{other.name}** ({map_type_label(other.type)}): "
                + ", ".join([f"`{i}`" for i in inter])
            )

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

    with st.expander("Atualizar interesses"):
        inter_str = ", ".join(user.interests)
        new_inter = st.text_input(
            "Interesses (separados por vírgula)", value=inter_str
        )
        if st.button("Salvar interesses"):
            user.interests = [i.strip() for i in new_inter.split(",") if i.strip()]
            save_state_to_file()
            st.success("Interesses atualizados.")
            st.experimental_rerun()

    with st.expander("Atualizar foto de perfil"):
        avatar_file = st.file_uploader("Nova foto (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if st.button("Salvar foto"):
            if avatar_file:
                img = Image.open(avatar_file).convert("RGB")
                uid = str(uuid.uuid4())
                avatar_path = AVATAR_DIR / f"{uid}.jpg"
                img.save(avatar_path, format="JPEG", quality=90)
                user.avatar_path = str(avatar_path)
                save_state_to_file()
                st.success("Foto atualizada.")
                st.experimental_rerun()
            else:
                st.warning("Selecione um arquivo.")

    st.write("---")
    st.subheader("Sessão")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Salvar dados agora"):
            save_state_to_file()
            st.success("Dados salvos.")
    with col2:
        if st.button("Sair da conta"):
            S["current_user_id"] = None
            save_state_to_file()
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

    view = S["current_view"]
    if view == "Feed social":
        view_feed()
    elif view == "Timeline / Etapas":
        view_timeline()
    elif view == "Pasta da pesquisa":
        view_docs()
    elif view == "Mapa mental":
        view_mindmap()
    elif view == "Canvas / Slides":
        view_canvas()
    elif view == "Análise inteligente":
        view_analysis()
    elif view == "Chat":
        view_chat()
    elif view == "Cadeia de ligação":
        view_network()
    elif view == "Configurações":
        view_settings()
    else:
        view_feed()

if __name__ == "__main__":
    main()
