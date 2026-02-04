import streamlit as st
import datetime
import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from collections import Counter
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
# CSS – AZUL ESCURO + LIQUID GLASS
# ======================================================

CSS = """
<style>
:root {
    --pqr-primary: #3b82f6;
    --pqr-primary-soft: rgba(59, 130, 246, 0.18);
    --pqr-accent: #22c55e;
    --pqr-bg: #020617;
    --pqr-bg-card: rgba(15,23,42,0.94);
    --pqr-border-soft: rgba(148,163,184,0.35);
    --pqr-text-main: #e5e7eb;
    --pqr-text-soft: #9ca3af;
}

/* esconder sidebar */
[data-testid="stSidebar"] { display:none; }

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
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:16px;
}
.pqr-logo-line { display:flex;align-items:center;gap:10px; }
.pqr-logo-avatar {
    width:36px;height:36px;border-radius:12px;
    background:conic-gradient(from 200deg,#3b82f6,#22c55e,#a855f7,#3b82f6);
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
    background:radial-gradient(circle at top left,rgba(15,23,42,0.95),rgba(15,23,42,0.9));
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

/* bell */
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

/* nav */
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

/* auth card */
.auth-card{
    margin-top:60px;
    max-width:420px;
    margin-left:auto;
    margin-right:auto;
    background:rgba(15,23,42,0.96);
    border-radius:18px;
    border:1px solid rgba(148,163,184,0.5);
    box-shadow:0 20px 45px rgba(15,23,42,0.95);
    padding:20px 22px;
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

/* liquid button */
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
.liquid-btn:hover{filter:brightness(1.1);}

/* input */
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
    type: str  # researcher, client, participant
    password: str
    interests: List[str] = field(default_factory=list)
    avatar_path: Optional[str] = None

@dataclass
class Post:
    id: str
    author_id: str
    text: str
    created_at: str
    likes: List[str] = field(default_factory=list)
    saved_by: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    comments: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class TimelineStep:
    id: str
    owner_id: str
    title: str
    description: str
    status: str  # ideação, campo, análise...
    created_at: str

@dataclass
class DocItem:
    id: str
    owner_id: str
    title: str
    content: str
    type: str  # nota, transcrição etc.
    created_at: str

@dataclass
class MindNode:
    id: str
    label: str
    children: List["MindNode"] = field(default_factory=list)

@dataclass
class ChatMessage:
    id: str
    from_id: str
    to_id: Optional[str]
    text: str
    time: str

# ======================================================
# ESTADO GLOBAL EM MEMÓRIA
# ======================================================

if "PQR_STATE" not in st.session_state:
    st.session_state.PQR_STATE = {
        "users": [],
        "posts": [],
        "timeline": [],
        "docs": [],
        "mindroot": None,
        "chat_messages": [],
        "current_user_id": None,
        "current_view": "Feed social",
        "notifications": [],
    }

S = st.session_state.PQR_STATE  # atalho

# ======================================================
# PERSISTÊNCIA EM ARQUIVO
# ======================================================

def save_state_to_file():
    data = {
        "users": [asdict(u) for u in S["users"]],
        "posts": [asdict(p) for p in S["posts"]],
        "timeline": [asdict(t) for t in S["timeline"]],
        "docs": [asdict(d) for d in S["docs"]],
        "mindroot": mindnode_to_dict(S["mindroot"]) if S["mindroot"] else None,
        "chat_messages": [asdict(m) for m in S["chat_messages"]],
        "current_user_id": S["current_user_id"],
        "notifications": S["notifications"],
    }
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # em ambientes sem escrita

def load_state_from_file():
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    # reconstruir
    S["users"] = [User(**u) for u in data.get("users", [])]
    S["posts"] = [Post(**p) for p in data.get("posts", [])]
    S["timeline"] = [TimelineStep(**t) for t in data.get("timeline", [])]
    S["docs"] = [DocItem(**d) for d in data.get("docs", [])]
    S["chat_messages"] = [ChatMessage(**m) for m in data.get("chat_messages", [])]
    S["current_user_id"] = data.get("current_user_id")
    S["notifications"] = data.get("notifications", [])
    mind = data.get("mindroot")
    if mind:
        S["mindroot"] = dict_to_mindnode(mind)

# ======================================================
# CONVERSÃO MINDNODE
# ======================================================

def mindnode_to_dict(n: MindNode) -> Dict[str, Any]:
    if not n:
        return {}
    return {
        "id": n.id,
        "label": n.label,
        "children": [mindnode_to_dict(c) for c in n.children],
    }

def dict_to_mindnode(d: Dict[str, Any]) -> MindNode:
    return MindNode(
        id=d.get("id", str(uuid.uuid4())),
        label=d.get("label", "Tópico"),
        children=[dict_to_mindnode(c) for c in d.get("children", [])],
    )

# carregar do arquivo uma vez
load_state_from_file()

# ======================================================
# HELPERS
# ======================================================

def map_type_label(t: str) -> str:
    return {
        "researcher": "Pesquisador(a)",
        "client": "Cliente",
        "participant": "Participante",
    }.get(t, t)

def get_current_user() -> Optional[User]:
    uid = S.get("current_user_id")
    if not uid:
        return None
    for u in S["users"]:
        if isinstance(u, User) and u.id == uid:
            return u
    return None

def avatar_html(user: User, size: int = 32) -> str:
    if user.avatar_path and os.path.exists(user.avatar_path):
        return f'<img src="{user.avatar_path}" style="width:{size}px;height:{size}px;border-radius:999px;object-fit:cover;">'
    return user.name[:1].upper()

# ======================================================
# AUTENTICAÇÃO – LOGIN / CADASTRO
# ======================================================

def auth_screen():
    st.markdown(
        """
        <div style="text-align:center;margin-top:30px;">
            <div class="pqr-logo-line" style="justify-content:center;">
                <div class="pqr-logo-avatar">P</div>
                <div class="pqr-title-text">
                    <div class="pqr-title-main">PQR</div>
                    <div class="pqr-title-sub">rede de pesquisa qualitativa</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="auth-card">', unsafe_allow_html=True)

    tabs = st.tabs(["Entrar", "Criar conta"])

    with tabs[0]:
        st.subheader("Entrar")
        email = st.text_input("E-mail", key="login_email")
        password = st.text_input("Senha", type="password", key="login_pwd")
        if st.button("Entrar agora", key="btn_login"):
            if not email or not password:
                st.error("Preencha e-mail e senha.")
            else:
                user = None
                for u in S["users"]:
                    if u.email.strip().lower() == email.strip().lower() and u.password == password:
                        user = u
                        break
                if user:
                    S["current_user_id"] = user.id
                    save_state_to_file()
                    st.experimental_rerun()
                else:
                    st.error("E-mail ou senha incorretos.")

    with tabs[1]:
        st.subheader("Criar conta")
        name = st.text_input("Nome completo", key="reg_name")
        email = st.text_input("E-mail", key="reg_email")
        user_type = st.selectbox(
            "Você é:",
            ["researcher", "client", "participant"],
            format_func=map_type_label,
            key="reg_type",
        )
        password = st.text_input("Crie uma senha", type="password", key="reg_pwd")
        password2 = st.text_input("Repita a senha", type="password", key="reg_pwd2")

        if st.button("Criar conta", key="btn_register"):
            if not name or not email or not password or not password2:
                st.error("Preencha todos os campos.")
            elif password != password2:
                st.error("As senhas não conferem.")
            elif any(u.email.strip().lower() == email.strip().lower() for u in S["users"]):
                st.error("Já existe uma conta com esse e-mail.")
            else:
                new_user = User(
                    id=str(uuid.uuid4()),
                    name=name.strip(),
                    email=email.strip().lower(),
                    type=user_type,
                    password=password,
                )
                S["users"].append(new_user)
                S["current_user_id"] = new_user.id
                save_state_to_file()
                st.success("Conta criada. Entrando...")
                st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# HEADER E NAVEGAÇÃO
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
    if "current_view" not in S or S["current_view"] not in VIEWS:
        S["current_view"] = "Feed social"

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
        st.markdown(
            f"""
            <div class="pqr-header-row" style="justify-content:flex-end;">
                <div class="user-pill-header">
                    <div class="user-pill-avatar">{av_html}</div>
                    <div style="font-size:0.78rem;">
                        {user.name.split(" ")[0]}<br/>
                        <span style="color:#9ca3af;">{map_type_label(user.type)}</span>
                    </div>
                </div>
                <div class="pqr-bell">
                    <span>🔔</span>
                    {'<div class="pqr-bell-dot"></div>' if S['notifications'] else ''}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # nav pills
    st.markdown('<div class="pqr-nav">', unsafe_allow_html=True)
    cols = st.columns(len(VIEWS))
    for i, (view_name, c) in enumerate(zip(VIEWS, cols)):
        with c:
            active = (S["current_view"] == view_name)
            label = view_name
            if st.button(label, key=f"nav_{i}"):
                S["current_view"] = view_name
                save_state_to_file()
                st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# VIEWS (FUNCIONALIDADES)
# ======================================================

# ---- FEED ----

def view_feed():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Feed social")
    user = get_current_user()
    if not user:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # criar post
    with st.expander("Criar novo post"):
        text = st.text_area("O que você está pesquisando/pensando?", key="new_post")
        tag_str = st.text_input("Tags (separadas por vírgula)", key="new_post_tags")
        if st.button("Publicar", key="btn_pub_post"):
            if text.strip():
                tags = [t.strip() for t in tag_str.split(",") if t.strip()]
                p = Post(
                    id=str(uuid.uuid4()),
                    author_id=user.id,
                    text=text.strip(),
                    created_at=datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
                    tags=tags,
                )
                S["posts"].insert(0, p)
                save_state_to_file()
                st.success("Publicado no feed.")
                st.experimental_rerun()
            else:
                st.warning("Escreva algo para postar.")

    st.write("---")
    if not S["posts"]:
        st.info("Nenhum post ainda. Comece compartilhando uma ideia de pesquisa.")
    else:
        for p in S["posts"]:
            author = next((u for u in S["users"] if u.id == p.author_id), None)
            author_name = author.name if author else "Usuário"
            av = avatar_html(author or user)
            liked = user.id in p.likes
            saved = user.id in p.saved_by

            st.markdown('<div class="post-card">', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="post-header">
                    <div class="user-pill-avatar" style="width:26px;height:26px;">{av}</div>
                    <div>
                        <strong>{author_name}</strong><br/>
                        <span class="post-meta">{p.created_at}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="post-body">{p.text}</div>', unsafe_allow_html=True)
            if p.tags:
                st.markdown(
                    "<div class='post-tags'>" +
                    " ".join([f"`{t}`" for t in p.tags]) +
                    "</div>",
                    unsafe_allow_html=True,
                )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button(("💙 Curtir" if liked else "🤍 Curtir"), key=f"like_{p.id}"):
                    if liked:
                        p.likes.remove(user.id)
                    else:
                        p.likes.append(user.id)
                    save_state_to_file()
                    st.experimental_rerun()
                st.caption(f"{len(p.likes)}")

            with col2:
                if st.button(("🔖 Salvo" if saved else "📎 Salvar"), key=f"save_{p.id}"):
                    if saved:
                        p.saved_by.remove(user.id)
                    else:
                        p.saved_by.append(user.id)
                    save_state_to_file()
                    st.experimental_rerun()
                st.caption(f"{len(p.saved_by)}")

            with col3:
                if st.button("💬 Comentar", key=f"c_{p.id}"):
                    st.session_state[f"show_comments_{p.id}"] = not st.session_state.get(
                        f"show_comments_{p.id}", False
                    )
            with col4:
                if st.button("📤 Compartilhar", key=f"share_{p.id}"):
                    S["notifications"].append(
                        f"{user.name} compartilhou um post de {author_name}"
                    )
                    save_state_to_file()
                    st.success("Compartilhado (simulado).")

            if st.session_state.get(f"show_comments_{p.id}", False):
                st.write("")
                for c in p.comments:
                    st.markdown(
                        f"**{c['author_name']}**: {c['text']}  "
                        f"*({c['time']})*"
                    )
                c_text = st.text_input(
                    "Seu comentário",
                    key=f"new_comment_{p.id}",
                )
                if st.button("Enviar comentário", key=f"btn_comment_{p.id}"):
                    if c_text.strip():
                        p.comments.append(
                            {
                                "author_id": user.id,
                                "author_name": user.name,
                                "text": c_text.strip(),
                                "time": datetime.datetime.now().strftime("%d/%m %H:%M"),
                            }
                        )
                        save_state_to_file()
                        st.experimental_rerun()
                    else:
                        st.warning("Escreva algo para comentar.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---- TIMELINE ----

def view_timeline():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Timeline / Etapas da pesquisa")
    user = get_current_user()
    if not user:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with st.expander("Adicionar etapa"):
        title = st.text_input("Título da etapa", key="step_title")
        desc = st.text_area("Descrição", key="step_desc")
        status = st.selectbox(
            "Status",
            ["ideação", "campo", "análise", "apresentação"],
            key="step_status",
        )
        if st.button("Adicionar etapa", key="btn_add_step"):
            if not title.strip():
                st.warning("Informe um título.")
            else:
                step = TimelineStep(
                    id=str(uuid.uuid4()),
                    owner_id=user.id,
                    title=title.strip(),
                    description=desc.strip(),
                    status=status,
                    created_at=datetime.datetime.now().strftime("%d/%m/%Y"),
                )
                S["timeline"].append(step)
                save_state_to_file()
                st.success("Etapa adicionada.")
                st.experimental_rerun()

    st.write("---")
    steps = [t for t in S["timeline"] if t.owner_id == user.id]
    if not steps:
        st.info("Nenhuma etapa ainda.")
    else:
        for t in steps:
            st.markdown('<div class="timeline-card">', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="timeline-card-header">
                    <span>{t.title}</span>
                    <span class="timeline-badge">{t.status}</span>
                </div>
                <div class="timeline-card-body">
                    {t.description or "Sem descrição."}<br/>
                    <span style="font-size:0.7rem;color:#6b7280;">
                        Criado em {t.created_at}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---- PASTA ----

def view_docs():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Pasta da pesquisa")
    user = get_current_user()
    if not user:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with st.expander("Adicionar documento"):
        title = st.text_input("Título", key="doc_title")
        typ = st.selectbox(
            "Tipo",
            ["nota", "transcrição", "insight", "roteiro"],
            key="doc_type",
        )
        content = st.text_area("Conteúdo", key="doc_content")
        if st.button("Salvar documento", key="btn_add_doc"):
            if not title.strip():
                st.warning("Informe um título.")
            else:
                d = DocItem(
                    id=str(uuid.uuid4()),
                    owner_id=user.id,
                    title=title.strip(),
                    content=content.strip(),
                    type=typ,
                    created_at=datetime.datetime.now().strftime("%d/%m/%Y"),
                )
                S["docs"].append(d)
                save_state_to_file()
                st.success("Documento salvo.")
                st.experimental_rerun()

    st.write("---")
    docs = [d for d in S["docs"] if d.owner_id == user.id]
    if not docs:
        st.info("Nenhum documento ainda.")
    else:
        for d in docs:
            st.markdown(f"**{d.title}**  (*{d.type}*, {d.created_at})")
            st.caption((d.content[:140] + "...") if len(d.content) > 140 else d.content)
            st.write("---")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- MAPA MENTAL (SIMPLES) ----

def view_mindmap():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Mapa mental (rústico)")
    user = get_current_user()
    if not user:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if S["mindroot"] is None:
        S["mindroot"] = MindNode(id=str(uuid.uuid4()), label="Tema central")
        save_state_to_file()

    root = S["mindroot"]
    st.write(f"**Nó raiz:** {root.label}")

    new_child = st.text_input("Novo sub-tópico ligado ao tema central", key="mm_new")
    if st.button("Adicionar tópico", key="mm_add"):
        if new_child.strip():
            root.children.append(
                MindNode(id=str(uuid.uuid4()), label=new_child.strip())
            )
            save_state_to_file()
            st.experimental_rerun()
        else:
            st.warning("Escreva algo.")

    st.write("---")
    st.markdown("#### Estrutura atual")
    def render_node(n: MindNode, level: int = 0):
        st.markdown("&nbsp;" * (level * 4) + f"- {n.label}", unsafe_allow_html=True)
        for c in n.children:
            render_node(c, level + 1)

    render_node(root)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- CANVAS / SLIDES (BEM SIMPLES) ----

def view_canvas():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Canvas / Slides (prototipagem simples)")

    st.info("Aqui é só um bloco de texto dividido em seções para você estruturar slides.")
    col1, col2 = st.columns(2)
    with col1:
        objetivo = st.text_area("Objetivo do estudo", key="canvas_obj")
        metodo = st.text_area("Método / amostra", key="canvas_met")
    with col2:
        achados = st.text_area("Principais achados", key="canvas_achados")
        recomend = st.text_area("Recomendações", key="canvas_rec")

    st.write("---")
    st.markdown("#### Prévia estilo slide")
    st.markdown(
        f"""
        **Objetivo:** {objetivo or "*não preenchido*"}  
        **Método:** {metodo or "*não preenchido*"}  

        **Achados:**  
        {achados or "*não preenchido*"}  

        **Recomendações:**  
        {recomend or "*não preenchido*"}
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---- ANÁLISE INTELIGENTE (SIMPLIFICADA) ----

def view_analysis():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Análise simples de atividade")

    user = get_current_user()
    if not user:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    my_posts = [p for p in S["posts"] if p.author_id == user.id]
    my_docs = [d for d in S["docs"] if d.owner_id == user.id]
    my_steps = [t for t in S["timeline"] if t.owner_id == user.id]

    col1, col2, col3 = st.columns(3)
    col1.metric("Posts publicados", len(my_posts))
    col2.metric("Docs na pasta", len(my_docs))
    col3.metric("Etapas na timeline", len(my_steps))

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
            st.write(f"- {s}: {qtd}")
    else:
        st.info("Nenhuma etapa criada ainda.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- CHAT ----

def view_chat():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Chat geral")

    user = get_current_user()
    if not user:
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
    if st.button("Enviar", key="btn_chat_send"):
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

# ---- CADEIA / REDE ----

def view_network():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Cadeia de ligação por interesses")

    user = get_current_user()
    if not user:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if not user.interests:
        st.info("Você ainda não declarou interesses. Vá em Configurações e atualize.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("#### Pessoas com interesses em comum")
    for other in S["users"]:
        if other.id == user.id:
            continue
        inter = list(set(user.interests) & set(other.interests))
        if inter:
            st.write(
                f"- **{other.name}** ({map_type_label(other.type)}): "
                + ", ".join([f"`{i}`" for i in inter])
            )

    st.markdown("</div>", unsafe_allow_html=True)

# ---- CONFIGURAÇÕES ----

def view_settings():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Configurações & perfil")

    user = get_current_user()
    if not user:
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
        new_inter = st.text_input("Interesses (separados por vírgula)", value=inter_str)
        if st.button("Salvar interesses", key="btn_save_inter"):
            user.interests = [i.strip() for i in new_inter.split(",") if i.strip()]
            save_state_to_file()
            st.success("Interesses atualizados.")
            st.experimental_rerun()

    with st.expander("Atualizar foto de perfil"):
        avatar_file = st.file_uploader("Nova foto (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if st.button("Salvar foto", key="btn_save_avatar"):
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
        if st.button("Salvar dados agora", key="btn_save_state"):
            save_state_to_file()
            st.success("Dados salvos.")
    with col2:
        if st.button("Sair da conta", key="btn_logout"):
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

    view = S.get("current_view", "Feed social")
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
