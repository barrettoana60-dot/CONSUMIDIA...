import streamlit as st
import datetime
import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from collections import Counter
import re
import uuid

# ======================================================
# CONFIG BÁSICA
# ======================================================

st.set_page_config(
    page_title="PQR – Rede de Pesquisa Qualitativa",
    layout="wide",
    initial_sidebar_state="collapsed",  # esconde sidebar
)

STATE_FILE = "pqr_state.json"

# ======================================================
# CSS – FUNDO BEGE, TEXTO ESCURO, GLASS, SEM SIDEBAR
# ======================================================

MODERN_CSS = """
<style>
:root {
    --pqr-primary: #2563eb;
    --pqr-primary-soft: rgba(37, 99, 235, 0.12);
    --pqr-accent: #10b981;
    --pqr-bg: #f4ecdf;           /* bege suave */
    --pqr-bg-card: rgba(255,255,255,0.85);
    --pqr-border-soft: rgba(17,24,39,0.12);
    --pqr-text-main: #111827;    /* quase preto */
    --pqr-text-soft: #4b5563;
}

/* esconder visualmente a sidebar (mas ela ainda existe para Streamlit) */
[data-testid="stSidebar"] {
    display: none;
}

/* fundo geral */
.stApp {
    background: radial-gradient(circle at top left, #fdf5e7, #f4ecdf 55%, #eadfcf);
    color: var(--pqr-text-main);
    font-family: system-ui,-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",sans-serif;
}

.block-container {
    padding-top: 0.5rem;
    padding-bottom: 0.8rem;
    max-width: 1100px;
}

/* header tipo instagram/linkedin no topo */
.pqr-header {
    position: sticky;
    top: 0;
    z-index: 999;
    padding: 10px 0 12px;
    background: linear-gradient(to bottom, rgba(244,236,223,0.96), rgba(244,236,223,0.90));
    backdrop-filter: blur(8px);
    border-bottom: 1px solid rgba(17,24,39,0.06);
}

/* linha com logo e perfil */
.pqr-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
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
    background: linear-gradient(135deg, #2563eb, #f97316);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #f9fafb;
    font-weight: 700;
    font-size: 0.9rem;
}
.pqr-title-text {
    display: flex;
    flex-direction: column;
}
.pqr-title-main {
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.pqr-title-sub {
    font-size: 0.8rem;
    color: var(--pqr-text-soft);
}

/* user pill / perfil no header */
.user-pill-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 8px;
    border-radius: 999px;
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(17,24,39,0.08);
}
.user-pill-avatar {
    width: 32px;
    height: 32px;
    border-radius: 999px;
    background: #e5e7eb;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #111827;
    font-weight: 600;
    overflow: hidden;
}
.user-pill-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 999px;
}

/* navegação tipo tabs horizontais */
.pqr-nav {
    margin-top: 8px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.pqr-nav-item {
    padding: 5px 12px;
    border-radius: 999px;
    border: 1px solid rgba(17,24,39,0.10);
    font-size: 0.78rem;
    color: var(--pqr-text-soft);
    background: rgba(255,255,255,0.7);
    cursor: pointer;
    user-select: none;
}
.pqr-nav-item-active {
    background: linear-gradient(135deg, #2563eb, #f97316);
    color: #f9fafb;
    border-color: rgba(17,24,39,0.1);
}

/* cartões glass principais */
.glass-main {
    margin-top: 14px;
    background: var(--pqr-bg-card);
    border-radius: 16px;
    border: 1px solid var(--pqr-border-soft);
    box-shadow: 0 18px 40px rgba(15,23,42,0.12);
    padding: 18px 22px;
    backdrop-filter: blur(14px);
}

/* seções internas mais discretas */
.glass-section {
    background: rgba(255,255,255,0.7);
    border-radius: 14px;
    border: 1px solid rgba(17,24,39,0.06);
    padding: 12px 14px;
}

/* posts estilo feed */
.post-card {
    background: rgba(255,255,255,0.85);
    border-radius: 14px;
    border: 1px solid rgba(17,24,39,0.08);
    padding: 10px 12px;
    margin-bottom: 10px;
}
.post-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
    font-size: 0.83rem;
}
.post-meta {
    font-size: 0.75rem;
    color: var(--pqr-text-soft);
}
.post-body {
    font-size: 0.86rem;
    margin: 4px 0 6px;
}
.post-actions {
    display: flex;
    gap: 12px;
    font-size: 0.78rem;
    color: var(--pqr-text-soft);
}

/* cards timeline */
.timeline-card {
    border-radius: 12px;
    padding: 8px 10px;
    margin-bottom: 6px;
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(17,24,39,0.08);
    font-size: 0.8rem;
}
.timeline-card-header {
    display: flex;
    justify-content: space-between;
    font-weight: 500;
    margin-bottom: 3px;
}
.timeline-card-body {
    color: var(--pqr-text-soft);
    font-size: 0.78rem;
}
.timeline-card-footer {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: var(--pqr-text-soft);
    margin-top: 4px;
}

/* chat bubble */
.chat-bubble {
    padding: 7px 9px;
    border-radius: 10px;
    margin-bottom: 6px;
    font-size: 0.84rem;
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(17,24,39,0.08);
}
.chat-meta {
    font-size: 0.70rem;
    color: var(--pqr-text-soft);
    margin-bottom: 2px;
}

/* mapa mental */
.mind-node {
    font-size: 0.84rem;
    margin: 2px 0;
}
.mind-node-label {
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(17,24,39,0.06);
}
.mind-node-selected {
    background: var(--pqr-primary-soft);
    color: var(--pqr-primary);
    border: 1px solid rgba(37,99,235,0.6);
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
    border:1px solid rgba(37,99,235,0.3);
    color:var(--pqr-primary);
}

/* inputs */
textarea, input, select {
    border-radius: 9px !important;
}

/* botões glass */
.stButton > button {
    border-radius: 999px;
    border: 1px solid rgba(17,24,39,0.14);
    background: radial-gradient(circle at top left, rgba(255,255,255,0.85), rgba(255,255,255,0.72));
    color: var(--pqr-text-main);
    font-size: 0.84rem;
    padding: 0.35rem 0.9rem;
    transition: all 0.15s ease-out;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(15,23,42,0.22);
    border-color: rgba(37,99,235,0.6);
}

/* radio nativo do Streamlit (não usamos mais para nav, mas deixo bonito) */
[data-baseweb="radio"] > div {
    gap: 4px;
}
</style>
"""

st.markdown(MODERN_CSS, unsafe_allow_html=True)

# ======================================================
# MODELOS DE DADOS
# ======================================================

@dataclass
class User:
    name: str
    email: str
    type: str
    password: str
    avatar_url: Optional[str] = None

@dataclass
class Card:
    id: str
    title: str
    description: str
    status: str
    deadline: Optional[str] = None
    created_at: str = ""

@dataclass
class ChatMessage:
    id: str
    user_name: str
    topic: str
    text: str
    time: str

@dataclass
class MindNode:
    id: str
    label: str
    note: str = ""
    tags: List[str] = field(default_factory=list)
    children: List["MindNode"] = field(default_factory=list)

@dataclass
class Post:
    id: str
    user_name: str
    user_avatar_url: Optional[str]
    text: str
    created_at: str
    likes: int = 0
    liked_by_me: bool = False

@dataclass
class Slide:
    id: str
    title: str
    content: str
    created_at: str

# ======================================================
# PERSISTÊNCIA
# ======================================================

def default_state_dict() -> Dict[str, Any]:
    return {
        "users": [],
        "current_user_email": None,
        "cards": [],
        "research_summary": "",
        "research_notes": "",
        "mind_root": {
            "id": "root",
            "label": "Tema central",
            "note": "",
            "tags": [],
            "children": [],
        },
        "mind_selected_id": "root",
        "chat_messages": [],
        "chat_topic": "metodologia",
        "posts": [],
        "slides": [],
    }

def load_persistent_state():
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

def mindnode_to_dict(n: MindNode) -> Dict[str, Any]:
    return {
        "id": getattr(n, "id", "no-id"),
        "label": getattr(n, "label", ""),
        "note": getattr(n, "note", ""),
        "tags": getattr(n, "tags", []),
        "children": [mindnode_to_dict(c) for c in getattr(n, "children", [])],
    }

def dict_to_mindnode(d: Dict[str, Any]) -> MindNode:
    return MindNode(
        id=d.get("id", "no-id"),
        label=d.get("label", ""),
        note=d.get("note", ""),
        tags=d.get("tags", []),
        children=[dict_to_mindnode(c) for c in d.get("children", [])],
    )

def save_persistent_state():
    data = {
        "users": [asdict(u) for u in st.session_state.users],
        "current_user_email": st.session_state.current_user_email,
        "cards": [asdict(c) for c in st.session_state.cards],
        "research_summary": st.session_state.research_summary,
        "research_notes": st.session_state.research_notes,
        "mind_root": mindnode_to_dict(st.session_state.mind_root),
        "mind_selected_id": st.session_state.mind_selected_id,
        "chat_messages": [asdict(m) for m in st.session_state.chat_messages],
        "chat_topic": st.session_state.chat_topic,
        "posts": [asdict(p) for p in st.session_state.posts],
        "slides": [asdict(s) for s in st.session_state.slides],
    }
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Não foi possível salvar em arquivo: {e}")

# ======================================================
# SESSION STATE
# ======================================================

def init_state():
    if "initialized" in st.session_state:
        return
    persisted = load_persistent_state()

    # users
    users_norm = []
    for u in persisted["users"]:
        u2 = dict(u)
        if "avatar_url" not in u2:
            u2["avatar_url"] = None
        users_norm.append(User(**u2))
    st.session_state.users = users_norm

    st.session_state.current_user_email = persisted["current_user_email"]
    st.session_state.cards = [Card(**c) for c in persisted["cards"]]
    st.session_state.research_summary = persisted["research_summary"]
    st.session_state.research_notes = persisted["research_notes"]

    # mind_root normalizado
    raw_root = persisted.get("mind_root", {})
    def normalize_mind_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        d2 = dict(d)
        if "note" not in d2:
            d2["note"] = ""
        if "tags" not in d2:
            d2["tags"] = []
        if "children" not in d2 or d2["children"] is None:
            d2["children"] = []
        else:
            d2["children"] = [normalize_mind_dict(c) for c in d2["children"]]
        return d2

    raw_root_norm = normalize_mind_dict(raw_root)
    st.session_state.mind_root = dict_to_mindnode(raw_root_norm)

    st.session_state.mind_selected_id = persisted["mind_selected_id"]
    st.session_state.chat_messages = [ChatMessage(**m) for m in persisted["chat_messages"]]
    st.session_state.chat_topic = persisted["chat_topic"]

    # posts
    raw_posts = persisted.get("posts", [])
    posts_norm = []
    for p in raw_posts:
        p2 = dict(p)
        if "likes" not in p2:
            p2["likes"] = 0
        if "liked_by_me" not in p2:
            p2["liked_by_me"] = False
        posts_norm.append(Post(**p2))
    st.session_state.posts = posts_norm

    # slides
    raw_slides = persisted.get("slides", [])
    slides_norm = [Slide(**s) for s in raw_slides]
    st.session_state.slides = slides_norm

    # view atual (navegação topo)
    st.session_state.current_view = st.session_state.get("current_view", "Feed social")

    st.session_state.initialized = True

init_state()

# ======================================================
# HELPERS
# ======================================================

def get_current_user() -> Optional[User]:
    if not st.session_state.current_user_email:
        return None
    for u in st.session_state.users:
        if u.email == st.session_state.current_user_email:
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

def extract_keywords(text: str, max_n: int = 6) -> List[str]:
    if not text:
        return []
    text_norm = (
        text.lower()
        .replace("á", "a").replace("à", "a").replace("ã", "a").replace("â", "a")
        .replace("é", "e").replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o").replace("ô", "o").replace("õ", "o")
        .replace("ú", "u")
    )
    words = re.findall(r"[a-z]{4,}", text_norm)
    stop = set([
        "como","para","onde","entre","sobre","dentro","dados",
        "estudo","pesquisa","analise","resultado","resultados",
        "qualitativa","qualitativo","uma","essa","esse","sera",
        "pelo","pela","com","tambem","que","isso","nao",
        "mais","menos","muito","pouco","sendo","assim",
    ])
    words = [w for w in words if w not in stop]
    if not words:
        return []
    freq = Counter(words)
    return [w for w, _ in freq.most_common(max_n)]

def mind_list_nodes(node: MindNode, prefix: str = "") -> List[MindNode]:
    nodes = [MindNode(id=node.id, label=prefix + node.label, note=node.note, tags=node.tags, children=[])]
    for ch in node.children:
        nodes += mind_list_nodes(ch, prefix + "  ")
    return nodes

def mind_find_node(node: MindNode, target_id: str) -> Optional[MindNode]:
    if node.id == target_id:
        return node
    for ch in node.children:
        found = mind_find_node(ch, target_id)
        if found:
            return found
    return None

def mind_remove_node(node: MindNode, target_id: str) -> bool:
    for i, ch in enumerate(node.children):
        if ch.id == target_id:
            node.children.pop(i)
            return True
        if mind_remove_node(ch, target_id):
            return True
    return False

def mind_print_tree(node: MindNode, indent: int = 0):
    pad = "&nbsp;" * indent
    sel_class = "mind-node-label"
    if node.id == st.session_state.mind_selected_id:
        sel_class += " mind-node-selected"
    tags_str = ""
    if node.tags:
        tags_str = " · " + ", ".join(f"#{t}" for t in node.tags)
    st.markdown(
        f'<div class="mind-node">{pad}<span class="{sel_class}">{node.label}</span>'
        f'<span style="font-size:0.7rem;color:#64748b;">{tags_str}</span></div>',
        unsafe_allow_html=True,
    )
    for ch in node.children:
        mind_print_tree(ch, indent + 4)

def cards_count_by_status(status: str) -> int:
    return len([c for c in st.session_state.cards if c.status == status])

def timeline_completion() -> int:
    total = len(st.session_state.cards)
    if total == 0:
        return 0
    done = len([c for c in st.session_state.cards if c.status == "redacao"])
    return round(done / total * 100)

# ======================================================
# AUTENTICAÇÃO (TELA INICIAL)
# ======================================================

def auth_screen():
    _, col, _ = st.columns([1, 2.3, 1])
    with col:
        st.markdown('<div class="glass-main">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="pqr-logo-line">
                <div class="pqr-logo-avatar">P</div>
                <div class="pqr-title-text">
                    <div class="pqr-title-main">PQR</div>
                    <div class="pqr-title-sub">Rede de pesquisa qualitativa com cara de social</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        tabs = st.tabs(["Entrar", "Criar conta"])

        with tabs[0]:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Senha", type="password", key="login_password")
            if st.button("Entrar", key="login_btn"):
                user = next((u for u in st.session_state.users if u.email == email), None)
                if not user or user.password != password:
                    st.error("Credenciais inválidas.")
                else:
                    st.session_state.current_user_email = email
                    st.experimental_rerun()

        with tabs[1]:
            name = st.text_input("Nome completo", key="cad_nome")
            email_c = st.text_input("Email institucional", key="cad_email")
            type_label = st.selectbox(
                "Tipo de bolsa",
                [
                    "Selecione…",
                    "IC – Iniciação Científica",
                    "Extensão",
                    "Doutorando",
                    "Voluntário",
                    "PRODIG",
                    "Mentoria",
                ],
                key="cad_tipo",
            )
            password_c = st.text_input(
                "Senha (mín. 6 caracteres)", type="password", key="cad_senha"
            )
            avatar_url = st.text_input(
                "URL da foto de perfil (opcional)",
                key="cad_avatar",
            )

            if st.button("Criar conta", key="cad_btn"):
                if type_label == "Selecione…":
                    st.warning("Escolha um tipo de bolsa.")
                elif len(password_c) < 6:
                    st.warning("Senha muito curta. Use ao menos 6 caracteres.")
                elif any(u.email == email_c for u in st.session_state.users):
                    st.error("Já existe usuário com este email.")
                else:
                    type_map = {
                        "IC – Iniciação Científica": "ic",
                        "Extensão": "extensao",
                        "Doutorando": "doutorando",
                        "Voluntário": "voluntario",
                        "PRODIG": "prodig",
                        "Mentoria": "mentoria",
                    }
                    t = type_map.get(type_label, "ic")
                    st.session_state.users.append(
                        User(
                            name=name,
                            email=email_c,
                            type=t,
                            password=password_c,
                            avatar_url=avatar_url.strip() or None,
                        )
                    )
                    st.session_state.current_user_email = email_c
                    save_persistent_state()
                    st.success("Conta criada.")
                    st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: FEED SOCIAL
# ======================================================

def view_social_feed():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)

    if "posts" not in st.session_state or st.session_state.posts is None:
        st.session_state.posts = []

    st.markdown("### Feed da rede")
    st.caption("Compartilhamentos rápidos de avanços, dúvidas e insights.")

    user = get_current_user()
    if not user:
        st.warning("Entre na sua conta para ver e publicar no feed.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with st.expander("Criar novo post", expanded=True):
        text = st.text_area("Escreva algo…", key="new_post_text")
        if st.button("Publicar no feed", key="btn_new_post"):
            if not text.strip():
                st.warning("Escreva algo antes de publicar.")
            else:
                st.session_state.posts.insert(
                    0,
                    Post(
                        id=str(uuid.uuid4()),
                        user_name=user.name.split(" ")[0] or user.name,
                        user_avatar_url=getattr(user, "avatar_url", None),
                        text=text.strip(),
                        created_at=datetime.datetime.now().isoformat(),
                    ),
                )
                save_persistent_state()
                st.success("Post publicado no feed.")

    st.write("---")
    if not st.session_state.posts:
        st.info("Ainda não há posts no feed. Publique o primeiro.")
    else:
        for p in st.session_state.posts:
            colA, colB = st.columns([0.13, 3])
            with colA:
                avatar_url = p.user_avatar_url
                if avatar_url:
                    st.markdown(
                        f"""
                        <div class="user-pill-avatar">
                            <img src="{avatar_url}">
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    ini = p.user_name[:1].upper() if p.user_name else "U"
                    st.markdown(
                        f"""
                        <div class="user-pill-avatar">
                            {ini}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            with colB:
                try:
                    dt = datetime.datetime.fromisoformat(p.created_at)
                    ts = dt.strftime("%d/%m %H:%M")
                except Exception:
                    ts = p.created_at
                st.markdown(
                    f"""
                    <div class="post-card">
                        <div class="post-header">
                            <strong>{p.user_name}</strong>
                        </div>
                        <div class="post-meta">{ts}</div>
                        <div class="post-body">{p.text}</div>
                    """,
                    unsafe_allow_html=True,
                )
                col_like, col_meta = st.columns([0.26, 3])
                with col_like:
                    liked_label = "💙 Curtido" if p.liked_by_me else "🤍 Curtir"
                    if st.button(liked_label, key=f"like_{p.id}"):
                        if p.liked_by_me:
                            p.liked_by_me = False
                            p.likes = max(0, p.likes - 1)
                        else:
                            p.liked_by_me = True
                            p.likes += 1
                        save_persistent_state()
                        st.experimental_rerun()
                with col_meta:
                    st.markdown(
                        f'<div class="post-actions">{p.likes} curtida(s)</div></div>',
                        unsafe_allow_html=True,
                    )

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: TIMELINE / ETAPAS
# ======================================================

def view_board():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Timeline da pesquisa")
    top_l, top_r = st.columns([3, 2.2])

    with top_l:
        st.caption("Sua jornada de pesquisa organizada em etapas.")
        st.markdown(
            '<span class="pqr-badge">PQR – PESQUISA QUALITATIVA DE RESULTADOS</span>',
            unsafe_allow_html=True,
        )
        query = st.text_input(
            "Busca global na pasta",
            key="global_search",
            placeholder="Procure termos em resumo, anotações e etapas…",
        )
        if st.button("Buscar", key="btn_search"):
            haystack = (
                st.session_state.research_summary
                + "\n"
                + st.session_state.research_notes
                + "\n"
                + "\n".join(f"{c.title} {c.description}" for c in st.session_state.cards)
            )
            found = query.strip() and query.lower() in haystack.lower()
            if not query.strip():
                st.warning("Digite um termo para buscar.")
            elif found:
                st.success(
                    f'A pasta contém referências a **"{query}"**. '
                    "Confira resumo, notas e timeline."
                )
            else:
                st.info(
                    f'Nenhuma ocorrência clara de **"{query}"** foi encontrada nas notas/etapas atuais.'
                )

    with top_r:
        with st.expander("Nova etapa / atualização"):
            title = st.text_input("Título da etapa", key="new_card_title")
            desc = st.text_area("Descrição", key="new_card_desc")
            deadline = st.date_input("Prazo", key="new_card_deadline")
            status = st.selectbox(
                "Fase",
                [
                    ("ideia", "Ideia / Delimitação"),
                    ("revisao", "Revisão de literatura"),
                    ("coleta", "Coleta de dados"),
                    ("analise", "Análise"),
                    ("redacao", "Redação / Resultados"),
                ],
                format_func=lambda x: x[1],
                key="new_card_status",
            )
            if st.button("Publicar etapa na timeline", key="btn_add_card"):
                if not title.strip():
                    st.warning("Dê um título para a etapa.")
                else:
                    st.session_state.cards.append(
                        Card(
                            id=str(uuid.uuid4()),
                            title=title.strip(),
                            description=desc.strip(),
                            status=status[0],
                            deadline=str(deadline),
                            created_at=datetime.datetime.now().isoformat(),
                        )
                    )
                    save_persistent_state()
                    st.success("Etapa adicionada à timeline.")

    st.write("---")

    statuses = [
        ("ideia", "Ideia / Delimitação"),
        ("revisao", "Revisão de literatura"),
        ("coleta", "Coleta de dados"),
        ("analise", "Análise"),
        ("redacao", "Redação / Resultados"),
    ]
    cols = st.columns(len(statuses))

    for (status_key, status_label), col in zip(statuses, cols):
        with col:
            st.markdown(f"**{status_label}**")
            st.caption(f"{cards_count_by_status(status_key)} etapa(s)")
            for c in st.session_state.cards:
                if c.status == status_key:
                    created_str = ""
                    if c.created_at:
                        try:
                            dt = datetime.datetime.fromisoformat(c.created_at)
                            created_str = dt.strftime("%d/%m %H:%M")
                        except Exception:
                            pass
                    st.markdown(
                        f"""
                        <div class="timeline-card">
                            <div class="timeline-card-header">
                                <span>{c.title}</span>
                                <span>{created_str}</span>
                            </div>
                            <div class="timeline-card-body">
                                {c.description or "<i>(sem descrição)</i>"}
                            </div>
                            <div class="timeline-card-footer">
                                <span>Prazo: {c.deadline}</span>
                                <span>{status_label}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.write("---")
    st.subheader("Progresso global da pesquisa")
    comp = timeline_completion()
    st.progress(comp / 100)
    st.caption(f"{comp}% concluído (estimativa via etapas em “Redação / Resultados”).")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: PASTA PRINCIPAL
# ======================================================

def view_research():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Pasta principal da pesquisa")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Resumo")
        st.session_state.research_summary = st.text_area(
            "Tema, objetivos, perguntas, contexto…",
            value=st.session_state.research_summary,
            height=240,
        )

    with col2:
        st.subheader("Anotações rápidas")
        st.session_state.research_notes = st.text_area(
            "Citações, ideias soltas, lembretes…",
            value=st.session_state.research_notes,
            height=240,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: MAPA MENTAL
# ======================================================

def view_mindmap():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Mapa mental (tópicos, notas e tags)")
    mind_print_tree(st.session_state.mind_root)
    st.write("---")

    nodes_list = mind_list_nodes(st.session_state.mind_root)
    ids = [n.id for n in nodes_list]
    label_map = {n.id: n.label for n in nodes_list}

    selected = st.selectbox(
        "Nó selecionado",
        ids,
        index=ids.index(st.session_state.mind_selected_id)
        if st.session_state.mind_selected_id in ids
        else 0,
        format_func=lambda x: label_map.get(x, x),
    )
    st.session_state.mind_selected_id = selected
    node = mind_find_node(st.session_state.mind_root, selected)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Editar nó")
        if node:
            new_label = st.text_input("Título do tópico", value=node.label)
            new_note = st.text_area("Nota / descrição do tópico", value=node.note)
            tags_str = st.text_input(
                "Tags (separadas por vírgula)",
                value=", ".join(node.tags),
            )
            if st.button("Salvar alterações no nó"):
                node.label = new_label.strip() or node.label
                node.note = new_note.strip()
                node.tags = [t.strip() for t in tags_str.split(",") if t.strip()]
                save_persistent_state()
                st.success("Nó atualizado.")

    with colB:
        st.markdown("#### Criar / remover nós")
        new_child_label = st.text_input("Adicionar sub‑tópico", key="mind_new_label")
        if st.button("Adicionar sub‑tópico"):
            if new_child_label.strip() and node:
                node.children.append(
                    MindNode(
                        id=str(uuid.uuid4()),
                        label=new_child_label.strip(),
                        note="",
                        tags=[],
                        children=[],
                    )
                )
                save_persistent_state()
                st.success("Sub‑tópico adicionado.")
        st.write("")
        if st.button("Remover nó selecionado"):
            if selected == "root":
                st.warning("Não é possível remover o nó raiz.")
            else:
                removed = mind_remove_node(st.session_state.mind_root, selected)
                if removed:
                    st.session_state.mind_selected_id = "root"
                    save_persistent_state()
                    st.info("Nó removido.")
                else:
                    st.error("Não foi possível remover o nó.")

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: SLIDES / CANVAS
# ======================================================

def view_slides():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Canvas / Slides da pesquisa")

    colA, colB = st.columns([1.5, 2])

    with colA:
        st.subheader("Criar / editar slide")
        slide_titles = [s.title for s in st.session_state.slides]
        slide_options = ["(Novo slide)"] + slide_titles
        choice = st.selectbox("Selecionar slide", slide_options)

        if choice == "(Novo slide)":
            s_title = st.text_input("Título do novo slide")
            s_content = st.text_area("Conteúdo (bullets, texto livre)")
            if st.button("Salvar slide novo"):
                if not s_title.strip():
                    st.warning("Dê um título para o slide.")
                else:
                    st.session_state.slides.append(
                        Slide(
                            id=str(uuid.uuid4()),
                            title=s_title.strip(),
                            content=s_content.strip(),
                            created_at=datetime.datetime.now().isoformat(),
                        )
                    )
                    save_persistent_state()
                    st.success("Slide criado.")
        else:
            slide = next(s for s in st.session_state.slides if s.title == choice)
            s_title = st.text_input("Título do slide", value=slide.title)
            s_content = st.text_area("Conteúdo do slide", value=slide.content, height=200)
            col_ed1, col_ed2 = st.columns(2)
            with col_ed1:
                if st.button("Atualizar slide"):
                    slide.title = s_title.strip() or slide.title
                    slide.content = s_content.strip()
                    save_persistent_state()
                    st.success("Slide atualizado.")
            with col_ed2:
                if st.button("Excluir slide"):
                    st.session_state.slides = [s for s in st.session_state.slides if s.id != slide.id]
                    save_persistent_state()
                    st.info("Slide excluído.")
                    st.experimental_rerun()

    with colB:
        st.subheader("Visualização rápida")
        if not st.session_state.slides:
            st.info("Nenhum slide criado ainda.")
        else:
            for s in st.session_state.slides:
                try:
                    dt = datetime.datetime.fromisoformat(s.created_at)
                    ts = dt.strftime("%d/%m %H:%M")
                except Exception:
                    ts = s.created_at
                st.markdown(
                    f"**{s.title}**  \n"
                    f"<span style='font-size:0.75rem;color:#64748b;'>Criado em {ts}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"> {s.content}")
                st.write("---")

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: ANÁLISE INTELIGENTE
# ======================================================

def view_analysis():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Análise inteligente (protótipo local)")

    if st.button("Rodar análise qualitativa agora"):
        summary = st.session_state.research_summary or ""
        notes = st.session_state.research_notes or ""
        total_cards = len(st.session_state.cards) or 1
        done_cards = len([c for c in st.session_state.cards if c.status == "redacao"])
        completion = round(done_cards / total_cards * 100)

        st.subheader("Síntese do que você já escreveu")
        if len(summary.strip()) < 60:
            st.write(
                "- Resumo ainda enxuto. Tente explicitar: contexto, problema, objetivos, perguntas."
            )
        else:
            st.write(
                "- Resumo com bom corpo. Veja se está explícita a abordagem qualitativa (quem, onde, como, por quê)."
            )

        st.subheader("Andamento do projeto")
        st.write(f"- Etapas concluídas (redação/resultados): {done_cards}/{total_cards}.")
        if completion < 30:
            st.write(
                "- Fase inicial: foque em problema, referencial e caminhos metodológicos."
            )
        elif completion < 70:
            st.write(
                "- Fase intermediária: revise coerência entre coleta (entrevista, grupo focal etc.) e perguntas."
            )
        else:
            st.write(
                "- Fase avançada: conecte dados, categorias e discussões com a literatura."
            )

        st.subheader("Pistas qualitativas")
        concat = (summary + " " + notes).lower()
        if "entrevista" in concat:
            st.write("- Entrevistas: explore análise temática / de conteúdo / narrativa.")
        if "grupo focal" in concat or "focal" in concat:
            st.write("- Grupo focal: considere o papel da interação na produção de sentidos.")
        if "questionário" in concat or "questionario" in concat:
            st.write("- Questionários abertos: trate respostas como narrativas a categorizar.")
        if len(notes) > 200:
            st.write("- Muitas anotações: hora de estruturar códigos/categorias preliminares.")
        if not summary.strip() and not notes.strip():
            st.info(
                "Ainda não há conteúdo suficiente para análise. Escreva pelo menos um parágrafo de resumo e algumas notas."
            )

        st.write("---")
        st.subheader("Progresso estimado")
        st.progress(completion / 100)
        st.caption(f"{completion}% concluído (estimativa via timeline).")
    else:
        st.info(
            "Clique em **Rodar análise qualitativa agora** para gerar um diagnóstico textual com base no que você já registrou."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: CHAT
# ======================================================

def view_chat():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Chat entre bolsistas (canais temáticos)")
    topics = {
        "metodologia": "Metodologia qualitativa",
        "referencias": "Referências e artigos",
        "duvidas-eticas": "Dúvidas éticas",
        "off-topic": "Off‑topic / descompressão",
    }

    col_topics, col_chat = st.columns([1.1, 2.3])

    with col_topics:
        st.markdown("**Canais**")
        for key, label in topics.items():
            if st.button(label, key=f"topic_{key}"):
                st.session_state.chat_topic = key
        st.caption(
            f"Canal ativo: **{topics.get(st.session_state.chat_topic, 'Metodologia qualitativa')}**"
        )

    with col_chat:
        user = get_current_user()
        if not user:
            st.warning("Entre na sua conta para participar do chat.")
        else:
            st.subheader("Timeline de mensagens")
            topic = st.session_state.chat_topic
            msgs = [m for m in st.session_state.chat_messages if m.topic == topic]
            if not msgs:
                st.caption("Ainda não há mensagens neste canal. Comece a conversa.")
            for m in msgs:
                try:
                    dt = datetime.datetime.fromisoformat(m.time)
                    time_str = dt.strftime("%d/%m %H:%M")
                except Exception:
                    time_str = m.time
                st.markdown(
                    f'<div class="chat-bubble">'
                    f'<div class="chat-meta">{m.user_name} • {time_str}</div>'
                    f'{m.text}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.write("---")
            with st.form("chat_form"):
                text = st.text_input("Escreva uma mensagem…", key="chat_text")
                send = st.form_submit_button("Publicar")
                if send and text.strip():
                    st.session_state.chat_messages.append(
                        ChatMessage(
                            id=str(uuid.uuid4()),
                            user_name=user.name.split(" ")[0] or user.name,
                            topic=topic,
                            text=text.strip(),
                            time=datetime.datetime.now().isoformat(),
                        )
                    )
                    save_persistent_state()
                    st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: REDE DE INTERESSES
# ======================================================

def view_network():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Cadeia de ligação conceitual")
    interest = st.text_input(
        "Interesse principal (tema eixo da rede)",
        key="net_interest",
        placeholder="ex.: inclusão digital, saúde mental, aprendizagem ativa",
    )
    if st.button("Gerar rede textual"):
        base_interest = interest.strip() or "Tema central"
        text = st.session_state.research_summary + " " + st.session_state.research_notes
        keywords = extract_keywords(text, max_n=8)
        st.subheader("Nós principais da rede")
        st.write(f"- **{base_interest}** (nó central)")
        if not keywords:
            st.warning(
                "Não identifiquei palavras‑chave suficientes. Escreva mais no resumo e nas anotações."
            )
        else:
            for kw in keywords:
                st.write(f"- {kw}")
            st.write("---")
            st.subheader("Ligações sugeridas")
            for kw in keywords:
                st.markdown(
                    f"- **{base_interest} ↔ {kw}** – analisar como esse conceito aparece nos dados, "
                    "como se relaciona a outras categorias e que tensões/contradições emergem."
                )
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: CONFIGURAÇÕES
# ======================================================

def view_settings():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("### Configurações, perfil & sessão")
    user = get_current_user()
    if user:
        avatar_url = getattr(user, "avatar_url", None)
        avatar_html = (
            f'<img src="{avatar_url}">'
            if avatar_url
            else user.name[:1].upper()
        )
        st.markdown(
            f"""
            <div class="user-pill-header" style="margin-bottom:8px;">
                <div class="user-pill-avatar">{avatar_html}</div>
                <div>
                    <strong>{user.name}</strong><br/>
                    <span style="font-size:0.78rem;color:#64748b;">
                        {user.email} – {map_type_label(user.type)}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Atualizar foto de perfil"):
            current_avatar = getattr(user, "avatar_url", None)
            new_url = st.text_input("URL da nova foto de avatar", value=current_avatar or "")
            if st.button("Salvar avatar"):
                user.avatar_url = new_url.strip() or None
                save_persistent_state()
                st.success("Avatar atualizado.")

    st.write("---")
    st.subheader("Sessão")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Salvar tudo agora e continuar", key="btn_save_only"):
            save_persistent_state()
    with col_b:
        if st.button("Salvar e sair", key="btn_save_logout"):
            save_persistent_state()
            st.session_state.current_user_email = None
            st.success("Dados salvos. Sessão encerrada.")
            st.experimental_rerun()

    st.write("---")
    st.caption(
        "Os dados são salvos em `pqr_state.json` (se o ambiente permitir gravação em disco)."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# HEADER E NAVEGAÇÃO (SEM SIDEBAR)
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
        avatar_url = getattr(user, "avatar_url", None)
        avatar_html = (
            f'<img src="{avatar_url}">'
            if avatar_url
            else user.name[:1].upper()
        )
        st.markdown(
            f"""
            <div class="pqr-header-row" style="justify-content:flex-end;">
                <div class="user-pill-header">
                    <div class="user-pill-avatar">{avatar_html}</div>
                    <div style="font-size:0.78rem;">
                        {user.name.split(" ")[0]}<br/>
                        <span style="color:#64748b;">{map_type_label(user.type)}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # nav horizontal
    st.markdown('<div class="pqr-nav">', unsafe_allow_html=True)
    nav_cols = st.columns(len(VIEWS))
    for i, (view_name, col) in enumerate(zip(VIEWS, nav_cols)):
        with col:
            active = (st.session_state.current_view == view_name)
            cls = "pqr-nav-item-active" if active else "pqr-nav-item"
            if st.button(view_name, key=f"nav_{i}"):
                st.session_state.current_view = view_name
                st.experimental_rerun()
            st.markdown(
                f'<div class="{cls}" style="display:none;">{view_name}</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

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

    if view == "Feed social":
        view_social_feed()
    elif view == "Timeline / Etapas":
        view_board()
    elif view == "Pasta da pesquisa":
        view_research()
    elif view == "Mapa mental":
        view_mindmap()
    elif view == "Canvas / Slides":
        view_slides()
    elif view == "Análise inteligente":
        view_analysis()
    elif view == "Chat":
        view_chat()
    elif view == "Cadeia de ligação":
        view_network()
    elif view == "Configurações":
        view_settings()

if __name__ == "__main__":
    main()
