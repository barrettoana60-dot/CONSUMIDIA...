import streamlit as st
import datetime
import json
import osa
import from dataclasses import dataclass, asdict, field
import from typing import List, Optional, Dict, Any
from collections import Counter
import re
import uuid

# ======================================================
# CONFIG BÁSICA
# ======================================================
st.set_page_config(
    page_title="PQR – Pesquisa Qualitativa de Resultados",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATE_FILE = "pqr_state.json"

# ======================================================
# CSS – TUDO EM LIQUID GLASS + AZUL ESCURO, ESTILO REDE SOCIAL
# ======================================================
LIQUID_CSS = """
<style>
:root {
    --pqr-accent: #55d6ff;
    --pqr-accent-soft: rgba(85, 214, 255, 0.18);
    --pqr-bg-dark: #050814;
    --pqr-bg-glass: rgba(5, 10, 25, 0.85);
    --pqr-border-soft: rgba(255,255,255,0.16);
    --pqr-text-main: #f7f9ff;
    --pqr-text-soft: #a6aec9;
}

/* Fundo geral com imagem + overlay azul escuro */
.stApp {
    background: var(--pqr-bg-dark);
    color: var(--pqr-text-main);
    font-family: system-ui,-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",sans-serif;
}
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: url("https://images.pexels.com/photos/237272/pexels-photo-237272.jpeg?auto=compress&cs=tinysrgb&w=1600");
    background-size: cover;
    background-position: center;
    filter: saturate(1.1) contrast(1.05) brightness(0.7);
    z-index: -2;
}
.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(circle at top left, rgba(15,40,85,0.9), transparent 50%),
        radial-gradient(circle at bottom right, rgba(2,4,15,0.96), #020309);
    z-index: -1;
}
.block-container {
    padding-top: 0.5rem;
    padding-bottom: 0.8rem;
}

/* Sidebar em glass azul */
[data-testid="stSidebar"] {
    background: linear-gradient(
        155deg,
        rgba(3, 9, 30, 0.96),
        rgba(3, 11, 40, 0.96)
    );
    border-right: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
}

/* Cartões glass principais */
.glass-main {
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    background: radial-gradient(circle at top, rgba(255,255,255,0.06), transparent 55%),
                rgba(8, 14, 38, 0.96);
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 28px 80px rgba(0,0,0,0.75);
    padding: 18px 22px;
}

/* Seções internas */
.glass-section {
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    background: linear-gradient(145deg, rgba(255,255,255,0.03), rgba(1,4,18,0.96));
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.12);
    padding: 14px 16px;
}

/* Título / badge tipo rede social */
.pqr-logo-line {
    display: flex;
    align-items: center;
    gap: 10px;
}
.pqr-logo-avatar {
    width: 36px;
    height: 36px;
    border-radius: 999px;
    background: radial-gradient(circle at 30% 20%, #55d6ff, #1960ff);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #02030a;
    font-weight: 700;
    font-size: 0.9rem;
    border: 2px solid rgba(255,255,255,0.6);
}
.pqr-title-text {
    display: flex;
    flex-direction: column;
}
.pqr-title-main {
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}
.pqr-title-sub {
    font-size: 0.82rem;
    color: var(--pqr-text-soft);
}

/* “Ficha” de usuário tipo perfil */
.user-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(3,10,30,0.92);
    border: 1px solid rgba(255,255,255,0.18);
    font-size: 0.82rem;
}
.user-pill-avatar {
    width: 24px;
    height: 24px;
    border-radius: 999px;
    background: radial-gradient(circle at 30% 20%, #55d6ff, #1960ff);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #02030a;
    font-weight: 600;
}

/* Botões pill glass */
.pqr-btn {
    border-radius: 999px !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    background: radial-gradient(circle at top left, rgba(255,255,255,0.15), transparent 45%),
                rgba(3,7,24,0.92) !important;
    color: var(--pqr-text-main) !important;
    font-size: 0.82rem !important;
    padding: 6px 14px !important;
}
.pqr-btn-primary {
    border: none !important;
    background: linear-gradient(135deg, #55d6ff, #1f7afe) !important;
    color: #02030a !important;
    box-shadow: 0 10px 24px rgba(42,168,255,0.55);
}
.pqr-btn-danger {
    border: none !important;
    background: linear-gradient(135deg, #ff6b81, #f03e5a) !important;
    color: #02030a !important;
    box-shadow: 0 10px 26px rgba(255,107,129,0.7);
}

/* Ajeitar botões padrão do Streamlit */
.stButton > button {
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.16);
    background: radial-gradient(circle at top left, rgba(255,255,255,0.12), transparent 45%),
                rgba(3,7,24,0.92);
    color: var(--pqr-text-main);
    font-size: 0.84rem;
}

/* “Timeline card” estilo social feed */
.timeline-card {
    border-radius: 16px;
    padding: 8px 10px;
    margin-bottom: 6px;
    background: rgba(5,10,30,0.92);
    border: 1px solid rgba(255,255,255,0.16);
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

/* Chat estilo feed */
.chat-bubble {
    padding: 7px 9px;
    border-radius: 14px;
    margin-bottom: 6px;
    font-size: 0.84rem;
    background: rgba(3, 8, 26, 0.98);
    border: 1px solid rgba(255,255,255,0.14);
}
.chat-meta {
    font-size: 0.70rem;
    color: var(--pqr-text-soft);
    margin-bottom: 2px;
}

/* Mindmap como lista visual */
.mind-node {
    font-size: 0.84rem;
    margin: 2px 0;
}
.mind-node-label {
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
}
.mind-node-selected {
    background: var(--pqr-accent-soft);
    color: var(--pqr-accent);
    border: 1px solid var(--pqr-accent);
}

/* Tabs para navegação de “pastas” tipo rede social */
.pqr-tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
}
.pqr-tab {
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.14);
    font-size: 0.8rem;
    cursor: pointer;
    background: rgba(3,7,24,0.8);
    color: var(--pqr-text-soft);
}
.pqr-tab-active {
    background: linear-gradient(135deg, rgba(85,214,255,0.18), rgba(38,105,255,0.4));
    border-color: rgba(85,214,255,0.75);
    color: var(--pqr-accent);
}

/* Pequenos ajustes de inputs */
textarea, input, select {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    background: rgba(2,5,20,0.9) !important;
    color: var(--pqr-text-main) !important;
    font-size: 0.85rem !important;
}

/* Badge simples */
.pqr-badge {
    display:inline-block;
    padding:3px 10px;
    border-radius:999px;
    font-size:0.72rem;
    letter-spacing:0.08em;
    text-transform:uppercase;
    background:rgba(85,214,255,0.12);
    border:1px solid rgba(85,214,255,0.7);
    color:var(--pqr-accent);
}
</style>
"""

st.markdown(LIQUID_CSS, unsafe_allow_html=True)

# ======================================================
# MODELOS DE DADOS
# ======================================================
@dataclass
class User:
    name: str
    email: str
    type: str
    password: str


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
    children: List["MindNode"] = field(default_factory=list)

# ======================================================
# PERSISTÊNCIA EM ARQUIVO (SALVAR / CARREGAR)
# ======================================================
def default_state_dict() -> Dict[str, Any]:
    return {
        "users": [],
        "current_user_email": None,
        "cards": [],
        "research_summary": "",
        "research_notes": "",
        "mind_root": {"id": "root", "label": "Tema central", "children": []},
        "mind_selected_id": "root",
        "chat_messages": [],
        "chat_topic": "metodologia",
    }


def load_persistent_state():
    if not os.path.exists(STATE_FILE):
        return default_state_dict()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {**default_state_dict(), **data}
    except Exception:
        return default_state_dict()


def mindnode_to_dict(n: MindNode) -> Dict[str, Any]:
    return {"id": n.id, "label": n.label, "children": [mindnode_to_dict(c) for c in n.children]}


def dict_to_mindnode(d: Dict[str, Any]) -> MindNode:
    return MindNode(
        id=d.get("id", "no-id"),
        label=d.get("label", ""),
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
    }
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        st.success("Dados salvos com sucesso.")
    except Exception as e:
        st.warning(f"Não foi possível salvar em arquivo: {e}")

# ======================================================
# SESSION STATE
# ======================================================
def init_state():
    if "initialized" in st.session_state:
        return
    persisted = load_persistent_state()

    st.session_state.users = [User(**u) for u in persisted["users"]]
    st.session_state.current_user_email = persisted["current_user_email"]
    st.session_state.cards = [Card(**c) for c in persisted["cards"]]
    st.session_state.research_summary = persisted["research_summary"]
    st.session_state.research_notes = persisted["research_notes"]
    st.session_state.mind_root = dict_to_mindnode(persisted["mind_root"])
    st.session_state.mind_selected_id = persisted["mind_selected_id"]
    st.session_state.chat_messages = [ChatMessage(**m) for m in persisted["chat_messages"]]
    st.session_state.chat_topic = persisted["chat_topic"]

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
    stop = set(
        [
            "como","para","onde","entre","sobre","dentro","dados",
            "estudo","pesquisa","analise","resultado","resultados",
            "qualitativa","qualitativo","uma","essa","esse","sera",
            "pelo","pela","com","tambem","que","isso","nao",
            "mais","menos","muito","pouco","sendo","assim",
        ]
    )
    words = [w for w in words if w not in stop]
    if not words:
        return []
    freq = Counter(words)
    return [w for w, _ in freq.most_common(max_n)]


def mind_list_nodes(node: MindNode, prefix: str = "") -> List[MindNode]:
    nodes = [MindNode(id=node.id, label=prefix + node.label, children=[])]
    for ch in node.children:
        nodes += mind_list_nodes(ch, prefix + " ")
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
    st.markdown(
        f'<div class="mind-node">{pad}<span class="{sel_class}">{node.label}</span></div>',
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
# AUTENTICAÇÃO
# ======================================================
def auth_screen():
    _, col, _ = st.columns([1, 2.3, 1])
    with col:
        st.markdown('<div class="glass-main">', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="pqr-logo-line">
                <div class="pqr-logo-avatar">PQR</div>
                <div class="pqr-title-text">
                    <div class="pqr-title-main">PQR</div>
                    <div class="pqr-title-sub">Pesquisa Qualitativa de Resultados</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        tabs = st.tabs(["Entrar", "Criar conta"])

        # LOGIN
        with tabs[0]:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Senha", type="password", key="login_password")
            if st.button("Entrar", key="login_btn", help="Acessar sua pasta PQR"):
                user = next((u for u in st.session_state.users if u.email == email), None)
                if not user or user.password != password:
                    st.error("Credenciais inválidas.")
                else:
                    st.session_state.current_user_email = email
                    st.experimental_rerun()

        # CADASTRO
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
            password_c = st.text_input("Senha (mín. 6 caracteres)", type="password", key="cad_senha")

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
                        User(name=name, email=email_c, type=t, password=password_c)
                    )
                    st.session_state.current_user_email = email_c
                    save_persistent_state()
                    st.success("Conta criada. Bem‑vindo(a) ao PQR!")
                    st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: BOARD – COMO FEED DE PROGRESSO
# ======================================================
def view_board():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    top_l, top_c, top_r = st.columns([2.6, 3.5, 2.2])

    with top_l:
        st.markdown("#### Timeline de pesquisa")
        st.caption("Sua jornada de pesquisa organizada em etapas, como um feed de progresso.")
        st.markdown(
            '<span class="pqr-badge">PQR – PESQUISA QUALITATIVA DE RESULTADOS</span>',
            unsafe_allow_html=True,
        )

    with top_c:
        st.write("")
        query = st.text_input(
            "Busca global na sua pasta (protótipo de IA interna)",
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
                    "Agora revise as seções para ver em que contexto isso aparece."
                )
            else:
                st.info(
                    f'Nenhuma ocorrência clara de **"{query}"** foi encontrada nas notas/etapas atuais.'
                )

    with top_r:
        st.write("")
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
                    novo_status = st.selectbox(
                        "Mover para",
                        statuses,
                        index=[s[0] for s in statuses].index(status_key),
                        format_func=lambda x: x[1],
                        key=f"move_{c.id}",
                    )
                    if novo_status[0] != c.status:
                        c.status = novo_status[0]
                        save_persistent_state()
                        st.info("Etapa movida na timeline.")

    st.write("---")
    st.subheader("Progresso global da pesquisa")
    comp = timeline_completion()
    st.progress(comp / 100)
    st.caption(f"{comp}% concluído (estimativa via etapas em “Redação / Resultados”).")

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: PESQUISA / PASTA PRINCIPAL
# ======================================================
def view_research():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("#### Pasta principal da sua pesquisa")

    tabs_html = """
    <div class="pqr-tabs">
        <div class="pqr-tab pqr-tab-active">Resumo & notas</div>
        <div class="pqr-tab">Artigos & buscas</div>
    </div>
    """
    st.markdown(tabs_html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Resumo da pesquisa")
        st.session_state.research_summary = st.text_area(
            "Tema, objetivos, perguntas, contexto…",
            value=st.session_state.research_summary,
            height=240,
        )

    with col2:
        st.subheader("Anotações rápidas (tipo mural privado)")
        st.session_state.research_notes = st.text_area(
            "Citações, ideias soltas, lembretes para você mesmo(a)…",
            value=st.session_state.research_notes,
            height=240,
        )

        st.write("")
        st.subheader("Atalhos para Google Acadêmico")
        keywords = st.text_input(
            "Palavras‑chave principais",
            key="ga_kw",
            placeholder="ex.: inclusão digital, aprendizagem ativa",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Artigos gerais"):
                if keywords.strip():
                    url = "https://scholar.google.com/scholar?q=" + keywords.replace(" ", "+")
                    st.markdown(f"[Abrir Google Acadêmico]({url})")
        with c2:
            if st.button("Últimos 5 anos"):
                if keywords.strip():
                    year = datetime.date.today().year - 5
                    url = (
                        "https://scholar.google.com/scholar?q="
                        + keywords.replace(" ", "+")
                        + f"&as_ylo={year}"
                    )
                    st.markdown(f"[Abrir (últimos 5 anos)]({url})")

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: MAPA MENTAL
# ======================================================
def view_mindmap():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("#### Mapa mental da sua pesquisa")

    st.markdown("Visualização hierárquica (protótipo textual em liquid glass):")
    mind_print_tree(st.session_state.mind_root)

    st.write("---")
    st.markdown("##### Editar nós do mapa")

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

    new_label = st.text_input("Novo tópico / sub‑tópico", key="mind_new_label")
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Adicionar sub‑tópico ao nó selecionado"):
            if new_label.strip():
                parent = mind_find_node(st.session_state.mind_root, selected)
                if parent:
                    parent.children.append(
                        MindNode(
                            id=str(uuid.uuid4()),
                            label=new_label.strip(),
                            children=[],
                        )
                    )
                    save_persistent_state()
                    st.success("Tópico adicionado ao mapa.")
                else:
                    st.error("Nó pai não encontrado.")

    with col_b:
        if st.button("Remover nó selecionado"):
            if selected == "root":
                st.warning("Não é possível remover o nó raiz.")
            else:
                removed = mind_remove_node(st.session_state.mind_root, selected)
                if removed:
                    st.session_state.mind_selected_id = "root"
                    save_persistent_state()
                    st.info("Nó removido do mapa mental.")
                else:
                    st.error("Não foi possível remover o nó.")

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: ANÁLISE INTELIGENTE (MOCK)
# ======================================================
def view_analysis():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("#### Análise inteligente (protótipo local)")

    if st.button("Rodar análise qualitativa agora"):
        summary = st.session_state.research_summary or ""
        notes = st.session_state.research_notes or ""
        total_cards = len(st.session_state.cards) or 1
        done_cards = len([c for c in st.session_state.cards if c.status == "redacao"])
        completion = round(done_cards / total_cards * 100)

        st.subheader("Síntese do que você já escreveu")
        if len(summary.strip()) < 60:
            st.write(
                "- O resumo ainda está enxuto. Tente explicitar: (1) contexto, (2) problema, (3) "
                "objetivos, (4) perguntas de pesquisa."
            )
        else:
            st.write(
                "- O resumo já tem um corpo interessante. Revise se está claro o recorte qualitativo "
                "(quem, onde, como, por quê)."
            )

        st.subheader("Andamento do projeto")
        st.write(f"- Etapas concluídas (redação/resultados): {done_cards}/{total_cards}.")
        if completion < 30:
            st.write(
                "- Fase inicial: foque em consolidar problema, referencial e possíveis caminhos "
                "metodológicos."
            )
        elif completion < 70:
            st.write(
                "- Fase intermediária: revise se a forma de coleta (entrevista, grupo focal, "
                "observação etc.) está coerente com o que você quer responder."
            )
        else:
            st.write(
                "- Fase avançada: agora é hora de conectar dados, categorias e discussões com a literatura."
            )

        st.subheader("Pistas qualitativas encontradas")
        concat = (summary + " " + notes).lower()
        if "entrevista" in concat:
            st.write(
                "- Há entrevistas: explore estratégias como análise temática, análise de conteúdo "
                "ou análise narrativa."
            )
        if "grupo focal" in concat or "focal" in concat:
            st.write(
                "- Menciona grupo focal: pense em como a interação entre participantes impacta os "
                "sentidos produzidos."
            )
        if "questionário" in concat or "questionario" in concat:
            st.write(
                "- Questionários aparecem no texto: se houver questões abertas, trate-as como "
                "narrativas/dizeres a serem categorizados."
            )
        if len(notes) > 200:
            st.write(
                "- Muitas anotações: excelente. Talvez seja momento de criar um quadro de códigos/"
                "categorias preliminares."
            )
        if not summary.strip() and not notes.strip():
            st.info(
                "Ainda não há conteúdo suficiente para análise. Escreva pelo menos um parágrafo de "
                "resumo e algumas notas."
            )

        st.write("---")
        st.subheader("Progresso estimado")
        st.progress(completion / 100)
        st.caption(f"{completion}% concluído (estimativa via timeline).")
    else:
        st.info(
            "Clique em **Rodar análise qualitativa agora** para gerar um diagnóstico textual com "
            "base no que você já registrou."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# VIEW: CHAT ESTILO REDE SOCIAL
# ======================================================
def view_chat():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("#### Chat entre bolsistas (estilo feed)")

    topics = {
        "metodologia": "Metodologia qualitativa",
        "referencias": "Referências e artigos",
        "duvidas-eticas": "Dúvidas éticas",
        "off-topic": "Off‑topic / descompressão",
    }

    col_topics, col_chat = st.columns([1.1, 2.3])

    with col_topics:
        st.markdown("**Canais temáticos**")
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
                st.caption("Ainda não há mensagens neste canal. Que tal iniciar a conversa?")
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
# VIEW: CADEIA DE LIGAÇÃO / REDE DE INTERESSES
# ======================================================
def view_network():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("#### Cadeia de ligação de pesquisas (rede conceitual)")

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
# VIEW: CONFIGURAÇÕES, SALVAR & SAIR
# ======================================================
def view_settings():
    st.markdown('<div class="glass-main">', unsafe_allow_html=True)
    st.markdown("#### Configurações, salvar & sair")

    user = get_current_user()
    if user:
        st.markdown(
            f"""
            <div class="user-pill">
                <div class="user-pill-avatar">{user.name[:1].upper()}</div>
                <div>
                    <strong>{user.name}</strong><br/>
                    <span style="font-size:0.78rem;color:#a6aec9;">
                        {user.email} – {map_type_label(user.type)}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        "O PQR salva os dados em um arquivo `pqr_state.json` (quando o ambiente permite gravação em disco)."
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# MAIN
# ======================================================
def main():
    user = get_current_user()

    if not user:
        auth_screen()
        return

    with st.sidebar:
        st.markdown(
            """
            <div class="pqr-logo-line">
                <div class="pqr-logo-avatar">P</div>
                <div class="pqr-title-text">
                    <div class="pqr-title-main">PQR</div>
                    <div class="pqr-title-sub">sua rede de pesquisa qualitativa</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"""
            <div class="user-pill">
                <div class="user-pill-avatar">{user.name[:1].upper()}</div>
                <div style="font-size:0.78rem;">
                    {user.name.split(" ")[0]}<br/>
                    <span style="color:#a6aec9;">{map_type_label(user.type)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        view = st.radio(
            "Navegação",
            [
                "Timeline / Feed",
                "Pasta da pesquisa",
                "Mapa mental",
                "Análise inteligente",
                "Chat",
                "Cadeia de ligação",
                "Configurações",
            ],
        )

    if view == "Timeline / Feed":
        view_board()
    elif view == "Pasta da pesquisa":
        view_research()
    elif view == "Mapa mental":
        view_mindmap()
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
