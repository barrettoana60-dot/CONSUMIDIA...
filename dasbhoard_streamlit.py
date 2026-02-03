import streamlit as st
import datetime
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import Counter
import re
import uuid

# ======================================================
# CONFIGURAÇÃO E CSS
# ======================================================
st.set_page_config(
    page_title="PQR – Pesquisa Qualitativa de Resultados",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_CSS = """
<style>
.stApp {
    background: #05070a;
    color: #f7f7fb;
    font-family: system-ui,-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",sans-serif;
}
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: url("https://images.pexels.com/photos/2166553/pexels-photo-2166553.jpeg?auto=compress&cs=tinysrgb&w=1600");
    background-size: cover;
    background-position: center;
    filter: saturate(1.05) contrast(1.05);
    z-index: -2;
}
.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(circle at top left, rgba(0,0,0,0.55), transparent 50%),
        radial-gradient(circle at bottom right, rgba(0,0,0,0.85), #020308);
    z-index: -1;
}
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 0.8rem;
}

/* Glass */
.glass-card {
    backdrop-filter: blur(26px);
    -webkit-backdrop-filter: blur(26px);
    background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(0,0,0,0.7));
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.7);
    padding: 16px 18px;
}
.glass-section {
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    background: linear-gradient(145deg, rgba(255,255,255,0.03), rgba(3,5,12,0.94));
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.10);
    padding: 14px 16px;
}

/* Título / badge */
.pqr-title {
    font-size: 1.8rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 700;
}
.pqr-subtitle {
    font-size: 0.9rem;
    color: #b4b7c2;
}
.pqr-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.70rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6cf5ff;
    border: 1px solid rgba(108,245,255,0.7);
    background: rgba(108,245,255,0.08);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        160deg,
        rgba(0,0,0,0.9),
        rgba(18,20,29,0.96)
    );
    border-right: 1px solid rgba(255,255,255,0.12);
}
[data-testid="stSidebar"] .sidebar-content {
    padding-top: 12px;
}

/* Chat bubbles */
.chat-bubble {
    padding: 6px 8px;
    border-radius: 10px;
    margin-bottom: 6px;
    font-size: 0.84rem;
    background: rgba(3, 5, 10, 0.9);
    border: 1px solid rgba(255,255,255,0.12);
}
.chat-meta {
    font-size: 0.70rem;
    color: #b4b7c2;
}
.chat-text {
    margin-top: 2px;
}

/* Mindmap “árvore” */
.mind-node {
    font-size: 0.86rem;
}
.mind-node-label {
    padding: 2px 6px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
}
.mind-node-selected {
    background: rgba(108,245,255,0.18);
    color: #6cf5ff;
}

/* Pequenas melhorias */
textarea, input, select {
    border-radius: 10px !important;
}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)

# ======================================================
# DATA CLASSES
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
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)


@dataclass
class ChatMessage:
    id: str
    user_name: str
    topic: str
    text: str
    time: datetime.datetime


@dataclass
class MindNode:
    id: str
    label: str
    children: List["MindNode"] = field(default_factory=list)


# ======================================================
# SESSION STATE
# ======================================================
def init_state():
    if "users" not in st.session_state:
        st.session_state.users: List[User] = []
    if "current_user_email" not in st.session_state:
        st.session_state.current_user_email: Optional[str] = None
    if "cards" not in st.session_state:
        st.session_state.cards: List[Card] = []
    if "research_summary" not in st.session_state:
        st.session_state.research_summary: str = ""
    if "research_notes" not in st.session_state:
        st.session_state.research_notes: str = ""
    if "mind_root" not in st.session_state:
        st.session_state.mind_root: MindNode = MindNode(
            id="root", label="Tema central", children=[]
        )
    if "mind_selected_id" not in st.session_state:
        st.session_state.mind_selected_id: str = "root"
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages: List[ChatMessage] = []
    if "chat_topic" not in st.session_state:
        st.session_state.chat_topic: str = "metodologia"
    if "font_scale" not in st.session_state:
        st.session_state.font_scale: int = 100


init_state()

# ======================================================
# FUNÇÕES AUXILIARES
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
            "como", "para", "onde", "entre", "sobre", "dentro", "dados",
            "estudo", "pesquisa", "analise", "resultado", "resultados",
            "qualitativa", "qualitativo", "uma", "essa", "esse", "sera",
            "pelo", "pela", "com", "tambem", "que", "isso", "nao",
            "mais", "menos", "muito", "pouco", "sendo", "assim",
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
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="pqr-title">PQR</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pqr-subtitle">Pesquisa Qualitativa de Resultados</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        tabs = st.tabs(["Login", "Cadastro"])

        # LOGIN
        with tabs[0]:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Senha", type="password", key="login_password")
            if st.button("Entrar", type="primary", key="login_btn"):
                user = next((u for u in st.session_state.users if u.email == email), None)
                if not user or user.password != password:
                    st.error("Credenciais inválidas.")
                else:
                    st.session_state.current_user_email = email
                    st.experimental_rerun()

        # CADASTRO
        with tabs[1]:
            name = st.text_input("Nome completo", key="cad_nome")
            email_c = st.text_input("Email", key="cad_email")
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
            password_c = st.text_input("Senha", type="password", key="cad_senha")

            if st.button("Criar conta", type="primary", key="cad_btn"):
                if type_label == "Selecione…":
                    st.warning("Escolha um tipo de bolsa.")
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
                    st.success("Conta criada. Entrando no sistema…")
                    st.experimental_rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# VIEW: BOARD / TIMELINE
# ======================================================
def view_board():
    st.markdown('<div class="glass-section">', unsafe_allow_html=True)
    top_left, top_mid, top_right = st.columns([2.4, 3.2, 2.4])

    with top_left:
        st.markdown("### Timeline / Board")
        st.markdown(
            '<span class="pqr-badge">PQR – PESQUISA QUALITATIVA DE RESULTADOS</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Visualização das etapas da sua pesquisa em colunas (da ideia até a redação)."
        )

    with top_mid:
        st.write("")
        st.write("")
        query = st.text_input(
            "Busca global (IA local, protótipo)",
            key="global_search",
            placeholder="Digite um termo para procurar em resumo, anotações e cards…",
        )
        if st.button("Buscar com IA local", key="btn_search"):
            haystack = (
                st.session_state.research_summary
                + "\n"
                + st.session_state.research_notes
                + "\n"
                + "\n".join(
                    f"{c.title} {c.description}" for c in st.session_state.cards
                )
            )
            found = query.strip() and query.lower() in haystack.lower()
            if not query.strip():
                st.warning("Digite um termo para buscar.")
            elif found:
                st.success(
                    f'Encontrei ocorrências de "{query}" no seu material. '
                    "Verifique resumo, anotações e cards para ver o contexto."
                )
            else:
                st.info(
                    f'Não encontrei "{query}" no que está registrado. '
                    "Talvez seja algo que ainda não foi documentado nas suas notas."
                )

    with top_right:
        st.write("")
        st.write("")
        with st.expander("Nova etapa da pesquisa"):
            title = st.text_input("Título", key="new_card_title")
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
            if st.button("Adicionar etapa", type="primary", key="btn_add_card"):
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
                        )
                    )
                    st.success("Etapa adicionada.")

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
                    with st.expander(f"• {c.title}"):
                        st.write(c.description or "_(sem descrição)_")
                        st.caption(f"Prazo: {c.deadline}")
                        novo_status = st.selectbox(
                            "Mover para",
                            statuses,
                            index=[s[0] for s in statuses].index(status_key),
                            format_func=lambda x: x[1],
                            key=f"move_{c.id}",
                        )
                        if novo_status[0] != c.status:
                            c.status = novo_status[0]
                            st.info("Etapa movida na timeline.")

    st.write("---")
    st.subheader("Resumo de andamento")
    comp = timeline_completion()
    st.progress(comp / 100)
    st.caption(f"{comp}% concluído (estimativa pela coluna de redação/resultados).")

    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# VIEW: PESQUISA / GOOGLE ACADÊMICO
# ======================================================
def view_research():
    st.markdown('<div class="glass-section">', unsafe_allow_html=True)
    st.markdown("### Pasta da pesquisa – texto, anotações e Google Acadêmico")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Resumo da pesquisa")
        st.session_state.research_summary = st.text_area(
            "Descreva tema, objetivos, perguntas, contexto, etc.",
            value=st.session_state.research_summary,
            height=220,
        )

    with col2:
        st.subheader("Atalhos para Google Acadêmico")
        keywords = st.text_input(
            "Palavras‑chave principais",
            key="ga_kw",
            placeholder="ex.: inclusão digital, aprendizagem ativa",
        )
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("Buscar artigos"):
                if keywords.strip():
                    url = "https://scholar.google.com/scholar?q=" + keywords.replace(
                        " ", "+"
                    )
                    st.markdown(f"[Abrir Google Acadêmico]({url})")
        with col_b2:
            if st.button("Buscar últimos 5 anos"):
                if keywords.strip():
                    year = datetime.date.today().year - 5
                    url = (
                        "https://scholar.google.com/scholar?q="
                        + keywords.replace(" ", "+")
                        + f"&as_ylo={year}"
                    )
                    st.markdown(f"[Abrir (últimos 5 anos)]({url})")

        st.subheader("Anotações rápidas")
        st.session_state.research_notes = st.text_area(
            "Cole citações, insights, trechos importantes, etc.",
            value=st.session_state.research_notes,
            height=150,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# VIEW: MAPA MENTAL
# ======================================================
def view_mindmap():
    st.markdown('<div class="glass-section">', unsafe_allow_html=True)
    st.markdown("### Mapa mental da pesquisa (protótipo textual)")

    # visualização
    st.markdown("#### Estrutura atual")
    mind_print_tree(st.session_state.mind_root)

    st.write("---")
    st.markdown("#### Manipular nós")

    # listar nós
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
        if st.button("Adicionar sub‑tópico"):
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
                    st.success("Tópico adicionado.")
                else:
                    st.error("Nó pai não encontrado (estranho, mas ok).")

    with col_b:
        if st.button("Remover nó selecionado"):
            if selected == "root":
                st.warning("Não é possível remover o nó raiz.")
            else:
                removed = mind_remove_node(st.session_state.mind_root, selected)
                if removed:
                    st.session_state.mind_selected_id = "root"
                    st.info("Nó removido.")
                else:
                    st.error("Não foi possível remover o nó.")

    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# VIEW: ANÁLISE INTELIGENTE (MOCK)
# ======================================================
def view_analysis():
    st.markdown('<div class="glass-section">', unsafe_allow_html=True)
    st.markdown("### Análise inteligente (protótipo local)")

    if st.button("Gerar análise agora", type="primary"):
        summary = st.session_state.research_summary or ""
        notes = st.session_state.research_notes or ""
        total_cards = len(st.session_state.cards) or 1
        done_cards = len(
            [c for c in st.session_state.cards if c.status == "redacao"]
        )
        completion = round(done_cards / total_cards * 100)

        st.subheader("Síntese geral")
        if len(summary.strip()) < 40:
            st.write(
                "- O resumo ainda está bem curto. Considere explicitar objetivos, perguntas de pesquisa e contexto em mais detalhes."
            )
        else:
            st.write(
                "- O resumo já tem um bom volume. Verifique se as perguntas de pesquisa estão claras e se o foco está bem delimitado."
            )

        st.subheader("Andamento do projeto")
        st.write(f"- Etapas concluídas (redação/resultados): {done_cards}/{total_cards}.")
        if completion < 30:
            st.write(
                "- A pesquisa está em fase inicial. Invista tempo em delimitação do problema e revisão de literatura."
            )
        elif completion < 70:
            st.write(
                "- O projeto está em andamento. Revise coerência entre coleta de dados, instrumento e abordagem analítica."
            )
        else:
            st.write(
                "- Fase avançada. Agora é o momento de lapidar resultados, discussão e implicações."
            )

        st.subheader("Pistas qualitativas")
        concat = (summary + " " + notes).lower()
        if "entrevista" in concat:
            st.write(
                "- Você menciona entrevistas: práticas comuns incluem análise temática, análise narrativa ou análise de conteúdo."
            )
        if "questionário" in concat or "questionario" in concat or "survey" in concat:
            st.write(
                "- Há questionários: mesmo em abordagem qualitativa, planeje como categorizar respostas e lidar com questões abertas."
            )
        if len(notes) > 200:
            st.write(
                "- Suas anotações são extensas. Talvez seja hora de criar códigos iniciais, categorias e subcategorias."
            )
        if not summary.strip() and not notes.strip():
            st.info(
                "Quase nada foi escrito em resumo/anotações. Registre alguns parágrafos para que a análise possa sugerir algo mais útil."
            )

        st.write("---")
        st.subheader("Progresso estimado")
        st.progress(completion / 100)
        st.caption(f"{completion}% concluído (estimativa via timeline).")
    else:
        st.info(
            "Quando houver resumo, anotações e etapas na timeline, clique em **Gerar análise agora** para ver sugestões qualitativas."
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# VIEW: CHAT ENTRE BOLSISTAS
# ======================================================
def view_chat():
    st.markdown('<div class="glass-section">', unsafe_allow_html=True)
    st.markdown("### Chat entre bolsistas (local, protótipo)")

    topics = {
        "metodologia": "Metodologia qualitativa",
        "referencias": "Referências e artigos",
        "duvidas-eticas": "Dúvidas éticas",
        "off-topic": "Off‑topic / descompressão",
    }

    col_topics, col_chat = st.columns([1.1, 2.1])

    with col_topics:
        st.markdown("**Threads por tema**")
        for key, label in topics.items():
            if st.button(label, key=f"topic_{key}"):
                st.session_state.chat_topic = key
        st.caption(
            f"Thread atual: **{topics.get(st.session_state.chat_topic, 'Metodologia qualitativa')}**"
        )

    with col_chat:
        user = get_current_user()
        if not user:
            st.warning("É preciso estar logado para participar do chat.")
        else:
            st.subheader("Mensagens")
            topic = st.session_state.chat_topic
            msgs = [
                m for m in st.session_state.chat_messages if m.topic == topic
            ]
            if not msgs:
                st.caption("Nenhuma mensagem ainda. Inicie uma conversa.")
            for m in msgs:
                st.markdown(
                    f'<div class="chat-bubble">'
                    f'<div class="chat-meta">{m.user_name} • {m.time.strftime("%d/%m %H:%M")}</div>'
                    f'<div class="chat-text">{m.text}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.write("---")
            with st.form("chat_form"):
                text = st.text_input("Digite uma mensagem…", key="chat_text")
                send = st.form_submit_button("Enviar")
                if send and text.strip():
                    st.session_state.chat_messages.append(
                        ChatMessage(
                            id=str(uuid.uuid4()),
                            user_name=user.name.split(" ")[0] or user.name,
                            topic=topic,
                            text=text.strip(),
                            time=datetime.datetime.now(),
                        )
                    )
                    st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# VIEW: CADEIA DE LIGAÇÃO (REDE SIMPLES)
# ======================================================
def view_network():
    st.markdown('<div class="glass-section">', unsafe_allow_html=True)
    st.markdown("### Cadeia de ligação de pesquisas")

    interest = st.text_input(
        "Interesse principal (tema eixo)",
        key="net_interest",
        placeholder="ex.: inclusão digital, saúde mental, aprendizagem ativa",
    )

    if st.button("Gerar rede (protótipo)", type="primary"):
        base_interest = interest.strip() or "Tema central"
        text = st.session_state.research_summary + " " + st.session_state.research_notes
        keywords = extract_keywords(text, max_n=8)

        st.subheader("Nós principais")
        st.write(f"- **{base_interest}** (nó central)")
        if not keywords:
            st.warning(
                "Pouco texto foi identificado para extrair palavras‑chave. Escreva mais no resumo e nas anotações."
            )
        else:
            for kw in keywords:
                st.write(f"- {kw}")

        st.write("---")
        if keywords:
            st.subheader("Ligações qualitativas (descritas em texto)")
            for kw in keywords:
                st.markdown(
                    f"- **{base_interest} ↔ {kw}** – explorar como esse conceito aparece nos dados (falas, documentos, observações) "
                    "e de que forma se articula com as categorias da análise."
                )

    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# VIEW: CONFIGURAÇÕES
# ======================================================
def view_settings():
    st.markdown('<div class="glass-section">', unsafe_allow_html=True)
    st.markdown("### Configurações e conta")

    user = get_current_user()
    if user:
        st.markdown(
            f"**Nome:** {user.name}  \n"
            f"**Email:** {user.email}  \n"
            f"**Tipo de bolsa:** {map_type_label(user.type)}"
        )

    st.write("---")
    st.subheader("Sessão")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Sair da conta"):
            st.session_state.current_user_email = None
            st.experimental_rerun()
    with col_b:
        st.caption(
            "Outras configurações de acessibilidade (tema, zoom) podem ser ajustadas pelo próprio navegador."
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
        st.markdown('<div class="pqr-title">PQR</div>', unsafe_allow_html=True)
        st.markdown(
            f"<small>{user.name.split(' ')[0]} – {map_type_label(user.type)}</small>",
            unsafe_allow_html=True,
        )
        st.write("---")
        view = st.radio(
            "Navegação",
            [
                "Timeline / Board",
                "Pesquisa",
                "Mapa mental",
                "Análise",
                "Chat",
                "Cadeia de ligação",
                "Configurações",
            ],
        )

    if view == "Timeline / Board":
        view_board()
    elif view == "Pesquisa":
        view_research()
    elif view == "Mapa mental":
        view_mindmap()
    elif view == "Análise":
        view_analysis()
    elif view == "Chat":
        view_chat()
    elif view == "Cadeia de ligação":
        view_network()
    elif view == "Configurações":
        view_settings()


if __name__ == "__main__":
    main()
