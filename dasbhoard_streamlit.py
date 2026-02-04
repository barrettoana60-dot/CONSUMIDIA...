from anvil import *
import anvil.server

import datetime
import uuid
import re
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


# ======================================================
# DATACLASSES / MODELOS
# ======================================================

@dataclass
class UserData:
  name: str
  email: str
  type: str
  password: str


@dataclass
class CardData:
  id: str
  title: str
  description: str
  status: str
  deadline: Optional[str] = None
  created_at: str = ""


@dataclass
class ChatMessageData:
  id: str
  user_name: str
  topic: str
  text: str
  time: str


@dataclass
class MindNodeData:
  id: str
  label: str
  children: List["MindNodeData"] = field(default_factory=list)


# ======================================================
# HELPERS GERAIS (equivalentes às funções do Streamlit)
# ======================================================

def default_state() -> Dict[str, Any]:
  return {
    "users": [],
    "current_user_email": None,
    "cards": [],
    "research_summary": "",
    "research_notes": "",
    "mind_root": MindNodeData(id="root", label="Tema central", children=[]),
    "mind_selected_id": "root",
    "chat_messages": [],
    "chat_topic": "metodologia",
  }


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
  stop = {
    "como","para","onde","entre","sobre","dentro","dados",
    "estudo","pesquisa","analise","resultado","resultados",
    "qualitativa","qualitativo","uma","essa","esse","sera",
    "pelo","pela","com","tambem","que","isso","nao",
    "mais","menos","muito","pouco","sendo","assim"
  }
  words = [w for w in words if w not in stop]
  if not words:
    return []
  freq = Counter(words)
  return [w for w, _ in freq.most_common(max_n)]


def mind_find_node(node: MindNodeData, target_id: str) -> Optional[MindNodeData]:
  if node.id == target_id:
    return node
  for ch in node.children:
    found = mind_find_node(ch, target_id)
    if found:
      return found
  return None


def mind_remove_node(node: MindNodeData, target_id: str) -> bool:
  for i, ch in enumerate(node.children):
    if ch.id == target_id:
      node.children.pop(i)
      return True
    if mind_remove_node(ch, target_id):
      return True
  return False


def mind_list_nodes(node: MindNodeData, prefix: str = "") -> List[MindNodeData]:
  nodes = [MindNodeData(id=node.id, label=prefix + node.label)]
  for ch in node.children:
    nodes += mind_list_nodes(ch, prefix + "  ")
  return nodes


def timeline_completion(cards: List[Dict[str, Any]]) -> int:
  total = len(cards)
  if total == 0:
    return 0
  done = len([c for c in cards if c["status"] == "redacao"])
  return round(done / total * 100)


# ======================================================
# FORM PRINCIPAL
# ======================================================

class Form1(Form1Template):

  def __init__(self, **properties):
    self.init_components(**properties)

    # Estado "global" da aplicação (em memória, tipo session_state)
    self.state: Dict[str, Any] = default_state()

    # Painel principal onde tudo vai ser desenhado
    # Crie um ColumnPanel chamado content_panel no designer, ou:
    if not hasattr(self, "content_panel"):
      self.content_panel = ColumnPanel()
      self.add_component(self.content_panel)

    self.build_layout()

  # ====================================================
  # AUX: Usuário atual
  # ====================================================

  def get_current_user(self) -> Optional[Dict[str, Any]]:
    email = self.state.get("current_user_email")
    if not email:
      return None
    for u in self.state["users"]:
      if u["email"] == email:
        return u
    return None

  # ====================================================
  # Decisão entre login e app principal
  # ====================================================

  def build_layout(self):
    self.content_panel.clear()
    user = self.get_current_user()
    if not user:
      self.auth_screen()
    else:
      self.main_app()

  # ====================================================
  # TELA DE AUTENTICAÇÃO
  # ====================================================

  def auth_screen(self):
    col = ColumnPanel()
    self.content_panel.add_component(col)

    col.add_component(
      Label(text="PQR – Pesquisa Qualitativa de Resultados",
            bold=True,
            align="center",
            role="headline")
    )

    # Dois botões simples funcionando como abas
    btn_row = FlowPanel(spacing="medium")
    col.add_component(btn_row)

    btn_login = Button(text="Entrar")
    btn_signup = Button(text="Criar conta")

    btn_row.add_component(btn_login)
    btn_row.add_component(btn_signup)

    self.auth_inner_panel = ColumnPanel()
    col.add_component(self.auth_inner_panel)

    def show_login(**e):
      self.auth_inner_panel.clear()
      self.build_login_panel(self.auth_inner_panel)

    def show_signup(**e):
      self.auth_inner_panel.clear()
      self.build_signup_panel(self.auth_inner_panel)

    btn_login.set_event_handler("click", show_login)
    btn_signup.set_event_handler("click", show_signup)

    show_login()

  def build_login_panel(self, parent: ColumnPanel):
    self.login_email_tb = TextBox(placeholder="Email")
    self.login_password_tb = TextBox(placeholder="Senha", hide_text=True)
    btn = Button(text="Entrar", role="primary-color")

    parent.add_component(self.login_email_tb)
    parent.add_component(self.login_password_tb)
    parent.add_component(btn)

    btn.set_event_handler("click", self.login_action)

  def build_signup_panel(self, parent: ColumnPanel):
    self.signup_name_tb = TextBox(placeholder="Nome completo")
    self.signup_email_tb = TextBox(placeholder="Email institucional")
    self.signup_type_dd = DropDown(
      items=[
        ("Selecione…", ""),
        ("IC – Iniciação Científica", "ic"),
        ("Extensão", "extensao"),
        ("Doutorando", "doutorando"),
        ("Voluntário", "voluntario"),
        ("PRODIG", "prodig"),
        ("Mentoria", "mentoria"),
      ]
    )
    self.signup_password_tb = TextBox(placeholder="Senha (mín. 6 caracteres)", hide_text=True)
    btn = Button(text="Criar conta", role="primary-color")

    parent.add_component(self.signup_name_tb)
    parent.add_component(self.signup_email_tb)
    parent.add_component(self.signup_type_dd)
    parent.add_component(self.signup_password_tb)
    parent.add_component(btn)

    btn.set_event_handler("click", self.signup_action)

  def login_action(self, **event_args):
    email = (self.login_email_tb.text or "").strip()
    password = (self.login_password_tb.text or "").strip()
    user = next((u for u in self.state["users"] if u["email"] == email), None)
    if not user or user["password"] != password:
      Notification("Credenciais inválidas.", style="danger").show()
      return
    self.state["current_user_email"] = email
    self.build_layout()

  def signup_action(self, **event_args):
    name = (self.signup_name_tb.text or "").strip()
    email = (self.signup_email_tb.text or "").strip()
    tipo = self.signup_type_dd.selected_value or ""
    password = (self.signup_password_tb.text or "").strip()

    if not name or not email or not tipo:
      Notification("Preencha todos os campos.", style="warning").show()
      return
    if len(password) < 6:
      Notification("Senha muito curta (mín. 6).", style="warning").show()
      return
    if any(u["email"] == email for u in self.state["users"]):
      Notification("Já existe usuário com este email.", style="danger").show()
      return

    self.state["users"].append(
      {"name": name, "email": email, "type": tipo, "password": password}
    )
    self.state["current_user_email"] = email
    Notification("Conta criada. Bem‑vinda ao PQR!", style="success").show()
    self.build_layout()

  # ====================================================
  # APP PRINCIPAL
  # ====================================================

  def main_app(self):
    user = self.get_current_user()
    main_panel = ColumnPanel()
    self.content_panel.add_component(main_panel)

    # "Perfil"
    main_panel.add_component(
      Label(text=f"{user['name']} • {map_type_label(user['type'])}",
            bold=True)
    )

    # Navegação por DropDown (equivalente às radios do sidebar)
    self.nav_dd = DropDown(
      items=[
        ("Timeline / Feed", "timeline"),
        ("Pasta da pesquisa", "research"),
        ("Mapa mental", "mindmap"),
        ("Análise inteligente", "analysis"),
        ("Chat", "chat"),
        ("Cadeia de ligação", "network"),
        ("Configurações", "settings"),
      ],
      selected_value="timeline"
    )
    main_panel.add_component(self.nav_dd)
    self.nav_dd.set_event_handler("change", self.render_view)

    # Painel de conteúdo
    self.view_panel = ColumnPanel()
    main_panel.add_component(self.view_panel)

    self.render_view()

  def render_view(self, **event_args):
    self.view_panel.clear()
    view = self.nav_dd.selected_value

    if view == "timeline":
      self.view_board()
    elif view == "research":
      self.view_research()
    elif view == "mindmap":
      self.view_mindmap()
    elif view == "analysis":
      self.view_analysis()
    elif view == "chat":
      self.view_chat()
    elif view == "network":
      self.view_network()
    elif view == "settings":
      self.view_settings()
    else:
      self.view_board()

  # ====================================================
  # VIEW: TIMELINE / FEED
  # ====================================================

  def view_board(self):
    p = self.view_panel
    p.add_component(Label(text="Timeline de pesquisa", bold=True))

    # Nova etapa
    title_tb = TextBox(placeholder="Título da etapa")
    desc_ta = TextArea(placeholder="Descrição")
    deadline_dp = DatePicker()
    status_dd = DropDown(
      items=[
        ("Ideia / Delimitação", "ideia"),
        ("Revisão de literatura", "revisao"),
        ("Coleta de dados", "coleta"),
        ("Análise", "analise"),
        ("Redação / Resultados", "redacao"),
      ],
      selected_value="ideia",
    )
    add_btn = Button(text="Publicar etapa na timeline", role="primary-color")

    p.add_component(title_tb)
    p.add_component(desc_ta)
    p.add_component(deadline_dp)
    p.add_component(status_dd)
    p.add_component(add_btn)

    def add_card(**e):
      title = (title_tb.text or "").strip()
      desc = (desc_ta.text or "").strip()
      if not title:
        Notification("Dê um título para a etapa.", style="warning").show()
        return
      deadline_str = str(deadline_dp.date) if deadline_dp.date else ""
      card = CardData(
        id=str(uuid.uuid4()),
        title=title,
        description=desc,
        status=status_dd.selected_value,
        deadline=deadline_str,
        created_at=datetime.datetime.now().isoformat(),
      )
      self.state["cards"].append(asdict(card))
      Notification("Etapa adicionada à timeline.", style="success").show()
      self.render_view()

    add_btn.set_event_handler("click", add_card)

    p.add_component(Label(text="Etapas por fase:", bold=True))

    statuses = [
      ("ideia", "Ideia / Delimitação"),
      ("revisao", "Revisão de literatura"),
      ("coleta", "Coleta de dados"),
      ("analise", "Análise"),
      ("redacao", "Redação / Resultados"),
    ]

    for status_key, status_label in statuses:
      cards = [c for c in self.state["cards"] if c["status"] == status_key]
      p.add_component(Label(text=f"{status_label} ({len(cards)})", bold=True))
      for c in cards:
        card_panel = ColumnPanel(border="1px solid #ccc", spacing="tiny")
        card_panel.add_component(Label(text=c["title"], bold=True))
        card_panel.add_component(Label(text=c["description"] or "(sem descrição)"))
        card_panel.add_component(Label(text=f"Prazo: {c['deadline']}"))

        move_dd = DropDown(
          items=statuses,
          selected_value=c["status"]
        )
        card_panel.add_component(move_dd)

        def mover(widget=move_dd, card=c, **e):
          new_status = widget.selected_value
          if new_status != card["status"]:
            card["status"] = new_status
            Notification("Etapa movida na timeline.", style="info").show()
            self.render_view()

        move_dd.set_event_handler("change", mover)

        p.add_component(card_panel)

    comp = timeline_completion(self.state["cards"])
    p.add_component(Label(text=f"Progresso global estimado: {comp}%"))

  # ====================================================
  # VIEW: PASTA DA PESQUISA
  # ====================================================

  def view_research(self):
    p = self.view_panel
    p.add_component(Label(text="Pasta principal da sua pesquisa", bold=True))

    self.summary_ta = TextArea(
      text=self.state["research_summary"],
      placeholder="Tema, objetivos, perguntas, contexto…",
      height=200
    )
    self.notes_ta = TextArea(
      text=self.state["research_notes"],
      placeholder="Citações, ideias, lembretes…",
      height=200
    )

    p.add_component(Label(text="Resumo da pesquisa", bold=True))
    p.add_component(self.summary_ta)
    p.add_component(Label(text="Anotações rápidas", bold=True))
    p.add_component(self.notes_ta)

    save_btn = Button(text="Salvar", role="primary-color")

    def save_notes(**e):
      self.state["research_summary"] = self.summary_ta.text or ""
      self.state["research_notes"] = self.notes_ta.text or ""
      Notification("Resumo e notas salvos.", style="success").show()

    save_btn.set_event_handler("click", save_notes)
    p.add_component(save_btn)

  # ====================================================
  # VIEW: MAPA MENTAL
  # ====================================================

  def view_mindmap(self):
    p = self.view_panel
    p.add_component(Label(text="Mapa mental da sua pesquisa", bold=True))

    root: MindNodeData = self.state["mind_root"]

    nodes_list = mind_list_nodes(root)
    ids = [n.id for n in nodes_list]
    labels = {n.id: n.label for n in nodes_list}

    selected_id = self.state.get("mind_selected_id", "root")
    if selected_id not in ids:
      selected_id = "root"
    self.mind_select_dd = DropDown(
      items=[(labels[i], i) for i in ids],
      selected_value=selected_id
    )
    p.add_component(self.mind_select_dd)

    self.mind_new_tb = TextBox(placeholder="Novo tópico / sub‑tópico")
    p.add_component(self.mind_new_tb)

    add_btn = Button(text="Adicionar sub‑tópico ao nó selecionado",
                     role="primary-color")
    del_btn = Button(text="Remover nó selecionado")

    p.add_component(add_btn)
    p.add_component(del_btn)

    def add_topic(**e):
      label = (self.mind_new_tb.text or "").strip()
      if not label:
        Notification("Digite o texto do tópico.", style="warning").show()
        return
      target = self.mind_select_dd.selected_value
      parent = mind_find_node(root, target)
      if not parent:
        Notification("Nó pai não encontrado.", style="danger").show()
        return
      parent.children.append(MindNodeData(id=str(uuid.uuid4()), label=label))
      self.state["mind_selected_id"] = target
      Notification("Tópico adicionado ao mapa.", style="success").show()
      self.render_view()

    def del_topic(**e):
      target = self.mind_select_dd.selected_value
      if target == "root":
        Notification("Não é possível remover o nó raiz.", style="warning").show()
        return
      removed = mind_remove_node(root, target)
      if removed:
        self.state["mind_selected_id"] = "root"
        Notification("Nó removido do mapa mental.", style="info").show()
        self.render_view()
      else:
        Notification("Não foi possível remover o nó.", style="danger").show()

    add_btn.set_event_handler("click", add_topic)
    del_btn.set_event_handler("click", del_topic)

  # ====================================================
  # VIEW: ANÁLISE INTELIGENTE
  # ====================================================

  def view_analysis(self):
    p = self.view_panel
    p.add_component(Label(text="Análise inteligente (protótipo local)", bold=True))

    run_btn = Button(text="Rodar análise qualitativa agora",
                     role="primary-color")
    p.add_component(run_btn)

    self.analysis_output = TextArea(read_only=True, height=200)
    p.add_component(self.analysis_output)

    def run_analysis(**e):
      summary = self.state["research_summary"] or ""
      notes = self.state["research_notes"] or ""
      cards = self.state["cards"]
      total = len(cards) or 1
      done = len([c for c in cards if c["status"] == "redacao"])
      completion = round(done / total * 100)

      lines = []

      if len(summary.strip()) < 60:
        lines.append("- O resumo está curto; explicite contexto, problema e objetivos.")
      else:
        lines.append("- O resumo já tem um corpo interessante; verifique clareza do recorte qualitativo.")

      if completion < 30:
        lines.append("- Fase inicial: foque em consolidar problema, referencial e caminhos metodológicos.")
      elif completion < 70:
        lines.append("- Fase intermediária: revise se a forma de coleta está coerente com o que você quer responder.")
      else:
        lines.append("- Fase avançada: conecte dados, categorias e discussões com a literatura.")

      concat = (summary + " " + notes).lower()
      if "entrevista" in concat:
        lines.append("- Há entrevistas: explore análise temática, de conteúdo ou narrativa.")
      if not summary.strip() and not notes.strip():
        lines.append("- Ainda não há conteúdo suficiente. Escreva um parágrafo de resumo e algumas notas.")

      self.analysis_output.text = "\n".join(lines) or "Sem dados suficientes."

    run_btn.set_event_handler("click", run_analysis)

  # ====================================================
  # VIEW: CHAT
  # ====================================================

  def view_chat(self):
    p = self.view_panel
    p.add_component(Label(text="Chat entre bolsistas (estilo feed)", bold=True))

    topics = {
      "metodologia": "Metodologia qualitativa",
      "referencias": "Referências e artigos",
      "duvidas-eticas": "Dúvidas éticas",
      "off-topic": "Off‑topic / descompressão",
    }

    current_topic = self.state.get("chat_topic", "metodologia")
    self.chat_topic_dd = DropDown(
      items=list(topics.items()),
      selected_value=current_topic
    )
    p.add_component(self.chat_topic_dd)

    def change_topic(**e):
      self.state["chat_topic"] = self.chat_topic_dd.selected_value
      self.render_view()

    self.chat_topic_dd.set_event_handler("change", change_topic)

    topic = self.state.get("chat_topic", "metodologia")
    msgs = [m for m in self.state["chat_messages"] if m["topic"] == topic]

    if not msgs:
      p.add_component(Label(text="Ainda não há mensagens neste canal. Inicie a conversa!"))

    for m in msgs:
      p.add_component(
        Label(text=f"{m['user_name']} • {m['time']}\n{m['text']}")
      )

    self.chat_text_tb = TextBox(placeholder="Escreva uma mensagem…", width="100%")
    send_btn = Button(text="Publicar", role="primary-color")
    p.add_component(self.chat_text_tb)
    p.add_component(send_btn)

    def send_msg(**e):
      text = (self.chat_text_tb.text or "").strip()
      if not text:
        Notification("Escreva uma mensagem.", style="warning").show()
        return
      user = self.get_current_user()
      msg = ChatMessageData(
        id=str(uuid.uuid4()),
        user_name=user["name"].split(" ")[0],
        topic=topic,
        text=text,
        time=datetime.datetime.now().strftime("%d/%m %H:%M"),
      )
      self.state["chat_messages"].append(asdict(msg))
      Notification("Mensagem publicada.", style="success").show()
      self.render_view()

    send_btn.set_event_handler("click", send_msg)

  # ====================================================
  # VIEW: CADEIA DE LIGAÇÃO / REDE
  # ====================================================

  def view_network(self):
    p = self.view_panel
    p.add_component(Label(text="Cadeia de ligação de pesquisas (rede conceitual)", bold=True))

    self.net_interest_tb = TextBox(placeholder="Interesse principal (tema eixo da rede)")
    p.add_component(self.net_interest_tb)

    self.net_output_ta = TextArea(read_only=True, height=200)
    p.add_component(self.net_output_ta)

    run_btn = Button(text="Gerar rede textual", role="primary-color")
    p.add_component(run_btn)

    def run_network(**e):
      interest = (self.net_interest_tb.text or "").strip() or "Tema central"
      text = self.state["research_summary"] + " " + self.state["research_notes"]
      keywords = extract_keywords(text, max_n=8)
      if not keywords:
        self.net_output_ta.text = "Não identifiquei palavras‑chave suficientes. Escreva mais no resumo e nas anotações."
        return
      lines = [f"{interest} ↔ {kw}" for kw in keywords]
      self.net_output_ta.text = "\n".join(lines)

    run_btn.set_event_handler("click", run_network)

  # ====================================================
  # VIEW: CONFIGURAÇÕES
  # ====================================================

  def view_settings(self):
    p = self.view_panel
    p.add_component(Label(text="Configurações, salvar & sair", bold=True))

    user = self.get_current_user()
    if user:
      p.add_component(
        Label(text=f"{user['name']} – {user['email']} – {map_type_label(user['type'])}")
      )

    save_btn = Button(text="Salvar tudo (protótipo em memória)", role="primary-color")
    logout_btn = Button(text="Salvar e sair")

    p.add_component(save_btn)
    p.add_component(logout_btn)

    def save_all(**e):
      # Aqui você poderia chamar server functions e Data Tables para salvar de verdade
      Notification("Neste protótipo, os dados ficam apenas em memória.", style="info").show()

    def logout(**e):
      self.state["current_user_email"] = None
      Notification("Sessão encerrada.", style="info").show()
      self.build_layout()

    save_btn.set_event_handler("click", save_all)
    logout_btn.set_event_handler("click", logout)
