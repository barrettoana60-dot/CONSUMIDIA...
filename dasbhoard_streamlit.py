# dashboard_consumidia_final.py
# CONSUMIDIA — Versão final com correções: cadastros, mensagens, busca, favoritos
import os
import re
import io
import json
import time
import random
import string
import unicodedata
import html
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from networkx.readwrite import json_graph
from fpdf import FPDF

# optional ML libs (silenciosamente não-fatal)
try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    joblib = None

# -------------------------
# Config & helpers
# -------------------------
st.set_page_config(page_title="CONSUMIDIA", layout="wide", initial_sidebar_state="expanded")

def safe_rerun():
    try:
        if hasattr(st, "rerun") and callable(getattr(st, "rerun")):
            st.rerun(); return
        if hasattr(st, "experimental_rerun") and callable(getattr(st, "experimental_rerun")):
            st.experimental_rerun(); return
    except Exception as e:
        try:
            st.error(f"safe_rerun: não foi possível reiniciar a app (erro: {e}). Verifique logs.")
        except Exception:
            pass
    try:
        st.stop()
    except Exception:
        raise RuntimeError("safe_rerun falhou e não foi possível chamar st.stop()")

# -------------------------
# CSS (visual uniforme)
# -------------------------
DEFAULT_CSS = r"""
:root{ --glass-bg: rgba(255,255,255,0.03); --muted-text: #bfc6cc; --icon-color:#fff; }
.css-1d391kg { background: linear-gradient(180deg,#071428 0%, #031926 100%) !important; }
.glass-box{ background: var(--glass-bg); border:1px solid rgba(255,255,255,0.04); border-radius:14px; padding:16px; box-shadow:0 8px 32px rgba(4,9,20,0.5); }
.stButton>button, .stDownloadButton>button{ background:transparent !important; color:var(--muted-text) !important; border:1px solid rgba(255,255,255,0.06) !important; padding:8px 12px !important; border-radius:10px !important; }
.card, .msg-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:12px; margin-bottom:10px; border:1px solid rgba(255,255,255,0.03); }
.avatar{width:40px;height:40px;border-radius:8px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;color:#fff;background:#6c5ce7;margin-right:8px}
.small-muted{font-size:12px;color:#bfc6cc;}
.card-title{font-weight:700;font-size:15px}
.card-mark{ background: rgba(255,255,0,0.12); padding: 0 2px; border-radius:2px; }
"""

try:
    css_path = Path("style.css")
    if not css_path.exists():
        css_path.write_text(DEFAULT_CSS, encoding='utf-8')
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
except Exception:
    st.markdown(f"<style>{DEFAULT_CSS}</style>", unsafe_allow_html=True)

st.markdown("<div style='max-width:1100px;margin:18px auto 8px;text-align:center;'><h1 style='font-weight:800;font-size:40px; background:linear-gradient(90deg,#8e44ad,#2979ff,#1abc9c,#ff8a00); -webkit-background-clip:text; color:transparent; margin:0;'>CONSUMIDIA</h1></div>", unsafe_allow_html=True)

# -------------------------
# Storage & fallback paths
# -------------------------
USERS_FILE = "users.json"
MESSAGES_FILE = "messages.json"
BACKUPS_DIR = Path("backups")
ATTACHMENTS_DIR = Path("user_files")
BACKUPS_DIR.mkdir(exist_ok=True)
ATTACHMENTS_DIR.mkdir(exist_ok=True)

# -------------------------
# Supabase stub (opcional)
# -------------------------
try:
    from supabase import create_client
except Exception:
    create_client = None
_supabase = None
# (se usares supabase, podes configurar em st.secrets ou variáveis de ambiente)
# por simplicidade esse arquivo usa fallback local quando _supabase for None

# -------------------------
# Utilidades gerais
# -------------------------
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn').lower()

def escape_html(s):
    return html.escape(str(s) if s is not None else "")

def gen_password(length=8):
    choices = string.ascii_letters + string.digits
    return ''.join(random.choice(choices) for _ in range(length))

# -------------------------
# load/save users (corrigido: atomic + Path.cwd())
# -------------------------
def load_users():
    if _supabase:
        return None
    users_path = Path.cwd() / USERS_FILE
    if users_path.exists():
        try:
            with users_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            print(f"[load_users] {users_path} inválido JSON. Recriando.")
            return {}
        except Exception as e:
            print(f"[load_users] Erro ao ler {users_path}: {e}")
            return {}
    return {}

def save_users(users: dict):
    if _supabase:
        return False
    users_path = Path.cwd() / USERS_FILE
    try:
        tmp_path = users_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(users_path)
        return True
    except Exception as e:
        print(f"[save_users] Erro ao salvar {users_path}: {e}")
        return False

# -------------------------
# Favorites helpers (session)
# -------------------------
def get_session_favorites():
    return st.session_state.get("favorites", [])

def add_to_favorites(result_data):
    favorites = get_session_favorites()
    result_id = f"{int(time.time())}_{random.randint(1000,9999)}"
    favorite_item = {"id": result_id, "data": result_data, "added_at": datetime.utcnow().isoformat()}
    temp_data_to_check = result_data.copy()
    temp_data_to_check.pop('_artemis_username', None)
    existing_contents = []
    for fav in favorites:
        temp_existing = fav["data"].copy()
        temp_existing.pop('_artemis_username', None)
        existing_contents.append(json.dumps(temp_existing, sort_keys=True))
    if json.dumps(temp_data_to_check, sort_keys=True) not in existing_contents:
        favorites.append(favorite_item)
        st.session_state.favorites = favorites
        return True
    return False

def remove_from_favorites(favorite_id):
    favorites = get_session_favorites()
    new_favorites = [fav for fav in favorites if fav["id"] != favorite_id]
    st.session_state.favorites = new_favorites
    return len(new_favorites) != len(favorites)

def clear_all_favorites():
    st.session_state.favorites = []
    return True

# -------------------------
# Messages local storage
# -------------------------
def _local_load_all_messages():
    p = Path.cwd() / MESSAGES_FILE
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _local_save_all_messages(msgs):
    p = Path.cwd() / MESSAGES_FILE
    try:
        tmp = p.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(msgs, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        tmp.replace(p)
    except Exception as e:
        print(f"[save_messages] erro: {e}")

def _local_upload_attachment(sender, attachment_file):
    safe_filename = re.sub(r'[^\w\.\-]', '_', attachment_file.name)
    unique_filename = f"{int(time.time())}_{sender}_{safe_filename}"
    save_path = ATTACHMENTS_DIR / unique_filename
    with open(save_path, "wb") as f:
        f.write(attachment_file.getbuffer())
    return {"name": attachment_file.name, "path": str(save_path)}

def _local_remove_attachment(path):
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except Exception:
        pass
    return False

def load_all_messages():
    if _supabase:
        try:
            res = _supabase.table("messages").select("*").order("ts", desc=True).execute()
            msgs = getattr(res, "data", None) or []
            return msgs
        except Exception:
            pass
    return _local_load_all_messages()

def send_message(sender, recipient, subject, body, attachment_file=None):
    mid = f"m_{int(time.time())}_{random.randint(1000,9999)}"
    entry = {"id": mid, "from": sender, "to": recipient, "subject": subject or "(sem assunto)", "body": body, "ts": datetime.utcnow().isoformat(), "read": False, "attachment": None}
    if attachment_file:
        try:
            entry["attachment"] = _local_upload_attachment(sender, attachment_file)
        except Exception:
            entry["attachment"] = None
    msgs = _local_load_all_messages()
    msgs.append(entry)
    _local_save_all_messages(msgs)
    return entry

def get_user_messages(username, box_type='inbox'):
    msgs = load_all_messages()
    if not msgs:
        return []
    key = "to" if box_type == 'inbox' else "from"
    user_msgs = [m for m in msgs if m.get(key) == username]
    user_msgs.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return user_msgs

def mark_message_read(message_id, username):
    msgs = _local_load_all_messages()
    changed = False
    for m in msgs:
        if m.get("id") == message_id and m.get("to") == username:
            if not m.get("read"):
                m["read"] = True; changed = True
            break
    if changed:
        _local_save_all_messages(msgs)
    return changed

def delete_message(message_id, username):
    msgs = _local_load_all_messages()
    msg_to_delete = next((m for m in msgs if m.get("id") == message_id and (m.get("to") == username or m.get("from") == username)), None)
    if msg_to_delete:
        if msg_to_delete.get("attachment"):
            try:
                apath = msg_to_delete["attachment"].get("path")
                if apath and os.path.exists(apath):
                    _local_remove_attachment(apath)
            except Exception:
                pass
        new_msgs = [m for m in msgs if m.get("id") != message_id]
        _local_save_all_messages(new_msgs)
        return True
    return False

# -------------------------
# Graph / reading / PDF utils (mantidos)
# -------------------------
def read_spreadsheet(uploaded_file):
    b = uploaded_file.read()
    buf = io.BytesIO(b)
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        for enc in ("utf-8", "latin1", "cp1252"):
            try:
                buf.seek(0)
                return pd.read_csv(buf, encoding=enc)
            except Exception:
                continue
        buf.seek(0)
        return pd.read_csv(buf, engine="python", on_bad_lines="skip")
    else:
        buf.seek(0)
        return pd.read_excel(buf)

def criar_grafo(df):
    G = nx.Graph()
    if df is None:
        return G
    cols_lower = {c.lower(): c for c in df.columns}
    for _, row in df.iterrows():
        autor = str(row.get(cols_lower.get("autor", ""), "") or "").strip()
        titulo = str(row.get(cols_lower.get("título", cols_lower.get("titulo", "")), "") or "").strip()
        ano = str(row.get(cols_lower.get("ano", ""), "") or "").strip()
        tema = str(row.get(cols_lower.get("tema", ""), "") or "").strip()
        if autor:
            G.add_node(f"Autor: {autor}", tipo="Autor", label=autor)
        if titulo:
            G.add_node(f"Título: {titulo}", tipo="Título", label=titulo)
        if ano:
            G.add_node(f"Ano: {ano}", tipo="Ano", label=ano)
        if tema:
            G.add_node(f"Tema: {tema}", tipo="Tema", label=tema)
        if autor and titulo:
            G.add_edge(f"Autor: {autor}", f"Título: {titulo}")
        if titulo and ano:
            G.add_edge(f"Título: {titulo}", f"Ano: {ano}")
        if titulo and tema:
            G.add_edge(f"Título: {titulo}", f"Tema: {tema}")
    return G

def generate_pdf_with_highlights(texto, highlight_hex="#ffd600"):
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=12); pdf.add_page(); pdf.set_font("Arial", size=12)
    for linha in (texto or "").split("\n"):
        parts = re.split(r"(==.*?==)", linha)
        for part in parts:
            if not part:
                continue
            if part.startswith("==") and part.endswith("=="):
                inner = part[2:-2]
                inner_safe = inner.replace("—", "-").replace("–", "-").encode("latin-1", "replace").decode("latin-1")
                hexv = (highlight_hex or "#ffd600").lstrip("#")
                if len(hexv) == 3:
                    hexv = ''.join([c*2 for c in hexv])
                try:
                    r, g, b = int(hexv[0:2], 16), int(hexv[2:4], 16), int(hexv[4:6], 16)
                except Exception:
                    r, g, b = (255, 214, 0)
                pdf.set_fill_color(r, g, b); pdf.set_text_color(0, 0, 0)
                w = pdf.get_string_width(inner_safe) + 2
                pdf.cell(w, 6, txt=inner_safe, border=0, ln=0, fill=True)
                pdf.set_text_color(0, 0, 0)
            else:
                safe_part = part.replace("—", "-").replace("–", "-").encode("latin-1", "replace").decode("latin-1")
                pdf.set_text_color(0, 0, 0)
                pdf.cell(pdf.get_string_width(safe_part), 6, txt=safe_part, border=0, ln=0)
        pdf.ln(6)
    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        return raw.encode("latin-1", "replace")
    elif isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    else:
        return str(raw).encode("latin-1", "replace")

# -------------------------
# Small UI helpers
# -------------------------
ICON_SVGS = {
    "register": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="8" r="3"/></svg>',
    "favoritos": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77z"/></svg>',
    "trash": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg>'
}

def icon_html_svg(key, size=18, color=None):
    svg = ICON_SVGS.get(key, "")
    col = color or "var(--muted-text)"
    style = f"color:{col}; width:{size}px; height:{size}px; display:inline-block; vertical-align:middle;"
    return f'<span style="{style}">{svg}</span>'

def action_button(label, icon_key, st_key, expanded_label=None):
    c_icon, c_btn = st.columns([0.12, 0.88])
    with c_icon:
        st.markdown(f"<div style='margin-top:6px'>{icon_html_svg(icon_key, size=18)}</div>", unsafe_allow_html=True)
    with c_btn:
        clicked = st.button(expanded_label or label, key=st_key, use_container_width=True)
    return clicked

# -------------------------
# Session defaults
# -------------------------
_defaults = {
    "authenticated": False, "username": None, "user_obj": None, "df": None,
    "G": nx.Graph(), "notes": "", "autosave": False, "page": "planilha",
    "restored_from_saved": False, "favorites": [], "reply_message_id": None,
    "search_results": pd.DataFrame(), "search_page": 1, "search_query_meta": {"col": None,"query":""},
    "search_view_index": None, "compose_inline": False, "compose_open": False
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------
# Authentication UI (local fallback)
# -------------------------
if not st.session_state.authenticated:
    st.markdown("<div class='glass-box auth' style='max-width:1100px;margin:0 auto;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Acesso — Faça login ou cadastre-se")
    tabs = st.tabs(["Entrar", "Cadastrar"])

    with tabs[0]:
        login_user = st.text_input("Usuário / Email", key="ui_login_user")
        login_pass = st.text_input("Senha", type="password", key="ui_login_pass")
        if st.button("Entrar", "btn_login_main"):
            users = load_users() or {}
            if not users:
                users = {"admin": {"name": "Administrador", "scholarship": "Admin", "password": "admin123", "created_at": datetime.utcnow().isoformat()}}
                save_users(users)
                st.warning("Nenhum usuário local encontrado. Usuário criado: 'admin' / 'admin123' (troque a senha).")
            if login_user in users and users[login_user].get("password") == login_pass:
                st.session_state.authenticated = True
                st.session_state.username = login_user
                st.session_state.user_obj = users[login_user]
                st.success("Login efetuado (local).")
                safe_rerun()
            else:
                st.warning("Usuário/Senha inválidos (local).")

    with tabs[1]:
        reg_name = st.text_input("Nome completo", key="ui_reg_name")
        reg_bolsa = st.selectbox("Tipo de bolsa", ["IC - Iniciação Científica", "BIA - Bolsa de Incentivo Acadêmico", "Extensão", "Doutorado"], key="ui_reg_bolsa")
        reg_user = st.text_input("Email (ou username para modo local)", key="ui_reg_user")
        if st.button("Cadastrar", "btn_register_main"):
            # local flow (supabase path left intact earlier)
            new_user = (reg_user or "").strip()
            if not new_user:
                st.warning("Informe um username/email válido")
            else:
                users = load_users() or {}
                if new_user in users:
                    st.warning("Username já existe (local).")
                else:
                    pwd = gen_password(8)
                    users[new_user] = {"name": reg_name or new_user, "scholarship": reg_bolsa, "password": pwd, "created_at": datetime.utcnow().isoformat()}
                    ok = save_users(users)
                    if ok:
                        st.success(f"Usuário criado. Username: {new_user} — Senha gerada: {pwd}")
                        st.info("Anote a senha e troque depois")
                        try:
                            safe_rerun()
                        except Exception:
                            st.info("Recarregue a página para ver o novo usuário.")
                    else:
                        st.error("Falha ao salvar o usuário localmente. Verifique permissões do diretório.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# -------------------------
# Post-auth: header, nav
# -------------------------
USERNAME = st.session_state.username
users_local = load_users() or {}
USER_OBJ = st.session_state.user_obj or users_local.get(USERNAME, {})
USER_STATE = Path.cwd() / f"artemis_state_{USERNAME}.json"

# restore per-user
if not st.session_state.restored_from_saved and USER_STATE.exists():
    try:
        with USER_STATE.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        # attempt restore
        st.session_state.notes = meta.get("notes", "")
        st.session_state.uploaded_name = meta.get("uploaded_name", None)
        st.session_state.favorites = meta.get("favorites", [])
        st.session_state.restored_from_saved = True
        st.success("Estado salvo do usuário restaurado automaticamente.")
    except Exception:
        pass

# unread count
UNREAD_COUNT = 0
try:
    all_msgs = load_all_messages()
    if isinstance(all_msgs, list):
        UNREAD_COUNT = sum(1 for m in all_msgs if m.get("to") == USERNAME and not m.get("read"))
except Exception:
    UNREAD_COUNT = 0

if "last_unread_count" not in st.session_state:
    st.session_state.last_unread_count = 0
if UNREAD_COUNT > st.session_state.last_unread_count:
    try:
        st.toast(f"Você tem {UNREAD_COUNT} nova(s) mensagem(ns) não lida(s).", icon="✉️")
    except Exception:
        pass
st.session_state.last_unread_count = UNREAD_COUNT
mens_label = f"✉️ Mensagens ({UNREAD_COUNT})" if UNREAD_COUNT > 0 else "✉️ Mensagens"

st.markdown("<div class='glass-box' style='padding-top:10px; padding-bottom:10px;'><div class='specular'></div>", unsafe_allow_html=True)
top1, top2 = st.columns([0.6, 0.4])
with top1:
    st.markdown(f"<div style='color:var(--muted-text);font-weight:700;padding-top:8px;'>Usuário: {USER_OBJ.get('name','')} — {USER_OBJ.get('scholarship','')}</div>", unsafe_allow_html=True)
with top2:
    nav_right1, nav_right2, nav_right3 = st.columns([1,1,1])
    with nav_right1:
        st.session_state.autosave = st.checkbox("Auto-save", value=st.session_state.autosave, key="ui_autosave")
    with nav_right2:
        if st.button("💾 Salvar", key="btn_save_now", use_container_width=True):
            # salva estado mínimo
            try:
                data = {"notes": st.session_state.get("notes",""), "uploaded_name": st.session_state.get("uploaded_name", None), "favorites": st.session_state.get("favorites", [])}
                tmp = USER_STATE.with_suffix(".tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.flush(); os.fsync(f.fileno())
                tmp.replace(USER_STATE)
                st.success("Progresso salvo.")
            except Exception as e:
                st.error(f"Erro ao salvar estado: {e}")
    with nav_right3:
        if st.button("🚪 Sair", key="btn_logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_obj = None
            st.session_state.reply_message_id = None
            safe_rerun()
st.markdown("</div>", unsafe_allow_html=True)

# Navigation buttons
st.markdown("<div style='margin-top:-20px'>", unsafe_allow_html=True)
nav_cols = st.columns(6)
nav_buttons = {"planilha": "📄 Planilha", "mapa": "🞠 Mapa", "anotacoes": "📝 Anotações", "graficos": "📊 Gráficos", "busca": "🔍 Busca", "mensagens": mens_label}
for i, (page_key, page_label) in enumerate(nav_buttons.items()):
    with nav_cols[i]:
        if st.button(page_label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state.page = page_key
            st.session_state.reply_message_id = None
            safe_rerun()
st.markdown("</div></div><hr>", unsafe_allow_html=True)

# -------------------------
# Cached backups collector
# -------------------------
@st.cache_data(ttl=300)
def collect_latest_backups():
    base = BACKUPS_DIR
    if not base.exists(): return None
    dfs = []
    for user_dir in sorted(base.iterdir()):
        if not user_dir.is_dir(): continue
        csvs = sorted(user_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs: continue
        try:
            df = pd.read_csv(csvs[0], encoding="utf-8", on_bad_lines='skip')
            df["_artemis_username"] = user_dir.name
            dfs.append(df)
        except Exception:
            try:
                df = pd.read_csv(csvs[0], encoding="latin1", on_bad_lines='skip')
                df["_artemis_username"] = user_dir.name
                dfs.append(df)
            except Exception:
                continue
    return pd.concat(dfs, ignore_index=True) if dfs else None

# -------------------------
# Highlight helper
# -------------------------
def highlight_search_terms(text, query, mark_class="card-mark"):
    if not text or not query:
        return escape_html(text)
    q = normalize_text(query)
    words = [w for w in re.split(r"\s+", q) if w]
    if not words:
        return escape_html(text)
    safe_text = escape_html(str(text))
    def repl(m):
        return f"<mark class='{mark_class}'>{escape_html(m.group(0))}</mark>"
    for w in sorted(words, key=lambda x: -len(x)):
        try:
            pattern = re.compile(re.escape(w), flags=re.IGNORECASE)
            safe_text = pattern.sub(repl, safe_text)
        except Exception:
            continue
    return safe_text

# -------------------------
# Page dispatcher
# -------------------------
if st.session_state.page == "planilha":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Planilha / Backup")
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("Restaurar estado salvo", key="btn_restore_state"):
            if USER_STATE.exists():
                try:
                    with USER_STATE.open("r", encoding="utf-8") as f:
                        meta = json.load(f)
                    st.session_state.notes = meta.get("notes","")
                    st.session_state.uploaded_name = meta.get("uploaded_name", None)
                    st.session_state.favorites = meta.get("favorites", [])
                    st.success("Estado salvo restaurado.")
                except Exception:
                    st.info("Erro ao restaurar estado.")
            else:
                st.info("Nenhum estado salvo encontrado.")
    with col2:
        meta = None
        if USER_STATE.exists():
            try:
                with USER_STATE.open("r", encoding="utf-8") as f: meta = json.load(f)
            except Exception: meta = None
        if meta and meta.get("backup_csv") and os.path.exists(meta.get("backup_csv")):
            st.write("Backup CSV encontrado:")
            st.text(meta.get("backup_csv"))
            with open(meta.get("backup_csv"), "rb") as fp:
                st.download_button("⬇ Baixar backup CSV", data=fp, file_name=os.path.basename(meta.get("backup_csv")), mime="text/csv")
        else:
            st.write("Nenhum backup CSV automático encontrado ainda.")

    uploaded = st.file_uploader("Carregue .csv ou .xlsx (cada linha será um nó)", type=["csv", "xlsx"], key=f"u_{USERNAME}")
    if uploaded:
        try:
            df = read_spreadsheet(uploaded)
            st.session_state.df = df
            st.session_state.uploaded_name = uploaded.name
            st.session_state.G = criar_grafo(df)
            st.success("Planilha carregada com sucesso.")
            if st.session_state.autosave:
                # gravar backup csv
                try:
                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    safe = re.sub(r"[^\w\-_.]", "_", st.session_state.get("uploaded_name") or "planilha")
                    backup_filename = f"{safe}_{ts}.csv"
                    p = BACKUPS_DIR / USERNAME
                    p.mkdir(parents=True, exist_ok=True)
                    path = p / backup_filename
                    st.session_state.df.to_csv(path, index=False, encoding="utf-8")
                    st.success("Backup salvo.")
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Erro ao ler planilha: {e}")

    if st.session_state.df is not None:
        st.write("Visualização da planilha:")
        st.dataframe(st.session_state.df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "mapa":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Mapa Mental 3D — Editor")
    if st.session_state.G.nodes():
        with st.expander("Editar Nós do Mapa"):
            left, right = st.columns([2,1])
            with left:
                new_node = st.text_input("Nome do novo nó", key=f"nm_name_{USERNAME}")
                new_tipo = st.selectbox("Tipo", ["Outro", "Autor", "Título", "Ano", "Tema"], key=f"nm_tipo_{USERNAME}")
                connect_to = st.selectbox("Conectar a (opcional)", ["Nenhum"] + list(st.session_state.G.nodes), key=f"nm_connect_{USERNAME}")
                if st.button("Adicionar nó", key=f"btn_add_{USERNAME}"):
                    n = new_node.strip()
                    if not n: st.warning("Nome inválido.")
                    elif n in st.session_state.G.nodes: st.warning("Nó já existe.")
                    else:
                        st.session_state.G.add_node(n, tipo=new_tipo, label=n)
                        if connect_to != "Nenhum":
                            st.session_state.G.add_edge(n, connect_to)
                        st.success(f"Nó '{n}' adicionado.")
                        if st.session_state.autosave: safe_rerun()
            with right:
                del_n = st.selectbox("Excluir nó", [""] + list(st.session_state.G.nodes), key=f"del_{USERNAME}")
                if st.button("Excluir nó", key=f"btn_del_{USERNAME}"):
                    if del_n and del_n in st.session_state.G:
                        st.session_state.G.remove_node(del_n)
                        st.success(f"Nó '{del_n}' removido.")
                        if st.session_state.autosave: safe_rerun()
                st.markdown("---")
                r_old = st.selectbox("Renomear: selecione nó", [""] + list(st.session_state.G.nodes), key=f"r_old_{USERNAME}")
                r_new = st.text_input("Novo nome", key=f"r_new_{USERNAME}")
                if st.button("Renomear", key=f"btn_ren_{USERNAME}"):
                    if r_old and r_new and r_old in st.session_state.G and r_new not in st.session_state.G:
                        nx.relabel_nodes(st.session_state.G, {r_old: r_new}, copy=False)
                        st.success(f"'{r_old}' → '{r_new}'")
                        if st.session_state.autosave: safe_rerun()
    st.markdown("### Visualização 3D")
    try:
        pos = nx.spring_layout(st.session_state.G, dim=3, seed=42)
        fig = go.Figure(); fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao renderizar grafo: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "anotacoes":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Anotações com Marca-texto")
    st.info("Use ==texto== para marcar (destacar) trechos que serão realçados no PDF.")
    notes = st.text_area("Digite suas anotações (use ==texto== para destacar)", value=st.session_state.notes, height=260, key=f"notes_{USERNAME}")
    st.session_state.notes = notes
    pdf_bytes = generate_pdf_with_highlights(st.session_state.notes)
    st.download_button("Baixar Anotações (PDF)", data=pdf_bytes, file_name="anotacoes_consumidia.pdf", mime="application/pdf")
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "graficos":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Gráficos Personalizados")
    if st.session_state.df is None:
        st.warning("Carregue uma planilha na aba 'Planilha' para gerar gráficos.")
    else:
        df = st.session_state.df.copy(); cols = df.columns.tolist()
        c1, c2 = st.columns(2)
        with c1:
            eixo_x = st.selectbox("Eixo X", options=cols, key=f"x_{USERNAME}")
        with c2:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            eixo_y = st.selectbox("Eixo Y (Opcional)", options=[None] + numeric_cols, key=f"y_{USERNAME}")
        if st.button("Gerar Gráfico"):
            try:
                if eixo_y:
                    fig = px.bar(df, x=eixo_x, y=eixo_y, title=f"{eixo_y} por {eixo_x}")
                else:
                    fig = px.histogram(df, x=eixo_x, title=f"Contagem por {eixo_x}")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#d6d9dc"))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao gerar gráficos: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# BUSCA (com destaque, paginação, detalhes e inline contact corrigido)
# -------------------------
elif st.session_state.page == "busca":
    st.markdown("<div class='glass-box' style='position:relative;padding:18px;'><div class='specular'></div>", unsafe_allow_html=True)
    tab_busca, tab_favoritos = st.tabs([f"🔍 Busca Inteligente", f"⭐ Favoritos ({len(get_session_favorites())})"])

    def extract_keywords(text, n=6):
        if not text: return []
        text = re.sub(r"[^\w\s]", " ", str(text or "")).lower()
        stop = {"de","da","do","e","a","o","em","para","por","com","os","as","um","uma","que","na","no"}
        words = [w for w in text.split() if len(w) > 2 and w not in stop]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        return [w for w, _ in sorted_words][:n]

    # BUSCA tab
    with tab_busca:
        st.markdown("<style>.card{}</style>", unsafe_allow_html=True)
        col_q, col_meta, col_actions = st.columns([0.6, 0.25, 0.15])
        with col_q:
            query = st.text_input("Termo de busca", key="ui_query_search", placeholder="Digite palavras-chave — ex: autor, título, tema...")
        with col_meta:
            backups_df_tmp = collect_latest_backups()
            all_cols = []
            if backups_df_tmp is not None:
                all_cols = [c for c in backups_df_tmp.columns if c.lower() not in ['_artemis_username', 'ano']]
            search_col = st.selectbox("Buscar em", options=all_cols or ["(nenhuma planilha encontrada)"], key="ui_search_col")
        with col_actions:
            per_page = st.selectbox("Por página", options=[5, 8, 12, 20], index=1, key="ui_search_pp")
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            search_clicked = st.button("🔎 Buscar", use_container_width=True, key="ui_search_btn")

        # init session keys
        if 'search_results' not in st.session_state:
            st.session_state.search_results = pd.DataFrame()
        if 'search_page' not in st.session_state:
            st.session_state.search_page = 1
        if 'search_view_index' not in st.session_state:
            st.session_state.search_view_index = None
        if 'compose_inline' not in st.session_state:
            st.session_state.compose_inline = False

        # perform search
        if search_clicked:
            st.session_state.search_view_index = None
            if (not query) or (not all_cols):
                st.info("Digite um termo e assegure que existam backups (salve progresso).")
                st.session_state.search_results = pd.DataFrame()
                st.session_state.search_query_meta = {"col": None, "query": ""}
            else:
                norm_query = normalize_text(query)
                ser = backups_df_tmp[search_col].astype(str).apply(normalize_text)
                hits = backups_df_tmp[ser.str.contains(norm_query, na=False)]
                st.session_state.search_results = hits.reset_index(drop=True)
                st.session_state.search_query_meta = {"col": search_col, "query": query}
                st.session_state.search_page = 1

        results_df = st.session_state.search_results
        if results_df is None or results_df.empty:
            if search_clicked:
                st.info("Nenhum resultado encontrado.")
            else:
                st.markdown("<div class='small-muted'>Resultados aparecerão aqui. Salve backups para ativar a busca.</div>", unsafe_allow_html=True)
        else:
            total = len(results_df)
            page = int(st.session_state.get("search_page", 1))
            max_pages = max(1, (total + per_page - 1) // per_page)
            page = max(1, min(page, max_pages))
            st.session_state.search_page = page

            start = (page - 1) * per_page
            end = min(start + per_page, total)
            page_df = results_df.iloc[start:end]

            st.markdown(f"**{total}** resultado(s) — exibindo {start+1} a {end}. (Página {page}/{max_pages})")
            st.markdown("---")
            q_for_highlight = st.session_state.search_query_meta.get("query", "")

            for orig_i in page_df.index:
                result_data = results_df.loc[orig_i].to_dict()
                user_src = result_data.get("_artemis_username", "N/A")
                initials = "".join([p[0].upper() for p in str(user_src).split()[:2]])[:2] or "U"
                title_raw = str(result_data.get('título') or result_data.get('titulo') or '(Sem título)')
                resumo_raw = str(result_data.get('resumo') or result_data.get('abstract') or "")
                title_html = highlight_search_terms(title_raw, q_for_highlight)
                resumo_html = highlight_search_terms(resumo_raw, q_for_highlight)
                author = str(result_data.get('autor') or '')
                year = str(result_data.get('ano') or '')

                card_html = f"""
                <div class="card">
                  <div style="display:flex; gap:12px; align-items:center;">
                    <div class="avatar">{escape_html(initials)}</div>
                    <div style="flex:1;">
                      <div class="card-title">{title_html}</div>
                      <div class="small-muted">Proveniente de <strong>{escape_html(user_src)}</strong> • {escape_html(author)}</div>
                      <div style="margin-top:6px;font-size:13px;color:#e6e8ea;">{resumo_html if resumo_raw else ''}</div>
                    </div>
                    <div style="text-align:right;">
                      <div class="small-muted">{escape_html(year)}</div>
                    </div>
                  </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

                # actions: Favoritar + Ver detalhes (usamos st.button fora de forms)
                a1, a2 = st.columns([0.28, 0.72])
                with a1:
                    if st.button("⭐ Favoritar", key=f"fav_{orig_i}", use_container_width=True):
                        if add_to_favorites(result_data):
                            st.toast("Adicionado aos favoritos!", icon="⭐")
                        else:
                            st.toast("Já está nos favoritos.")
                with a2:
                    if st.button("🔎 Ver detalhes", key=f"view_{orig_i}", use_container_width=True):
                        st.session_state.search_view_index = int(orig_i)
                        st.session_state.compose_inline = False
                        safe_rerun()

            # pagination
            st.markdown("---")
            p1, p2, p3 = st.columns([0.33, 0.34, 0.33])
            with p1:
                if st.button("◀ Anterior", key="search_prev") and st.session_state.search_page > 1:
                    st.session_state.search_page = max(1, st.session_state.search_page - 1)
                    st.session_state.search_view_index = None
                    safe_rerun()
            with p2:
                st.markdown(f"**Página {st.session_state.search_page} / {max_pages}**", unsafe_allow_html=True)
            with p3:
                if st.button("Próxima ▶", key="search_next") and st.session_state.search_page < max_pages:
                    st.session_state.search_page = min(max_pages, st.session_state.search_page + 1)
                    st.session_state.search_view_index = None
                    safe_rerun()

            # DETAILS (fora do loop)
            if st.session_state.get("search_view_index") is not None:
                vi = int(st.session_state.search_view_index)
                if 0 <= vi < len(results_df):
                    det = results_df.loc[vi].to_dict()
                    origin_user = det.get("_artemis_username", "N/A")
                    st.markdown("## Detalhes do Registro")
                    st.markdown(f"**Título:** {escape_html(det.get('título') or det.get('titulo') or '(Sem título)')}")
                    st.markdown(f"**Autor:** {escape_html(det.get('autor') or '(não informado)')} • **Origem:** {escape_html(origin_user)}")
                    st.markdown(f"**Ano:** {escape_html(det.get('ano') or '')}")
                    st.markdown("---")
                    expl = []
                    if det.get("título") or det.get("titulo"):
                        expl.append("Este registro parece ser um trabalho (título) relacionado à pesquisa ou produção indicada.")
                    if det.get("autor"):
                        expl.append("O campo 'autor' identifica o responsável ou autor principal do item.")
                    if det.get("tema"):
                        expl.append("O campo 'tema' ajuda a categorizar o conteúdo e facilitar buscas similares.")
                    if det.get("resumo") or det.get("abstract"):
                        expl.append("Há um resumo/abstract — leia-o para entender melhor o conteúdo do trabalho.")
                    if not expl:
                        expl = ["Registro importado de backup — verifique os campos para confirmar informação."]
                    st.info(" ".join(expl))
                    combined = f"{det.get('título') or ''} {det.get('tema') or ''} {det.get('resumo') or det.get('abstract') or ''}"
                    keywords = extract_keywords(combined, n=8)
                    if keywords:
                        st.markdown("**Palavras-chave (sugeridas):** " + ", ".join(keywords))
                    else:
                        st.markdown("**Palavras-chave (sugeridas):** (não identificadas)")
                    st.markdown("---")
                    for k, v in det.items():
                        if k == "_artemis_username": continue
                        st.markdown(f"- **{escape_html(k)}:** {escape_html(v)}")
                    st.markdown("---")
                    st.markdown("### ✉️ Contatar autor / origem")

                    # INLINE contact form inside a st.form (usar form_submit_button para ENVIAR e CANCELAR)
                    if origin_user and origin_user != "N/A":
                        st.markdown(f"Enviar mensagem diretamente para **{escape_html(origin_user)}** sobre este registro.")
                        to_pref = origin_user
                        subj_pref = f"Sobre o registro: {det.get('título') or det.get('titulo') or '(Sem título)'}"
                        body_pref = f"Olá {origin_user},\n\nEncontrei este registro na plataforma e gostaria de conversar sobre: {det.get('título') or det.get('titulo') or ''}\n\n"
                        with st.form(key=f"inline_compose_form_{vi}", clear_on_submit=False):
                            to_fill = st.text_input("Para:", value=to_pref, key=f"inline_to_{vi}")
                            subj_fill = st.text_input("Assunto:", value=subj_pref, key=f"inline_subj_{vi}")
                            body_fill = st.text_area("Mensagem:", value=body_pref, height=180, key=f"inline_body_{vi}")
                            attach_inline = st.file_uploader("Anexar arquivo (opcional):", key=f"inline_attach_{vi}")
                            send_clicked = st.form_submit_button("✉️ Enviar mensagem agora")
                            cancel_clicked = st.form_submit_button("Cancelar envio inline")
                            if send_clicked:
                                try:
                                    send_message(USERNAME, to_fill, subj_fill, body_fill, attachment_file=attach_inline)
                                    st.success(f"Mensagem enviada para {to_fill}.")
                                    if st.session_state.autosave:
                                        # save minimal state
                                        try:
                                            data = {"notes": st.session_state.get("notes",""), "uploaded_name": st.session_state.get("uploaded_name", None), "favorites": st.session_state.get("favorites", [])}
                                            tmp = USER_STATE.with_suffix(".tmp")
                                            with tmp.open("w", encoding="utf-8") as f:
                                                json.dump(data, f, ensure_ascii=False, indent=2); f.flush(); os.fsync(f.fileno())
                                            tmp.replace(USER_STATE)
                                        except Exception:
                                            pass
                                    safe_rerun()
                                except Exception as e:
                                    st.error(f"Erro ao enviar: {e}")
                            if cancel_clicked:
                                st.info("Envio cancelado.")
                                safe_rerun()
                    else:
                        st.warning("Origem/usuário não disponível para contato.")

    # FAVORITOS tab (cards)
    with tab_favoritos:
        st.header("Seus Resultados Salvos")
        favorites = get_session_favorites()
        if not favorites:
            st.info("Você ainda não favoritou nenhum resultado.")
        else:
            _, col_clear = st.columns([0.75, 0.25])
            with col_clear:
                if action_button("Limpar Todos", "trash", "clear_favs"):
                    clear_all_favorites()
                    safe_rerun()
            st.markdown("---")
            sorted_favorites = sorted(favorites, key=lambda x: x['added_at'], reverse=True)
            q_for_highlight = st.session_state.search_query_meta.get("query", "")
            for fav in sorted_favorites:
                fav_data = fav['data'].copy()
                source_user = fav_data.pop('_artemis_username', 'N/A')
                title_raw = str(fav_data.get('título') or fav_data.get('titulo') or '(Sem título)')
                resumo_raw = str(fav_data.get('resumo') or fav_data.get('abstract') or "")
                title_html = highlight_search_terms(title_raw, q_for_highlight)
                resumo_html = highlight_search_terms(resumo_raw, q_for_highlight)
                initials = "".join([p[0].upper() for p in str(source_user).split()[:2]])[:2] or "U"
                card_html = f"""
                <div class="card">
                  <div style="display:flex; gap:12px; align-items:center;">
                    <div class="avatar">{escape_html(initials)}</div>
                    <div style="flex:1;">
                      <div class="card-title">{title_html}</div>
                      <div class="small-muted">Proveniente de <strong>{escape_html(source_user)}</strong></div>
                      <div style="margin-top:6px;font-size:13px;color:#e6e8ea;">{resumo_html if resumo_raw else ''}</div>
                    </div>
                  </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                c1, c2 = st.columns([0.75, 0.25])
                with c1:
                    if st.button("🔎 Ver detalhes", key=f"fav_view_{fav['id']}", use_container_width=True):
                        st.session_state.fav_detail = fav['data']
                        safe_rerun()
                with c2:
                    if st.button("Remover", key=f"fav_del_{fav['id']}", use_container_width=True):
                        remove_from_favorites(fav['id'])
                        safe_rerun()
            if st.session_state.get("fav_detail"):
                det = st.session_state.pop("fav_detail")
                origin_user = det.get("_artemis_username", "N/A")
                st.markdown("## Detalhes do Favorito")
                st.markdown(f"**Título:** {escape_html(det.get('título') or det.get('titulo') or '(Sem título)')}")
                st.markdown(f"**Autor:** {escape_html(det.get('autor') or '(não informado)')} • **Origem:** {escape_html(origin_user)}")
                st.markdown("---")
                st.info("Este item foi salvo em seus Favoritos. Use o botão abaixo para contatar a origem.")
                for k, v in det.items():
                    if k == "_artemis_username": continue
                    st.markdown(f"- **{escape_html(k)}:** {escape_html(v)}")
                if origin_user and origin_user != "N/A":
                    if st.button(f"✉️ Contatar {escape_html(origin_user)}", key="fav_contact_now"):
                        st.session_state.compose_open = True
                        st.session_state.compose_to = origin_user
                        st.session_state.compose_subject = f"Sobre o registro: {det.get('título') or det.get('titulo') or '(Sem título)'}"
                        st.session_state.compose_prefill = f"Olá {origin_user},\n\nEncontrei este registro nos favoritos e gostaria de falar sobre: {det.get('título') or det.get('titulo') or ''}\n\n"
                        st.session_state.page = "mensagens"
                        safe_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# MENSAGENS (cartões, 1 botão Abrir, detalhes/responder)
# -------------------------
elif st.session_state.page == "mensagens":
    st.markdown("<div class='glass-box' style='position:relative;padding:18px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.markdown("<style>.msg-card{}</style>", unsafe_allow_html=True)
    st.subheader("✉️ Central de Mensagens")
    inbox = get_user_messages(USERNAME, 'inbox')
    outbox = get_user_messages(USERNAME, 'outbox')
    tab_inbox, tab_compose, tab_sent = st.tabs([f"📥 Caixa ({sum(1 for m in inbox if not m.get('read'))})", "✍️ Escrever", f"📤 Enviadas ({len(outbox)})"])

    # INBOX
    with tab_inbox:
        st.markdown("#### Mensagens Recebidas")
        if not inbox:
            st.info("Sua caixa de entrada está vazia.")
        else:
            for m in inbox:
                mid = m.get("id")
                read = m.get("read", False)
                badge = "✅" if read else "🔵"
                subj = escape_html(m.get("subject") or "(sem assunto)")
                fromu = escape_html(m.get("from") or "anônimo")
                ts = escape_html(m.get("ts") or "")
                preview = escape_html((m.get("body") or "")[:200])
                card = f"""
                <div class="msg-card">
                  <div style="display:flex; gap:10px; align-items:center;">
                    <div style="width:44px;height:44px;border-radius:8px;background:#6c5ce7;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700">{fromu[:2].upper()}</div>
                    <div style="flex:1;">
                      <div class="msg-sub">{badge} {subj}</div>
                      <div class="msg-meta">De: <strong>{fromu}</strong> • {ts}</div>
                      <div style="margin-top:6px;color:#e6e8ea;font-size:13px;">{preview}{'...' if len(m.get('body',''))>200 else ''}</div>
                    </div>
                    <div style="width:120px;text-align:right;">
                    </div>
                  </div>
                </div>
                """
                st.markdown(card, unsafe_allow_html=True)
                # SINGLE streamlit button "Abrir" (não um botão HTML)
                if st.button("Abrir", key=f"open_inbox_{mid}"):
                    st.session_state.reply_message_id = mid
                    safe_rerun()

    # COMPOSE
    with tab_compose:
        st.markdown("### ✍️ Escrever nova mensagem")
        with st.form(key="compose_form_main", clear_on_submit=True):
            users_dict = load_users() or {}
            all_usernames = [u for u in users_dict.keys() if u != USERNAME]
            if all_usernames:
                to_user = st.selectbox("Para:", options=["(escolha)"] + all_usernames, index=0, key="compose_select_to_main")
                if to_user == "(escolha)": to_user = ""
            else:
                to_user = st.text_input("Para (username):", value=st.session_state.get("compose_to", ""))
            subj = st.text_input("Assunto:", value=st.session_state.get("compose_subject", ""))
            body = st.text_area("Mensagem:", value=st.session_state.get("compose_prefill", ""), height=200)
            attachment = st.file_uploader("Anexar arquivo (opcional):", key="compose_attach_main")
            submitted = st.form_submit_button("✉️ Enviar")
            if submitted:
                if not to_user:
                    st.error("Informe o destinatário.")
                else:
                    send_message(USERNAME, to_user, subj, body, attachment_file=attachment)
                    st.success(f"Mensagem enviada para {to_user}.")
                    if st.session_state.autosave:
                        try:
                            data = {"notes": st.session_state.get("notes",""), "uploaded_name": st.session_state.get("uploaded_name", None), "favorites": st.session_state.get("favorites", [])}
                            tmp = USER_STATE.with_suffix(".tmp")
                            with tmp.open("w", encoding="utf-8") as f:
                                json.dump(data, f, ensure_ascii=False, indent=2); f.flush(); os.fsync(f.fileno())
                            tmp.replace(USER_STATE)
                        except Exception:
                            pass
                    safe_rerun()

    # SENT
    with tab_sent:
        st.markdown("#### Mensagens Enviadas")
        if not outbox:
            st.info("Nenhuma mensagem enviada ainda.")
        else:
            for m in outbox:
                mid = m.get("id")
                subj = escape_html(m.get("subject") or "(sem assunto)")
                to = escape_html(m.get("to") or "")
                ts = escape_html(m.get("ts") or "")
                preview = escape_html((m.get("body") or "")[:200])
                card = f"""
                <div class="msg-card">
                  <div style="display:flex; gap:10px; align-items:center;">
                    <div style="width:44px;height:44px;border-radius:8px;background:#1abc9c;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700">{(to[:2]).upper()}</div>
                    <div style="flex:1;">
                      <div class="msg-sub">📤 {subj}</div>
                      <div class="msg-meta">Para: <strong>{to}</strong> • {ts}</div>
                      <div style="margin-top:6px;color:#e6e8ea;font-size:13px;">{preview}{'...' if len(m.get('body',''))>200 else ''}</div>
                    </div>
                    <div style="width:120px;text-align:right;">
                    </div>
                  </div>
                </div>
                """
                st.markdown(card, unsafe_allow_html=True)
                if st.button("Abrir", key=f"open_sent_{mid}"):
                    st.session_state.reply_message_id = mid
                    safe_rerun()

    # DETAILS / ACTIONS
    st.markdown("---")
    st.markdown("### Detalhes / Ações")
    selected = st.session_state.get("reply_message_id")
    if not selected:
        st.markdown("<div class='small-muted'>Selecione uma mensagem (Abrir) para ver detalhes e responder.</div>", unsafe_allow_html=True)
    else:
        msg = next((m for m in load_all_messages() if m.get("id") == selected), None)
        if not msg:
            st.info("Mensagem não encontrada (talvez tenha sido apagada).")
        else:
            subj = escape_html(msg.get("subject") or "(sem assunto)")
            fr = escape_html(msg.get("from") or "")
            to = escape_html(msg.get("to") or "")
            ts = escape_html(msg.get("ts") or "")
            body = escape_html(msg.get("body") or "")
            st.markdown(f"**Assunto:** {subj}")
            st.markdown(f"**De:** {fr} • **Para:** {to} • **Enviada em:** `{ts}`")
            st.markdown("---")
            st.markdown(f"<div style='padding:12px;border-radius:8px;background:rgba(255,255,255,0.01);'>{body.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
            st.markdown("---")
            if msg.get("attachment"):
                att = msg.get("attachment")
                localp = att.get("path")
                if localp and os.path.exists(localp):
                    with open(localp, "rb") as fp:
                        st.download_button(label=f"⬇️ Baixar Anexo: {att.get('name')}", data=fp, file_name=att.get('name'))
            cA, cB, cC = st.columns([0.33,0.33,0.34])
            with cA:
                if st.button("✉️ Responder", key=f"detail_reply_{selected}"):
                    st.session_state.compose_open = True
                    st.session_state.compose_to = msg.get("from")
                    st.session_state.compose_subject = f"Re: {msg.get('subject')}"
                    st.session_state.compose_prefill = f"\n\n---\nEm {msg.get('ts')}, {msg.get('from')} escreveu:\n> " + "\n> ".join(str(msg.get('body','')).split('\n'))
                    safe_rerun()
            with cB:
                if st.button("🗑️ Apagar", key=f"detail_del_{selected}"):
                    if msg.get("to") == USERNAME or msg.get("from") == USERNAME:
                        delete_message(selected, USERNAME)
                        st.toast("Mensagem apagada.")
                        st.session_state.reply_message_id = None
                        safe_rerun()
                    else:
                        st.warning("Você só pode apagar mensagens enviadas/recebidas por você.")
            with cC:
                if msg.get("to") == USERNAME and not msg.get("read", False):
                    if st.button("Marcar como lida", key=f"detail_mark_{selected}"):
                        mark_message_read(selected, USERNAME)
                        st.toast("Marcada como lida.")
                        safe_rerun()

    # Quick compose if requested
    if st.session_state.get("compose_open"):
        st.markdown("---")
        st.markdown("### ✍️ Compor Mensagem Rápida")
        with st.form("quick_compose", clear_on_submit=True):
            to_default = st.session_state.pop("compose_to", "")
            subj_default = st.session_state.pop("compose_subject", "")
            body_default = st.session_state.pop("compose_prefill", "")
            users_dict = load_users() or {}
            all_usernames = [u for u in users_dict.keys() if u != USERNAME]
            if all_usernames:
                to_user = st.selectbox("Para:", options=["(escolha)"] + all_usernames, index=0, key="compose_select_quick")
                if to_user == "(escolha)": to_user = to_default or ""
            else:
                to_user = st.text_input("Para (username):", value=to_default)
            subj = st.text_input("Assunto:", value=subj_default)
            body = st.text_area("Mensagem:", value=body_default, height=200)
            attach = st.file_uploader("Anexar arquivo (opcional):", key="compose_attach_quick")
            submitted = st.form_submit_button("✉️ Enviar")
            if submitted:
                if not to_user:
                    st.error("Informe o destinatário.")
                else:
                    send_message(USERNAME, to_user, subj, body, attachment_file=attach)
                    st.success(f"Mensagem enviada para {to_user}.")
                    st.session_state.compose_open = False
                    safe_rerun()
        if st.button("Cancelar composição"):
            st.session_state.compose_open = False
            safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)
