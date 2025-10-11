# dashboard_nugep_pqr_final_complete.py
# NUGEP-PQR — Versão final (login com CPF) + Mapa Mental 3D (separável/interativo)
# Ajustado: Lógica de criação de grafo refeita para GARANTIR a criação de linhas a partir de qualquer planilha.

import os
import re
import io
import json
import time
import random
import string
import unicodedata
import html
import math
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
st.set_page_config(page_title="NUGEP-PQR", layout="wide", initial_sidebar_state="expanded")


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
# Base CSS
# -------------------------
BASE_CSS = r"""
:root{ --glass-bg-dark: rgba(255,255,255,0.03); --muted-text-dark:#bfc6cc; }
body { transition: background-color .25s ease, color .25s ease; }
.glass-box{ border-radius:12px; padding:16px; box-shadow:0 8px 32px rgba(4,9,20,0.15); }
.card, .msg-card { border-radius:10px; padding:12px; margin-bottom:10px; }
.avatar{width:40px;height:40px;border-radius:8px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;margin-right:8px}
.small-muted{font-size:12px;}
.card-title{font-weight:700;font-size:15px}
.card-mark{ background: rgba(255,255,0,0.12); padding: 0 2px; border-radius:2px; }
/* Estilos para botões interativos */
.stButton>button, .stDownloadButton>button {
    transition: transform 0.1s ease, opacity 0.1s ease, background-color 0.15s ease !important;
}
.stButton>button:active, .stDownloadButton>button:active {
    transform: scale(0.97);
    opacity: 0.8;
}
"""

# default dark CSS
DEFAULT_CSS = r"""
.css-1d391kg { background: linear-gradient(180deg,#071428 0%, #031926 100%) !important; }
.glass-box{ background: rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.04); box-shadow:0 8px 32px rgba(4,9,20,0.5); }
.stButton>button, .stDownloadButton>button{ background:transparent !important; color:#bfc6cc !important; border:1px solid rgba(255,255,255,0.06) !important; padding:8px 12px !important; border-radius:10px !important; }
.stButton>button:hover, .stDownloadButton>button:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
}
.card, .msg-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:12px; margin-bottom:10px; border:1px solid rgba(255,255,255,0.03); }
.avatar{color:#fff;background:#6c5ce7}
.small-muted{color:#bfc6cc;}
.card-title{color:#fff}
"""

# inject base CSS
st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)
# inject dark default
st.markdown(f"<style>{DEFAULT_CSS}</style>", unsafe_allow_html=True)

# header
st.markdown("<div style='max-width:1100px;margin:18px auto 8px;text-align:center;'><h1 style='font-weight:800;font-size:40px; background:linear-gradient(90deg,#8e44ad,#2979ff,#1abc9c,#ff8a00); -webkit-background-clip:text; color:transparent; margin:0;'>NUGEP-PQR</h1></div>", unsafe_allow_html=True)

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

def apply_global_styles(font_scale=1.0):
    """Applies global CSS for font scaling and ensures dark theme body styles."""
    try:
        # Static dark theme body style
        dark_body_style = "<style>body { background-color: #071428; color: #d6d9dc; }</style>"
        st.markdown(dark_body_style, unsafe_allow_html=True)

        # Dynamic font scaling
        font_css = f"html {{ font-size: {font_scale * 100}%; }}"
        st.markdown(f"<style>{font_css}</style>", unsafe_allow_html=True)
    except Exception:
        pass

# helper: render credential box with copy & download
def _render_credentials_box(username, password, note=None, key_prefix="cred"):
    st.markdown("---")
    st.success("Usuário criado com sucesso — anote/guarde a senha abaixo:")
    col1, col2 = st.columns([3,1])
    with col1:
        st.text_input("CPF", value=username, key=f"{key_prefix}_user", disabled=True)
        st.text_input("Senha gerada", value=password, key=f"{key_prefix}_pwd", disabled=True)
        if note:
            st.info(note)
    with col2:
        creds_txt = f"cpf: {username}\npassword: {password}\n"
        st.download_button("⬇️ Baixar credenciais", data=creds_txt, file_name=f"credenciais_{username}.txt", mime="text/plain")
        js = f"""
        <script>
        function copyToClipboard_{key_prefix}(){{
            navigator.clipboard.writeText(`cpf: {username}\\npassword: {password}`);
            const el = document.getElementById('copy_hint_{key_prefix}');
            if(el) el.innerText = 'Copiado!';
        }}
        </script>
        <button onclick="copyToClipboard_{key_prefix}()">📋 Copiar para área de transferência</button>
        <div id='copy_hint_{key_prefix}' style='margin-top:6px;font-size:13px;color:#bfc6cc'></div>
        """
        st.markdown(js, unsafe_allow_html=True)
    st.markdown("---")


# -------------------------
# load/save users (atomic)
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
            f.flush(); os.fsync(f.fileno())
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

def criar_grafo(df, silent=False):
    """
    Função à prova de falhas para criar um grafo a partir de um DataFrame.
    Cria um nó central para cada LINHA da planilha e conecta todas as células
    daquela linha a este nó central. GARANTE a criação de linhas.
    """
    G = nx.Graph()
    if df is None:
        return G

    created_edges = 0
    # Itera em cada linha da planilha
    for index, row in df.iterrows():
        # Cria um nó central para representar a linha (ex: "Registro 1")
        row_node_id = f"Registro {index + 1}"
        G.add_node(row_node_id, tipo="Registro", label=f"Registro {index + 1}")

        # Itera em cada célula da linha
        for col_name, cell_value in row.items():
            val = str(cell_value or '').strip()
            if val:
                # Cria um nó para o conteúdo da célula
                tipo = str(col_name).strip().capitalize()
                node_id = f"{tipo}: {val}"
                G.add_node(node_id, tipo=tipo, label=val)
                
                # Conecta o nó da célula ao nó central da linha
                G.add_edge(row_node_id, node_id)
                created_edges += 1

    if not silent:
        if created_edges > 0:
            st.success(f"Grafo criado com sucesso, com {created_edges} linhas de conexão.")
        else:
            st.info("Grafo de pontos criado, mas a planilha parece estar vazia ou não gerou conexões.")
            
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
# Session defaults & settings
# -------------------------
_defaults = {
    "authenticated": False, "username": None, "user_obj": None, "df": None,
    "G": nx.Graph(), "notes": "", "autosave": False, "page": "planilha",
    "restored_from_saved": False, "favorites": [], "reply_message_id": None,
    "search_results": pd.DataFrame(), "search_page": 1, "search_query_meta": {"col": None,"query":""},
    "search_view_index": None, "compose_inline": False, "compose_open": False,
    "last_backup_path": None,
    "settings": {
        "plot_height": 720,
        "font_scale": 1.0,
        "node_opacity": 0.95,
    }
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def get_settings():
    return st.session_state.get("settings", _defaults["settings"])

def clean_for_json(d):
    """
    Função recursiva para 'limpar' um dicionário ou lista, convertendo
    tipos de dados do numpy/pandas para tipos nativos do Python que podem
    ser salvos em JSON.
    """
    if isinstance(d, dict):
        return {k: clean_for_json(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_for_json(i) for i in d]
    elif isinstance(d, (np.int64, np.int32, np.int8)):
        return int(d)
    elif isinstance(d, (np.float64, np.float32)):
        return None if np.isnan(d) else float(d)
    elif pd.isna(d):
        return None
    else:
        return d

def save_user_state_minimal(USER_STATE):
    """
    Salva o estado essencial do usuário em um arquivo JSON.
    Agora inclui limpeza de dados e feedback de erro na tela.
    """
    try:
        data = {
            "notes": st.session_state.get("notes",""),
            "uploaded_name": st.session_state.get("uploaded_name", None),
            "favorites": st.session_state.get("favorites", []),
            "settings": st.session_state.get("settings", {}),
            "last_backup_path": st.session_state.get("last_backup_path")
        }
        # Limpa os dados para garantir que são compatíveis com JSON
        clean_data = clean_for_json(data)

        tmp = USER_STATE.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        tmp.replace(USER_STATE)
        return True
    except Exception as e:
        # Mostra o erro diretamente na interface do usuário
        st.error(f"FALHA AO SALVAR O ESTADO: {e}")
        print(f"Erro ao salvar estado do usuário: {e}") # Mantém no log para debug
        return False


# -------------------------
# Authentication UI (local fallback)
# -------------------------
if not st.session_state.authenticated:
    st.markdown("<div class='glass-box auth' style='max-width:1100px;margin:0 auto;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Acesso — Faça login ou cadastre-se")
    tabs = st.tabs(["Entrar", "Cadastrar"])

    with tabs[0]:
        login_cpf = st.text_input("CPF", key="ui_login_user")
        login_pass = st.text_input("Senha", type="password", key="ui_login_pass")

        users = load_users() or {}
        if not users:
            admin_user = "admin"
            admin_pwd = "admin123"
            users[admin_user] = {"name": "Administrador", "scholarship": "Admin", "password": admin_pwd, "created_at": datetime.utcnow().isoformat()}
            save_users(users)
            st.warning("Nenhum usuário local encontrado. Um usuário administrativo foi criado temporariamente.")
            st.session_state.new_user_created = {"user": admin_user, "pwd": admin_pwd, "note": "Este é um usuário administrativo temporário. Para testes, use 'admin' como CPF."}

        if st.button("Entrar", "btn_login_main"):
            users = load_users() or {}
            if login_cpf in users and users[login_cpf].get("password") == login_pass:
                st.session_state.authenticated = True
                st.session_state.username = login_cpf
                st.session_state.user_obj = users[login_cpf]
                st.success("Login efetuado (local).")
                safe_rerun()
            else:
                st.warning("CPF/Senha inválidos (local).")

        if st.session_state.get("new_user_created"):
            nu = st.session_state.get("new_user_created")
            _render_credentials_box(nu["user"], nu["pwd"], note=nu.get("note",""), key_prefix="admin_fallback")
            if st.button("Entendido — fechar aviso", key="close_admin_fallback"):
                st.session_state.pop("new_user_created", None)
                safe_rerun()

    with tabs[1]:
        reg_name = st.text_input("Nome completo", key="ui_reg_name")
        reg_bolsa = st.selectbox("Tipo de bolsa", ["IC - Iniciação Científica", "BIA - Bolsa de Incentivo Acadêmico", "Extensão", "Doutorado"], key="ui_reg_bolsa")
        reg_cpf = st.text_input("CPF", key="ui_reg_user")
        reg_pass = st.text_input("Crie sua senha", type="password", key="ui_reg_pass")
        reg_pass_confirm = st.text_input("Confirme sua senha", type="password", key="ui_reg_pass_confirm")

        if st.button("Cadastrar", "btn_register_main"):
            new_cpf = (reg_cpf or "").strip()
            new_pass = (reg_pass or "").strip()

            if not new_cpf:
                st.warning("Informe um CPF válido.")
            elif not new_pass:
                st.warning("A senha não pode estar em branco.")
            elif len(new_pass) < 6:
                st.warning("A senha deve ter pelo menos 6 caracteres.")
            elif new_pass != reg_pass_confirm:
                st.error("As senhas não coincidem. Tente novamente.")
            else:
                users = load_users() or {}
                if new_cpf in users:
                    st.warning("CPF já cadastrado (local).")
                else:
                    users[new_cpf] = {"name": reg_name or new_cpf, "scholarship": reg_bolsa, "password": new_pass, "created_at": datetime.utcnow().isoformat()}
                    ok = save_users(users)
                    if ok:
                        st.success("Usuário cadastrado com sucesso! Você já pode fazer o login na aba 'Entrar'.")
                        if "new_user_created" in st.session_state:
                            del st.session_state["new_user_created"]
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

# restore per-user saved state
if not st.session_state.restored_from_saved and USER_STATE.exists():
    try:
        with USER_STATE.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        # Restore simple values
        st.session_state.notes = meta.get("notes", st.session_state.notes)
        st.session_state.uploaded_name = meta.get("uploaded_name", st.session_state.get("uploaded_name"))
        st.session_state.favorites = meta.get("favorites", st.session_state.favorites)
        st.session_state.last_backup_path = meta.get("last_backup_path", st.session_state.last_backup_path)
        
        # Restore settings
        if "settings" in meta:
            st.session_state.settings.update(meta.get("settings", {}))
        
        # *** NEW: AUTOMATICALLY RESTORE SPREADSHEET AND GRAPH ***
        backup_path = st.session_state.get("last_backup_path")
        if backup_path and os.path.exists(backup_path):
            try:
                df = pd.read_csv(backup_path)
                st.session_state.df = df
                st.session_state.G = criar_grafo(df, silent=True) # silent=True to avoid messages on login
                st.toast(f"Planilha '{os.path.basename(backup_path)}' restaurada automaticamente.", icon="📄")
            except Exception as e:
                st.error(f"Falha ao restaurar o backup da sua planilha: {e}")
                st.session_state.last_backup_path = None # Invalidate bad path
        
        st.session_state.restored_from_saved = True
        st.toast("Progresso anterior restaurado.", icon="👍")
    except Exception as e:
        # CRITICAL: Show error if loading fails
        st.error(f"Erro ao restaurar seu progresso: o arquivo de estado pode estar corrompido. Erro: {e}")


# apply theme and font CSS based on settings immediately
s = get_settings()
apply_global_styles(s.get("font_scale", 1.0))

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
    st.markdown(f"<div style='color:var(--muted-text-dark);font-weight:700;padding-top:8px;'>Usuário: {USER_OBJ.get('name','')} — {USER_OBJ.get('scholarship','')}</div>", unsafe_allow_html=True)
with top2:
    nav_right1, nav_right2, nav_right3 = st.columns([1,1,1])
    with nav_right1:
        st.session_state.autosave = st.checkbox("Auto-save", value=st.session_state.autosave, key="ui_autosave")
    with nav_right2:
        if st.button("💾 Salvar", key="btn_save_now", use_container_width=True):
            ok = save_user_state_minimal(USER_STATE)
            if ok: 
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.success(f"Progresso salvo com sucesso às {timestamp}.")
    with nav_right3:
        if st.button("🚪 Sair", key="btn_logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_obj = None
            st.session_state.reply_message_id = None
            safe_rerun()
st.markdown("</div>", unsafe_allow_html=True)

# Navigation buttons (inclui Configurações)
st.markdown("<div style='margin-top:-20px'>", unsafe_allow_html=True)
nav_buttons = {
    "planilha": "📄 Planilha",
    "mapa": "🞠 Mapa",
    "anotacoes": "📝 Anotações",
    "graficos": "📊 Gráficos",
    "busca": "🔍 Busca",
    "mensagens": mens_label,
    "config": "⚙️ Configurações"
}
nav_cols = st.columns(len(nav_buttons))
for i, (page_key, page_label) in enumerate(nav_buttons.items()):
    with nav_cols[i]:
        if st.button(page_label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state.page = page_key
            st.session_state.reply_message_id = None
            safe_rerun()
st.markdown("</div></div><hr>", unsafe_allow_html=True)

# -------------------------
# Page: Planilha
# -------------------------
if st.session_state.page == "planilha":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Planilha / Backup")
    col1, col2 = st.columns([1,3])
    with col1:
        # Este botão agora só restaura o estado que não inclui a planilha (que já carrega sozinha)
        if st.button("Restaurar estado salvo", key="btn_restore_state"):
            if USER_STATE.exists():
                st.info("O estado da sua planilha já é restaurado automaticamente ao entrar. Este botão recarrega outras informações como anotações e favoritos.")
                # Força um re-run para garantir que o processo de load no topo da página seja re-executado.
                st.session_state.restored_from_saved = False 
                safe_rerun()
            else:
                st.info("Nenhum estado salvo encontrado.")
    with col2:
        # Mostra o caminho do backup que está em uso no momento
        current_backup_path = st.session_state.get("last_backup_path")
        if current_backup_path and os.path.exists(current_backup_path):
            st.write("Backup CSV em uso:")
            st.text(os.path.basename(current_backup_path))
            with open(current_backup_path, "rb") as fp:
                st.download_button("⬇ Baixar backup CSV", data=fp, file_name=os.path.basename(current_backup_path), mime="text/csv")
        else:
            st.write("Nenhum backup de planilha ativo. Carregue um arquivo para começar.")

    uploaded = st.file_uploader("Carregue .csv ou .xlsx para criar um novo mapa ou substituir o atual", type=["csv", "xlsx"], key=f"u_{USERNAME}")
    if uploaded:
        try:
            df = read_spreadsheet(uploaded)
            st.session_state.df = df
            st.session_state.uploaded_name = uploaded.name
            st.session_state.G = criar_grafo(df) # Usa a nova função robusta
            
            # Cria o backup da planilha recém-carregada
            try:
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                safe_name = re.sub(r"[^\w\-_.]", "_", uploaded.name)
                backup_filename = f"{safe_name}_{ts}.csv"
                p = BACKUPS_DIR / USERNAME
                p.mkdir(parents=True, exist_ok=True)
                path = p / backup_filename
                df.to_csv(path, index=False, encoding="utf-8")
                
                # ATUALIZA o caminho do backup em uso e salva o estado se autosave estiver ativo
                st.session_state.last_backup_path = str(path)
                st.success(f"Backup '{backup_filename}' criado com sucesso.")
                if st.session_state.autosave:
                    save_user_state_minimal(USER_STATE)

            except Exception as e:
                st.error(f"Erro ao salvar backup automático da planilha: {e}")

        except Exception as e:
            st.error(f"Erro ao ler a planilha: {e}")

    if st.session_state.df is not None:
        st.write("Visualização da planilha em uso:")
        st.dataframe(st.session_state.df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: Mapa (3D interativo, separável, edições por nó)
# -------------------------
elif st.session_state.page == "mapa":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Mapa Mental 3D")

    with st.expander("Editar Nós do Mapa"):
        left, right = st.columns([2,1])
        with left:
            new_node = st.text_input("Nome do novo nó", key=f"nm_name_{USERNAME}")
            new_tipo = st.selectbox("Tipo", ["Outro", "Autor", "Título", "Ano", "Tema"], key=f"nm_tipo_{USERNAME}")
            connect_to = st.selectbox("Conectar a (opcional)", ["Nenhum"] + list(st.session_state.G.nodes), key=f"nm_connect_{USERNAME}")
            if st.button("Adicionar nó", key=f"btn_add_{USERNAME}"):
                n = new_node.strip()
                if not n:
                    st.warning("Nome inválido.")
                elif n in st.session_state.G.nodes:
                    st.warning("Nó já existe.")
                else:
                    st.session_state.G.add_node(n, tipo=new_tipo, label=n)
                    if connect_to != "Nenhum":
                        st.session_state.G.add_edge(n, connect_to)
                    st.success(f"Nó '{n}' adicionado.")
                    if st.session_state.autosave: save_user_state_minimal(USER_STATE)
        with right:
            del_n = st.selectbox("Excluir nó", [""] + list(st.session_state.G.nodes), key=f"del_{USERNAME}")
            if st.button("Excluir nó", key=f"btn_del_{USERNAME}"):
                if del_n and del_n in st.session_state.G:
                    st.session_state.G.remove_node(del_n)
                    st.success(f"Nó '{del_n}' removido.")
                    if st.session_state.autosave: save_user_state_minimal(USER_STATE)
            st.markdown("---")
            r_old = st.selectbox("Renomear: selecione nó", [""] + list(st.session_state.G.nodes), key=f"r_old_{USERNAME}")
            r_new = st.text_input("Novo nome", key=f"r_new_{USERNAME}")
            if st.button("Renomear", key=f"btn_ren_{USERNAME}"):
                if r_old and r_new and r_old in st.session_state.G and r_new not in st.session_state.G:
                    nx.relabel_nodes(st.session_state.G, {r_old: r_new}, copy=False)
                    st.success(f"'{r_old}' → '{r_new}'")
                    if st.session_state.autosave: save_user_state_minimal(USER_STATE)

    st.markdown("### Visualização 3D")

    G = st.session_state.G or nx.Graph()

    show_labels = st.checkbox("Mostrar rótulos fixos (pode sobrepor)", value=False, key=f"show_labels_{USERNAME}")

    # Diagnóstico: Informar o status do grafo atual
    st.info(f"O grafo atual tem **{G.number_of_nodes()}** nós (pontos) e **{G.number_of_edges()}** arestas (linhas).")


    try:
        if G.number_of_nodes() == 0:
            st.warning("Para começar, adicione nós na seção 'Editar Nós do Mapa' acima ou carregue uma planilha.")
        elif G.number_of_edges() == 0 and G.number_of_nodes() > 0:
            st.warning("O mapa não tem linhas porque nenhum nó está conectado. Conecte os nós ao criá-los ou carregue uma planilha com dados.")

        if G.number_of_nodes() > 0:
            # 1. Calculate layout once
            k_val = 2.0 / math.sqrt(G.number_of_nodes()) if G.number_of_nodes() > 0 else None
            pos = nx.spring_layout(G, dim=3, seed=42, iterations=200, k=k_val)

            # 2. Prepare ONE trace for ALL edges
            edge_x, edge_y, edge_z = [], [], []
            for u, v in G.edges():
                x0, y0, z0 = pos.get(u, (0,0,0))
                x1, y1, z1 = pos.get(v, (0,0,0))
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])

            edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='#888', width=2), hoverinfo='none')

            # 3. Prepare ONE trace for ALL nodes
            node_x, node_y, node_z = [], [], []
            node_colors, node_sizes, node_texts = [], [], []
            
            tipo_order = ["Registro", "Autor", "Título", "Ano", "Tema", "Outro"]
            palette = px.colors.qualitative.Plotly
            tipo_color_map = {t: palette[i % len(palette)] for i, t in enumerate(tipo_order)}
            
            for node, data in G.nodes(data=True):
                x, y, z = pos.get(node, (0,0,0))
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                
                node_tipo = data.get('tipo', 'Outro')
                node_colors.append(tipo_color_map.get(node_tipo, '#cccccc'))
                
                degree = G.degree(node)
                node_sizes.append(max(8, (degree + 1) * 8))
                
                label = data.get("label", node)
                hover_text = f"<b>{escape_html(label)}</b><br>Tipo: {escape_html(node_tipo)}<br>Grau: {degree}"
                node_texts.append(hover_text)
                
            node_trace = go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers+text' if show_labels else 'markers',
                text=[d.get('label', '') for n, d in G.nodes(data=True)] if show_labels else None,
                textposition="top center",
                hoverinfo='text',
                hovertext=node_texts,
                marker=dict(color=node_colors, size=node_sizes, line_width=0.5)
            )

            # 4. Create figure with ONLY TWO traces: lines first, then nodes on top
            fig = go.Figure(data=[edge_trace, node_trace])

            # 5. Apply layout
            fig.update_layout(
                height=int(get_settings().get("plot_height", 720)),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    aspectmode='auto'
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao renderizar o grafo: {e}")
        st.exception(e)

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# Page: Anotações
# -------------------------
elif st.session_state.page == "anotacoes":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Anotações com Marca-texto")
    st.info("Use ==texto== para marcar (destacar) trechos que serão realçados no PDF.")
    notes = st.text_area("Digite suas anotações (use ==texto== para destacar)", value=st.session_state.notes, height=260, key=f"notes_{USERNAME}")
    st.session_state.notes = notes
    pdf_bytes = generate_pdf_with_highlights(st.session_state.notes)
    st.download_button("Baixar Anotações (PDF)", data=pdf_bytes, file_name="anotacoes_nugep_pqr.pdf", mime="application/pdf")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: Gráficos
# -------------------------
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
                # Fixed dark theme
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#d6d9dc"))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao gerar gráficos: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: Busca (mantido)
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

        if 'search_results' not in st.session_state: st.session_state.search_results = pd.DataFrame()
        if 'search_page' not in st.session_state: st.session_state.search_page = 1
        if 'search_view_index' not in st.session_state: st.session_state.search_view_index = None
        if 'compose_inline' not in st.session_state: st.session_state.compose_inline = False

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
            if search_clicked: st.info("Nenhum resultado encontrado.")
            else: st.markdown("<div class='small-muted'>Resultados aparecerão aqui. Salve backups para ativar a busca.</div>", unsafe_allow_html=True)
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
                        <div style="text-align:right;"><div class="small-muted">{escape_html(year)}</div></div>
                    </div>
                </div>"""
                st.markdown(card_html, unsafe_allow_html=True)
                a1, a2 = st.columns([0.28, 0.72])
                with a1:
                    if st.button("⭐ Favoritar", key=f"fav_{orig_i}", use_container_width=True):
                        if add_to_favorites(result_data): st.toast("Adicionado aos favoritos!", icon="⭐")
                        else: st.toast("Já está nos favoritos.")
                with a2:
                    if st.button("🔎 Ver detalhes", key=f"view_{orig_i}", use_container_width=True):
                        st.session_state.search_view_index = int(orig_i)
                        st.session_state.compose_inline = False
                        safe_rerun()

            st.markdown("---")
            p1, p2, p3 = st.columns([0.33, 0.34, 0.33])
            with p1:
                if st.button("◀ Anterior", key="search_prev", disabled=(st.session_state.search_page <= 1)):
                    st.session_state.search_page -= 1
                    st.session_state.search_view_index = None
                    safe_rerun()
            with p2:
                st.markdown(f"<div style='text-align:center; padding-top:8px'><b>Página {st.session_state.search_page} / {max_pages}</b></div>", unsafe_allow_html=True)
            with p3:
                if st.button("Próxima ▶", key="search_next", disabled=(st.session_state.search_page >= max_pages)):
                    st.session_state.search_page += 1
                    st.session_state.search_view_index = None
                    safe_rerun()

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
                    st.info("Análise do registro: " + " ".join(filter(None, [
                        "Parece ser um trabalho acadêmico." if det.get("título") else None,
                        "Identifica o autor principal." if det.get("autor") else None,
                        "Categorizado por tema." if det.get("tema") else None,
                        "Contém um resumo para leitura rápida." if det.get("resumo") or det.get("abstract") else None
                    ])) or "Registro importado de backup.")
                    keywords = extract_keywords(f"{det.get('título') or ''} {det.get('tema') or ''} {det.get('resumo') or det.get('abstract') or ''}", n=8)
                    st.markdown(f"**Palavras-chave (sugeridas):** {', '.join(keywords) if keywords else '(não identificadas)'}")
                    st.markdown("---")
                    for k, v in det.items():
                        if k != "_artemis_username": st.markdown(f"- **{escape_html(k)}:** {escape_html(v)}")
                    st.markdown("---")
                    st.markdown("### ✉️ Contatar autor / origem")
                    if origin_user and origin_user != "N/A":
                        st.markdown(f"Enviar mensagem diretamente para **{escape_html(origin_user)}** sobre este registro.")
                        with st.form(key=f"inline_compose_form_{vi}"):
                            to_fill = st.text_input("Para:", value=origin_user, key=f"inline_to_{vi}")
                            subj_fill = st.text_input("Assunto:", value=f"Sobre o registro: {det.get('título') or det.get('titulo') or '(Sem título)'}", key=f"inline_subj_{vi}")
                            body_fill = st.text_area("Mensagem:", value=f"Olá {origin_user},\n\nEncontrei este registro na plataforma e gostaria de conversar sobre: {det.get('título') or det.get('titulo') or ''}\n\n", height=180, key=f"inline_body_{vi}")
                            attach_inline = st.file_uploader("Anexar arquivo (opcional):", key=f"inline_attach_{vi}")
                            c1, c2 = st.columns(2)
                            with c1:
                                if st.form_submit_button("✉️ Enviar mensagem agora"):
                                    try:
                                        send_message(USERNAME, to_fill, subj_fill, body_fill, attachment_file=attach_inline)
                                        st.success(f"Mensagem enviada para {to_fill}.")
                                        time.sleep(2)
                                        safe_rerun()
                                    except Exception as e:
                                        st.error(f"Erro ao enviar: {e}")
                            with c2:
                                if st.form_submit_button("Cancelar"):
                                    st.session_state.search_view_index = None
                                    safe_rerun()
                    else:
                        st.warning("Origem/usuário não disponível para contato.")

    # FAVORITOS tab
    with tab_favoritos:
        st.header("Seus Resultados Salvos")
        favorites = get_session_favorites()
        if not favorites:
            st.info("Você ainda não favoritou nenhum resultado.")
        else:
            if st.button("🗑️ Limpar Todos", key="clear_favs"):
                clear_all_favorites()
                safe_rerun()
            st.markdown("---")
            for fav in sorted(favorites, key=lambda x: x['added_at'], reverse=True):
                fav_data = fav['data']
                source_user = fav_data.get('_artemis_username', 'N/A')
                title_raw = str(fav_data.get('título') or fav_data.get('titulo') or '(Sem título)')
                resumo_raw = str(fav_data.get('resumo') or fav_data.get('abstract') or "")
                initials = "".join([p[0].upper() for p in str(source_user).split()[:2]])[:2] or "U"
                card_html = f"""
                <div class="card">
                    <div style="display:flex; gap:12px; align-items:center;">
                        <div class="avatar">{escape_html(initials)}</div>
                        <div style="flex:1;">
                            <div class="card-title">{escape_html(title_raw)}</div>
                            <div class="small-muted">Proveniente de <strong>{escape_html(source_user)}</strong></div>
                            {f'<div style="margin-top:6px;font-size:13px;color:#e6e8ea;">{escape_html(resumo_raw)}</div>' if resumo_raw else ''}
                        </div>
                    </div>
                </div>"""
                st.markdown(card_html, unsafe_allow_html=True)
                c1, c2 = st.columns([0.75, 0.25])
                with c1:
                    if st.button("🔎 Ver detalhes", key=f"fav_view_{fav['id']}", use_container_width=True):
                        st.session_state.fav_detail = fav['data']
                with c2:
                    if st.button("Remover", key=f"fav_del_{fav['id']}", use_container_width=True):
                        remove_from_favorites(fav['id'])
                        safe_rerun()
            if 'fav_detail' in st.session_state and st.session_state.fav_detail:
                det = st.session_state.pop("fav_detail")
                origin_user = det.get("_artemis_username", "N/A")
                st.markdown("## Detalhes do Favorito")
                st.markdown(f"**Título:** {escape_html(det.get('título') or det.get('titulo') or '(Sem título)')}")
                st.markdown(f"**Autor:** {escape_html(det.get('autor') or '(não informado)')} • **Origem:** {escape_html(origin_user)}")
                st.markdown("---")
                for k, v in det.items():
                    if k != "_artemis_username": st.markdown(f"- **{escape_html(k)}:** {escape_html(v)}")
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
# Page: Mensagens (mantido)
# -------------------------
elif st.session_state.page == "mensagens":
    st.markdown("<div class='glass-box' style='position:relative;padding:18px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("✉️ Central de Mensagens")
    inbox = get_user_messages(USERNAME, 'inbox')
    outbox = get_user_messages(USERNAME, 'outbox')
    tab_inbox, tab_compose, tab_sent = st.tabs([f"📥 Caixa de Entrada ({sum(1 for m in inbox if not m.get('read'))})", "✍️ Escrever", f"📤 Enviadas ({len(outbox)})"])

    with tab_inbox:
        if not inbox: st.info("Sua caixa de entrada está vazia.")
        else:
            for m in inbox:
                badge = "🔵" if not m.get('read') else ""
                card = f"""<div class="msg-card" style="border-left: 3px solid {'#6c5ce7' if not m.get('read') else 'transparent'};">
                    <div style="display:flex; gap:10px; align-items:center;">
                        <div style="flex:1;">
                            <div class="msg-sub">{badge} {escape_html(m.get("subject"))}</div>
                            <div class="msg-meta">De: <strong>{escape_html(m.get("from"))}</strong> • {escape_html(m.get("ts"))}</div>
                        </div>
                        <div style="width:120px;text-align:right;"></div>
                    </div></div>"""
                st.markdown(card, unsafe_allow_html=True)
                if st.button("Abrir", key=f"open_inbox_{m.get('id')}"):
                    st.session_state.reply_message_id = m.get('id')
                    mark_message_read(m.get('id'), USERNAME)
                    safe_rerun()

    with tab_compose:
        with st.form(key="compose_form_main", clear_on_submit=True):
            users_dict = load_users() or {}
            all_usernames = [u for u in users_dict.keys() if u != USERNAME]
            to_user = st.selectbox("Para:", [""] + all_usernames, key="compose_to")
            subj = st.text_input("Assunto:", value=st.session_state.pop("compose_subject", ""))
            body = st.text_area("Mensagem:", value=st.session_state.pop("compose_prefill", ""), height=200)
            attachment = st.file_uploader("Anexar arquivo (opcional):", key="compose_attach_main")
            if st.form_submit_button("✉️ Enviar"):
                if not to_user: st.error("Informe o destinatário.")
                else:
                    send_message(USERNAME, to_user, subj, body, attachment)
                    st.success(f"Mensagem enviada para {to_user}.")
                    time.sleep(1); safe_rerun()

    with tab_sent:
        if not outbox: st.info("Nenhuma mensagem enviada ainda.")
        else:
            for m in outbox:
                card = f"""<div class="msg-card">
                    <div style="display:flex; gap:10px; align-items:center;">
                        <div style="flex:1;">
                            <div class="msg-sub">📤 {escape_html(m.get("subject"))}</div>
                            <div class="msg-meta">Para: <strong>{escape_html(m.get("to"))}</strong> • {escape_html(m.get("ts"))}</div>
                        </div></div></div>"""
                st.markdown(card, unsafe_allow_html=True)
                if st.button("Ver Enviada", key=f"open_sent_{m.get('id')}"):
                    st.session_state.reply_message_id = m.get('id')
                    safe_rerun()

    if st.session_state.get("reply_message_id"):
        selected_id = st.session_state.reply_message_id
        msg = next((m for m in load_all_messages() if m.get("id") == selected_id), None)
        if msg:
            st.markdown("---")
            st.markdown(f"**Assunto:** {escape_html(msg.get('subject'))}")
            st.markdown(f"**De:** {escape_html(msg.get('from'))} | **Para:** {escape_html(msg.get('to'))}")
            st.markdown(f"<div style='padding:12px;border-radius:8px;background:rgba(255,255,255,0.01);margin-top:10px;'>{escape_html(msg.get('body')).replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
            if msg.get("attachment"):
                att = msg.get("attachment")
                if att.get("path") and os.path.exists(att["path"]):
                    with open(att["path"], "rb") as fp:
                        st.download_button(f"⬇️ Baixar: {att.get('name')}", fp, att.get('name'))
            c1, c2, c3 = st.columns(3)
            if msg.get("to") == USERNAME:
                with c1:
                    if st.button("Responder", key=f"detail_reply_{selected_id}"):
                        st.session_state.compose_subject = f"Re: {msg.get('subject')}"
                        st.session_state.compose_prefill = f"\n\n---\nEm {msg.get('ts')}, {msg.get('from')} escreveu:\n> " + "\n> ".join(str(msg.get('body','')).split('\n'))
                        st.session_state.page = "mensagens"
                        safe_rerun()
            with c2:
                if st.button("🗑️ Apagar", key=f"detail_del_{selected_id}"):
                    delete_message(selected_id, USERNAME)
                    st.toast("Mensagem apagada.")
                    st.session_state.reply_message_id = None
                    safe_rerun()
            with c3:
                if st.button("Fechar", key=f"detail_close_{selected_id}"):
                    st.session_state.reply_message_id = None
                    safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: Configurações / Acessibilidade (SIMPLIFICADO)
# -------------------------
elif st.session_state.page == "config":
    st.markdown("<div class='glass-box' style='position:relative;padding:12px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("⚙️ Configurações")
    s = get_settings()

    plot_height = st.number_input("Altura do gráfico (px)", value=int(s.get("plot_height",720)), step=10, key="cfg_plot_height")
    font_scale = st.slider("Escala de fonte (aplicada a todo o programa)", min_value=0.7, max_value=2.0, value=float(s.get("font_scale",1.0)), step=0.1, key="cfg_font_scale")

    if st.button("Aplicar configurações"):
        st.session_state.settings["plot_height"] = int(plot_height)
        st.session_state.settings["font_scale"] = float(font_scale)

        ok = save_user_state_minimal(USER_STATE)
        apply_global_styles(font_scale)

        if ok:
            st.success("Configurações aplicadas e salvas.")
        # O erro já é mostrado dentro da função save_user_state_minimal

        time.sleep(0.5)
        safe_rerun()

    st.markdown("---")
    st.markdown("**Acessibilidade**")
    st.markdown("- Use *Escala de fonte* para aumentar ou diminuir o tamanho do texto em todo o programa.")
    st.markdown("- O programa utiliza um tema escuro fixo para garantir bom contraste.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Fallback (should not happen)
# -------------------------
else:
    st.info("Página não encontrada — selecione uma aba no topo.")
