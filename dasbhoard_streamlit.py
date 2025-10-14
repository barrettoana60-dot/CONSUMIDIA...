# app_nugep_pqr_full.py
# NUGEP-PQR — Versão completa com correções nas Recomendações e CSS do título

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
import networkx as nx
from fpdf import FPDF

from streamlit_agraph import agraph, Node, Edge, Config
import requests

# optional ML libs (silenciosamente não-fatal)
try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    joblib = None
    TfidfVectorizer = None
    cosine_similarity = None

# tentativa segura de importar matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception as _e:
    plt = None
    MATPLOTLIB_AVAILABLE = False
    print(f"matplotlib não disponível: {_e}")

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
/* evita barra azul larga: título como texto inline */
.card-title{display:inline-block;font-weight:700;font-size:15px; background: linear-gradient(90deg,#0077ff,#00a3ff); -webkit-background-clip: text; color: transparent; margin:0; padding:0;}
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
/* CAIXAS COM FUNDO SÓLIDO (SEM EFEITO TRANSLÚCIDO) */
.glass-box{ background: #0E192A; border:1px solid #2A3B52; box-shadow:0 4px 12px rgba(0,0,0,0.3); }
.stButton>button, .stDownloadButton>button{ background:#1C2D4A !important; color:#bfc6cc !important; border:1px solid #2A3B52 !important; padding:8px 12px !important; border-radius:10px !important; }
.stButton>button:hover, .stDownloadButton>button:hover {
    background: #2A3B52 !important;
    border-color: #3C5070 !important;
}
.card, .msg-card { background: #0E192A; border-radius:12px; padding:12px; margin-bottom:10px; border:1px solid #2A3B52; }
.avatar{color:#fff;background:#6c5ce7}
.small-muted{color:#bfc6cc;}
.card-title{color:transparent}
"""

# inject base CSS
st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)
# inject dark default
st.markdown(f"<style>{DEFAULT_CSS}</style>", unsafe_allow_html=True)

# header
st.markdown("<div style='max-width:1100px;margin:18px auto 8px;text-align:center;'><h1 style='font-weight:800;font-size:40px; background:linear-gradient(90deg,#0077ff,#00a3ff); -webkit-background-clip:text; color:transparent; margin:0;'>NUGEP-PQR</h1></div>", unsafe_allow_html=True)

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
# Utilidades gerais
# -------------------------
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn').lower()

def escape_html(s):
    return html.escape(str(s) if s is not None else "")

def hex_to_rgba(h, alpha):
    h = h.lstrip('#')
    return f"rgba({', '.join(str(i) for i in tuple(int(h[i:i+2], 16) for i in (0, 2, 4)))}, {alpha})"


def gen_password(length=8):
    choices = string.ascii_letters + string.digits
    return ''.join(random.choice(choices) for _ in range(length))

def apply_global_styles(font_scale=1.0):
    try:
        dark_body_style = "<style>body { background-color: #071428; color: #d6d9dc; }</style>"
        st.markdown(dark_body_style, unsafe_allow_html=True)
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
# Funções de Busca & Recomendação
# -------------------------
@st.cache_data(ttl=600)
def collect_latest_backups():
    all_dfs = []
    base_path = Path(BACKUPS_DIR)
    if not base_path.exists():
        return pd.DataFrame()

    for user_dir in base_path.iterdir():
        if user_dir.is_dir():
            username = user_dir.name
            for csv_file in user_dir.glob("*.csv"):
                try:
                    df_temp = pd.read_csv(csv_file)
                    if not df_temp.empty:
                        df_temp['_artemis_username'] = username
                        all_dfs.append(df_temp)
                except Exception as e:
                    print(f"Skipping unreadable backup {csv_file}: {e}")
                    continue
    
    if not all_dfs:
        return pd.DataFrame()

    try:
        return pd.concat(all_dfs, ignore_index=True)
    except Exception as e:
        print(f"Error concatenating DataFrames: {e}")
        return pd.DataFrame()

def highlight_search_terms(text, query):
    if not query or not text or not isinstance(text, str):
        return escape_html(text)
    safe_text = escape_html(text)
    highlighted_text = re.sub(f'({re.escape(query)})', r'<span class="card-mark">\1</span>', safe_text, flags=re.IGNORECASE)
    return highlighted_text

def recomendar_artigos(temas_selecionados, df_total, query_text=None, top_n=50):
    # se sklearn não está disponível, retorna DataFrame vazio
    if TfidfVectorizer is None or cosine_similarity is None:
        st.error("Bibliotecas de Machine Learning (scikit-learn) não estão instaladas.")
        return pd.DataFrame()

    if df_total.empty or (not temas_selecionados and not query_text):
        return pd.DataFrame()

    corpus_series = pd.Series([''] * len(df_total), index=df_total.index, dtype=str)
    
    for col in ['título', 'tema', 'resumo', 'titulo', 'abstract']:
        if col in df_total.columns:
            corpus_series += df_total[col].fillna('') + ' '
    
    df_total['corpus'] = corpus_series.str.lower()
    
    if df_total['corpus'].str.strip().eq('').all():
        return pd.DataFrame()
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df_total['corpus'])
    
    query_parts = []
    if temas_selecionados: query_parts.extend(temas_selecionados)
    if query_text: query_parts.append(query_text)
    
    query_final = ' '.join(query_parts).lower()
    query_vector = vectorizer.transform([query_final])
    
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    similar_indices = [i for i in related_docs_indices if cosine_similarities[i] > 0.05]
    
    if not similar_indices:
        return pd.DataFrame()

    recomendados_df = df_total.iloc[similar_indices].copy()
    recomendados_df['similarity'] = cosine_similarities[similar_indices]
    
    return recomendados_df.drop(columns=['corpus']).reset_index(drop=True)

PORTUGUESE_STOP_WORDS = [
    'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 
    'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 
    'quando', 'muito', 'há', 'nos', 'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 
    'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'estão', 
    'você', 'tinha', 'foram', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'têm', 'numa', 'pelos', 'elas', 
    'havia', 'seja', 'qual', 'será', 'nós', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fosse', 
    'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 
    'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 
    'isto', 'aquilo'
]

@st.cache_data(ttl=600)
def extract_popular_themes_from_data(df_total, top_n=30):
    if TfidfVectorizer is None: return []
    if df_total.empty: return []

    corpus_series = pd.Series([''] * len(df_total), index=df_total.index, dtype=str)
    for col in ['título', 'tema', 'resumo', 'titulo', 'abstract']:
        if col in df_total.columns:
            corpus_series += df_total[col].fillna('') + ' '
    
    df_total['corpus'] = corpus_series.str.lower()

    if df_total['corpus'].str.strip().eq('').all(): return []

    try:
        vectorizer = TfidfVectorizer(stop_words=PORTUGUESE_STOP_WORDS, max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(df_total['corpus'])
        sum_tfidf = tfidf_matrix.sum(axis=0)
        words = vectorizer.get_feature_names_out()
        tfidf_scores = [(words[i], sum_tfidf[0, i]) for i in range(len(words))]
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_scores[:top_n]]
    except Exception as e:
        print(f"Erro ao extrair temas populares: {e}")
        return []

def load_users():
    users_path = Path.cwd() / USERS_FILE
    if users_path.exists():
        try:
            with users_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"[load_users] Erro: {e}")
            return {}
    return {}

def save_users(users: dict):
    users_path = Path.cwd() / USERS_FILE
    try:
        tmp_path = users_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        tmp_path.replace(users_path)
        return True
    except Exception as e:
        print(f"[save_users] Erro: {e}")
        return False

def get_session_favorites(): return st.session_state.get("favorites", [])

def add_to_favorites(result_data):
    favorites = get_session_favorites()
    favorite_item = {"id": f"{int(time.time())}_{random.randint(1000,9999)}", "data": result_data, "added_at": datetime.utcnow().isoformat()}
    
    temp_data_to_check = {k: v for k, v in result_data.items() if k not in ['_artemis_username', 'similarity']}
    
    existing_contents = [json.dumps({k: v for k, v in fav["data"].items() if k not in ['_artemis_username', 'similarity']}, sort_keys=True) for fav in favorites]
    
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
        return False

def load_all_messages():
    return _local_load_all_messages()

def send_message(sender, recipient, subject, body, attachment_file=None):
    entry = {
        "id": f"m_{int(time.time())}_{random.randint(1000,9999)}", 
        "from": sender, "to": recipient, "subject": subject or "(sem assunto)", 
        "body": body, "ts": datetime.utcnow().isoformat(), "read": False, "attachment": None
    }
    if attachment_file:
        try:
            entry["attachment"] = _local_upload_attachment(sender, attachment_file)
        except Exception:
            pass
    msgs = _local_load_all_messages()
    msgs.append(entry)
    _local_save_all_messages(msgs)
    return entry

def get_user_messages(username, box_type='inbox'):
    msgs = load_all_messages()
    key = "to" if box_type == 'inbox' else "from"
    user_msgs = sorted([m for m in msgs if m.get(key) == username], key=lambda x: x.get("ts", ""), reverse=True)
    return user_msgs

def mark_message_read(message_id, username):
    msgs = _local_load_all_messages()
    changed = False
    for m in msgs:
        if m.get("id") == message_id and m.get("to") == username and not m.get("read"):
            m["read"] = True; changed = True
            break
    if changed:
        _local_save_all_messages(msgs)
    return changed

def delete_message(message_id, username):
    msgs = _local_load_all_messages()
    msg_to_delete = next((m for m in msgs if m.get("id") == message_id and (m.get("to") == username or m.get("from") == username)), None)
    if msg_to_delete:
        if isinstance(msg_to_delete.get("attachment"), dict) and msg_to_delete.get("attachment", {}).get("path"):
            _local_remove_attachment(msg_to_delete["attachment"]["path"])
        new_msgs = [m for m in msgs if m.get("id") != message_id]
        _local_save_all_messages(new_msgs)
        return True
    return False

def read_spreadsheet(uploaded_file):
    buf = io.BytesIO(uploaded_file.read())
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

def generate_pdf_with_highlights(texto, highlight_hex="#ffd600"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for linha in (texto or "").split("\n"):
        parts = re.split(r"(==.*?==)", linha)
        for part in parts:
            if not part: continue
            if part.startswith("==") and part.endswith("=="):
                inner = part[2:-2].replace("—", "-").replace("–", "-").encode("latin-1", "replace").decode("latin-1")
                hexv = (highlight_hex or "#ffd600").lstrip("#")
                if len(hexv) == 3: hexv = ''.join([c*2 for c in hexv])
                try: r, g, b = int(hexv[0:2], 16), int(hexv[2:4], 16), int(hexv[4:6], 16)
                except Exception: r, g, b = (255, 214, 0)
                pdf.set_fill_color(r, g, b)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 6, txt=inner, border=0, fill=True)
            else:
                safe_part = part.replace("—", "-").replace("–", "-").encode("latin-1", "replace").decode("latin-1")
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 6, txt=safe_part, border=0)
    return pdf.output(dest="S").encode("latin-1")

# export PNG (usa matplotlib se disponível)
def export_graph_png_bytes(G):
    if not MATPLOTLIB_AVAILABLE:
        print("Exportação PNG requisitada, mas matplotlib não está instalado.")
        return None

    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#071428')
        ax.set_facecolor('#071428')
        try:
            pos = nx.spring_layout(G, seed=42, k=1.2)
        except Exception:
            pos = nx.random_layout(G, seed=42)

        node_labels = {n: G.nodes[n].get('label', n) for n in G.nodes()}
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, edge_color="#B0B0B0")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#22252A", node_size=3000)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_color="white", font_size=10)

        ax.axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"Erro exportando PNG: {e}")
        return None

_defaults = {
    "authenticated": False, "username": None, "user_obj": None, "df": None,
    "notes": "", "autosave": False, "page": "planilha",
    "restored_from_saved": False, "favorites": [], "reply_message_id": None,
    "view_message_id": None, "sent_messages_view": False,
    "search_results": pd.DataFrame(), "search_page": 1, "search_query_meta": {"col": None,"query":""},
    "search_view_index": None, "compose_inline": False, "compose_open": False,
    "last_backup_path": None, "selected_node": None,
    "tutorial_completed": False, 
    "recommendations": pd.DataFrame(), "recommendation_page": 1, "recommendation_view_index": None,
    "recommendation_onboarding_complete": False,
    "settings": {
        "plot_height": 720, "font_scale": 1.0, "node_opacity": 1.0,
    }
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def get_settings(): return st.session_state.get("settings", _defaults["settings"])

def clean_for_json(d):
    if isinstance(d, dict): return {k: clean_for_json(v) for k, v in d.items()}
    if isinstance(d, list): return [clean_for_json(i) for i in d]
    if isinstance(d, (np.int64, np.int32, np.int8)): return int(d)
    if isinstance(d, (np.float64, np.float32)): return None if np.isnan(d) else float(d)
    if pd.isna(d): return None
    return d

def save_user_state_minimal(USER_STATE):
    try:
        data = {
            "notes": st.session_state.get("notes",""),
            "uploaded_name": st.session_state.get("uploaded_name", None),
            "favorites": st.session_state.get("favorites", []),
            "settings": st.session_state.get("settings", {}),
            "last_backup_path": st.session_state.get("last_backup_path"),
            "tutorial_completed": st.session_state.get("tutorial_completed", False),
            "recommendation_onboarding_complete": st.session_state.get("recommendation_onboarding_complete", False)
        }
        clean_data = clean_for_json(data)

        tmp = USER_STATE.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(clean_data, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        tmp.replace(USER_STATE)
        return True
    except Exception as e:
        st.error(f"FALHA AO SALVAR O ESTADO: {e}")
        return False

# -------------------------
# Autenticação (com validação de CPF - apenas números e 11 dígitos)
# -------------------------
if not st.session_state.authenticated:
    st.markdown("<div class='glass-box auth' style='max-width:1100px;margin:0 auto;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Acesso — Faça login ou cadastre-se")
    tabs = st.tabs(["Entrar", "Cadastrar"])

    with tabs[0]:
        login_cpf = st.text_input("CPF (somente números)", key="ui_login_user")
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
            cpf_clean = re.sub(r'\D', '', (login_cpf or ""))
            if len(cpf_clean) != 11:
                st.warning("CPF deve ter 11 números.")
            else:
                if cpf_clean in users and users[cpf_clean].get("password") == login_pass:
                    st.session_state.authenticated = True
                    st.session_state.username = cpf_clean
                    st.session_state.user_obj = users[cpf_clean]
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
        reg_cpf = st.text_input("CPF (somente números)", key="ui_reg_user")
        reg_pass = st.text_input("Crie sua senha", type="password", key="ui_reg_pass")
        reg_pass_confirm = st.text_input("Confirme sua senha", type="password", key="ui_reg_pass_confirm")

        if st.button("Cadastrar", "btn_register_main"):
            new_cpf_raw = (reg_cpf or "").strip()
            new_cpf = re.sub(r'\D', '', new_cpf_raw)
            new_pass = (reg_pass or "").strip()

            if not new_cpf or len(new_cpf) != 11:
                st.warning("Informe um CPF válido (11 números).")
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
                    if save_users(users):
                        st.success("Usuário cadastrado com sucesso! Você já pode fazer o login na aba 'Entrar'.")
                        if "new_user_created" in st.session_state:
                            del st.session_state["new_user_created"]
                    else:
                        st.error("Falha ao salvar o usuário localmente.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

USERNAME = st.session_state.username
USER_OBJ = st.session_state.user_obj or load_users().get(USERNAME, {})
USER_STATE = Path.cwd() / f"artemis_state_{USERNAME}.json"

if not st.session_state.restored_from_saved and USER_STATE.exists():
    try:
        with USER_STATE.open("r", encoding="utf-8") as f: meta = json.load(f)
        for key, value in meta.items():
            if key in st.session_state:
                st.session_state[key] = value
        if "settings" in meta: st.session_state.settings.update(meta.get("settings", {}))
        
        backup_path = st.session_state.get("last_backup_path")
        if backup_path and Path(backup_path).exists():
            try:
                st.session_state.df = pd.read_csv(backup_path)
                st.toast(f"Planilha '{Path(backup_path).name}' restaurada.", icon="📄")
            except Exception as e:
                st.error(f"Falha ao restaurar backup: {e}")
                st.session_state.last_backup_path = None
        
        st.session_state.restored_from_saved = True
        st.toast("Progresso anterior restaurado.", icon="👍")
    except Exception as e:
        st.error(f"Erro ao restaurar seu progresso: {e}")

s = get_settings()
apply_global_styles(s.get("font_scale", 1.0))

all_msgs = load_all_messages()
UNREAD_COUNT = sum(1 for m in all_msgs if m.get("to") == USERNAME and not m.get("read"))
if "last_unread_count" not in st.session_state: st.session_state.last_unread_count = 0
if UNREAD_COUNT > st.session_state.last_unread_count:
    st.toast(f"Você tem {UNREAD_COUNT} nova(s) mensagem(ns) não lida(s).", icon="✉️")
st.session_state.last_unread_count = UNREAD_COUNT
mens_label = f"✉️ Mensagens ({UNREAD_COUNT})" if UNREAD_COUNT > 0 else "✉️ Mensagens"

st.markdown("<div class='glass-box' style='padding-top:10px; padding-bottom:10px;'><div class='specular'></div>", unsafe_allow_html=True)
top1, top2 = st.columns([0.6, 0.4])
with top1:
    st.markdown(f"<div style='color:var(--muted-text-dark);font-weight:700;padding-top:8px;'>Usuário: {USER_OBJ.get('name','')} — {USER_OBJ.get('scholarship','')}</div>", unsafe_allow_html=True)
with top2:
    nav_right1, nav_right2, nav_right3 = st.columns([1,1,1])
    with nav_right1: st.session_state.autosave = st.checkbox("Auto-save", value=st.session_state.autosave, key="ui_autosave")
    with nav_right2:
        if st.button("💾 Salvar", key="btn_save_now", use_container_width=True):
            if save_user_state_minimal(USER_STATE): st.success(f"Progresso salvo às {datetime.now().strftime('%H:%M:%S')}.")
    with nav_right3:
        if st.button("🚪 Sair", key="btn_logout", use_container_width=True):
            for key in list(st.session_state.keys()): del st.session_state[key]
            safe_rerun()
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-top:-20px'>", unsafe_allow_html=True)
nav_buttons = {"planilha": "📄 Planilha", "recomendacoes": "💡 Recomendações", "mapa": "🞠 Mapa",
               "anotacoes": "📝 Anotações", "graficos": "📊 Gráficos", "busca": "🔍 Busca",
               "mensagens": mens_label, "config": "⚙️ Configurações"}
nav_cols = st.columns(len(nav_buttons))
for i, (page_key, page_label) in enumerate(nav_buttons.items()):
    with nav_cols[i]:
        if st.button(page_label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state.page = page_key
            st.session_state.selected_node = None 
            safe_rerun()
st.markdown("</div></div><hr>", unsafe_allow_html=True)

if not st.session_state.get("tutorial_completed"):
    with st.expander("👋 Bem-vindo ao NUGEP-PQR! Um Guia Rápido Para Você", expanded=True):
        st.markdown("""
        **O que cada botão faz?**
        * **📄 Planilha**: Carregue sua planilha (.csv ou .xlsx). Os dados dela alimentarão os gráficos e as buscas.
        * **💡 Recomendações**: Explore artigos e trabalhos de outros usuários com base em temas de interesse.
        * **🞠 Mapa**: Visualize e edite um mapa de ideias no formato hierárquico. Você pode adicionar, conectar e remover nós.
        * **📝 Anotações**: Um bloco de notas para destacar texto com `==sinais de igual==` e exportar como PDF.
        * **📊 Gráficos**: Gere gráficos personalizados a partir da sua planilha.
        * **🔍 Busca**: Pesquise em todas as planilhas carregadas na plataforma.
        * **✉️ Mensagens**: Comunique-se com outros pesquisadores.
        * **⚙️ Configurações**: Personalize a aparência do aplicativo.
        """)
        if st.button("Entendido, começar a usar!", use_container_width=True):
            st.session_state.tutorial_completed = True
            save_user_state_minimal(USER_STATE) 
            st.balloons()
            time.sleep(1); safe_rerun()
    st.markdown("---")

# -------------------------
# Páginas
# -------------------------
if st.session_state.page == "planilha":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("📄 Planilha / Backup")
    
    uploaded = st.file_uploader("Carregue .csv ou .xlsx para usar nas buscas e gráficos", type=["csv", "xlsx"], key=f"u_{USERNAME}")
    if uploaded:
        try:
            df = read_spreadsheet(uploaded)
            st.session_state.df = df
            st.session_state.uploaded_name = uploaded.name
            
            try:
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                safe_name = re.sub(r"[^\w\-_.]", "_", uploaded.name)
                backup_filename = f"{safe_name}_{ts}.csv"
                p = BACKUPS_DIR / USERNAME
                p.mkdir(parents=True, exist_ok=True)
                path = p / backup_filename
                df.to_csv(path, index=False, encoding="utf-8")
                
                st.session_state.last_backup_path = str(path)
                st.success(f"Backup '{backup_filename}' criado com sucesso.")
                if st.session_state.autosave: save_user_state_minimal(USER_STATE)
                safe_rerun()
            except Exception as e:
                st.error(f"Erro ao salvar backup: {e}")
        except Exception as e:
            st.error(f"Erro ao ler a planilha: {e}")

    if st.session_state.df is not None:
        st.write("Visualização da planilha em uso:")
        st.dataframe(st.session_state.df, use_container_width=True)
        
        current_backup_path = st.session_state.get("last_backup_path")
        if current_backup_path and Path(current_backup_path).exists():
            st.write("Backup CSV em uso:")
            st.text(Path(current_backup_path).name)
            with open(current_backup_path, "rb") as fp:
                st.download_button("⬇ Baixar backup CSV", data=fp, file_name=Path(current_backup_path).name, mime="text/csv")
    else:
        st.info("Nenhuma planilha carregada.")

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "recomendacoes":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("💡 Recomendações de Artigos")

    # Coleta sempre os backups mais recentes ao abrir a aba
    try:
        with st.spinner("Analisando..."):
            df_total = collect_latest_backups()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        df_total = pd.DataFrame()

    temas_populares = extract_popular_themes_from_data(df_total) if not df_total.empty else []

    # Sempre exibe a seleção de temas e busca (evita travar por flag)
    st.markdown("#### Descoberta Inteligente")
    st.write("Selecione tópicos de interesse ou busque por palavra-chave para gerar recomendações.")
    # uso de keys únicas por usuário para evitar retenção de valores entre sessões/abas
    temas_key = f"temas_recomendacao_{USERNAME}"
    palavra_key = f"palavra_chave_recomendacao_{USERNAME}"
    temas_selecionados = st.multiselect("Selecione um ou mais temas:", options=temas_populares, key=temas_key)
    palavra_chave = st.text_input("Buscar por palavra-chave:", placeholder="...", key=palavra_key)

    if st.button("🔍 Buscar Recomendações", use_container_width=True, key=f"buscar_rec_btn_{USERNAME}"):
        # garante que sempre use o df_total atual
        if temas_selecionados or palavra_chave:
            with st.spinner("Analisando..."):
                if 'titulo' in df_total.columns and 'título' not in df_total.columns:
                    df_total = df_total.rename(columns={'titulo': 'título'})
                recommended_df = recomendar_artigos(temas_selecionados, df_total, palavra_chave)
                st.session_state.recommendations = recommended_df
                st.session_state.recommendation_page = 1
                st.session_state.recommendation_view_index = None
                # não alteramos a flag de onboarding aqui (mantemos para UX, mas não dependemos dela)
                safe_rerun()
        else:
            st.warning("Selecione um tema ou digite uma palavra-chave.")

    # Mostrar resultados + favoritos integrados
    results_df = st.session_state.get('recommendations', pd.DataFrame())
    if not results_df.empty:
        if st.session_state.get("recommendation_view_index") is not None:
            vi = st.session_state.recommendation_view_index
            if 0 <= vi < len(results_df):
                det = results_df.iloc[vi].to_dict()
                st.markdown("### 📄 Detalhes do Artigo Recomendado")
                if st.button("⬅️ Voltar para a lista"):
                    st.session_state.recommendation_view_index = None
                    safe_rerun()

                # campos importantes: título, autor, ano, tema, resumo
                for campo in ['título', 'autor', 'ano', 'tema', 'resumo']:
                    if campo in det and pd.notna(det[campo]):
                        st.markdown(f"**{campo.capitalize()}:** {escape_html(str(det[campo]))}")
                
                st.markdown("**Outras informações:**")
                for k, v in det.items():
                    if k not in ['similarity', 'corpus', 'título', 'autor', 'ano', 'tema', 'resumo'] and pd.notna(v):
                        st.markdown(f"- **{str(k).capitalize()}:** {escape_html(str(v))}")
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("⭐ Adicionar aos Favoritos", use_container_width=True, key=f"fav_detail_rec_{vi}"):
                        if add_to_favorites(det): st.toast("Adicionado aos favoritos!", icon="⭐")
                        else: st.toast("Este artigo já está nos favoritos.")
                with col_btn2:
                    if st.button("📝 Ver Anotações", use_container_width=True, key=f"notes_rec_{vi}"):
                        st.session_state.page = "anotacoes"
                        safe_rerun()

        else:
            per_page = 5
            total = len(results_df)
            max_pages = max(1, (total + per_page - 1) // per_page)
            page = max(1, min(st.session_state.get("recommendation_page", 1), max_pages))
            start, end = (page - 1) * per_page, min(page * per_page, total)
            page_df = results_df.iloc[start:end]

            st.markdown(f"**🎯 {total}** artigo(s) recomendado(s) — exibindo {start+1} a {end}.")

            for idx, row in page_df.iterrows():
                user_src = row.get("_artemis_username", "N/A")
                # buscar nome do usuário se existir nos users
                users_map = load_users()
                if user_src == "web":
                    user_display = "Fonte: Web"
                    initials = "W"
                else:
                    uobj = users_map.get(str(user_src), {})
                    user_display = uobj.get("name") if uobj and uobj.get("name") else str(user_src)
                    initials = "".join([p[0] for p in str(user_display).split()[:2]]).upper() or "U"

                title = str(row.get('título') or row.get('titulo') or '(Sem título)')
                similarity = row.get('similarity', 0)
                
                st.markdown(f"""
                <div class="card">
                    <div style="display:flex; gap:12px; align-items:flex-start;">
                        <div class="avatar" style="background:#6c5ce7; color:white; font-weight:bold;">{escape_html(initials)}</div>
                        <div style="flex:1;">
                            <div class="card-title">{escape_html(title)}</div>
                            <div class="small-muted">De <strong>{escape_html(user_display)}</strong> • Similaridade: <strong>{similarity:.2f}</strong></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if st.button("⭐ Favoritar", key=f"fav_rec_{idx}", use_container_width=True):
                        if add_to_favorites(row.to_dict()): st.toast("Adicionado aos favoritos!", icon="⭐")
                        else: st.toast("Já está nos favoritos.")
                with b_col2:
                    if st.button("🔎 Ver detalhes", key=f"view_rec_{idx}", use_container_width=True):
                        st.session_state.recommendation_view_index = idx
                        safe_rerun()
                st.markdown("---")
            
            p1, p2, p3 = st.columns([1, 1, 1])
            with p1:
                if st.button("◀ Anterior", key="rec_prev", disabled=(page <= 1), use_container_width=True):
                    st.session_state.recommendation_page -= 1
                    safe_rerun()
            with p2: st.markdown(f"<div style='text-align:center; padding-top:8px'><b>Página {page} / {max_pages}</b></div>", unsafe_allow_html=True)
            with p3:
                if st.button("Próxima ▶", key="rec_next", disabled=(page >= max_pages), use_container_width=True):
                    st.session_state.recommendation_page += 1
                    safe_rerun()

    else:
        st.info("Nenhum resultado encontrado. Tente outros temas ou palavras-chave.")

    # seção de favoritos dentro da aba de Recomendações
    st.markdown("---")
    st.markdown("### ⭐ Seus Favoritos (Recomendações)")
    favorites = get_session_favorites()
    if not favorites:
        st.info("Nenhum favorito salvo.")
    else:
        for fav in sorted(favorites, key=lambda x: x['added_at'], reverse=True):
            fav_data = fav['data']
            author_display = fav_data.get('_artemis_username', 'N/A')
            users_map = load_users()
            if author_display != "web":
                user_obj = users_map.get(str(author_display), {})
                author_display_name = user_obj.get("name") if user_obj and user_obj.get("name") else str(author_display)
            else:
                author_display_name = "Fonte: Web"
            st.markdown(f"""
            <div class="card">
                <div style="display:flex; gap:12px; align-items:center;">
                    <div class="avatar"> {escape_html('★')}</div>
                    <div style="flex:1;">
                        <div class="card-title">{escape_html(fav_data.get('título', '(Sem título)'))}</div>
                        <div class="small-muted">De <strong>{escape_html(author_display_name)}</strong></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
            c1, c2 = st.columns([0.8, 0.2])
            with c1:
                if st.button("Ver detalhes", key=f"favrec_view_{fav['id']}"):
                    st.session_state.recommendation_view_index = None
                    st.session_state.fav_detail = fav['data']
            with c2:
                if st.button("Remover", key=f"favrec_del_{fav['id']}"):
                    remove_from_favorites(fav['id'])
                    safe_rerun()
        if 'fav_detail' in st.session_state and st.session_state.fav_detail:
            det = st.session_state.pop("fav_detail")
            st.markdown("## Detalhes do Favorito")
            for k, v in det.items(): st.markdown(f"- **{escape_html(k)}:** {escape_html(v)}")

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "mapa":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("🞠 Mapa de Ideias Editável")

    if 'mapa_G' not in st.session_state:
        st.session_state.mapa_G = nx.DiGraph()

    G = st.session_state.mapa_G

    if not G.nodes():
        default_nodes = {
            "ALTO": {"label": "🟢  ALTO", "tipo": "Categoria"},
            "MÉDIO-ALTO": {"label": "🟡  MÉDIO-ALTO", "tipo": "Categoria"},
            "MÉDIO": {"label": "🟠  MÉDIO", "tipo": "Categoria"},
            "Open Access": {"label": "Open Access Massivo", "tipo": "Item"},
            "IA Curadoria": {"label": "IA para\ncuradoria/personalização", "tipo": "Item"},
            "Ferramentas Interativas": {"label": "Ferramentas interativas\nfísicas-digitais", "tipo": "Item"},
            "Foco Educacional": {"label": "Foco educacional", "tipo": "Item"},
            "Digitalização Técnica": {"label": "Digitalização técnica\navançada", "tipo": "Item"},
            "Projetos VR": {"label": "Projetos pontuais de VR", "tipo": "Item"},
            "Falta Transparência": {"label": "Falta de transparência", "tipo": "Item"},
        }
        for node_id, attrs in default_nodes.items():
            G.add_node(node_id, **attrs)
        
        default_edges = [
            ("ALTO", "Open Access"), ("ALTO", "IA Curadoria"), ("ALTO", "Ferramentas Interativas"),
            ("MÉDIO-ALTO", "Foco Educacional"), ("MÉDIO-ALTO", "Digitalização Técnica"),
            ("MÉDIO", "Projetos VR"), ("MÉDIO", "Falta Transparência")
        ]
        G.add_edges_from(default_edges)
        st.session_state.mapa_G = G 

    with st.expander("Opções e Edição do Mapa"):
        edit_c1, edit_c2 = st.columns(2)
        with edit_c1:
            with st.form("create_node_form", clear_on_submit=True):
                st.write("**1. Criar Novo Nó**")
                new_node_label = st.text_input("Rótulo do nó")
                new_node_id = st.text_input("ID único (sem espaços)")
                new_node_type = st.selectbox("Tipo do nó", options=["Categoria", "Item"])
                if st.form_submit_button("➕ Criar Nó"):
                    if new_node_label and new_node_id and new_node_type:
                        if new_node_id not in G:
                            G.add_node(new_node_id, label=new_node_label, tipo=new_node_type)
                            st.success(f"Nó '{new_node_label}' criado!")
                            st.session_state.mapa_G = G
                            time.sleep(0.5); safe_rerun()
                        else: st.warning("Este ID de nó já existe.")
                    else: st.warning("Preencha todos os campos.")
        with edit_c2:
            with st.form("connect_nodes_form", clear_on_submit=True):
                st.write("**2. Conectar Nós**")
                nodes_list = list(G.nodes())
                node1 = st.selectbox("De:", options=[""] + nodes_list, key="connect1")
                node2 = st.selectbox("Para:", options=[""] + nodes_list, key="connect2")
                if st.form_submit_button("🔗 Conectar"):
                    if node1 and node2 and node1 != node2:
                        if not G.has_edge(node1, node2):
                           G.add_edge(node1, node2)
                           st.success("Nós conectados.")
                           st.session_state.mapa_G = G
                           time.sleep(0.5); safe_rerun()
                        else: st.info("Esses nós já estão conectados.")
                    else: st.warning("Selecione dois nós diferentes.")

    if G.nodes():
        nodes = []
        for node_id, data in G.nodes(data=True):
            node_args = data.copy()
            node_args['id'] = node_id
            node_args.pop('tipo', None) # CORREÇÃO DO TypeError
            # força fonte branca
            if 'font' not in node_args:
                node_args['font'] = {"color": "#FFFFFF", "size": 16}
            else:
                node_args['font']['color'] = "#FFFFFF"
            nodes.append(Node(**node_args))

        edges = [Edge(source=u, target=v) for u, v in G.edges()]
        
        config = Config(width="100%", height=800, directed=True, physics=False, hierarchical=True,
                        layout_settings={"hierarchical": {"direction": "LR", "sortMethod": "directed", "levelSeparation": 300, "nodeSpacing": 120}},
                        node_style={"shape": "box", "borderWidth": 2, "color": "#22252A", "font": {"color": "#FFFFFF", "size": 16}},
                        edge_style={"color": "#B0B0B0", "width": 2})

        clicked_node_id = agraph(nodes=nodes, edges=edges, config=config)
        if clicked_node_id: st.session_state.selected_node = clicked_node_id
    else:
        st.warning("O mapa está vazio.")

    selected_node_name = st.session_state.get("selected_node")
    if selected_node_name and selected_node_name in G:
        node_data = G.nodes[selected_node_name]
        st.markdown("---")
        st.subheader(f"🔍 Detalhes do Nó: {node_data.get('label', selected_node_name)}")
        col1, col2 = st.columns([3, 1])
        with col1:
            connections = list(nx.all_neighbors(G, selected_node_name))
            st.markdown(f"**Tipo:** {escape_html(node_data.get('tipo', 'N/A'))}")
            st.markdown(f"**Conexões:** {len(connections)}")
        with col2:
            if st.button("🗑️ Excluir Nó", use_container_width=True):
                G.remove_node(selected_node_name)
                st.session_state.selected_node = None
                st.session_state.mapa_G = G
                st.toast(f"Nó '{selected_node_name}' removido.")
                time.sleep(1); safe_rerun()
        if connections:
            st.write("**Conectado a:**")
            for neighbor in sorted(connections):
                st.markdown(f"- {G.nodes[neighbor].get('label', neighbor)}")
        else:
            st.write("Este nó não possui conexões.")

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "anotacoes":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("📝 Anotações")
    st.info("Use ==texto== para destacar.")
    notes = st.text_area("Digite suas anotações", value=st.session_state.notes, height=260, key=f"notes_{USERNAME}")
    st.session_state.notes = notes
    pdf_bytes = generate_pdf_with_highlights(st.session_state.notes)
    st.download_button("Baixar Anotações (PDF)", data=pdf_bytes, file_name="anotacoes.pdf", mime="application/pdf")
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "graficos":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("📊 Gráficos Personalizados")
    if st.session_state.df is None:
        st.warning("Carregue uma planilha para gerar gráficos.")
    else:
        df = st.session_state.df.copy()
        cols = df.columns.tolist()
        c1, c2 = st.columns(2)
        with c1: eixo_x = st.selectbox("Eixo X", options=cols, key=f"x_{USERNAME}")
        with c2: eixo_y = st.selectbox("Eixo Y (Opcional)", options=[None] + df.select_dtypes(include=np.number).columns.tolist(), key=f"y_{USERNAME}")
        if st.button("Gerar Gráfico"):
            try:
                fig = px.bar(df, x=eixo_x, y=eixo_y, title=f"{eixo_y} por {eixo_x}") if eixo_y else px.histogram(df, x=eixo_x, title=f"Contagem por {eixo_x}")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#d6d9dc"))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao gerar gráficos: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "busca":
    st.markdown("<div class='glass-box' style='position:relative;padding:18px;'><div class='specular'></div>", unsafe_allow_html=True)
    tab_busca, tab_favoritos = st.tabs([f"🔍 Busca Inteligente", f"⭐ Favoritos ({len(get_session_favorites())})"])

    def extract_keywords(text, n=6):
        if not text: return []
        text = re.sub(r"[^\w\s]", " ", str(text or "")).lower()
        stop = {"de","da","do","e","a","o","em","para","por","com"}
        words = [w for w in text.split() if len(w) > 2 and w not in stop]
        freq = {w: words.count(w) for w in set(words)}
        return [w for w, _ in sorted(freq.items(), key=lambda item: item[1], reverse=True)][:n]

    with tab_busca:
        col_q, col_meta, col_actions = st.columns([0.6, 0.25, 0.15])
        with col_q: query = st.text_input("Termo de busca", key="ui_query_search", placeholder="...")
        with col_meta:
            backups_df_tmp = collect_latest_backups()
            all_cols = list(backups_df_tmp.columns) if not backups_df_tmp.empty else []
            search_col = st.selectbox("Buscar em", options=[c for c in all_cols if c != '_artemis_username'] or ["(sem dados)"], key="ui_search_col")
        with col_actions:
            per_page = st.selectbox("Por página", [5, 8, 12, 20], index=1, key="ui_search_pp")
            search_clicked = st.button("🔎 Buscar", use_container_width=True, key="ui_search_btn")

        if search_clicked:
            st.session_state.search_view_index = None
            if not query or backups_df_tmp.empty:
                st.info("Digite um termo e certifique-se de que há dados para pesquisar.")
                st.session_state.search_results = pd.DataFrame()
            else:
                norm_query = normalize_text(query)
                ser = backups_df_tmp[search_col].astype(str).apply(normalize_text)
                hits = backups_df_tmp[ser.str.contains(norm_query, na=False)]
                st.session_state.search_results = hits.reset_index(drop=True)
                st.session_state.search_query_meta = {"col": search_col, "query": query}
                st.session_state.search_page = 1
        
        results_df = st.session_state.get('search_results', pd.DataFrame())
        if not results_df.empty:
            total = len(results_df)
            max_pages = max(1, (total + per_page - 1) // per_page)
            page = max(1, min(st.session_state.get("search_page", 1), max_pages))
            start, end = (page - 1) * per_page, min(page * per_page, total)
            page_df = results_df.iloc[start:end]

            st.markdown(f"**{total}** resultado(s) — exibindo {start+1} a {end}.")
            for orig_i in page_df.index:
                result_data = results_df.loc[orig_i].to_dict()
                user_src = result_data.get("_artemis_username", "N/A")
                initials = "".join([p[0] for p in str(user_src).split()[:2]]).upper() or "U"
                title_raw = str(result_data.get('título') or result_data.get('titulo') or '(Sem título)')
                resumo_raw = str(result_data.get('resumo') or result_data.get('abstract') or "")
                
                # Mostrar NOME do usuário (se disponível) ou "Fonte: Web"
                if user_src == "web":
                    user_display_name = "Fonte: Web"
                else:
                    user_obj = load_users().get(str(user_src), {})
                    user_display_name = user_obj.get("name") if user_obj and user_obj.get("name") else str(user_src)

                st.markdown(f"""
                <div class="card">
                    <div style="display:flex; gap:12px; align-items:center;">
                        <div class="avatar">{escape_html(initials)}</div>
                        <div style="flex:1;">
                            <div class="card-title">{highlight_search_terms(title_raw, query)}</div>
                            <div class="small-muted">De <strong>{escape_html(user_display_name)}</strong> • {escape_html(result_data.get('autor', ''))}</div>
                            <div style="margin-top:6px;font-size:13px;color:#e6e8ea;">{highlight_search_terms(resumo_raw, query) if resumo_raw else ''}</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                
                a1, a2 = st.columns([0.28, 0.72])
                with a1:
                    if st.button("⭐ Favoritar", key=f"fav_{orig_i}", use_container_width=True):
                        if add_to_favorites(result_data): st.toast("Adicionado!", icon="⭐")
                        else: st.toast("Já está nos favoritos.")
                with a2:
                    if st.button("🔎 Ver detalhes", key=f"view_{orig_i}", use_container_width=True):
                        st.session_state.search_view_index = int(orig_i)
                        safe_rerun()
            
            st.markdown("---")
            p1, p2, p3 = st.columns([1,1,1])
            with p1: 
                if st.button("◀", disabled=(page <= 1), key="search_prev"):
                    st.session_state.search_page -= 1; safe_rerun()
            with p2: st.markdown(f"<div style='text-align:center; padding-top:8px'><b>Página {page}/{max_pages}</b></div>", unsafe_allow_html=True)
            with p3: 
                if st.button("▶", disabled=(page >= max_pages), key="search_next"):
                    st.session_state.search_page += 1; safe_rerun()

            if st.session_state.get("search_view_index") is not None:
                vi = int(st.session_state.search_view_index)
                if 0 <= vi < len(results_df):
                    det = results_df.loc[vi].to_dict()
                    origin_user = det.get("_artemis_username", "N/A")
                    st.markdown("## Detalhes do Registro")
                    for k, v in det.items():
                        if k != "_artemis_username": st.markdown(f"- **{escape_html(k)}:** {escape_html(v)}")
                    st.markdown("---")
                    st.markdown("### ✉️ Contatar autor")
                    if origin_user != "N/A":
                        with st.form(key=f"inline_compose_{vi}"):
                            subj_fill = st.text_input("Assunto:", value=f"Sobre: {det.get('título', '')[:50]}...")
                            body_fill = st.text_area("Mensagem:", value=f"Olá {origin_user},\n\nVi seu registro na plataforma e gostaria de conversar.\n\n")
                            c1, c2 = st.columns(2)
                            with c1:
                                if st.form_submit_button("✉️ Enviar"):
                                    send_message(USERNAME, origin_user, subj_fill, body_fill)
                                    st.success(f"Mensagem enviada para {origin_user}.")
                                    time.sleep(2); safe_rerun()
                            with c2:
                                if st.form_submit_button("Cancelar"):
                                    st.session_state.search_view_index = None; safe_rerun()
                    else:
                        st.warning("Origem indisponível para contato.")

    with tab_favoritos:
        st.header("Seus Resultados Salvos")
        favorites = get_session_favorites()
        if not favorites:
            st.info("Nenhum resultado salvo.")
        else:
            if st.button("🗑️ Limpar Todos", key="clear_favs"):
                clear_all_favorites(); safe_rerun()
            st.markdown("---")
            for fav in sorted(favorites, key=lambda x: x['added_at'], reverse=True):
                fav_data = fav['data']
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">{escape_html(fav_data.get('título', '(Sem título)'))}</div>
                    <div class="small-muted">De <strong>{escape_html(fav_data.get('_artemis_username', 'N/A'))}</strong></div>
                </div>""", unsafe_allow_html=True)
                c1, c2 = st.columns([0.75, 0.25])
                with c1:
                    if st.button("Ver", key=f"fav_view_{fav['id']}", use_container_width=True):
                        st.session_state.fav_detail = fav['data']
                with c2:
                    if st.button("Remover", key=f"fav_del_{fav['id']}", use_container_width=True):
                        remove_from_favorites(fav['id']); safe_rerun()
            if 'fav_detail' in st.session_state and st.session_state.fav_detail:
                det = st.session_state.pop("fav_detail")
                st.markdown("## Detalhes do Favorito")
                for k, v in det.items(): st.markdown(f"- **{escape_html(k)}:** {escape_html(v)}")

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "mensagens":
    st.markdown("<div class='glass-box' style='position:relative;padding:12px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("✉️ Mensagens")

    tab_inbox, tab_sent, tab_compose = st.tabs(["Caixa de Entrada", "Enviados", "Escrever Nova"])

    with tab_inbox:
        if st.session_state.get("view_message_id"):
            msg = next((m for m in all_msgs if m.get('id') == st.session_state.view_message_id), None)
            if msg and isinstance(msg, dict):
                mark_message_read(msg.get('id'), USERNAME)
                st.markdown(f"**De:** {escape_html(msg.get('from'))}\n\n**Assunto:** {escape_html(msg.get('subject'))}")
                st.markdown(f"<div style='white-space:pre-wrap; padding:10px; border-radius:8px; background:rgba(0,0,0,0.2);'>{escape_html(msg.get('body'))}</div>", unsafe_allow_html=True)
                if isinstance(msg.get('attachment'), dict) and msg.get('attachment', {}).get('path') and Path(msg['attachment']['path']).exists():
                    with open(msg['attachment']['path'], "rb") as fp:
                        st.download_button(f"⬇️ Baixar anexo: {msg['attachment'].get('name','anexo')}", data=fp, file_name=msg['attachment'].get('name','anexo'))
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("↩️ Voltar", key="back_inbox"):
                        st.session_state.view_message_id = None; safe_rerun()
                with c2:
                    if st.button("↪️ Responder", key="reply_msg"):
                        st.session_state.reply_message_id = msg['id']; st.session_state.view_message_id = None; safe_rerun()
                with c3:
                    if st.button("🗑️ Excluir", key="del_inbox_msg"):
                        delete_message(msg['id'], USERNAME); st.session_state.view_message_id = None; st.toast("Excluída."); safe_rerun()
            else:
                st.warning("Mensagem não encontrada."); st.session_state.view_message_id = None
        else:
            inbox_msgs = get_user_messages(USERNAME, 'inbox')
            if not inbox_msgs:
                st.info("Caixa de entrada vazia.")
            for msg in inbox_msgs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    read_marker = "" if msg.get('read', False) else "🔵 "
                    st.markdown(f"**{read_marker}{escape_html(msg.get('subject', '(sem assunto)'))}**")
                    st.markdown(f"<span class='small-muted'>De: {escape_html(msg.get('from', '...'))} em {datetime.fromisoformat(msg.get('ts')).strftime('%d/%m/%Y %H:%M')}</span>", unsafe_allow_html=True)
                with col2:
                    if st.button("Ler", key=f"read_{msg['id']}", use_container_width=True):
                        st.session_state.view_message_id = msg['id']; safe_rerun()
                st.markdown("---")

    with tab_sent:
        sent_msgs = get_user_messages(USERNAME, 'sent')
        if not sent_msgs:
            st.info("Nenhuma mensagem enviada.")
        for msg in sent_msgs:
            st.markdown(f"**{escape_html(msg.get('subject', '(sem assunto)'))}**")
            st.markdown(f"<span class='small-muted'>Para: {escape_html(msg.get('to', '...'))} em {datetime.fromisoformat(msg.get('ts')).strftime('%d/%m/%Y %H:%M')}</span>", unsafe_allow_html=True)
            if st.button("🗑️ Excluir", key=f"del_sent_{msg['id']}"):
                delete_message(msg['id'], USERNAME); st.toast("Excluída."); safe_rerun()
            st.markdown("---")

    with tab_compose:
        if st.session_state.get("reply_message_id"):
            original_msg = next((m for m in all_msgs if m.get('id') == st.session_state.reply_message_id), None)
            if original_msg:
                st.info(f"Respondendo a: {original_msg['from']}")
                default_to, default_subj, default_body = original_msg['from'], f"Re: {original_msg['subject']}", f"\n\n---\nEm resposta a:\n{original_msg['body']}"
            else:
                st.session_state.reply_message_id = None; default_to, default_subj, default_body = "", "", ""
        elif st.session_state.get("compose_open"):
            default_to = st.session_state.get("compose_to", "")
            default_subj = st.session_state.get("compose_subject", "")
            default_body = st.session_state.get("compose_prefill", "")
            st.session_state.compose_open = False
        else:
            default_to, default_subj, default_body = "", "", ""

        with st.form("compose_form", clear_on_submit=True):
            all_users = [u for u in load_users().keys() if u != USERNAME]
            to_user = st.selectbox("Para:", options=all_users, index=all_users.index(default_to) if default_to in all_users else 0) if all_users else st.text_input("Para (CPF):", value=default_to)
            subject = st.text_input("Assunto:", value=default_subj)
            body = st.text_area("Mensagem:", height=200, value=default_body)
            attachment = st.file_uploader("Anexo (opcional)")
            
            if st.form_submit_button("✉️ Enviar Mensagem"):
                if to_user:
                    send_message(USERNAME, to_user, subject, body, attachment)
                    st.success(f"Mensagem enviada para {to_user}!")
                    st.session_state.reply_message_id = None
                    time.sleep(1); safe_rerun()
                else:
                    st.warning("Selecione um destinatário.")
    st.markdown("</div>", unsafe_allow_html=True)
    
elif st.session_state.page == "config":
    st.markdown("<div class='glass-box' style='position:relative;padding:12px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("⚙️ Configurações")
    s = get_settings()

    font_scale = st.slider("Escala de fonte", 0.7, 2.0, float(s.get("font_scale",1.0)), 0.1, key="cfg_font_scale")

    if st.button("Aplicar configurações"):
        st.session_state.settings["font_scale"] = float(font_scale)
        save_user_state_minimal(USER_STATE)
        apply_global_styles(font_scale)
        st.success("Configurações aplicadas e salvas.")
        time.sleep(0.5); safe_rerun()

    st.markdown("---")
    st.markdown("**Acessibilidade**\n\n- Use *Escala de fonte* para aumentar ou diminuir o tamanho do texto.\n- O programa utiliza um tema escuro fixo para garantir bom contraste.")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Página não encontrada — selecione uma aba no topo.")
