# app_nugep_pqr_full_final.py

# NUGEP-PQR — versão final com ajustes: título centralizado/branco, remover download JSON, manter apenas PNG,
# e edição de nós com renomear/excluir diretamente no painel de edição.

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

import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Para evitar problemas com a interface gráfica

from streamlit_agraph import agraph, Node, Edge, Config

# optional ML libs (silenciosamente não-fatal)
try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    joblib = None
    TfidfVectorizer = None
    cosine_similarity = None

# bcrypt for password hashing (optional)
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except Exception:
    bcrypt = None
    BCRYPT_AVAILABLE = False

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
.card-title{color:#fff}
"""

st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)
st.markdown(f"<style>{DEFAULT_CSS}</style>", unsafe_allow_html=True)

# TÍTULO - centralizado (geral) e estilo padronizado
st.markdown("<div style='text-align:center; padding-top:8px; padding-bottom:6px;'><h1 style='margin:0;color:#ffffff;'>NUGEP-PQR</h1></div>", unsafe_allow_html=True)
st.markdown("<hr style='margin-top:6px;margin-bottom:16px;border-color:#233447'/>", unsafe_allow_html=True)

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
# Utilities: CPF / hashing / formatting
# -------------------------
def normalize_cpf(raw_cpf: str) -> str:
    if not raw_cpf:
        return ""
    return re.sub(r'\D', '', str(raw_cpf))

def format_cpf_display(cpf: str) -> str:
    s = normalize_cpf(cpf)
    if len(s) != 11:
        return cpf or ""
    return f"{s[0:3]}.{s[3:6]}.{s[6:9]}-{s[9:11]}"

def is_valid_cpf(cpf: str) -> bool:
    s = normalize_cpf(cpf)
    if len(s) != 11:
        return False
    if s == s[0] * 11:
        return False
    nums = [int(ch) for ch in s]
    sum1 = sum([(10 - i) * nums[i] for i in range(9)])
    r1 = sum1 % 11
    d1 = 0 if r1 < 2 else 11 - r1
    if nums[9] != d1:
        return False
    sum2 = sum([(11 - i) * nums[i] for i in range(10)])
    r2 = sum2 % 11
    d2 = 0 if r2 < 2 else 11 - r2
    if nums[10] != d2:
        return False
    return True

def hash_password(plain: str) -> str:
    if BCRYPT_AVAILABLE:
        return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()
    return plain

def check_password(stored_hash_or_plain: str, plain: str) -> bool:
    if not stored_hash_or_plain:
        return False
    s = str(stored_hash_or_plain)
    if BCRYPT_AVAILABLE and (s.startswith("$2b$") or s.startswith("$2y$") or s.startswith("$2a$")):
        try:
            return bcrypt.checkpw(plain.encode(), s.encode())
        except Exception:
            return False
    return s == plain

# -------------------------
# Small helpers
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
    try:
        dark_body_style = "<style>body { background-color: #071428; color: #d6d9dc; }</style>"
        st.markdown(dark_body_style, unsafe_allow_html=True)
        font_css = f"html {{ font-size: {font_scale * 100}%; }}"
        st.markdown(f"<style>{font_css}</style>", unsafe_allow_html=True)
    except Exception:
        pass

def _render_credentials_box(username, password, note=None, key_prefix="cred"):
    st.markdown("---")
    st.success("Usuário criado com sucesso — anote/guarde a senha abaixo:")
    col1, col2 = st.columns([3,1])
    with col1:
        st.text_input("CPF", value=format_cpf_display(username), key=f"{key_prefix}_user", disabled=True)
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
# Stop words
# -------------------------
PORTUGUESE_STOP_WORDS = [
    'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'una', 'os', 'no', 'se', 'na', 
    'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 
    'quando', 'muito', 'há', 'nos', 'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 
    'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'estão', 
    'você', 'tinha', 'foram', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'têm', 'numa', 'pelos', 'elas', 
    'havia', 'seja', 'qual', 'será', 'nós', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fosse', 
    'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 
    'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 
    'isto', 'aquilo'
]

# -------------------------
# Storage helpers
# -------------------------
def load_users():
    users_path = Path.cwd() / USERS_FILE
    if users_path.exists():
        try:
            with users_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, Exception) as e:
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
        if msg_to_delete.get("attachment") and isinstance(msg_to_delete["attachment"], dict) and msg_to_delete["attachment"].get('path'):
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

# -------------------------
# Recommendation & search
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
    
    vectorizer = TfidfVectorizer(stop_words=PORTUGUESE_STOP_WORDS, max_features=5000)
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
    
    if 'titulo' in recomendados_df.columns and 'título' not in recomendados_df.columns:
        recomendados_df = recomendados_df.rename(columns={'titulo': 'título'})
    if 'autor' not in recomendados_df.columns and 'autores' in recomendados_df.columns:
        recomendados_df = recomendados_df.rename(columns={'autores': 'autor'})

    return recomendados_df.drop(columns=['corpus']).reset_index(drop=True)

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

def search_crossref(query, rows=6):
    url = "https://api.crossref.org/works"
    params = {"query.title": query, "rows": rows, "sort": "relevance"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("message", {}).get("items", [])
    except Exception as e:
        print(f"[search_crossref] erro: {e}")
        return []

    results = []
    for it in data:
        title = " ".join(it.get("title", [])) if it.get("title") else ""
        authors = []
        for a in it.get("author", [])[:6]:
            fam = a.get("family") or ""
            given = a.get("given") or ""
            authors.append((given + " " + fam).strip())
        year = None
        if it.get("issued") and it["issued"].get("date-parts"):
            year = it["issued"]["date-parts"][0][0]
        doi = it.get("DOI")
        url_ = it.get("URL")
        abstr = it.get("abstract") or ""
        results.append({
            "título": title,
            "autor": "; ".join(authors),
            "ano": year,
            "doi": doi,
            "url": url_,
            "resumo": re.sub(r'<[^>]+>', '', abstr) if abstr else ""
        })
    return results

# -------------------------
# Metadata enrichment
# -------------------------
import html as _html

def _safe_strip_html(s):
    if not s: return ""
    return re.sub(r'<[^>]+>', '', str(s)).strip()

def _format_authors_field(auth_field):
    if not auth_field: return "— Autor(es) não informado(s) —"
    if isinstance(auth_field, (list, tuple)):
        return "; ".join([str(a).strip() for a in auth_field if a])
    s = str(auth_field)
    s = s.replace("|", ";")
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s or "— Autor(es) não informado(s) —"

def enrich_article_metadata(det):
    if not isinstance(det, dict):
        return det or {}

    lower_map = {}
    for k in list(det.keys()):
        if isinstance(k, str) and k.lower() != k:
            lower_map[k.lower()] = det.pop(k)
    det.update(lower_map)

    titulo = det.get('título') or det.get('title') or det.get('titulo')
    autor = det.get('autor') or det.get('autores')
    resumo = det.get('resumo') or det.get('abstract')
    if titulo and autor and resumo:
        return det

    doi = det.get('doi') or det.get('DOI') or None
    if doi:
        try:
            doi_clean = str(doi).strip()
            cr_url = f"https://api.crossref.org/works/{requests.utils.requote_uri(doi_clean)}"
            r = requests.get(cr_url, timeout=8)
            if r.status_code == 200:
                msg = r.json().get("message", {})
                if not titulo:
                    t = " ".join(msg.get("title", [])) if msg.get("title") else ""
                    if t: det['título'] = _safe_strip_html(t)
                if not autor:
                    authors = []
                    for a in msg.get("author", [])[:10]:
                        given = a.get("given") or ""
                        fam = a.get("family") or ""
                        name = (given + " " + fam).strip()
                        if name:
                            authors.append(name)
                    if authors:
                        det['autor'] = "; ".join(authors)
                if not resumo:
                    abstr = msg.get("abstract") or ""
                    if abstr:
                        det['resumo'] = _safe_strip_html(abstr)
        except Exception:
            pass

    if (not det.get('título') or not det.get('resumo')) and det.get('url'):
        try:
            r = requests.get(det.get('url'), timeout=6, headers={"User-Agent": "nugrp-pqr-bot/1.0"})
            if r.status_code == 200:
                html_text = r.text
                if not det.get('título'):
                    m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html_text, flags=re.I)
                    if m:
                        det['título'] = _html.unescape(m.group(1).strip())
                    else:
                        m2 = re.search(r'<title>([^<]+)</title>', html_text, flags=re.I)
                        if m2:
                            det['título'] = _html.unescape(m2.group(1).strip())
                if not det.get('resumo'):
                    m = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html_text, flags=re.I)
                    if m:
                        det['resumo'] = _html.unescape(m.group(1).strip())
                    else:
                        m2 = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']', html_text, flags=re.I)
                        if m2:
                            det['resumo'] = _html.unescape(m2.group(1).strip())
        except Exception:
            pass

    if not det.get('título'):
        det['título'] = det.get('title') or det.get('titulo') or "— Título não disponível —"
    if not det.get('autor'):
        det['autor'] = _format_authors_field(det.get('autor') or det.get('autores'))
    if not det.get('resumo'):
        det['resumo'] = det.get('abstract') or "Resumo não disponível."

    for k in ('título','autor','resumo'):
        if k in det and isinstance(det[k], str):
            det[k] = _safe_strip_html(det[k])

    return det

# -------------------------
# Defaults & session state
# -------------------------
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
# Authentication UI (login & register) with CPF normalization + bcrypt migration
# -------------------------
if not st.session_state.authenticated:
    st.markdown("<div class='glass-box auth' style='max-width:1100px;margin:0 auto;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Acesso — Faça login ou cadastre-se")
    tabs = st.tabs(["Entrar", "Cadastrar"])

    with tabs[0]:
        login_cpf_raw = st.text_input("CPF", key="ui_login_user")
        login_pass = st.text_input("Senha", type="password", key="ui_login_pass")

        users = load_users() or {}
        if not users:
            admin_user = "admin"
            admin_pwd = "admin123"
            users[admin_user] = {"name": "Administrador", "scholarship": "Admin", "password_hash": hash_password(admin_pwd), "created_at": datetime.utcnow().isoformat()}
            save_users(users)
            st.warning("Nenhum usuário local encontrado. Um usuário administrativo foi criado temporariamente.")
            st.session_state.new_user_created = {"user": admin_user, "pwd": admin_pwd, "note": "Este é um usuário administrativo temporário. Para testes, use 'admin' como CPF."}

        if st.button("Entrar", "btn_login_main"):
            login_cpf_norm = normalize_cpf(login_cpf_raw)
            users = load_users() or {}

            matched_user = None
            candidate_keys = []
            if login_cpf_norm:
                candidate_keys.append(login_cpf_norm)
            if login_cpf_raw:
                candidate_keys.append(login_cpf_raw)
            candidate_keys.append("admin")
            for key in candidate_keys:
                if not key: continue
                if key in users:
                    u = users[key]
                    if u.get("password_hash"):
                        if check_password(u.get("password_hash"), login_pass):
                            matched_user = key
                            break
                    if u.get("password") and check_password(u.get("password"), login_pass):
                        matched_user = key
                        if BCRYPT_AVAILABLE:
                            try:
                                users[key]["password_hash"] = hash_password(login_pass)
                                users[key].pop("password", None)
                                save_users(users)
                            except Exception:
                                pass
                        break

            if matched_user:
                st.session_state.authenticated = True
                st.session_state.username = matched_user
                st.session_state.user_obj = users[matched_user]
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
        reg_cpf_raw = st.text_input("CPF", key="ui_reg_user")
        reg_pass = st.text_input("Crie sua senha", type="password", key="ui_reg_pass")
        reg_pass_confirm = st.text_input("Confirme sua senha", type="password", key="ui_reg_pass_confirm")

        if st.button("Cadastrar", "btn_register_main"):
            new_cpf_norm = normalize_cpf(reg_cpf_raw)
            new_pass = (reg_pass or "").strip()

            if not new_cpf_norm:
                st.warning("Informe um CPF (somente números).")
            elif len(new_cpf_norm) != 11:
                st.warning("CPF deve ter 11 dígitos (apenas números).")
            elif not is_valid_cpf(new_cpf_norm):
                st.error("CPF inválido (verificador incorreto). Verifique os números.")
            elif len(new_pass) < 6:
                st.warning("A senha deve ter pelo menos 6 caracteres.")
            elif new_pass != reg_pass_confirm:
                st.error("As senhas não coincidem. Tente novamente.")
            else:
                users = load_users() or {}
                if new_cpf_norm in users:
                    st.warning("CPF já cadastrado (local).")
                else:
                    password_hash = hash_password(new_pass)
                    users[new_cpf_norm] = {"name": reg_name or new_cpf_norm, "scholarship": reg_bolsa, "password_hash": password_hash, "created_at": datetime.utcnow().isoformat()}
                    if save_users(users):
                        st.success("Usuário cadastrado com sucesso! Você já pode fazer o login na aba 'Entrar'.")
                        if "new_user_created" in st.session_state:
                            del st.session_state["new_user_created"]
                        _render_credentials_box(new_cpf_norm, new_pass, note="Guarde sua senha. Ela é salva de forma segura (hash).", key_prefix=f"cred_{new_cpf_norm}")
                    else:
                        st.error("Falha ao salvar o usuário localmente.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# -------------------------
# After login: restore, settings, onboarding, pages...
# -------------------------
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
    st.toast(f"Você tem {UNREAD_COUNT} nova(s) mensagem(n) não lida(s).", icon="✉️")
st.session_state.last_unread_count = UNREAD_COUNT
mens_label = f"✉️ Mensagens ({UNREAD_COUNT})" if UNREAD_COUNT > 0 else "✉️ Mensagens"

# -------------------------
# Onboarding (first contact)
# -------------------------
if st.session_state.authenticated and not st.session_state.recommendation_onboarding_complete:
    st.markdown("<div class='glass-box' style='position:relative;margin-bottom:12px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("✨ Primeiro contato — escolha seus interesses")
    default_themes = [
        "documentação",
        "documentação participativa",
        "cultura de inovação",
        "nft",
        "inovação social",
        "inovação tecnológica"
    ]
    sel = st.multiselect("Selecione temas de interesse (pelo menos 1):", options=default_themes, key="onb_themes")
    only_pt = st.checkbox("Priorizar resultados em português (quando possível)", value=True, key="onb_only_pt")
    max_per_theme = st.slider("Resultados por tema", 1, 8, 4, key="onb_rows")

    if st.button("🔍 Buscar artigos sugeridos", key="onb_search"):
        if not sel:
            st.error("Escolha pelo menos um tema.")
        else:
            all_hits = []
            with st.spinner("Buscando artigos..."):
                for theme in sel:
                    hits = search_crossref(theme, rows=max_per_theme)
                    for h in hits:
                        h["_artemis_username"] = "web"
                        h["_tema_origem"] = theme
                    all_hits.extend(hits)
            if all_hits:
                rec_df = pd.DataFrame(all_hits)
                st.session_state.recommendations = rec_df
                st.session_state.recommendation_onboarding_complete = True
                st.session_state.recommendation_page = 1
                st.toast(f"{len(rec_df)} recomendação(ões) carregada(s).")
                safe_rerun()
            else:
                st.info("Nenhum artigo encontrado automaticamente. Tente outros termos.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Top navigation and pages
# -------------------------
st.markdown("<div class='glass-box' style='padding-top:10px; padding-bottom:10px;'><div class='specular'></div>", unsafe_allow_html=True)
top1, top2 = st.columns([0.6, 0.4])
with top1:
    # Exibição do nome centralizada e em branco (solicitado)
    st.markdown(f"<div style='text-align:center;color:#ffffff;font-weight:700;padding-top:4px;padding-bottom:4px'>{escape_html(USER_OBJ.get('name',''))} — {escape_html(USER_OBJ.get('scholarship',''))}</div>", unsafe_allow_html=True)
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
        if st.button(page_label, key=f"nav_{page_key}_{USERNAME}", use_container_width=True):
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
# Page: planilha
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

# -------------------------
# Page: recomendacoes (mantém Favoritos aqui)
# -------------------------
elif st.session_state.page == "recomendacoes":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("💡 Recomendações de Artigos")

    try:
        with st.spinner("Analisando..."):
            df_total = collect_latest_backups()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        df_total = pd.DataFrame()

    temas_populares = extract_popular_themes_from_data(df_total) if not df_total.empty else []

    # favorites panel (moved here)
    with st.expander(f"⭐ Favoritos ({len(get_session_favorites())})", expanded=False):
        favorites = get_session_favorites()
        if not favorites:
            st.info("Nenhum favorito salvo.")
        else:
            if st.button("🗑️ Limpar Todos os Favoritos", key=f"clear_favs_rec_{USERNAME}"):
                clear_all_favorites(); st.session_state.recommendation_page = 1; safe_rerun()
            for fav in sorted(favorites, key=lambda x: x['added_at'], reverse=True):
                fav_data = fav['data']
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">{escape_html(fav_data.get('título', '(Sem título)'))}</div>
                    <div class="small-muted">De <strong>{escape_html(fav_data.get('_artemis_username', 'N/A'))}</strong></div>
                </div>""", unsafe_allow_html=True)
                c1, c2 = st.columns([0.75, 0.25])
                with c1:
                    if st.button("Ver", key=f"fav_view_rec_{fav['id']}_{USERNAME}", use_container_width=True):
                        st.session_state.fav_detail = fav['data']
                with c2:
                    if st.button("Remover", key=f"fav_del_rec_{fav['id']}_{USERNAME}", use_container_width=True):
                        remove_from_favorites(fav['id']); safe_rerun()
            if 'fav_detail' in st.session_state and st.session_state.fav_detail:
                det_fav = st.session_state.pop("fav_detail")
                det_fav = enrich_article_metadata(det_fav)
                st.markdown("## Detalhes do Favorito")
                st.markdown(f"**{escape_html(det_fav.get('título','— Sem título —'))}**")
                st.markdown(f"_Autor(es):_ {escape_html(det_fav.get('autor','— —'))}")
                st.markdown("---")
                st.markdown(escape_html(det_fav.get('resumo','Resumo não disponível.')))

    # recommendation onboarding or refine
    if not st.session_state.recommendation_onboarding_complete:
        if df_total.empty:
            st.warning("Ainda não há dados suficientes para gerar recomendações automaticamente. Use o onboarding no topo para obter recomendações iniciais.")
        elif not temas_populares:
            st.warning("Não foi possível identificar temas populares.")
        else:
            st.markdown("#### Bem-vindo à Descoberta Inteligente!")
            st.write("Selecione tópicos de interesse para encontrarmos artigos para você.")
            temas_selecionados = st.multiselect("Selecione um ou mais temas:", options=temas_populares, key="temas_onboarding")
            
            if st.button("🔍 Gerar Recomendações", key=f"gen_rec_{USERNAME}"):
                if temas_selecionados:
                    with st.spinner("Buscando..."):
                        if 'titulo' in df_total.columns and 'título' not in df_total.columns:
                            df_total = df_total.rename(columns={'titulo': 'título'})
                        
                        recommended_df = recomendar_artigos(temas_selecionados, df_total)
                        st.session_state.recommendations = recommended_df
                        st.session_state.recommendation_page = 1
                        st.session_state.recommendation_view_index = None
                        st.session_state.recommendation_onboarding_complete = True
                        safe_rerun()
                else:
                    st.error("Selecione pelo menos um tema.")
    else:
        st.write("Refine suas recomendações ou explore novos temas.")
        
        col1, col2 = st.columns([3, 2])
        with col1:
            temas_options = temas_populares or []
            temas_selecionados = st.multiselect("Selecione temas:", options=temas_options, key="temas_recomendacao", help="Se não houver dados locais, use temas livres na caixa ao lado")
        with col2:
            palavra_chave = st.text_input("Buscar por palavra-chave (ou escreva qualquer tema):", placeholder="ex.: documentação participativa", key="palavra_chave_recomendacao")

        if st.button("🔍 Buscar Recomendações", use_container_width=True, key=f"btn_recom_search_{USERNAME}"):
            if temas_selecionados or palavra_chave:
                with st.spinner("Analisando..."):
                    if not df_total.empty:
                        q_parts = temas_selecionados[:] if temas_selecionados else []
                        if palavra_chave: q_parts.append(palavra_chave)
                        recommended_df = recomendar_artigos(q_parts, df_total, palavra_chave if palavra_chave else None)
                        if recommended_df.empty:
                            hits = []
                            queries = temas_selecionados if temas_selecionados else ([palavra_chave] if palavra_chave else [])
                            for q in queries:
                                hits += search_crossref(q, rows=6)
                            if palavra_chave and not temas_selecionados:
                                hits += search_crossref(palavra_chave, rows=6)
                            for h in hits:
                                h["_artemis_username"] = "web"
                            rec_df = pd.DataFrame(hits) if hits else pd.DataFrame()
                            st.session_state.recommendations = rec_df
                        else:
                            st.session_state.recommendations = recommended_df
                    else:
                        hits = []
                        if temas_selecionados:
                            for t in temas_selecionados:
                                hits += search_crossref(t, rows=6)
                        if palavra_chave:
                            hits += search_crossref(palavra_chave, rows=6)
                        for h in hits:
                            h["_artemis_username"] = "web"
                        rec_df = pd.DataFrame(hits) if hits else pd.DataFrame()
                        st.session_state.recommendations = rec_df

                    st.session_state.recommendation_page = 1
                    st.session_state.recommendation_view_index = None
                    st.session_state.recommendation_onboarding_complete = True
                    safe_rerun()
            else:
                st.warning("Selecione um tema ou digite uma palavra-chave.")

    results_df = st.session_state.get('recommendations', pd.DataFrame())
    
    if not results_df.empty:
        if st.session_state.get("recommendation_view_index") is not None:
            vi = st.session_state.recommendation_view_index
            if 0 <= vi < len(results_df):
                det = results_df.iloc[vi].to_dict()
                det = enrich_article_metadata(det)

                st.markdown("### 📄 Detalhes do Artigo Recomendado")
                if st.button("⬅️ Voltar para a lista", key=f"rec_back_{USERNAME}"):
                    st.session_state.recommendation_view_index = None
                    safe_rerun()

                # MELHORIA: Exibição mais organizada dos dados
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{escape_html(det.get('título','— Sem título —'))}**")
                    st.markdown(f"**Autor(es):** {escape_html(det.get('autor','— Não informado —'))}")
                    st.markdown(f"**Ano:** {escape_html(str(det.get('ano', det.get('year','— —'))))}")
                    st.markdown(f"**País:** {escape_html(det.get('país', det.get('pais', det.get('country','— —'))))}")
                    
                    if det.get('doi'):
                        doi_link = f"https://doi.org/{det.get('doi')}"
                        st.markdown(f"**DOI:** [{det.get('doi')}]({doi_link})")
                    elif det.get('url'):
                        st.markdown(f"**Link:** [{det.get('url')}]({det.get('url')})")
                    
                    st.markdown("---")
                    st.markdown("**Resumo**")
                    st.markdown(escape_html(det.get('resumo', 'Resumo não disponível.')))
                
                with col2:
                    # Informações adicionais
                    st.markdown("**Informações Adicionais**")
                    if det.get('similarity'):
                        st.metric("Similaridade", f"{det['similarity']:.2f}")
                    
                    if det.get('_artemis_username'):
                        st.write(f"Fonte: {det['_artemis_username']}")

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("⭐ Adicionar aos Favoritos", use_container_width=True, key=f"fav_detail_rec_{vi}_{USERNAME}"):
                        if add_to_favorites(det): st.toast("Adicionado aos favoritos!", icon="⭐")
                        else: st.toast("Este artigo já está nos favoritos.")
                with col_btn2:
                    if st.button("📝 Ver Anotações", use_container_width=True, key=f"notes_rec_{vi}_{USERNAME}"):
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
                initials = "".join([p[0] for p in str(user_src).split()[:2]]).upper() or "U"
                title = str(row.get('título') or row.get('titulo') or '(Sem título)')
                similarity = row.get('similarity', 0)
                author_snippet = row.get('autor') or ""
                year = row.get('ano') or row.get('year') or ""
                country = row.get('país') or row.get('pais') or row.get('country') or ""
                link = row.get('url') or row.get('link') or row.get('doi') or ""
                
                # MELHORIA: Card mais informativo
                st.markdown(f"""
                <div class="card">
                    <div style="display:flex; gap:12px; align-items:flex-start;">
                        <div class="avatar" style="background:#6c5ce7; color:white; font-weight:bold;">{escape_html(initials)}</div>
                        <div style="flex:1;">
                            <div class="card-title">{escape_html(title)}</div>
                            <div class="small-muted">De <strong>{escape_html(user_src)}</strong> • {escape_html(author_snippet)}</div>
                            <div class="small-muted">Ano: {escape_html(str(year))} • País: {escape_html(country)}</div>
                            <div class="small-muted">Link: {escape_html(link)}</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if st.button("⭐ Favoritar", key=f"fav_rec_{idx}_{USERNAME}", use_container_width=True):
                        if add_to_favorites(row.to_dict()): st.toast("Adicionado aos favoritos!", icon="⭐")
                        else: st.toast("Já está nos favoritos.")
                with b_col2:
                    if st.button("🔎 Ver detalhes", key=f"view_rec_{idx}_{USERNAME}", use_container_width=True):
                        st.session_state.recommendation_view_index = idx
                        safe_rerun()
                st.markdown("---")
            
            p1, p2, p3 = st.columns([1, 1, 1])
            with p1:
                if st.button("◀ Anterior", key=f"rec_prev_{USERNAME}", disabled=(page <= 1), use_container_width=True):
                    st.session_state.recommendation_page -= 1
                    safe_rerun()
            with p2: st.markdown(f"<div style='text-align:center; padding-top:8px'><b>Página {page} / {max_pages}</b></div>", unsafe_allow_html=True)
            with p3:
                if st.button("Próxima ▶", key=f"rec_next_{USERNAME}", disabled=(page >= max_pages), use_container_width=True):
                    st.session_state.recommendation_page += 1
                    safe_rerun()

    elif st.session_state.recommendation_onboarding_complete:
        st.info("Nenhum resultado encontrado. Tente outros temas.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: mapa (força texto branco) - COM EDIÇÃO DE NÓS RENOMEAR/EXCLUIR NO PAINEL
# -------------------------
elif st.session_state.page == "mapa":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("🞠 Mapa de Ideias Editável")

    if 'mapa_G' not in st.session_state:
        st.session_state.mapa_G = nx.DiGraph()

    G = st.session_state.mapa_G

    if not G.nodes():
        default_nodes = {
            "ALTO": {"label": "🟢 ALTO", "tipo": "Categoria"},
            "MÉDIO-ALTO": {"label": "🟡 MÉDIO-ALTO", "tipo": "Categoria"},
            "MÉDIO": {"label": "🟠 MÉDIO", "tipo": "Categoria"},
            "Open Access": {"label": "Open Access Massivo", "tipo": "Item"},
            "IA Curadoria": {"label": "IA para curadoria", "tipo": "Item"},
            "Ferramentas Interativas": {"label": "Ferramentas interativas", "tipo": "Item"},
            "Foco Educacional": {"label": "Foco educacional", "tipo": "Item"},
            "Digitalização Técnica": {"label": "Digitalização técnica", "tipo": "Item"},
            "Projetos VR": {"label": "Projetos de VR", "tipo": "Item"},
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

    with st.expander("Opções e Edição do Mapa", expanded=True):
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
                node1 = st.selectbox("De:", options=[""] + nodes_list, key=f"connect1_{USERNAME}")
                node2 = st.selectbox("Para:", options=[""] + nodes_list, key=f"connect2_{USERNAME}")
                if st.form_submit_button("🔗 Conectar"):
                    if node1 and node2 and node1 != node2:
                        if not G.has_edge(node1, node2):
                           G.add_edge(node1, node2)
                           st.success("Nós conectados.")
                           st.session_state.mapa_G = G
                           time.sleep(0.5); safe_rerun()
                        else: st.info("Esses nós já estão conectados.")
                    else: st.warning("Selecione dois nós diferentes.")

        st.markdown("---")
        st.write("**Edição Rápida dos Nós**")
        
        # Mostrar lista de nós com controle de renomear/excluir direto
        nodes_list = list(G.nodes())
        if nodes_list:
            st.write(f"Total de {len(nodes_list)} nós no mapa:")
            for nid in nodes_list:
                nlabel = G.nodes[nid].get('label', nid)
                ntipo = G.nodes[nid].get('tipo', 'Item')
                
                col_a, col_b, col_c = st.columns([4, 2, 1])
                with col_a:
                    new_label = st.text_input(f"Rótulo ({nid})", value=nlabel, key=f"edit_label_{nid}")
                with col_b:
                    # Atualizar tipo
                    new_tipo = st.selectbox(f"Tipo", options=["Categoria", "Item"], 
                                          index=0 if ntipo == "Categoria" else 1, 
                                          key=f"edit_tipo_{nid}")
                    if new_tipo != ntipo:
                        G.nodes[nid]['tipo'] = new_tipo
                        st.session_state.mapa_G = G
                        st.toast(f"Tipo do nó '{nid}' atualizado.")
                with col_c:
                    if st.button("🗑️", key=f"btn_del_{nid}", help=f"Excluir nó {nid}"):
                        try:
                            G.remove_node(nid)
                            st.session_state.mapa_G = G
                            if st.session_state.get("selected_node") == nid:
                                st.session_state.selected_node = None
                            st.toast(f"Nó '{nid}' excluído.")
                            time.sleep(0.5); safe_rerun()
                        except Exception as e:
                            st.error(f"Erro ao excluir nó: {e}")
                
                # Botão de renomear separado
                if st.button("✏️ Renomear", key=f"btn_rename_{nid}", use_container_width=True):
                    if new_label.strip():
                        G.nodes[nid]['label'] = new_label.strip()
                        st.session_state.mapa_G = G
                        st.toast(f"Nó renomeado para '{new_label}'.")
                        time.sleep(0.5); safe_rerun()
                    else:
                        st.warning("O rótulo não pode ser vazio.")
                
                st.markdown("---")
        else:
            st.info("Nenhum nó para editar.")

        st.markdown("---")
        st.write("**3. Baixar Mapa (PNG)**")

        # Função para exportar o mapa como PNG (em memória)
        def export_map_to_png_bytes(G):
            fig = plt.figure(figsize=(14, 10), dpi=200)
            try:
                pos = nx.spring_layout(G, k=1.5, iterations=100)
            except Exception:
                pos = nx.spring_layout(G)

            # Cores diferentes para categorias e itens
            node_colors = []
            node_sizes = []
            for node in G.nodes():
                if G.nodes[node].get('tipo') == 'Categoria':
                    node_colors.append('#FF6B6B')  # Vermelho para categorias
                    node_sizes.append(3000)
                else:
                    node_colors.append('#4ECDC4')  # Verde para itens
                    node_sizes.append(2000)

            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                                 alpha=0.95, edgecolors='white', linewidths=2)
            
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                                 arrowsize=25, width=2, connectionstyle='arc3,rad=0.1')

            # Labels com quebra de linha
            labels = {}
            for node in G.nodes():
                label = G.nodes[node].get('label', node)
                # Quebra de linha automática para labels longos
                if len(label) > 15:
                    words = label.split()
                    lines = []
                    current_line = []
                    for word in words:
                        if len(' '.join(current_line + [word])) <= 15:
                            current_line.append(word)
                        else:
                            if current_line:
                                lines.append(' '.join(current_line))
                            current_line = [word]
                    if current_line:
                        lines.append(' '.join(current_line))
                    label = '\n'.join(lines)
                labels[node] = label

            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, 
                                  font_color='white', font_weight='bold')

            plt.title("Mapa Mental", color='white', fontsize=16, pad=20)
            plt.axis('off')
            plt.tight_layout()
            buf = io.BytesIO()
            fig.patch.set_facecolor('#0E192A')
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                       facecolor=fig.get_facecolor(), transparent=False)
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()

        try:
            png_bytes = export_map_to_png_bytes(G)
            st.download_button(
                label="⬇️ Baixar Mapa como PNG",
                data=png_bytes,
                file_name=f"mapa_mental_{USERNAME}_{int(time.time())}.png",
                mime="image/png",
                use_container_width=True,
                help="Salva uma imagem PNG do mapa mental."
            )
        except Exception as e:
            st.error(f"Não foi possível gerar a imagem PNG: {e}")

    # Visualização interativa
    if G.nodes():
        nodes = []
        for node_id, data in G.nodes(data=True):
            node_args = {
                'id': node_id,
                'label': data.get('label', node_id),
                'color': '#FF6B6B' if data.get('tipo') == 'Categoria' else '#4ECDC4',
                'font': {'color': '#FFFFFF', 'size': 16, 'face': 'Arial'},
                'size': 25 if data.get('tipo') == 'Categoria' else 20
            }
            nodes.append(Node(**node_args))

        edges = [Edge(source=u, target=v, color='#B0B0B0', width=2) for u, v in G.edges()]
        
        config = Config(
            width="100%", 
            height=780, 
            directed=True, 
            physics=True,
            hierarchical=False,
            node_highlight_behavior=True,
            highlight_color="#F8F8F8",
            collapsible=True,
            node={'labelProperty': 'label'},
            link={'labelProperty': 'label', 'renderLabel': True}
        )

        try:
            clicked_node_id = agraph(nodes=nodes, edges=edges, config=config)
            if clicked_node_id: 
                st.session_state.selected_node = clicked_node_id
        except Exception as e:
            st.warning(f"Erro na visualização interativa: {e}")
            # Fallback para visualização estática
            st.info("Visualização interativa temporariamente indisponível. Use o download PNG.")
    else:
        st.warning("O mapa está vazio. Adicione alguns nós para começar.")

    selected_node_name = st.session_state.get("selected_node")
    if selected_node_name and selected_node_name in G:
        node_data = G.nodes[selected_node_name]
        st.markdown("---")
        st.subheader(f"🔍 Nó Selecionado: {node_data.get('label', selected_node_name)}")
        
        # Painel de informações do nó
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**ID:** `{selected_node_name}`")
            st.markdown(f"**Tipo:** {escape_html(node_data.get('tipo', 'N/A'))}")
        with col2:
            # Conexões
            in_connections = list(G.predecessors(selected_node_name))
            out_connections = list(G.successors(selected_node_name))
            st.markdown(f"**Conexões de entrada:** {len(in_connections)}")
            st.markdown(f"**Conexões de saída:** {len(out_connections)}")

        st.markdown("---")
        
        # Formulário para edição avançada
        with st.form(f"edit_node_{selected_node_name}"):
            st.write("**Editar Nó**")
            new_label = st.text_input(
                "Novo Rótulo:", 
                value=node_data.get('label', selected_node_name), 
                key=f"rename_{selected_node_name}"
            )
            new_tipo = st.selectbox(
                "Novo Tipo:",
                options=["Categoria", "Item"],
                index=0 if node_data.get('tipo') == "Categoria" else 1,
                key=f"tipo_{selected_node_name}"
            )
            
            col_edit1, col_edit2 = st.columns(2)
            with col_edit1:
                if st.form_submit_button("💾 Atualizar Nó", use_container_width=True):
                    if new_label.strip():
                        G.nodes[selected_node_name]['label'] = new_label.strip()
                        G.nodes[selected_node_name]['tipo'] = new_tipo
                        st.session_state.mapa_G = G
                        st.toast("Nó atualizado com sucesso!")
                        time.sleep(1); safe_rerun()
                    else:
                        st.warning("O rótulo não pode ser vazio.")
            
            with col_edit2:
                if st.form_submit_button("🗑️ Excluir Nó", use_container_width=True):
                    G.remove_node(selected_node_name)
                    st.session_state.selected_node = None
                    st.session_state.mapa_G = G
                    st.toast(f"Nó '{selected_node_name}' removido.")
                    time.sleep(1); safe_rerun()

        # Mostrar conexões
        if in_connections or out_connections:
            st.write("**Conexões:**")
            col_conn1, col_conn2 = st.columns(2)
            with col_conn1:
                if in_connections:
                    st.write("**Entrada de:**")
                    for neighbor in sorted(in_connections):
                        st.markdown(f"- {G.nodes[neighbor].get('label', neighbor)}")
            with col_conn2:
                if out_connections:
                    st.write("**Saída para:**")
                    for neighbor in sorted(out_connections):
                        st.markdown(f"- {G.nodes[neighbor].get('label', neighbor)}")
        else:
            st.write("Este nó não possui conexões.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: anotacoes
# -------------------------
elif st.session_state.page == "anotacoes":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("📝 Anotações")
    st.info("Use ==texto== para destacar.")
    notes = st.text_area("Digite suas anotações", value=st.session_state.notes, height=260, key=f"notes_{USERNAME}")
    st.session_state.notes = notes
    pdf_bytes = generate_pdf_with_highlights(st.session_state.notes)
    st.download_button("Baixar Anotações (PDF)", data=pdf_bytes, file_name="anotacoes.pdf", mime="application/pdf")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: graficos (MELHORIA: Gráficos mais inteligentes)
# -------------------------
elif st.session_state.page == "graficos":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("📊 Gráficos Inteligentes")
    
    if st.session_state.df is None:
        st.warning("Carregue uma planilha na página 'Planilha' para gerar gráficos.")
    else:
        df = st.session_state.df.copy()
        
        # Análise automática dos dados
        st.write("### 📈 Análise dos Dados")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Registros", len(df))
        with col2:
            st.metric("Colunas", len(df.columns))
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.metric("Colunas Numéricas", len(numeric_cols))
        
        # Seleção de tipo de gráfico
        st.write("### 🎨 Tipo de Gráfico")
        chart_type = st.selectbox(
            "Selecione o tipo de gráfico:",
            ["Barra", "Linha", "Dispersão", "Histograma", "Pizza", "Boxplot", "Heatmap", "Treemap"],
            key=f"chart_type_{USERNAME}"
        )
        
        cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if chart_type in ["Barra", "Linha"]:
            col1, col2 = st.columns(2)
            with col1:
                eixo_x = st.selectbox("Eixo X", options=categorical_cols + numeric_cols, key=f"x_{USERNAME}")
            with col2:
                if numeric_cols:
                    eixo_y = st.selectbox("Eixo Y", options=[None] + numeric_cols, key=f"y_{USERNAME}")
                else:
                    eixo_y = None
                    st.info("Nenhuma coluna numérica encontrada")
            
            # Agrupamento para gráficos de barra
            if chart_type == "Barra" and eixo_y is None:
                group_by = st.selectbox("Agrupar por (opcional)", options=[None] + categorical_cols, key=f"group_{USERNAME}")
            else:
                group_by = None
                
        elif chart_type == "Dispersão":
            col1, col2, col3 = st.columns(3)
            with col1:
                eixo_x = st.selectbox("Eixo X", options=numeric_cols if numeric_cols else cols, key=f"scatter_x_{USERNAME}")
            with col2:
                eixo_y = st.selectbox("Eixo Y", options=numeric_cols if numeric_cols else cols, key=f"scatter_y_{USERNAME}")
            with col3:
                color_by = st.selectbox("Cor por (opcional)", options=[None] + categorical_cols, key=f"scatter_color_{USERNAME}")
                
        elif chart_type == "Histograma":
            eixo_x = st.selectbox("Selecione a coluna:", options=numeric_cols if numeric_cols else cols, key=f"hist_x_{USERNAME}")
            bins = st.slider("Número de bins", 5, 100, 20, key=f"bins_{USERNAME}")
            
        elif chart_type == "Pizza":
            eixo_x = st.selectbox("Categorias", options=categorical_cols, key=f"pie_x_{USERNAME}")
            if numeric_cols:
                eixo_y = st.selectbox("Valores", options=numeric_cols, key=f"pie_y_{USERNAME}")
            else:
                eixo_y = None
                st.info("Nenhuma coluna numérica para valores")
                
        elif chart_type == "Boxplot":
            eixo_x = st.selectbox("Categorias (opcional)", options=[None] + categorical_cols, key=f"box_x_{USERNAME}")
            eixo_y = st.selectbox("Valores", options=numeric_cols if numeric_cols else cols, key=f"box_y_{USERNAME}")
            
        elif chart_type == "Heatmap":
            st.info("Heatmap mostra correlação entre variáveis numéricas")
            if len(numeric_cols) >= 2:
                selected_cols = st.multiselect("Selecione colunas para heatmap", 
                                             options=numeric_cols, default=numeric_cols[:5],
                                             key=f"heatmap_cols_{USERNAME}")
            else:
                selected_cols = numeric_cols
                st.warning("Heatmap requer pelo menos 2 colunas numéricas")
                
        elif chart_type == "Treemap":
            path_cols = st.multiselect("Hierarquia (caminho)", options=categorical_cols, 
                                     key=f"treemap_path_{USERNAME}", max_selections=3)
            if numeric_cols:
                value_col = st.selectbox("Valor", options=numeric_cols, key=f"treemap_value_{USERNAME}")
            else:
                value_col = None
                st.info("Nenhuma coluna numérica para valores")

        if st.button("Gerar Gráfico", key=f"gen_chart_{USERNAME}", use_container_width=True):
            try:
                fig = None
                
                if chart_type == "Barra":
                    if eixo_y is None:
                        # Gráfico de contagem
                        if group_by:
                            fig = px.histogram(df, x=eixo_x, color=group_by, title=f"Contagem de {eixo_x} por {group_by}")
                        else:
                            fig = px.histogram(df, x=eixo_x, title=f"Contagem de {eixo_x}")
                    else:
                        if group_by:
                            fig = px.bar(df, x=eixo_x, y=eixo_y, color=group_by, title=f"{eixo_y} por {eixo_x}")
                        else:
                            fig = px.bar(df, x=eixo_x, y=eixo_y, title=f"{eixo_y} por {eixo_x}")
                            
                elif chart_type == "Linha":
                    if eixo_y:
                        fig = px.line(df, x=eixo_x, y=eixo_y, title=f"{eixo_y} por {eixo_x}")
                    else:
                        st.error("Selecione uma coluna numérica para o eixo Y")
                        
                elif chart_type == "Dispersão":
                    if color_by:
                        fig = px.scatter(df, x=eixo_x, y=eixo_y, color=color_by, 
                                       title=f"{eixo_y} vs {eixo_x}")
                    else:
                        fig = px.scatter(df, x=eixo_x, y=eixo_y, 
                                       title=f"{eixo_y} vs {eixo_x}")
                                       
                elif chart_type == "Histograma":
                    fig = px.histogram(df, x=eixo_x, nbins=bins, 
                                     title=f"Distribuição de {eixo_x}")
                                     
                elif chart_type == "Pizza":
                    if eixo_y:
                        fig = px.pie(df, names=eixo_x, values=eixo_y, 
                                   title=f"Proporção de {eixo_x}")
                    else:
                        # Contagem simples
                        contagem = df[eixo_x].value_counts()
                        fig = px.pie(values=contagem.values, names=contagem.index,
                                   title=f"Distribuição de {eixo_x}")
                                   
                elif chart_type == "Boxplot":
                    if eixo_x:
                        fig = px.box(df, x=eixo_x, y=eixo_y, 
                                   title=f"Distribuição de {eixo_y} por {eixo_x}")
                    else:
                        fig = px.box(df, y=eixo_y, title=f"Distribuição de {eixo_y}")
                        
                elif chart_type == "Heatmap" and len(selected_cols) >= 2:
                    corr_matrix = df[selected_cols].corr()
                    fig = px.imshow(corr_matrix, 
                                  title="Matriz de Correlação",
                                  color_continuous_scale='RdBu_r',
                                  aspect="auto")
                    fig.update_layout(height=600)
                    
                elif chart_type == "Treemap" and path_cols and value_col:
                    fig = px.treemap(df, path=path_cols, values=value_col,
                                   title=f"Treemap de {value_col}")
                    fig.update_layout(height=600)

                if fig:
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", 
                        paper_bgcolor="rgba(0,0,0,0)", 
                        font=dict(color="#d6d9dc"),
                        title_font=dict(size=20, color="#ffffff"),
                        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#d6d9dc"))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Estatísticas descritivas
                    if chart_type in ["Barra", "Linha", "Dispersão", "Histograma", "Boxplot"] and eixo_y in numeric_cols:
                        with st.expander("📊 Estatísticas Descritivas"):
                            st.write(f"**Estatísticas para {eixo_y}:**")
                            stats = df[eixo_y].describe()
                            st.dataframe(stats)
                            
                else:
                    st.warning("Não foi possível gerar o gráfico com os parâmetros selecionados.")

            except Exception as e:
                st.error(f"Erro ao gerar gráfico: {e}")
                
        # Gráficos automáticos sugeridos
        if len(numeric_cols) > 0:
            with st.expander("🚀 Gráficos Automáticos Sugeridos"):
                st.write("Gráficos gerados automaticamente com base nos dados:")
                
                # Gráfico de correlação
                if len(numeric_cols) >= 2:
                    if st.button("📈 Matriz de Correlação", key=f"auto_corr_{USERNAME}"):
                        corr_matrix = df[numeric_cols].corr()
                        fig = px.imshow(corr_matrix, 
                                      title="Matriz de Correlação (Automática)",
                                      color_continuous_scale='RdBu_r',
                                      aspect="auto")
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Distribuição das principais colunas numéricas
                if numeric_cols:
                    col_numeric = st.selectbox("Selecione coluna para distribuição:", 
                                             options=numeric_cols, key=f"auto_dist_{USERNAME}")
                    if st.button("📊 Distribuição", key=f"auto_hist_{USERNAME}"):
                        fig = px.histogram(df, x=col_numeric, 
                                         title=f"Distribuição de {col_numeric}")
                        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: busca (AQUI foi a alteração principal)
# -------------------------
elif st.session_state.page == "busca":
    st.markdown("<div class='glass-box' style='position:relative;padding:18px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("🔍 Busca Inteligente")

    def extract_keywords(text, n=6):
        if not text: return []
        text = re.sub(r"[^\w\s]", " ", str(text or "")).lower()
        stop = {"de","da","do","e","a","o","em","para","por","com"}
        words = [w for w in text.split() if len(w) > 2 and w not in stop]
        freq = {w: words.count(w) for w in set(words)}
        return [w for w, _ in sorted(freq.items(), key=lambda item: item[1], reverse=True)][:n]

    col_q, col_meta, col_actions = st.columns([0.6, 0.25, 0.15])
    with col_q: query = st.text_input("Termo de busca", key="ui_query_search", placeholder="...")
    with col_meta:
        backups_df_tmp = collect_latest_backups()
        all_cols = list(backups_df_tmp.columns) if not backups_df_tmp.empty else []
        search_col = st.selectbox("Buscar em", options=[c for c in all_cols if c != '_artemis_username'] or ["(sem dados)"], key="ui_search_col")
    with col_actions:
        per_page = st.selectbox("Por página", [5, 8, 12, 20], index=1, key="ui_search_pp")
        search_clicked = st.button("🔎 Buscar", use_container_width=True, key=f"ui_search_btn_{USERNAME}")

    if search_clicked:
        st.session_state.search_view_index = None
        if not query or backups_df_tmp.empty:
            st.info("Digite um termo e certifique-se de que há dados para pesquisar.")
            st.session_state.search_results = pd.DataFrame()
        else:
            norm_query = normalize_text(query)
            if search_col not in backups_df_tmp.columns:
                st.info("Coluna inválida para busca. Selecione outra coluna.")
                st.session_state.search_results = pd.DataFrame()
            else:
                ser = backups_df_tmp[search_col].astype(str).apply(normalize_text)
                hits = backups_df_tmp[ser.str.contains(norm_query, na=False)]
                st.session_state.search_results = hits.reset_index(drop=True)
                st.session_state.search_query_meta = {"col": search_col, "query": query}
                st.session_state.search_page = 1

    results_df = st.session_state.get('search_results', pd.DataFrame())
    users_map = load_users()  # carregamos os usuários para mapear CPF -> nome

    if not results_df.empty:
        total = len(results_df)
        max_pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(st.session_state.get("search_page", 1), max_pages))
        start, end = (page - 1) * per_page, min(page * per_page, total)
        page_df = results_df.iloc[start:end]

        st.markdown(f"**{total}** resultado(s) — exibindo {start+1} a {end}.")
        for orig_i in page_df.index:
            result_data = results_df.loc[orig_i].to_dict()
            origin_uid = result_data.get("_artemis_username", "N/A")
            # Aqui: mostrar o NOME do usuário (se existir), senão uma indicação ("Usuário desconhecido" / "Web")
            if origin_uid == "web":
                user_display_name = "Fonte: Web"
            else:
                user_obj = users_map.get(str(origin_uid), {})
                user_display_name = user_obj.get("name") if user_obj and user_obj.get("name") else str(origin_uid)

            initials = "".join([p[0] for p in str(user_display_name).split()[:2]]).upper() or "U"
            title_raw = str(result_data.get('título') or result_data.get('titulo') or '(Sem título)')
            resumo_raw = str(result_data.get('resumo') or result_data.get('abstract') or "")
            year = result_data.get('ano') or result_data.get('year') or ""
            country = result_data.get('país') or result_data.get('pais') or result_data.get('country') or ""
            
            st.markdown(f"""
            <div class="card">
                <div style="display:flex; gap:12px; align-items:center;">
                    <div class="avatar">{escape_html(initials)}</div>
                    <div style="flex:1;">
                        <div class="card-title">{highlight_search_terms(title_raw, query)}</div>
                        <div class="small-muted">De <strong>{escape_html(user_display_name)}</strong> • {escape_html(result_data.get('autor', ''))}</div>
                        <div class="small-muted">Ano: {escape_html(str(year))} • País: {escape_html(country)}</div>
                        <div style="margin-top:6px;font-size:13px;color:#e6e8ea;">{highlight_search_terms(resumo_raw, query) if resumo_raw else ''}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
            
            a1, a2 = st.columns([0.28, 0.72])
            with a1:
                if st.button("⭐ Favoritar", key=f"fav_search_{orig_i}_{USERNAME}", use_container_width=True):
                    if add_to_favorites(result_data): st.toast("Adicionado!", icon="⭐")
                    else: st.toast("Já está nos favoritos.")
            with a2:
                if st.button("🔎 Ver detalhes", key=f"view_search_{orig_i}_{USERNAME}", use_container_width=True):
                    st.session_state.search_view_index = int(orig_i)
                    safe_rerun()
        
        st.markdown("---")
        p1, p2, p3 = st.columns([1,1,1])
        with p1: 
            if st.button("◀", disabled=(page <= 1), key=f"search_prev_{USERNAME}"):
                st.session_state.search_page -= 1; safe_rerun()
        with p2: st.markdown(f"<div style='text-align:center; padding-top:8px'><b>Página {page}/{max_pages}</b></div>", unsafe_allow_html=True)
        with p3: 
            if st.button("▶", disabled=(page >= max_pages), key=f"search_next_{USERNAME}"):
                st.session_state.search_page += 1; safe_rerun()

        if st.session_state.get("search_view_index") is not None:
            vi = int(st.session_state.search_view_index)
            if 0 <= vi < len(results_df):
                det = results_df.loc[vi].to_dict()
                det = enrich_article_metadata(det)
                origin_user = det.get("_artemis_username", "N/A")
                # display-friendly name
                if origin_user == "web":
                    origin_display = "Fonte: Web"
                else:
                    ou = users_map.get(str(origin_user), {})
                    origin_display = ou.get("name") if ou and ou.get("name") else str(origin_user)
                st.markdown("## Detalhes do Registro")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{escape_html(det.get('título','— Sem título —'))}**")
                    st.markdown(f"**Autor(es):** {escape_html(det.get('autor','— —'))}")
                    st.markdown(f"**Ano:** {escape_html(str(det.get('ano', det.get('year','— —'))))}")
                    st.markdown(f"**País:** {escape_html(det.get('país', det.get('pais', det.get('country','— —'))))}")
                    st.markdown(f"**Fonte:** {escape_html(origin_display)}")
                    
                    if det.get('doi'):
                        doi_link = f"https://doi.org/{det.get('doi')}"
                        st.markdown(f"**DOI:** [{det.get('doi')}]({doi_link})")
                    elif det.get('url'):
                        st.markdown(f"**Link:** [{det.get('url')}]({det.get('url')})")
                    
                    st.markdown("---")
                    st.markdown("**Resumo**")
                    st.markdown(escape_html(det.get('resumo','Resumo não disponível.')))
                
                with col2:
                    # Informações adicionais
                    st.markdown("**Informações**")
                    if det.get('similarity'):
                        st.metric("Similaridade", f"{det['similarity']:.2f}")
                    
                    # Botões de ação
                    if st.button("⭐ Adicionar aos Favoritos", key=f"fav_search_detail_{vi}", use_container_width=True):
                        if add_to_favorites(det): 
                            st.toast("Adicionado aos favoritos!", icon="⭐")
                        else: 
                            st.toast("Já está nos favoritos.")
                    
                    if st.button("📝 Anotações", key=f"notes_search_{vi}", use_container_width=True):
                        st.session_state.page = "anotacoes"
                        safe_rerun()

                st.markdown("---")
                st.markdown("### ✉️ Contatar autor")
                if origin_user != "N/A" and origin_user != "web":
                    with st.form(key=f"inline_compose_{vi}_{USERNAME}"):
                        subj_fill = st.text_input("Assunto:", value=f"Sobre: {det.get('título', '')[:50]}...")
                        body_fill = st.text_area("Mensagem:", value=f"Olá {origin_display},\n\nVi seu registro '{det.get('título', '')}' na plataforma e gostaria de conversar.\n\n")
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.form_submit_button("✉️ Enviar"):
                                # aqui enviamos para o CPF (origin_user) internamente
                                send_message(USERNAME, str(origin_user), subj_fill, body_fill)
                                st.success(f"Mensagem enviada para {origin_display}.")
                                time.sleep(2); safe_rerun()
                        with c2:
                            if st.form_submit_button("❌ Cancelar"):
                                st.session_state.search_view_index = None; safe_rerun()
                else:
                    st.warning("Origem indisponível para contato (registro público/web).")
                    
                if st.button("⬅️ Voltar para resultados", key=f"back_search_{vi}"):
                    st.session_state.search_view_index = None
                    safe_rerun()
    else:
        st.info("Nenhum resultado de busca (executar uma pesquisa com dados carregados).")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: mensagens (CORREÇÃO do erro de attachment)
# -------------------------
elif st.session_state.page == "mensagens":
    st.markdown("<div class='glass-box' style='position:relative;padding:12px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("✉️ Mensagens")

    tab_inbox, tab_sent, tab_compose = st.tabs(["Caixa de Entrada", "Enviados", "Escrever Nova"])

    with tab_inbox:
        if st.session_state.get("view_message_id"):
            msg = next((m for m in all_msgs if m['id'] == st.session_state.view_message_id), None)
            if msg:
                mark_message_read(msg['id'], USERNAME)
                st.markdown(f"**De:** {escape_html(msg.get('from'))}")
                st.markdown(f"**Assunto:** {escape_html(msg.get('subject'))}")
                st.markdown(f"**Data:** {datetime.fromisoformat(msg.get('ts')).strftime('%d/%m/%Y %H:%M')}")
                st.markdown("---")
                st.markdown(f"<div style='white-space:pre-wrap; padding:10px; border-radius:8px; background:rgba(0,0,0,0.2);'>{escape_html(msg.get('body'))}</div>", unsafe_allow_html=True)
                
                # CORREÇÃO: Verificação segura do anexo
                if msg.get('attachment') and isinstance(msg['attachment'], dict) and msg['attachment'].get('path') and Path(msg['attachment']['path']).exists():
                    with open(msg['attachment']['path'], "rb") as fp:
                        st.download_button(
                            f"⬇️ Baixar anexo: {msg['attachment']['name']}", 
                            data=fp, 
                            file_name=msg['attachment']['name'],
                            use_container_width=True
                        )

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("↩️ Voltar", key=f"back_inbox_{USERNAME}", use_container_width=True):
                        st.session_state.view_message_id = None; safe_rerun()
                with c2:
                    if st.button("↪️ Responder", key=f"reply_msg_{USERNAME}", use_container_width=True):
                        st.session_state.reply_message_id = msg['id']; st.session_state.view_message_id = None; safe_rerun()
                with c3:
                    if st.button("🗑️ Excluir", key=f"del_inbox_msg_{msg['id']}_{USERNAME}", use_container_width=True):
                        if delete_message(msg['id'], USERNAME): 
                            st.session_state.view_message_id = None; 
                            st.toast("Mensagem excluída.")
                            safe_rerun()
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
                    if st.button("Ler", key=f"read_{msg['id']}_{USERNAME}", use_container_width=True):
                        st.session_state.view_message_id = msg['id']; safe_rerun()
                st.markdown("---")

    with tab_sent:
        sent_msgs = get_user_messages(USERNAME, 'sent')
        if not sent_msgs:
            st.info("Nenhuma mensagem enviada.")
        for msg in sent_msgs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{escape_html(msg.get('subject', '(sem assunto)'))}**")
                st.markdown(f"<span class='small-muted'>Para: {escape_html(msg.get('to', '...'))} em {datetime.fromisoformat(msg.get('ts')).strftime('%d/%m/%Y %H:%M')}</span>", unsafe_allow_html=True)
            with col2:
                if st.button("🗑️ Excluir", key=f"del_sent_{msg['id']}_{USERNAME}", use_container_width=True):
                    if delete_message(msg['id'], USERNAME): 
                        st.toast("Mensagem excluída.")
                        safe_rerun()
            st.markdown("---")

    with tab_compose:
        if st.session_state.get("reply_message_id"):
            original_msg = next((m for m in all_msgs if m['id'] == st.session_state.reply_message_id), None)
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
            if not all_users:
                st.warning("Nenhum outro usuário cadastrado — não é possível enviar mensagens.")
            else:
                to_user = st.selectbox("Para:", options=all_users, index=all_users.index(default_to) if default_to in all_users else 0)
                subject = st.text_input("Assunto:", value=default_subj)
                body = st.text_area("Mensagem:", height=200, value=default_body)
                attachment = st.file_uploader("Anexo (opcional)", type=['pdf', 'txt', 'doc', 'docx', 'xls', 'xlsx'])
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("✉️ Enviar Mensagem", use_container_width=True):
                        if to_user:
                            send_message(USERNAME, to_user, subject, body, attachment)
                            st.success(f"Mensagem enviada para {to_user}!")
                            st.session_state.reply_message_id = None
                            time.sleep(1); safe_rerun()
                        else:
                            st.warning("Selecione um destinatário.")
                with col2:
                    if st.form_submit_button("❌ Cancelar", use_container_width=True):
                        st.session_state.reply_message_id = None
                        safe_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: config
# -------------------------
elif st.session_state.page == "config":
    st.markdown("<div class='glass-box' style='position:relative;padding:12px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("⚙️ Configurações")
    s = get_settings()

    font_scale = st.slider("Escala de fonte", 0.7, 2.0, float(s.get("font_scale",1.0)), 0.1, key="cfg_font_scale")

    if st.button("Aplicar configurações", key=f"apply_cfg_{USERNAME}"):
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
