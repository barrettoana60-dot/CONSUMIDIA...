# dashboard_nugep_pqr_final.py
# VERSÃO FINAL CORRIGIDA (13/10/2025)
# COM: Tutorial de primeiro uso, página de Recomendações dedicada e Mapa Mental Interativo com nós clicáveis.
# ALTERAÇÕES: Adicionado guia para novos usuários. Refatorada a lógica de Recomendações com descoberta inteligente.

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
    TfidfVectorizer = None
    cosine_similarity = None

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

def hex_to_rgba(h, alpha):
    """Converte cor hex (ex: #ffffff) para string rgba (ex: rgba(255,255,255,0.5))."""
    h = h.lstrip('#')
    return f"rgba({', '.join(str(i) for i in tuple(int(h[i:i+2], 16) for i in (0, 2, 4)))}, {alpha})"


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
# Funções de Busca & Recomendação
# -------------------------
@st.cache_data(ttl=600) # Cache para não re-processar a cada clique
def collect_latest_backups():
    """
    Scans the BACKUPS_DIR, finds all user backup CSVs,
    and consolidates them into a single DataFrame for searching.
    """
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
                    df_temp['_artemis_username'] = username
                    all_dfs.append(df_temp)
                except Exception as e:
                    print(f"Skipping unreadable backup {csv_file}: {e}")
                    continue
    
    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)

def highlight_search_terms(text, query):
    if not query or not text or not isinstance(text, str):
        return escape_html(text)
    
    safe_text = escape_html(text)
    
    highlighted_text = re.sub(
        f'({re.escape(query)})', 
        r'<span class="card-mark">\1</span>', 
        safe_text, 
        flags=re.IGNORECASE
    )
    return highlighted_text

def recomendar_artigos(temas_selecionados, df_total, top_n=10):
    """
    Recomenda artigos de um DataFrame com base em temas, usando TF-IDF e similaridade de cosseno.
    """
    if TfidfVectorizer is None or cosine_similarity is None:
        st.error("Bibliotecas de Machine Learning (scikit-learn) não estão instaladas. A recomendação não funcionará.")
        return pd.DataFrame()

    if df_total.empty or not temas_selecionados:
        return pd.DataFrame()

    # --- INÍCIO DA CORREÇÃO (AttributeError) ---
    # Inicializa um Pandas Series vazio para garantir que o tipo seja consistente
    corpus_series = pd.Series([''] * len(df_total), index=df_total.index, dtype=str)
    
    if 'título' in df_total.columns:
        corpus_series += df_total['título'].fillna('') + ' '
    if 'tema' in df_total.columns:
        corpus_series += df_total['tema'].fillna('') + ' '
    if 'resumo' in df_total.columns:
        corpus_series += df_total['resumo'].fillna('')
    
    df_total['corpus'] = corpus_series.str.lower()
    # --- FIM DA CORREÇÃO ---
    
    # Se o corpus estiver vazio após a tentativa, retorna um DataFrame vazio
    if df_total['corpus'].str.strip().eq('').all():
        st.warning("Nenhum conteúdo textual (título, tema, resumo) encontrado nos dados para gerar recomendações.")
        return pd.DataFrame()
    
    # Vetorizar o corpus
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df_total['corpus'])
    
    # Criar um vetor para a consulta (temas selecionados)
    query_text = ' '.join(temas_selecionados).lower()
    query_vector = vectorizer.transform([query_text])
    
    # Calcular a similaridade de cosseno
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Obter os índices dos artigos mais similares
    related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    
    # Filtrar para mostrar apenas os resultados com alguma similaridade
    similar_indices = [i for i in related_docs_indices if cosine_similarities[i] > 0.05]
    
    if not similar_indices:
        return pd.DataFrame()

    # Retornar o DataFrame com os artigos recomendados
    recomendados_df = df_total.iloc[similar_indices].copy()
    recomendados_df['similarity'] = cosine_similarities[similar_indices]
    
    return recomendados_df.drop(columns=['corpus'])

# Lista de stop words para melhorar a extração de temas
PORTUGUESE_STOP_WORDS = [
    'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma',
    'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao',
    'ele', 'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há',
    'nos', 'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela',
    'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas',
    'me', 'esse', 'eles', 'estão', 'tinha', 'foram', 'essa', 'num', 'nem',
    'suas', 'meu', 'às', 'minha', 'têm', 'numa', 'pelos', 'elas', 'havia', 'seja',
    'qual', 'será', 'nós', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas',
    'este', 'fosse', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas',
    'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela',
    'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas',
    'isto', 'aquilo', 'estou', 'está', 'estamos', 'estão', 'estive', 'esteve',
    'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam', 'estivera',
    'estivéramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos',
    'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'há', 'havemos',
    'hão', 'houve', 'houvemos', 'houveram', 'houvera', 'houvéramos', 'haja',
    'hajamos', 'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver',
    'houvermos', 'houverem', 'houverei', 'houverá', 'houveremos', 'houverão',
    'houveria', 'houveríamos', 'houveriam', 'sou', 'somos', 'são', 'era', 'éramos',
    'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'fôramos', 'seja', 'sejamos',
    'sejam', 'fosse', 'fôssemos', 'fossem', 'for', 'formos', 'forem', 'serei',
    'será', 'seremos', 'serão', 'seria', 'seríamos', 'seriam', 'tenho', 'tem',
    'temos', 'tém', 'tinha', 'tínhamos', 'tinham', 'tive', 'teve', 'tivemos',
    'tiveram', 'tivera', 'tivéramos', 'tenha', 'tenhamos', 'tenham', 'tivesse',
    'tivéssemos', 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'terá',
    'teremos', 'terão', 'teria', 'teríamos', 'teriam'
]

@st.cache_data(ttl=600) # Cache para não re-processar a cada clique
def extract_popular_themes_from_data(df_total, top_n=30):
    """
    Extrai os temas/palavras-chave mais populares de um DataFrame consolidado usando TF-IDF.
    """
    if TfidfVectorizer is None:
        return [] # Retorna vazio se a biblioteca não estiver disponível

    if df_total.empty:
        return []

    corpus_series = pd.Series([''] * len(df_total), index=df_total.index, dtype=str)
    
    # Concatena colunas relevantes para formar o corpus de texto
    for col in ['título', 'tema', 'resumo', 'titulo', 'abstract']:
        if col in df_total.columns:
            corpus_series += df_total[col].fillna('') + ' '
    
    df_total['corpus'] = corpus_series.str.lower()

    if df_total['corpus'].str.strip().eq('').all():
        return []

    try:
        # Usa TF-IDF para encontrar os termos mais relevantes
        vectorizer = TfidfVectorizer(stop_words=PORTUGUESE_STOP_WORDS, max_features=1000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(df_total['corpus'])
        
        # Soma os scores de TF-IDF de cada termo em todos os documentos
        sum_tfidf = tfidf_matrix.sum(axis=0)
        
        # Mapeia os índices dos termos para seus nomes
        words = vectorizer.get_feature_names_out()
        tfidf_scores = [(words[i], sum_tfidf[0, i]) for i in range(len(words))]
        
        # Ordena por score e retorna os top_n termos
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        
        return [word for word, score in sorted_scores[:top_n]]
    except Exception as e:
        print(f"Erro ao extrair temas populares: {e}")
        return [] # Em caso de erro, retorna uma lista vazia
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
    temp_data_to_check.pop('similarity', None) # Ignorar campo de similaridade
    existing_contents = []
    for fav in favorites:
        temp_existing = fav["data"].copy()
        temp_existing.pop('_artemis_username', None)
        temp_existing.pop('similarity', None)
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
    G = nx.Graph()
    if df is None:
        return G

    created_edges = 0
    for index, row in df.iterrows():
        row_nodes = []
        
        for col_name, cell_value in row.items():
            val = str(cell_value or '').strip()
            if val and col_name.lower() not in ["registro", "outro"]:
                tipo = str(col_name).strip().capitalize()
                node_id = f"{tipo}: {val}"
                G.add_node(node_id, tipo=tipo, label=val)
                row_nodes.append(node_id)
        
        for i in range(len(row_nodes)):
            for j in range(i + 1, len(row_nodes)):
                if not G.has_edge(row_nodes[i], row_nodes[j]):
                    G.add_edge(row_nodes[i], row_nodes[j])
                    created_edges += 1

    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    if not silent:
        if created_edges > 0:
            st.success(f"Grafo criado com sucesso, com {G.number_of_nodes()} nós e {created_edges} arestas.")
        else:
            st.info("Grafo de pontos criado, mas a planilha parece estar vazia ou não gerou conexões significativas.")
            
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
# Session defaults & settings
# -------------------------
_defaults = {
    "authenticated": False, "username": None, "user_obj": None, "df": None,
    "G": nx.Graph(), "notes": "", "autosave": False, "page": "planilha",
    "restored_from_saved": False, "favorites": [], "reply_message_id": None,
    "view_message_id": None, "sent_messages_view": False,
    "search_results": pd.DataFrame(), "search_page": 1, "search_query_meta": {"col": None,"query":""},
    "search_view_index": None, "compose_inline": False, "compose_open": False,
    "last_backup_path": None, "selected_node": None,
    "tutorial_completed": False, # Flag para o novo tutorial
    "settings": {
        "plot_height": 720,
        "font_scale": 1.0,
        "node_opacity": 1.0,
    }
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def get_settings():
    return st.session_state.get("settings", _defaults["settings"])

def clean_for_json(d):
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
    try:
        data = {
            "notes": st.session_state.get("notes",""),
            "uploaded_name": st.session_state.get("uploaded_name", None),
            "favorites": st.session_state.get("favorites", []),
            "settings": st.session_state.get("settings", {}),
            "last_backup_path": st.session_state.get("last_backup_path"),
            "tutorial_completed": st.session_state.get("tutorial_completed", False) # Salva o estado do tutorial
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
        print(f"Erro ao salvar estado do usuário: {e}")
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

        st.session_state.notes = meta.get("notes", st.session_state.notes)
        st.session_state.uploaded_name = meta.get("uploaded_name", st.session_state.get("uploaded_name"))
        st.session_state.favorites = meta.get("favorites", st.session_state.favorites)
        st.session_state.last_backup_path = meta.get("last_backup_path", st.session_state.last_backup_path)
        st.session_state.tutorial_completed = meta.get("tutorial_completed", False) # Restaura o estado do tutorial
        
        if "settings" in meta:
            st.session_state.settings.update(meta.get("settings", {}))
        
        backup_path = st.session_state.get("last_backup_path")
        if backup_path and os.path.exists(backup_path):
            try:
                df = pd.read_csv(backup_path)
                st.session_state.df = df
                st.session_state.G = criar_grafo(df, silent=True)
                st.toast(f"Planilha '{os.path.basename(backup_path)}' restaurada automaticamente.", icon="📄")
            except Exception as e:
                st.error(f"Falha ao restaurar o backup da sua planilha: {e}")
                st.session_state.last_backup_path = None
        
        st.session_state.restored_from_saved = True
        st.toast("Progresso anterior restaurado.", icon="👍")
    except Exception as e:
        st.error(f"Erro ao restaurar seu progresso: o arquivo de estado pode estar corrompido. Erro: {e}")


# apply theme and font CSS based on settings immediately
s = get_settings()
apply_global_styles(s.get("font_scale", 1.0))

# -------------------------
# First-run: garantir interesses do usuário
# -------------------------
def ensure_user_interests_fixed():
    users = load_users() or {}
    saved = st.session_state.get("user_interests") or users.get(USERNAME, {}).get("interests") or []

    fixed_defaults = [
        "documentação",
        "inovação tecnológica",
        "nft",
        "cultura de inovação",
        "inovação social"
    ]

    # mostrar apenas se não houver interesses salvos
    if not saved:
        st.markdown("<div class='glass-box' style='max-width:900px;margin:10px auto;padding:12px;'>", unsafe_allow_html=True)
        st.subheader("Escolha seus interesses iniciais")
        st.caption("Selecione os temas que deseja priorizar em Recomendações.")
        chosen = st.multiselect("Interesses (mínimo 1):", options=fixed_defaults, default=[], key="ui_initial_interests_fixed")
        cols = st.columns([3,1])
        with cols[0]:
            if st.button("Salvar interesses"):
                chosen_list = st.session_state.get("ui_initial_interests_fixed", [])
                if not chosen_list:
                    st.warning("Selecione pelo menos um interesse.")
                else:
                    users = load_users() or {}
                    users.setdefault(USERNAME, {}).update(users.get(USERNAME, {}))
                    users[USERNAME]["interests"] = chosen_list
                    save_users(users)
                    st.session_state["user_interests"] = chosen_list
                    try:
                        save_user_state_minimal(USER_STATE)
                    except Exception:
                        pass
                    st.success("Interesses salvos — serão usados para filtrar suas recomendações.")
                    safe_rerun()
        with cols[1]:
            if st.button("Pular por enquanto"):
                st.session_state["user_interests"] = []
                try:
                    save_user_state_minimal(USER_STATE)
                except Exception:
                    pass
                safe_rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.session_state["user_interests"] = saved

# chama no carregamento
ensure_user_interests_fixed()

# -------------------------
# Função de recomendações por usuário (utilizada na página de Recomendações)
# -------------------------
def get_recommendations_for_current_user(df_total, top_n=10):
    temas = st.session_state.get("user_interests", []) or []
    if not temas:
        return pd.DataFrame()

    # primeiro tenta pipeline TF-IDF se disponível
    try:
        recs = recomendar_artigos(temas, df_total, top_n=top_n) if (TfidfVectorizer is not None and cosine_similarity is not None) else pd.DataFrame()
    except Exception:
        recs = pd.DataFrame()

    # fallback simples: filtrar linhas que contenham qualquer termo em colunas textuais
    if recs.empty:
        cols_to_search = [c for c in ['título','titulo','tema','resumo','abstract'] if c in df_total.columns]
        if not cols_to_search:
            return pd.DataFrame()
        mask = pd.Series(False, index=df_total.index)
        for t in temas:
            t_low = str(t).lower()
            for c in cols_to_search:
                try:
                    mask = mask | df_total[c].fillna("").astype(str).str.lower().str.contains(re.escape(t_low))
                except Exception:
                    try:
                        mask = mask | df_total[c].fillna("").astype(str).str.lower().str.contains(t_low)
                    except Exception:
                        pass
        filtered = df_total[mask]
        return filtered.head(top_n).copy()
    else:
        # garante presença de ao menos um tema
        def matches_theme_row(row):
            text = " ".join(str(row.get(c, "")) for c in ['título', 'titulo', 'tema', 'resumo', 'abstract']).lower()
            return any(t.lower() in text for t in temas)
        mask = recs.apply(matches_theme_row, axis=1)
        recs = recs[mask].copy()
        return recs.head(top_n)

# -------------------------
# (o resto do arquivo segue com suas páginas e lógica originais)
# -------------------------

# Render top navigation (exemplo simples)
nav_cols = st.columns([1,1,1,1,1])
with nav_cols[0]:
    if st.button("Planilha"):
        st.session_state.page = "planilha"
        safe_rerun()
with nav_cols[1]:
    if st.button("Mapa Mental"):
        st.session_state.page = "mapa"
        safe_rerun()
with nav_cols[2]:
    if st.button("Recomendações"):
        st.session_state.page = "recomendações"
        safe_rerun()
with nav_cols[3]:
    if st.button("Mensagens"):
        st.session_state.page = "mensagens"
        safe_rerun()
with nav_cols[4]:
    if st.button("Configurações"):
        st.session_state.page = "config"
        safe_rerun()

st.markdown("---")

# -------------------------
# Page: Planilha (upload / criar grafo)
# -------------------------
if st.session_state.page == "planilha":
    st.markdown("<div class='glass-box' style='position:relative;padding:12px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Upload de planilha / Visualização")
    uploaded = st.file_uploader("Envie sua planilha (CSV/XLSX)", type=['csv','xls','xlsx'])
    if uploaded:
        try:
            df = read_spreadsheet(uploaded)
            st.session_state.df = df
            st.session_state.uploaded_name = uploaded.name
            st.success("Planilha carregada.")
            # salvar backup simples
            user_backup_dir = BACKUPS_DIR / USERNAME
            user_backup_dir.mkdir(exist_ok=True)
            backup_path = user_backup_dir / f"{int(time.time())}_{uploaded.name}"
            try:
                df.to_csv(backup_path, index=False)
                st.session_state.last_backup_path = str(backup_path)
            except Exception:
                pass
            # criar grafo
            st.session_state.G = criar_grafo(df)
            save_user_state_minimal(USER_STATE)
        except Exception as e:
            st.error(f"Erro ao ler a planilha: {e}")
    else:
        st.info("Nenhum arquivo carregado. Você pode restaurar o último backup ou enviar um novo arquivo.")
        if st.session_state.get("last_backup_path"):
            st.markdown(f"Último backup: {st.session_state.get('last_backup_path')}")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: Mapa Mental (preserva visual e adiciona edição embutida)
# -------------------------
elif st.session_state.page == "mapa":
    st.markdown("<div class='glass-box' style='position:relative;padding:12px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Mapa Mental")

    G = st.session_state.get("G", nx.Graph())
    if G is None or G.number_of_nodes() == 0:
        st.info("Mapa mental vazio. Carregue uma planilha para gerar o grafo.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # --- visualização (mantida igual ao original) ---
        pos = nx.spring_layout(G, seed=42)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', hoverinfo='none', line=dict(width=1))

        node_x, node_y, labels = [], [], []
        for n in G.nodes():
            x,y = pos[n]
            node_x.append(x); node_y.append(y); labels.append(G.nodes[n].get('label', n))
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=labels, textposition="top center",
            marker=dict(size=16, opacity=float(st.session_state.settings.get("node_opacity",1.0)))
        )

        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest'))
        fig.update_layout(height=int(st.session_state.settings.get("plot_height",720)))
        st.plotly_chart(fig, use_container_width=True)

        # --- edição embutida (expander dentro da mesma área) ---
        with st.expander("Editar nó (edição embutida na visualização)"):
            node_list = list(G.nodes())
            if not node_list:
                st.info("Nenhum nó disponível para edição.")
            else:
                selected = st.selectbox("Selecione nó:", options=node_list, index=node_list.index(st.session_state.get("selected_node")) if st.session_state.get("selected_node") in node_list else 0)
                st.session_state.selected_node = selected

                edit_label = st.text_input("Etiqueta (label):", value=G.nodes[selected].get("label", selected), key="edit_node_label")
                colA, colB = st.columns([1,1])
                with colA:
                    if st.button("Salvar alterações no nó"):
                        newlabel = st.session_state.get("edit_node_label", G.nodes[selected].get("label", selected))
                        G.nodes[selected]['label'] = newlabel
                        st.session_state.G = G
                        save_user_state_minimal(USER_STATE)
                        st.success("Rótulo do nó atualizado.")
                        safe_rerun()
                with colB:
                    if st.button("Remover nó selecionado"):
                        G.remove_node(selected)
                        st.session_state.G = G
                        save_user_state_minimal(USER_STATE)
                        st.success("Nó removido.")
                        safe_rerun()

                st.markdown("---")
                st.markdown("### Anexar / ver imagem 3D (apenas visualização de imagem fornecida)")
                uploaded = st.file_uploader("Enviar imagem (jpg/png) para anexar ao nó", type=['jpg','jpeg','png'])
                if uploaded:
                    info = _local_upload_attachment(USERNAME, uploaded)
                    G.nodes[selected]['image'] = info['path']
                    st.session_state.G = G
                    save_user_state_minimal(USER_STATE)
                    st.success("Imagem anexada ao nó.")

                # mostrar imagem anexada (se houver)
                imgpath = G.nodes[selected].get('image')
                if imgpath and os.path.exists(imgpath):
                    st.image(imgpath, caption=f"Imagem anexada ao nó: {selected}", use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: Recomendações (varredura filtrada por interesses)
# -------------------------
elif st.session_state.page == "recomendações":
    st.markdown("<div class='glass-box' style='padding:12px;'>", unsafe_allow_html=True)
    st.subheader("🔎 Recomendações filtradas pelos seus interesses")
    df_total = collect_latest_backups()  # junta backups
    # também considerar planilha carregada na sessão
    if st.session_state.get("df") is not None and not st.session_state.df.empty:
        try:
            df_total = pd.concat([df_total, st.session_state.df], ignore_index=True) if not df_total.empty else st.session_state.df.copy()
        except Exception:
            pass

    if df_total is None or df_total.empty:
        st.info("Não há planilhas disponíveis (backups ou upload). Faça upload de uma planilha para gerar recomendações.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        user_interests = st.session_state.get("user_interests", [])
        if not user_interests:
            st.info("Você ainda não selecionou interesses. Vá para a tela inicial para marcá-los.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Seus interesses salvos:** {', '.join(user_interests)}")
            # permitir escolher um subconjunto dos interesses salvos para esta busca
            chosen_now = st.multiselect("Filtrar por (apenas os selecionados serão usados agora):", options=user_interests, default=user_interests, key="ui_reco_filter_now")
            if st.button("Mostrar Recomendações"):
                temas = chosen_now or user_interests
                # busca simples: qualquer interesse presente em colunas relevantes
                cols_to_search = [c for c in ['título','titulo','tema','resumo','abstract'] if c in df_total.columns]
                if not cols_to_search:
                    st.warning("As planilhas não possuem colunas textuais (título/tema/resumo).")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    mask = pd.Series(False, index=df_total.index)
                    for tema in temas:
                        # busca case-insensitive em cada coluna
                        t = str(tema).lower()
                        col_mask = pd.Series(False, index=df_total.index)
                        for c in cols_to_search:
                            try:
                                col_mask = col_mask | df_total[c].fillna("").astype(str).str.lower().str.contains(re.escape(t))
                            except Exception:
                                # fallback: contains simples sem regex
                                try:
                                    col_mask = col_mask | df_total[c].fillna("").astype(str).str.lower().str.contains(t)
                                except Exception:
                                    pass
                        mask = mask | col_mask

                    results = df_total[mask].copy()
                    if results.empty:
                        st.info("Nenhum item nas planilhas corresponde aos interesses selecionados.")
                    else:
                        st.write(f"Resultados encontrados: {len(results)}")
                        # apresentar colunas relevantes (priorizar título/tema/resumo)
                        display_cols = [c for c in ['título','titulo','tema','resumo','abstract','_artemis_username'] if c in results.columns]
                        st.dataframe(results[display_cols].head(200))
                        # permitir salvar linha nos favoritos
                        for i, row in results.head(50).iterrows():
                            cols = st.columns([4,1])
                            with cols[0]:
                                st.markdown(f"**{ row.get('título') or row.get('titulo') or row.get('tema') or '—' }**")
                                if 'resumo' in row and pd.notna(row.get('resumo')):
                                    st.caption(str(row.get('resumo'))[:200] + "...")
                            with cols[1]:
                                if st.button(f"⭐ Favoritar {i}", key=f"fav_reco_{i}"):
                                    added = add_to_favorites(row.to_dict())
                                    if added:
                                        st.success("Adicionado aos favoritos.")
                                    else:
                                        st.info("Já estava nos favoritos.")
            st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: Mensagens (inbox + compose)
# -------------------------
elif st.session_state.page == "mensagens":
    st.markdown("<div class='glass-box' style='position:relative;padding:12px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Mensagens")
    tab_inbox, tab_compose, tab_sent = st.tabs(["Caixa de Entrada", "Escrever", "Enviadas"])

    with tab_inbox:
        inbox = get_user_messages(USERNAME, box_type='inbox')
        if not inbox:
            st.info("Nenhuma mensagem na sua caixa de entrada.")
        else:
            for m in inbox:
                cols = st.columns([6,1,1])
                with cols[0]:
                    st.markdown(f"**De:** {m.get('from')} — **Assunto:** {m.get('subject')}")
                    st.markdown(f"{m.get('body')[:300]}...")
                with cols[1]:
                    if st.button("Visualizar", key=f"view_{m.get('id')}"):
                        st.session_state.view_message_id = m.get('id')
                        safe_rerun()
                with cols[2]:
                    if st.button("Responder", key=f"reply_{m.get('id')}"):
                        st.session_state.reply_message_id = m.get('id')
                        st.session_state.page = "mensagens"
                        st.session_state.compose_open = True
                        safe_rerun()
        # visualizar mensagem selecionada
        if st.session_state.get("view_message_id"):
            vm = next((x for x in inbox if x.get('id') == st.session_state.get("view_message_id")), None)
            if vm:
                st.markdown("---")
                st.markdown(f"**De:** {vm.get('from')} — **Assunto:** {vm.get('subject')}")
                st.markdown(vm.get('body'))
                if vm.get('attachment'):
                    ap = vm.get('attachment').get('path')
                    if ap and os.path.exists(ap):
                        st.markdown(f"Anexo: {vm.get('attachment').get('name')}")
    with tab_sent:
        sent = get_user_messages(USERNAME, box_type='sent') if False else [m for m in load_all_messages() if m.get('from') == USERNAME]
        if not sent:
            st.info("Você não enviou mensagens.")
        else:
            for m in sent:
                st.markdown(f"Para: {m.get('to')} — {m.get('subject')} — {m.get('ts')}")
    with tab_compose:
        if st.session_state.get("reply_message_id"):
            reply_to_id = st.session_state.reply_message_id
            all_msgs = load_all_messages()
            original_msg = next((m for m in all_msgs if m['id'] == reply_to_id), None)
            
            if original_msg:
                st.info(f"Respondendo a: {original_msg['from']}")
                default_to = original_msg['from']
                default_subj = f"Re: {original_msg['subject']}"
                default_body = f"\n\n---\nEm resposta a:\n{original_msg['body']}"
            else:
                st.session_state.reply_message_id = None
                default_to, default_subj, default_body = "", "", ""
        elif st.session_state.get("compose_open"):
            default_to = st.session_state.get("compose_to", "")
            default_subj = st.session_state.get("compose_subject", "")
            default_body = st.session_state.get("compose_prefill", "")
            st.session_state.compose_open = False
        else:
            default_to, default_subj, default_body = "", "", ""

        with st.form("compose_form", clear_on_submit=True):
            users = load_users() or {}
            all_users = [u for u in users.keys() if u != USERNAME]
            
            to_user = st.selectbox("Para:", options=all_users, index=all_users.index(default_to) if default_to in all_users else 0)
            subject = st.text_input("Assunto:", value=default_subj)
            body = st.text_area("Mensagem:", height=200, value=default_body)
            attachment = st.file_uploader("Anexo (opcional)")
            
            if st.form_submit_button("✉️ Enviar Mensagem"):
                if not to_user:
                    st.warning("Selecione um destinatário.")
                else:
                    send_message(USERNAME, to_user, subject, body, attachment)
                    st.success(f"Mensagem enviada para {to_user}!")
                    st.session_state.reply_message_id = None
                    time.sleep(1)
                    safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    
# -------------------------
# Page: Configurações
# -------------------------
elif st.session_state.page == "config":
    st.markdown("<div class='glass-box' style='position:relative;padding:12px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("⚙️ Configurações")
    s = get_settings()

    plot_height = st.number_input("Altura do gráfico (px)", value=int(s.get("plot_height",720)), step=10, key="cfg_plot_height")
    font_scale = st.slider("Escala de fonte (aplicada a todo o programa)", min_value=0.7, max_value=2.0, value=float(s.get("font_scale",1.0)), step=0.1, key="cfg_font_scale")

    st.markdown("---")
    st.markdown("**Configurações de Visualização do Mapa**")
    node_opacity = st.slider("Opacidade dos Nós no Mapa", min_value=0.1, max_value=1.0, value=float(s.get("node_opacity", 1.0)), step=0.05, key="cfg_node_opacity")


    if st.button("Aplicar configurações"):
        st.session_state.settings["plot_height"] = int(plot_height)
        st.session_state.settings["font_scale"] = float(font_scale)
        st.session_state.settings["node_opacity"] = float(node_opacity)

        ok = save_user_state_minimal(USER_STATE)
        apply_global_styles(font_scale)

        if ok:
            st.success("Configurações aplicadas e salvas.")
        
        time.sleep(0.5)
        safe_rerun()

    st.markdown("---")
    st.markdown("**Acessibilidade**")
    st.markdown("- Use *Escala de fonte* para aumentar ou diminuir o tamanho do texto em todo o programa.")
    st.markdown("- O programa utiliza um tema escuro fixo para garantir bom contraste.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Fallback
# -------------------------
else:
    st.info("Página não encontrada — selecione uma aba no topo.")
