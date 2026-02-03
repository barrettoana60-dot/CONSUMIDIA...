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
matplotlib.use('Agg')

from streamlit_agraph import agraph, Node, Edge, Config

# Optional ML libs
try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.manifold import TSNE
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from textblob import TextBlob
except Exception:
    joblib = None
    TfidfVectorizer = None
    cosine_similarity = None
    KMeans = None
    LatentDirichletAllocation = None
    TSNE = None
    RandomForestRegressor = None
    LinearRegression = None

# bcrypt for password hashing
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
        st.rerun()
    except:
        st.experimental_rerun()

# --- NEW: Global Stop Words for Portuguese ---
PORTUGUESE_STOP_WORDS = [
    "a", "à", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "às", "até", "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "depois", "do", "dos", "e", "é", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "essa", "essas", "esse", "esses", "esta", "está", "estas", "este", "estes", "eu", "foi", "fomos", "for", "foram", "fosse", "fossem", "fui", "há", "isso", "isto", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas", "muito", "na", "não", "nas", "nem", "no", "nos", "nossa", "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual", "quando", "que", "quem", "se", "sem", "ser", "será", "serei", "seremos", "seria", "seriam", "seu", "seus", "só", "somos", "sua", "suas", "também", "te", "tem", "têm", "tinha", "tinham", "tive", "tivemos", "tiver", "tiveram", "tivesse", "tivessem", "tu", "tua", "tuas", "um", "uma", "você", "vocês", "vos"
]

# -------------------------
# Base CSS - Aprimorado para Liquid Glass e Social Feed (CORRIGIDO E SIMPLIFICADO)
# -------------------------
BASE_CSS = r"""
:root{
    --glass-bg-dark: rgba(255,255,255,0.03);
    --muted-text-dark:#bfc6cc;
    --primary-color: #6c5ce7; /* Um roxo vibrante */
    --secondary-color: #00cec9; /* Um ciano para contraste */
    --background-gradient: linear-gradient(180deg, #071428 0%, #031926 100%);
    --card-bg: rgba(14, 25, 42, 0.7); /* Fundo do card semi-transparente */
    --border-color: rgba(42, 59, 82, 0.5); /* Borda mais suave */
    --text-color: #e0e0e0;
    --highlight-color: #fdbb2d; /* Amarelo para destaques */
}

body {
    transition: background-color .25s ease, color .25s ease;
    font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    color: var(--text-color);
}

.stApp {
    background: var(--background-gradient);
}

/* Liquid Glass Effect for main elements */
.glass-box, .card, .msg-card, .ai-response, .vision-analysis, .social-post-card, .folder-card {
    background: var(--card-bg);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(4, 9, 20, 0.37);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
    margin-bottom: 15px;
}

.glass-box:hover, .card:hover, .msg-card:hover, .social-post-card:hover, .folder-card:hover {
    box-shadow: 0 12px 40px 0 rgba(4, 9, 20, 0.5);
    border-color: rgba(var(--primary-color), 0.7);
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background: var(--primary-color) !important;
    color: white !important;
    border: none !important;
    padding: 10px 18px !important;
    border-radius: 10px !important;
    font-weight: 600;
    transition: transform 0.1s ease, opacity 0.1s ease, background-color 0.15s ease !important;
    box-shadow: 0 4px 10px rgba(108, 92, 231, 0.3);
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: #5a4cd0 !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(108, 92, 231, 0.4);
}
.stButton button[kind="secondary"] {
    background: var(--border-color) !important;
    color: var(--text-color) !important;
    box-shadow: none;
}

/* Inputs */
.stTextInput>div>div>input, .stSelectbox>div>div>select, .stTextArea>div>div>textarea, .stFileUploader>div>div {
    background-color: rgba(14, 25, 42, 0.5);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    color: var(--text-color);
    padding: 10px;
}
.stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus, .stTextArea>div>div>textarea:focus, .stFileUploader>div>div:focus-within {
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(108, 92, 231, 0.3);
}

/* Text and Headers */
h1, h2, h3, h4, h5, h6, p, label {
    color: var(--text-color);
}
.small-muted {
    color: var(--muted-text-dark);
}

/* Sidebar specific styling for glass effect */
.css-1lcbmhc { /* This targets the sidebar container */
    background: rgba(14, 25, 42, 0.9) !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-right: 1px solid var(--border-color);
}
/* Ensure all text elements within the sidebar inherit the correct color */
.css-1lcbmhc .st-emotion-cache-1r6slb0, .css-1lcbmhc .st-emotion-cache-1r6slb0 * {
    color: var(--text-color) !important;
}

/* Social Post Card (for Timeline) */
.social-post-card {
    border-left: 5px solid var(--primary-color);
}
.social-post-card .post-header {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
}
.social-post-card .post-avatar {
    width: 45px;
    height: 45px;
    border-radius: 50%;
    background: var(--secondary-color);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    color: white;
    margin-right: 10px;
    font-size: 1.1em;
}
.social-post-card .post-author {
    font-weight: 600;
    color: var(--text-color);
}
.social-post-card .post-timestamp {
    font-size: 0.75em;
    color: var(--muted-text-dark);
    margin-left: 8px;
}
.social-post-card .post-content {
    margin-top: 10px;
    color: var(--text-color);
}
.social-post-card .post-actions {
    display: flex;
    gap: 10px;
    margin-top: 15px;
}
.social-post-card .post-actions button {
    background: none !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-color) !important;
    box-shadow: none;
    padding: 6px 12px !important;
    border-radius: 8px !important;
    font-size: 0.9em;
}
.social-post-card .post-actions button:hover {
    background: rgba(var(--primary-color), 0.2) !important;
    border-color: var(--primary-color) !important;
    transform: translateY(-1px);
}

/* Folder Card (for My Research) */
.folder-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 15px;
    border-left: 5px solid var(--secondary-color);
}
.folder-card .folder-name {
    font-weight: 600;
    font-size: 1.1em;
    color: var(--text-color);
}
.folder-card .folder-meta {
    font-size: 0.8em;
    color: var(--muted-text-dark);
}
.folder-card .folder-actions {
    display: flex;
    gap: 8px;
}
.folder-card .folder-actions button {
    padding: 5px 10px !important;
    font-size: 0.8em;
    border-radius: 6px !important;
}
"""

# DEFAULT_CSS is no longer needed as BASE_CSS is comprehensive
# st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)
# st.markdown(f"<style>{DEFAULT_CSS}</style>", unsafe_allow_html=True) # REMOVED

# Apply the simplified BASE_CSS
st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)


st.markdown("<div style='text-align:center; padding-top:8px; padding-bottom:6px;'><h1 style='margin:0;color:#ffffff;'>NUGEP-PQR</h1></div>", unsafe_allow_html=True)
st.markdown("<hr style='margin-top:6px;margin-bottom:16px;border-color:#233447'/><br>", unsafe_allow_html=True) # Adicionado <br> para espaçamento

# -------------------------
# Storage & fallback paths (UPDATED for Email-based users)
# -------------------------
USERS_FILE = "users_email.json" # Changed to reflect email-based users
MESSAGES_FILE = "messages_email.json" # Changed to reflect email-based users
RESEARCH_DATA_FILE = "research_data.json" # New file for research data and folders
BACKUPS_DIR = Path("backups")
ATTACHMENTS_DIR = Path("user_files")
BACKUPS_DIR.mkdir(exist_ok=True)
ATTACHMENTS_DIR.mkdir(exist_ok=True)

# -------------------------
# User Management Functions (UPDATED for Email-based users)
# -------------------------
def load_users():
    if not Path(USERS_FILE).exists():
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

def hash_password(password):
    if BCRYPT_AVAILABLE:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    return password # Fallback if bcrypt is not available (NOT RECOMMENDED FOR PRODUCTION)

def check_password(password, hashed):
    if BCRYPT_AVAILABLE:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    return password == hashed # Fallback

def register_user(email, password, name, scholarship):
    users = load_users()
    if email in users:
        return False, "Este e-mail já está cadastrado."

    # Basic email format validation
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False, "Formato de e-mail inválido."

    hashed_password = hash_password(password)
    users[email] = {
        "password": hashed_password,
        "name": name,
        "scholarship": scholarship,
        "created_at": datetime.now().isoformat(),
        "settings": {},
        "notes": "",
        "favorites": [],
        "tutorial_completed": False,
        "folders": {}, # NEW: User's research folders
        "connections": [] # NEW: Social connections
    }
    save_users(users)
    return True, "Usuário registrado com sucesso!"

def authenticate_user(email, password):
    users = load_users()
    user = users.get(email)
    if user and check_password(password, user["password"]):
        return user
    return None

def get_user_data(email):
    users = load_users()
    return users.get(email)

def update_user_data(email, data):
    users = load_users()
    if email in users:
        users[email].update(data)
        save_users(users)
        return True
    return False

# -------------------------
# Session State Initialization (UPDATED for Email-based users and new features)
# -------------------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'user_obj' not in st.session_state:
    st.session_state.user_obj = None
if 'page' not in st.session_state:
    st.session_state.page = "login" # Default to login page
if 'df' not in st.session_state:
    st.session_state.df = None # The currently loaded research DataFrame
if 'current_folder' not in st.session_state:
    st.session_state.current_folder = "Minhas Pesquisas" # Default folder
if 'settings' not in st.session_state:
    st.session_state.settings = {}
if 'notes' not in st.session_state:
    st.session_state.notes = ""
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'tutorial_completed' not in st.session_state:
    st.session_state.tutorial_completed = False
if 'autosave' not in st.session_state:
    st.session_state.autosave = True
if 'last_backup_path' not in st.session_state:
    st.session_state.last_backup_path = None
if 'reply_message_id' not in st.session_state:
    st.session_state.reply_message_id = None
if 'compose_inline' not in st.session_state:
    st.session_state.compose_inline = False
if 'ai_analysis_result' not in st.session_state:
    st.session_state.ai_analysis_result = None
if 'mindmap_nodes' not in st.session_state:
    st.session_state.mindmap_nodes = []
if 'mindmap_edges' not in st.session_state:
    st.session_state.mindmap_edges = []
if 'mindmap_config' not in st.session_state:
    st.session_state.mindmap_config = Config(width=700, height=500, directed=True, physics=True,
                                            nodeHighlightBehavior=True, highlightColor="#F7A7A6",
                                            collapsible=True, node={'labelProperty':'label'},
                                            link={'labelProperty': 'label', 'renderLabel': True})
if 'vision_analysis_result' not in st.session_state:
    st.session_state.vision_analysis_result = None
if 'selected_research_id' not in st.session_state: # To manage which research is being viewed/edited
    st.session_state.selected_research_id = None
if 'all_research_data' not in st.session_state: # Stores all research data for the logged-in user
    st.session_state.all_research_data = {} # Structure: {folder_name: {research_id: {name, df_path, metadata, shared_with}}}
if 'timeline_posts' not in st.session_state: # Global timeline posts
    st.session_state.timeline_posts = [] # List of {post_id, user_email, user_name, research_id, research_name, timestamp, likes, comments}

# -------------------------
# Load User State and Research Data (UPDATED)
# -------------------------
def load_user_state(email):
    user_data = get_user_data(email)
    if user_data:
        st.session_state.settings = user_data.get("settings", {})
        st.session_state.notes = user_data.get("notes", "")
        st.session_state.favorites = user_data.get("favorites", [])
        st.session_state.tutorial_completed = user_data.get("tutorial_completed", False)
        st.session_state.all_research_data = user_data.get("research_data", {}) # Load all research data
        st.session_state.user_obj = user_data # Store full user object

        # Apply font scale from settings
        apply_global_styles(st.session_state.settings.get("font_scale", 1.0))
    else:
        # Reset to defaults if user data not found (shouldn't happen after successful login)
        st.session_state.settings = {}
        st.session_state.notes = ""
        st.session_state.favorites = []
        st.session_state.tutorial_completed = False
        st.session_state.all_research_data = {}
        st.session_state.user_obj = None

def save_user_state(email):
    if email:
        user_data = {
            "settings": st.session_state.settings,
            "notes": st.session_state.notes,
            "favorites": st.session_state.favorites,
            "tutorial_completed": st.session_state.tutorial_completed,
            "research_data": st.session_state.all_research_data, # Save all research data
            "connections": st.session_state.user_obj.get("connections", []) # Save connections
        }
        update_user_data(email, user_data)

def apply_global_styles(font_scale):
    # This function will apply a dynamic font scale to the entire app
    # We'll inject CSS to override font sizes based on the slider value
    dynamic_css = f"""
    <style>
    html {{
        font-size: {font_scale * 100}%; /* Base font size */
    }}
    </style>
    """
    st.markdown(dynamic_css, unsafe_allow_html=True)

# -------------------------
# Research Data Management (NEW: Folders and Research Items)
# -------------------------
def create_folder(user_email, folder_name):
    users = load_users()
    if user_email in users:
        if "research_data" not in users[user_email]:
            users[user_email]["research_data"] = {}
        if folder_name not in users[user_email]["research_data"]:
            users[user_email]["research_data"][folder_name] = {}
            save_users(users)
            st.session_state.all_research_data = users[user_email]["research_data"] # Update session state
            return True
    return False

def delete_folder(user_email, folder_name):
    users = load_users()
    if user_email in users and folder_name in users[user_email].get("research_data", {}):
        del users[user_email]["research_data"][folder_name]
        save_users(users)
        st.session_state.all_research_data = users[user_email]["research_data"]
        if st.session_state.current_folder == folder_name:
            st.session_state.current_folder = "Minhas Pesquisas" # Reset if current folder deleted
        return True
    return False

def add_research_to_folder(user_email, folder_name, research_name, uploaded_file):
    users = load_users()
    if user_email in users and folder_name in users[user_email].get("research_data", {}):
        research_id = str(uuid.uuid4()) # Generate a unique ID for the research
        file_extension = Path(uploaded_file.name).suffix
        file_path = ATTACHMENTS_DIR / f"{research_id}{file_extension}"

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        users[user_email]["research_data"][folder_name][research_id] = {
            "name": research_name,
            "df_path": str(file_path),
            "original_filename": uploaded_file.name,
            "upload_timestamp": datetime.now().isoformat(),
            "metadata": {}, # To store AI analysis results, etc.
            "shared_with": [] # List of user_emails this research is shared with
        }
        save_users(users)
        st.session_state.all_research_data = users[user_email]["research_data"]
        return True
    return False

def load_research_dataframe(research_item):
    """Loads a pandas DataFrame from a research item's df_path."""
    if not research_item or not research_item.get("df_path"):
        return None
    file_path = Path(research_item["df_path"])
    if not file_path.exists():
        st.error(f"Arquivo de pesquisa não encontrado: {file_path}")
        return None

    try:
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        # Add other formats as needed
        else:
            st.warning(f"Formato de arquivo não suportado para carregamento: {file_path.suffix}")
            return None
    except Exception as e:
        st.error(f"Erro ao carregar a planilha: {e}")
        return None

# -------------------------
# Message Management Functions (UPDATED for Email-based users)
# -------------------------
def load_messages():
    if not Path(MESSAGES_FILE).exists():
        return []
    with open(MESSAGES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_messages(messages):
    with open(MESSAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)

def send_message(sender_email, recipient_email, subject, body, attachment=None):
    messages = load_messages()
    message_id = str(uuid.uuid4())

    att_info = None
    if attachment:
        att_filename = f"{message_id}_{attachment.name}"
        att_path = ATTACHMENTS_DIR / att_filename
        with open(att_path, "wb") as f:
            f.write(attachment.getbuffer())
        att_info = {"name": attachment.name, "path": str(att_path)}

    new_message = {
        "id": message_id,
        "from": sender_email,
        "to": recipient_email,
        "subject": subject,
        "body": body,
        "ts": datetime.now().isoformat(),
        "read_by": [],
        "attachment": att_info
    }
    messages.append(new_message)
    save_messages(messages)
    return True

def get_user_messages(user_email, msg_type='inbox'):
    all_msgs = load_messages()
    if msg_type == 'inbox':
        return sorted([m for m in all_msgs if m['to'] == user_email], key=lambda x: x['ts'], reverse=True)
    elif msg_type == 'sent':
        return sorted([m for m in all_msgs if m['from'] == user_email], key=lambda x: x['ts'], reverse=True)
    return []

def mark_message_read(message_id, user_email):
    messages = load_messages()
    for msg in messages:
        if msg['id'] == message_id and user_email not in msg['read_by']:
            msg['read_by'].append(user_email)
            save_messages(messages)
            return True
    return False

def delete_message(message_id, user_email):
    messages = load_messages()
    original_len = len(messages)
    # Filter out the message if the user is the sender or recipient
    # For a full social system, this would be more complex (e.g., soft delete, delete for sender only)
    messages = [msg for msg in messages if not (msg['id'] == message_id and (msg['from'] == user_email or msg['to'] == user_email))]
    if len(messages) < original_len:
        save_messages(messages)
        return True
    return False

# -------------------------
# AI Helper Functions - DataAnalyzer (Mantido, mas será integrado com a nova estrutura de pesquisa)
# -------------------------
class DataAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.insights = []

    def generate_comprehensive_analysis(self):
        """Gera uma análise completa e inteligente dos dados"""
        analysis = ""
        analysis += self._basic_analysis()
        analysis += self._author_analysis()
        analysis += self._temporal_analysis()
        analysis += self._thematic_analysis()
        analysis += self._collaboration_analysis()
        analysis += self._geographic_analysis()
        analysis += self._trend_analysis()
        return analysis

    def _basic_analysis(self):
        text = "### 📊 Visão Geral\n\n"
        text += f"- **Total de registros**: {len(self.df)}\n"
        text += f"- **Colunas disponíveis**: {', '.join(self.df.columns.tolist())}\n"
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        text += f"- **Colunas numéricas**: {len(numeric_cols)}\n"
        text += f"- **Colunas de texto**: {len(text_cols)}\n\n"
        return text

    def _author_analysis(self):
        text = "### 👥 Análise de Autores\n\n"
        author_col = None
        possible_author_cols = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['autor', 'author', 'pesquisador', 'escritor', 'writer', 'nome']):
                possible_author_cols.append(col)
                sample_data = self.df[col].dropna().head(5)
                if len(sample_data) > 0:
                    has_multiple_authors = any(';' in str(val) or ',' in str(val) for val in sample_data)
                    if has_multiple_authors or any(len(str(val).split()) >= 2 for val in sample_data):
                        author_col = col
                        break
        if not author_col and possible_author_cols:
            author_col = possible_author_cols[0]
        if not author_col:
            return "❌ **Autores**: Nenhuma coluna de autores identificada. Verifique se há colunas como 'autor', 'autores', 'author' na sua planilha.\n\n"
        text += f"**Coluna utilizada**: '{author_col}'\n\n"
        all_authors = []
        authors_found = 0
        for authors_str in self.df[author_col].dropna():
            if isinstance(authors_str, str) and authors_str.strip():
                authors_found += 1
                authors = re.split(r'[;,]|\be\b|\band\b|&', authors_str)
                for author in authors:
                    author_clean = author.strip()
                    if (author_clean and len(author_clean) > 2 and
                        author_clean.lower() not in ['', 'e', 'and', 'et', 'de', 'da', 'do', 'dos', 'das'] and
                        not author_clean.isdigit() and
                        not author_clean.replace('.', '').isdigit()):
                        all_authors.append(author_clean)
        if all_authors:
            author_counts = pd.Series(all_authors).value_counts()
            text += "**Principais autores identificados:**\n"
            for author, count in author_counts.head(8).items():
                text += f"- **{author}**: {count} publicação(ões)\n"
            collaborations = 0
            for authors_str in self.df[author_col].dropna():
                if isinstance(authors_str, str) and len(re.split(r'[;,]|\be\b|\band\b|&', authors_str)) > 1:
                    collaborations += 1
            if authors_found > 0: # Avoid division by zero
                collaboration_rate = (collaborations / authors_found) * 100
                text += f"\n**Colaborações**: {collaborations} trabalhos com coautoria ({collaboration_rate:.1f}%)\n"
            else:
                text += f"\n**Colaborações**: Nenhuma colaboração identificada\n"
            text += f"\n**Total de registros com autores**: {authors_found}\n"
            text += f"**Total de nomes extraídos**: {len(all_authors)}\n\n"
        else:
            text += f"⚠️ **Autores**: Coluna '{author_col}' encontrada mas não foi possível extrair autores válidos\n\n"
            text += f"**Dica**: Verifique o formato dos dados na coluna '{author_col}'\n\n"
        return text

    def _temporal_analysis(self):
        text = "### 📈 Análise Temporal\n\n"
        year_col = None
        year_data_found = False
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['ano', 'year', 'data', 'date', 'publication']):
                year_col = col
                year_data_found = True
                break
        if not year_data_found:
            for col in self.df.select_dtypes(include=[np.number]).columns:
                sample_data = self.df[col].dropna().head(10)
                if len(sample_data) > 0:
                    current_year = datetime.now().year
                    if all(1900 <= val <= current_year for val in sample_data if pd.notnull(val)):
                        year_col = col
                        year_data_found = True
                        text += f"⚠️ **Atenção**: Usando coluna '{col}' para análise temporal (detecção automática)\n\n"
                        break
        if not year_col:
            return "❌ **Anos**: Nenhuma coluna de anos identificada na planilha\n\n"
        try:
            years = pd.to_numeric(self.df[year_col], errors='coerce').dropna()
        except:
            years = pd.Series(dtype=float)
        if len(years) > 0:
            min_year = int(years.min())
            max_year = int(years.max())
            year_range = max_year - min_year
            text += f"- **Período analisado**: {min_year} - {max_year} ({year_range} anos)\n"
            year_counts = years.value_counts()
            if not year_counts.empty:
                most_frequent_year = int(year_counts.index[0])
                most_frequent_count = int(year_counts.iloc[0])
                text += f"- **Ano com mais publicações**: {most_frequent_year} ({most_frequent_count} publicações)\n"
            if year_range > 20:
                decades = (years // 10) * 10
                decade_counts = decades.value_counts().sort_index()
                if len(decade_counts) > 1:
                    text += "\n**Distribuição por década:**\n"
                    for decade, count in decade_counts.head(5).items():
                        text += f"- {int(decade)}s: {int(count)} publicação(ões)\n"
            if len(years) > 5:
                recent_threshold = max_year - 5
                recent_years = years[years >= recent_threshold]
                older_years = years[years < recent_threshold]
                if len(recent_years) > 0 and len(older_years) > 0:
                    recent_avg = len(recent_years) / 5
                    older_avg = len(older_years) / max(1, (recent_threshold - min_year))
                    if recent_avg > older_avg * 1.2:
                        text += "- **Tendência**: 📈 Crescimento na produção recente\n"
                    elif recent_avg < older_avg * 0.8:
                        text += "- **Tendência**: 📉 Produção mais concentrada no passado\n"
                    else:
                        text += "- **Tendência**: ➡️ Produção constante ao longo do tempo\n"
            text += f"\n**Total de registros com anos**: {len(years)}\n\n"
        else:
            text += f"⚠️ **Anos**: Coluna '{year_col}' encontrada mas sem dados numéricos válidos\n\n"
        return text

    def _thematic_analysis(self):
        text = "### 🔍 Análise Temática\n\n"
        texto_completo = ""
        text_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
        for col in text_cols[:4]:
            col_text = self.df[col].fillna('').astype(str).str.cat(sep=' ')
            if len(col_text) > 100:
                texto_completo += col_text + " "
        if not texto_completo.strip():
            return "❌ **Temas**: Nenhuma coluna de texto com conteúdo suficiente para análise temática.\n\n"
        if TfidfVectorizer and KMeans:
            try:
                vectorizer = TfidfVectorizer(stop_words=PORTUGUESE_STOP_WORDS, max_features=1000, ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform([texto_completo])
                feature_names = vectorizer.get_feature_names_out()

                # Simple keyword extraction
                if tfidf_matrix.shape[0] > 0:
                    top_n = 10
                    sorted_items = sorted(zip(vectorizer.idf_, feature_names), key=lambda x: x[0])
                    keywords = [item[1] for item in sorted_items[:top_n]]
                    text += "**Palavras-chave mais relevantes**: " + ", ".join(keywords) + "\n\n"

                # Topic modeling (if enough data)
                if tfidf_matrix.shape[0] > 10 and LatentDirichletAllocation: # LDA needs more than 1 document
                    num_topics = min(5, tfidf_matrix.shape[0] // 2) # Adjust num_topics dynamically
                    if num_topics > 1:
                        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
                        lda.fit(tfidf_matrix)
                        text += "**Tópicos Principais:**\n"
                        for idx, topic in enumerate(lda.components_):
                            top_words = [feature_names[i] for i in topic.argsort()[:-5 - 1:-1]]
                            text += f"- Tópico {idx+1}: {', '.join(top_words)}\n"
                        text += "\n"
                    else:
                        text += "ℹ️ Não há dados suficientes para modelagem de tópicos avançada.\n\n"
                else:
                    text += "ℹ️ Não há dados suficientes para modelagem de tópicos avançada.\n\n"

            except Exception as e:
                text += f"⚠️ Erro na análise temática avançada: {e}\n\n"
        else:
            text += "⚠️ Bibliotecas de ML (TfidfVectorizer, KMeans, LDA) não disponíveis para análise temática avançada.\n\n"
        return text

    def _collaboration_analysis(self):
        text = "### 🤝 Análise de Colaboração (Rede)\n\n"
        author_col = None
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['autor', 'author', 'pesquisador', 'escritor', 'writer', 'nome']):
                author_col = col
                break
        if not author_col:
            return "❌ **Colaboração**: Nenhuma coluna de autores identificada para análise de rede.\n\n"

        G = nx.Graph()
        for authors_str in self.df[author_col].dropna():
            if isinstance(authors_str, str) and authors_str.strip():
                authors = [a.strip() for a in re.split(r'[;,]|\be\b|\band\b|&', authors_str) if a.strip() and len(a.strip()) > 2]
                if len(authors) > 1:
                    for i in range(len(authors)):
                        for j in range(i + 1, len(authors)):
                            author1 = authors[i]
                            author2 = authors[j]
                            if G.has_edge(author1, author2):
                                G[author1][author2]['weight'] += 1
                            else:
                                G.add_edge(author1, author2, weight=1)
        if G.number_of_nodes() > 1:
            text += f"- **Total de autores na rede**: {G.number_of_nodes()}\n"
            text += f"- **Total de colaborações**: {G.number_of_edges()}\n\n"

            # Top collaborators
            degree_centrality = nx.degree_centrality(G)
            sorted_authors = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)
            text += "**Autores mais conectados (Top 5):**\n"
            for author, centrality in sorted_authors[:5]:
                text += f"- {author} (Conexões: {G.degree(author)})\n"

            # Communities (simple detection)
            if nx.is_connected(G):
                try:
                    from networkx.algorithms import community
                    communities_generator = community.girvan_newman(G)
                    top_communities = next(communities_generator)
                    sorted_communities = sorted(map(sorted, top_communities))
                    text += "\n**Comunidades de Colaboração (Exemplo):**\n"
                    for i, comm in enumerate(sorted_communities[:3]): # Show top 3 communities
                        if len(comm) > 1:
                            text += f"- Comunidade {i+1}: {', '.join(comm[:5])}{'...' if len(comm) > 5 else ''}\n"
                except Exception as e:
                    text += f"⚠️ Erro ao detectar comunidades: {e}\n"
            else:
                text += "ℹ️ A rede de colaboração não é totalmente conectada, o que pode indicar grupos isolados.\n"
        else:
            text += "ℹ️ Não há dados suficientes para construir uma rede de colaboração significativa.\n\n"
        return text

    def _geographic_analysis(self):
        text = "### 🌍 Análise Geográfica\n\n"
        geo_cols = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['pais', 'país', 'country', 'cidade', 'city', 'estado', 'state', 'instituicao', 'instituição', 'affiliation', 'local']):
                geo_cols.append(col)

        if not geo_cols:
            return "❌ **Geografia**: Nenhuma coluna geográfica identificada (ex: 'país', 'cidade', 'instituição').\n\n"

        text += f"**Colunas consideradas**: {', '.join(geo_cols)}\n\n"

        # Prioritize 'country' or 'país'
        primary_geo_col = next((col for col in geo_cols if 'pais' in col.lower() or 'country' in col.lower()), geo_cols[0])

        if primary_geo_col and not self.df[primary_geo_col].dropna().empty:
            top_locations = self.df[primary_geo_col].value_counts().head(5)
            text += f"**Principais {primary_geo_col}s/Localidades:**\n"
            for loc, count in top_locations.items():
                text += f"- **{loc}**: {count} ocorrência(s)\n"
            text += "\n"
        else:
            text += "⚠️ Nenhuma informação geográfica válida encontrada nas colunas identificadas.\n\n"

        return text

    def _trend_analysis(self):
        text = "### 🚀 Análise de Tendências Futuras\n\n"
        # This is a placeholder for more advanced predictive analytics
        # For now, it will summarize based on temporal and thematic analysis.

        # Re-use temporal analysis insights
        temporal_text = self._temporal_analysis()
        if "Crescimento na produção recente" in temporal_text:
            text += "- A produção de pesquisa mostra uma **tendência de crescimento**, indicando áreas ativas e promissoras.\n"
        elif "Produção mais concentrada no passado" in temporal_text:
            text += "- A produção parece ter sido mais intensa no passado, sugerindo que novas direções ou focos podem estar surgindo.\n"
        else:
            text += "- A produção tem sido **constante**, o que pode indicar uma área de pesquisa estável ou madura.\n"

        # Re-use thematic analysis insights
        thematic_text = self._thematic_analysis()
        if "Palavras-chave mais relevantes" in thematic_text:
            keywords_match = re.search(r"\*\*Palavras-chave mais relevantes\*\*: (.*?)\n", thematic_text)
            if keywords_match:
                keywords = keywords_match.group(1)
                text += f"- As **palavras-chave dominantes** como {keywords} apontam para os focos atuais e potenciais direções de inovação.\n"

        text += "\n**Recomendações de IA para o futuro:**\n"
        text += "- **Explorar novas colaborações**: A IA pode sugerir pesquisadores com perfis complementares com base em temas e histórico.\n"
        text += "- **Identificar lacunas de pesquisa**: Analisar temas menos abordados ou conexões fracas na rede pode revelar oportunidades.\n"
        text += "- **Monitorar tendências emergentes**: Ficar atento a termos e conceitos que começam a aparecer com mais frequência pode indicar novas áreas quentes.\n"
        text += "- **Diversificar fontes de dados**: Integrar dados de diferentes plataformas pode enriquecer a análise e a detecção de tendências.\n\n"

        return text

# -------------------------
# Vision AI Helper Functions (Placeholder for future expansion)
# -------------------------
def analyze_image_with_ai(image_file):
    """
    Placeholder for a more advanced AI vision analysis function.
    In a real application, this would integrate with a cloud AI vision API
    (e.g., Google Cloud Vision, Azure Cognitive Services, OpenAI Vision).
    """
    if image_file is None:
        return "Nenhuma imagem fornecida para análise."

    # Simulate AI analysis
    analysis_results = {
        "description": "Análise simulada: Esta imagem parece conter gráficos e texto relacionados a dados de pesquisa. Possíveis elementos detectados: barras, linhas, legendas, títulos de artigos.",
        "keywords": ["gráfico", "dados", "pesquisa", "visualização", "documento"],
        "sentiment": "Neutro",
        "confidence": 0.85
    }

    # Save the image temporarily to display or process
    image_path = ATTACHMENTS_DIR / f"vision_temp_{image_file.name}"
    with open(image_path, "wb") as f:
        f.write(image_file.getbuffer())

    st.session_state.vision_analysis_result = {
        "image_path": str(image_path),
        "results": analysis_results
    }
    return analysis_results

# -------------------------
# Timeline & Social Features (NEW)
# -------------------------
def create_post(user_email, research_id, research_name, content, visibility="public"):
    posts = st.session_state.get('timeline_posts', [])
    post_id = str(uuid.uuid4())
    user_obj = get_user_data(user_email)
    user_name = user_obj.get('name', user_email)

    new_post = {
        "post_id": post_id,
        "user_email": user_email,
        "user_name": user_name,
        "research_id": research_id,
        "research_name": research_name,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "visibility": visibility, # "public", "connections", "private"
        "likes": [], # List of user_emails who liked
        "comments": [] # List of {user_email, user_name, comment_text, timestamp}
    }
    posts.append(new_post)
    st.session_state.timeline_posts = posts # Update session state
    # In a real app, this would be saved to a persistent storage (e.g., a JSON file for posts)
    # For now, it's in session_state, meaning it resets on full app refresh.
    # To make it persistent, you'd need a `timeline_posts.json` file and load/save functions.
    return True

def get_timeline_posts(user_email):
    all_posts = st.session_state.get('timeline_posts', [])
    user_connections = st.session_state.user_obj.get('connections', [])

    # Filter posts based on visibility
    filtered_posts = []
    for post in all_posts:
        if post['visibility'] == 'public':
            filtered_posts.append(post)
        elif post['visibility'] == 'connections' and (post['user_email'] == user_email or post['user_email'] in user_connections):
            filtered_posts.append(post)
        elif post['visibility'] == 'private' and post['user_email'] == user_email:
            filtered_posts.append(post)

    return sorted(filtered_posts, key=lambda x: x['timestamp'], reverse=True)

def toggle_like_post(post_id, user_email):
    posts = st.session_state.get('timeline_posts', [])
    for post in posts:
        if post['post_id'] == post_id:
            if user_email in post['likes']:
                post['likes'].remove(user_email)
            else:
                post['likes'].append(user_email)
            st.session_state.timeline_posts = posts
            return True
    return False

def add_comment_to_post(post_id, user_email, comment_text):
    posts = st.session_state.get('timeline_posts', [])
    user_obj = get_user_data(user_email)
    user_name = user_obj.get('name', user_email)
    for post in posts:
        if post['post_id'] == post_id:
            post['comments'].append({
                "user_email": user_email,
                "user_name": user_name,
                "comment_text": comment_text,
                "timestamp": datetime.now().isoformat()
            })
            st.session_state.timeline_posts = posts
            return True
    return False

def get_user_avatar_initials(user_name):
    if not user_name:
        return "?"
    parts = user_name.split()
    if len(parts) > 1:
        return (parts[0][0] + parts[-1][0]).upper()
    return parts[0][0].upper()

# -------------------------
# Global Variables (UPDATED for Email-based users)
# -------------------------
USERNAME = st.session_state.user_email
USER_OBJ = st.session_state.user_obj
USER_STATE_FILE = Path(f"user_state_{USERNAME}.json") # This will be replaced by direct user_obj updates
UNREAD_COUNT = 0
if USERNAME:
    inbox_msgs = get_user_messages(USERNAME, 'inbox')
    UNREAD_COUNT = sum(1 for msg in inbox_msgs if USERNAME not in msg['read_by'])

# -------------------------
# Navigation (UPDATED for new pages)
# -------------------------
def navigate_to(page):
    st.session_state.page = page
    safe_rerun()

# -------------------------
# Login/Registration Page (UPDATED for Email-based users)
# -------------------------
if not st.session_state.authenticated:
    st.sidebar.empty() # Clear sidebar for login page

    st.markdown("<div class='glass-box' style='max-width:500px; margin: 50px auto; padding: 30px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:var(--primary-color);'>Bem-vindo ao NUGEP-PQR</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:var(--muted-text-dark);'>Sua plataforma avançada de gestão de pesquisa e análise.</p>", unsafe_allow_html=True)

    login_tab, register_tab = st.tabs(["Login", "Registrar"])

    with login_tab:
        st.subheader("Acessar sua conta")
        with st.form("login_form"):
            email = st.text_input("E-mail", key="login_email")
            password = st.text_input("Senha", type="password", key="login_password")
            submitted = st.form_submit_button("Entrar", use_container_width=True)

            if submitted:
                user = authenticate_user(email, password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.session_state.user_obj = user
                    load_user_state(email) # Load user-specific data
                    st.success(f"Bem-vindo(a), {user.get('name', email)}!")
                    navigate_to("timeline") # Redirect to timeline after login
                else:
                    st.error("E-mail ou senha incorretos.")

    with register_tab:
        st.subheader("Criar nova conta")
        with st.form("register_form"):
            new_name = st.text_input("Nome Completo", key="register_name")
            new_email = st.text_input("E-mail", key="register_email")
            new_password = st.text_input("Senha", type="password", key="register_password")
            confirm_password = st.text_input("Confirmar Senha", type="password", key="confirm_password")
            new_scholarship = st.text_input("Bolsa/Afiliação (Opcional)", key="register_scholarship")

            register_submitted = st.form_submit_button("Registrar", use_container_width=True)

            if register_submitted:
                if not new_name or not new_email or not new_password or not confirm_password:
                    st.error("Por favor, preencha todos os campos obrigatórios.")
                elif new_password != confirm_password:
                    st.error("As senhas não coincidem.")
                else:
                    success, message = register_user(new_email, new_password, new_name, new_scholarship)
                    if success:
                        st.success(message + " Agora você pode fazer login.")
                        # Optionally pre-fill login fields or switch tab
                        st.session_state.login_email = new_email
                        st.session_state.login_password = "" # Clear password
                        st.session_state.page = "login" # Stay on login tab
                        safe_rerun()
                    else:
                        st.error(message)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Main Application Pages (Authenticated)
# -------------------------
else:
    # Sidebar Navigation
    st.sidebar.markdown(f"<div style='text-align:center; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='post-avatar' style='margin: 0 auto;'>{get_user_avatar_initials(USER_OBJ.get('name', USERNAME))}</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<h3 style='text-align:center; margin-top: 10px;'>{USER_OBJ.get('name', USERNAME)}</h3>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<p class='small-muted' style='text-align:center;'>{USER_OBJ.get('scholarship', 'Pesquisador(a)')}</p>", unsafe_allow_html=True)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.header("Navegação")

    # New navigation structure
    if st.sidebar.button("🏠 Timeline", use_container_width=True, key="nav_timeline"):
        navigate_to("timeline")
    if st.sidebar.button("🗂️ Minhas Pesquisas", use_container_width=True, key="nav_my_research"):
        navigate_to("my_research")
    if st.sidebar.button("🧠 Análise IA", use_container_width=True, key="nav_ai_analysis"):
        navigate_to("ai_analysis")
    if st.sidebar.button("👁️ Visão Computacional", use_container_width=True, key="nav_vision_ai"):
        navigate_to("vision_ai")
    if st.sidebar.button(f"📧 Mensagens ({UNREAD_COUNT} não lidas)", use_container_width=True, key="nav_messages"):
        navigate_to("messages")
    if st.sidebar.button("⚙️ Configurações", use_container_width=True, key="nav_config"):
        navigate_to("config")

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Sair", type="secondary", use_container_width=True, key="logout_button"):
        save_user_state(USERNAME) # Save state before logging out
        st.session_state.clear()
        st.session_state.authenticated = False
        st.session_state.page = "login"
        safe_rerun()

    # -------------------------
    # Page: Timeline (NEW)
    # -------------------------
    if st.session_state.page == "timeline":
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.subheader("🏠 Sua Timeline de Pesquisas")
        st.write("Veja as últimas pesquisas compartilhadas por você e sua rede.")

        # Post new research section
        with st.expander("✨ Compartilhar Nova Pesquisa", expanded=False):
            user_research_options = {}
            for folder_name, researches in st.session_state.all_research_data.items():
                for research_id, research_info in researches.items():
                    user_research_options[f"{folder_name} / {research_info['name']}"] = research_id

            if not user_research_options:
                st.info("Você precisa ter pesquisas carregadas em suas pastas para poder compartilhar.")
            else:
                selected_research_display = st.selectbox("Selecione a pesquisa para compartilhar:", options=list(user_research_options.keys()))
                post_content = st.text_area("O que você gostaria de dizer sobre esta pesquisa?", height=100)
                visibility_option = st.radio("Visibilidade:", ["Público", "Conexões", "Privado"], index=0)

                if st.button("Postar na Timeline", use_container_width=True):
                    if selected_research_display and post_content:
                        selected_research_id = user_research_options[selected_research_display]
                        selected_research_name = selected_research_display.split(' / ')[1]
                        create_post(USERNAME, selected_research_id, selected_research_name, post_content, visibility_option.lower())
                        st.success("Pesquisa postada na timeline!")
                        safe_rerun()
                    else:
                        st.error("Por favor, selecione uma pesquisa e digite um conteúdo para o post.")

        st.markdown("---")
        st.subheader("Últimas Postagens")

        timeline_posts = get_timeline_posts(USERNAME)

        if not timeline_posts:
            st.info("Nenhuma postagem na timeline ainda. Comece a compartilhar suas pesquisas!")
        else:
            for post in timeline_posts:
                with st.container():
                    st.markdown(f"<div class='social-post-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='post-header'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='post-avatar'>{get_user_avatar_initials(post['user_name'])}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div><span class='post-author'>{post['user_name']}</span> <span class='post-timestamp'>postou em {datetime.fromisoformat(post['timestamp']).strftime('%d/%m/%Y %H:%M')}</span></div>", unsafe_allow_html=True)
                    st.markdown(f"</div>", unsafe_allow_html=True) # End post-header

                    st.markdown(f"**Pesquisa:** {post['research_name']}")
                    st.markdown(f"<p class='post-content'>{post['content']}</p>", unsafe_allow_html=True)

                    col1, col2, col3 = st.columns([1,1,3])
                    with col1:
                        if st.button(f"👍 {len(post['likes'])} Curtir", key=f"like_{post['post_id']}"):
                            toggle_like_post(post['post_id'], USERNAME)
                            safe_rerun()
                    with col2:
                        if st.button(f"💬 {len(post['comments'])} Comentar", key=f"comment_btn_{post['post_id']}"):
                            st.session_state[f"show_comment_input_{post['post_id']}"] = not st.session_state.get(f"show_comment_input_{post['post_id']}", False)
                            safe_rerun()

                    if st.session_state.get(f"show_comment_input_{post['post_id']}", False):
                        comment_text = st.text_input("Seu comentário:", key=f"comment_text_{post['post_id']}")
                        if st.button("Enviar Comentário", key=f"submit_comment_{post['post_id']}"):
                            if comment_text:
                                add_comment_to_post(post['post_id'], USERNAME, comment_text)
                                st.success("Comentário adicionado!")
                                st.session_state[f"show_comment_input_{post['post_id']}"] = False
                                safe_rerun()
                            else:
                                st.warning("Por favor, digite um comentário.")

                    if post['comments']:
                        st.markdown("---")
                        st.markdown("**Comentários:**")
                        for comment in post['comments']:
                            st.markdown(f"<div style='margin-left: 20px; font-size: 0.9em; border-left: 2px solid var(--border-color); padding-left: 10px; margin-bottom: 5px;'>", unsafe_allow_html=True)
                            st.markdown(f"**{comment['user_name']}** ({datetime.fromisoformat(comment['timestamp']).strftime('%H:%M')}): {comment['comment_text']}", unsafe_allow_html=True)
                            st.markdown(f"</div>", unsafe_allow_html=True)

                    st.markdown(f"</div>", unsafe_allow_html=True) # End social-post-card
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # Page: Minhas Pesquisas (NEW: Folder-based research management)
    # -------------------------
    elif st.session_state.page == "my_research":
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.subheader("🗂️ Minhas Pesquisas")
        st.write("Organize suas planilhas e dados de pesquisa em pastas.")

        # Folder management
        st.markdown("### Gerenciar Pastas")
        col1, col2 = st.columns([3,1])
        with col1:
            new_folder_name = st.text_input("Nome da nova pasta:", key="new_folder_name")
        with col2:
            if st.button("Criar Pasta", use_container_width=True, key="create_folder_btn"):
                if new_folder_name:
                    if create_folder(USERNAME, new_folder_name):
                        st.success(f"Pasta '{new_folder_name}' criada com sucesso!")
                        st.session_state.current_folder = new_folder_name # Set new folder as current
                        safe_rerun()
                    else:
                        st.error("Erro ao criar pasta ou pasta já existe.")
                else:
                    st.warning("Por favor, digite um nome para a pasta.")

        st.markdown("---")
        st.markdown("### Suas Pastas")

        user_folders = st.session_state.all_research_data.keys()
        if not user_folders:
            st.info("Você ainda não tem pastas. Crie uma para começar a organizar suas pesquisas!")
        else:
            folder_options = list(user_folders)
            selected_folder = st.selectbox("Selecione uma pasta:", options=folder_options, key="select_folder", index=folder_options.index(st.session_state.current_folder) if st.session_state.current_folder in folder_options else 0)
            st.session_state.current_folder = selected_folder

            st.markdown(f"#### Pasta Atual: **{st.session_state.current_folder}**")

            # Actions for the selected folder
            col_folder_actions = st.columns(3)
            with col_folder_actions[0]:
                if st.button("➕ Adicionar Pesquisa", use_container_width=True, key="add_research_to_folder_btn"):
                    st.session_state.show_add_research_form = True
            with col_folder_actions[1]:
                if st.button("✏️ Renomear Pasta", use_container_width=True, key="rename_folder_btn"):
                    st.session_state.show_rename_folder_form = True
            with col_folder_actions[2]:
                if st.button("🗑️ Excluir Pasta", use_container_width=True, key="delete_folder_btn", type="secondary"):
                    if st.warning(f"Tem certeza que deseja excluir a pasta '{st.session_state.current_folder}' e todo o seu conteúdo?"):
                        if delete_folder(USERNAME, st.session_state.current_folder):
                            st.success(f"Pasta '{st.session_state.current_folder}' excluída.")
                            safe_rerun()
                        else:
                            st.error("Erro ao excluir pasta.")

            # Add Research Form
            if st.session_state.get("show_add_research_form", False):
                with st.form("add_research_form", clear_on_submit=True):
                    st.markdown(f"##### Adicionar nova pesquisa à pasta '{st.session_state.current_folder}'")
                    research_name = st.text_input("Nome da Pesquisa:", key="new_research_name")
                    uploaded_file = st.file_uploader("Carregar Planilha (CSV, Excel):", type=["csv", "xls", "xlsx"], key="new_research_file")
                    add_research_submitted = st.form_submit_button("Salvar Pesquisa", use_container_width=True)

                    if add_research_submitted:
                        if research_name and uploaded_file:
                            if add_research_to_folder(USERNAME, st.session_state.current_folder, research_name, uploaded_file):
                                st.success(f"Pesquisa '{research_name}' adicionada à pasta '{st.session_state.current_folder}'.")
                                st.session_state.show_add_research_form = False
                                safe_rerun()
                            else:
                                st.error("Erro ao adicionar pesquisa.")
                        else:
                            st.warning("Por favor, preencha o nome e carregue um arquivo.")

            # Rename Folder Form (Placeholder for now, needs more robust implementation)
            if st.session_state.get("show_rename_folder_form", False):
                st.info("Funcionalidade de renomear pasta em desenvolvimento.")
                if st.button("Fechar Renomear", key="close_rename_folder"):
                    st.session_state.show_rename_folder_form = False
                    safe_rerun()

            st.markdown("---")
            st.markdown(f"### Pesquisas em '{st.session_state.current_folder}'")

            current_folder_researches = st.session_state.all_research_data.get(st.session_state.current_folder, {})
            if not current_folder_researches:
                st.info("Esta pasta está vazia. Adicione uma pesquisa acima!")
            else:
                for research_id, research_info in current_folder_researches.items():
                    with st.container():
                        st.markdown(f"<div class='folder-card'>", unsafe_allow_html=True)
                        st.markdown(f"<div><span class='folder-name'>📄 {research_info['name']}</span><br><span class='folder-meta'>Upload: {datetime.fromisoformat(research_info['upload_timestamp']).strftime('%d/%m/%Y')} | Arquivo: {research_info['original_filename']}</span></div>", unsafe_allow_html=True)

                        col_research_actions = st.columns(4)
                        with col_research_actions[0]:
                            if st.button("Ver/Editar", key=f"view_research_{research_id}"):
                                st.session_state.selected_research_id = research_id
                                st.session_state.df = load_research_dataframe(research_info) # Load DF for viewing
                                navigate_to("ai_analysis") # Redirect to AI Analysis to view/analyze
                        with col_research_actions[1]:
                            if st.download_button("Baixar", data=open(research_info['df_path'], 'rb').read(), file_name=research_info['original_filename'], mime="application/octet-stream", key=f"download_research_{research_id}"):
                                pass
                        with col_research_actions[2]:
                            if st.button("Compartilhar", key=f"share_research_{research_id}"):
                                st.info("Funcionalidade de compartilhamento em desenvolvimento.")
                        with col_research_actions[3]:
                            if st.button("Excluir", key=f"delete_research_{research_id}", type="secondary"):
                                # Implement delete logic for research item
                                if st.warning(f"Tem certeza que deseja excluir a pesquisa '{research_info['name']}'?"):
                                    # Delete file from disk
                                    Path(research_info['df_path']).unlink(missing_ok=True)
                                    # Remove from user's research_data
                                    del st.session_state.all_research_data[st.session_state.current_folder][research_id]
                                    save_user_state(USERNAME)
                                    st.success("Pesquisa excluída.")
                                    safe_rerun()
                        st.markdown(f"</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # Page: AI Analysis (UPDATED to use selected research)
    # -------------------------
    elif st.session_state.page == "ai_analysis":
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.subheader("🧠 Análise Inteligente de Dados")
        st.write("Utilize a IA para obter insights profundos sobre suas pesquisas.")

        # Selector for research
        research_options = {"Selecione uma pesquisa...": None}
        for folder_name, researches in st.session_state.all_research_data.items():
            for research_id, research_info in researches.items():
                research_options[f"{folder_name} / {research_info['name']}"] = research_id

        # Pre-select if a research was chosen from "Minhas Pesquisas"
        default_index = 0
        if st.session_state.selected_research_id:
            for i, (display_name, r_id) in enumerate(research_options.items()):
                if r_id == st.session_state.selected_research_id:
                    default_index = i
                    break

        selected_research_display = st.selectbox(
            "Escolha uma pesquisa para analisar:",
            options=list(research_options.keys()),
            index=default_index,
            key="ai_analysis_research_selector"
        )

        selected_research_id = research_options[selected_research_display]

        if selected_research_id:
            # Find the full research_info
            current_research_info = None
            for folder_name, researches in st.session_state.all_research_data.items():
                if selected_research_id in researches:
                    current_research_info = researches[selected_research_id]
                    break

            if current_research_info:
                st.session_state.df = load_research_dataframe(current_research_info) # Load the DataFrame
                st.markdown(f"#### Analisando: **{current_research_info['name']}**")
                st.write(f"Arquivo original: {current_research_info['original_filename']}")

                if st.session_state.df is not None:
                    st.dataframe(st.session_state.df.head()) # Show a preview of the DataFrame

                    if st.button("🚀 Gerar Análise Completa da IA", use_container_width=True):
                        with st.spinner("A IA está trabalhando duro para analisar seus dados..."):
                            analyzer = DataAnalyzer(st.session_state.df)
                            analysis_text = analyzer.generate_comprehensive_analysis()
                            st.session_state.ai_analysis_result = analysis_text
                            # Save analysis result to research metadata
                            current_research_info['metadata']['last_ai_analysis'] = analysis_text
                            save_user_state(USERNAME)
                        st.success("Análise concluída!")
                        safe_rerun() # Rerun to display results

                    if st.session_state.ai_analysis_result:
                        st.markdown("<div class='ai-response'>", unsafe_allow_html=True)
                        st.markdown("### Resultados da Análise da IA:")
                        st.markdown(st.session_state.ai_analysis_result)
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("Não foi possível carregar o DataFrame para análise. Verifique o arquivo.")
            else:
                st.warning("Pesquisa selecionada não encontrada. Por favor, selecione outra.")
        else:
            st.info("Por favor, selecione uma pesquisa para iniciar a análise da IA.")
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # Page: Vision AI (Placeholder)
    # -------------------------
    elif st.session_state.page == "vision_ai":
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.subheader("👁️ Visão Computacional Avançada")
        st.write("Carregue imagens de gráficos, documentos ou diagramas para a IA analisar e extrair informações.")

        uploaded_image = st.file_uploader("Carregar Imagem para Análise (JPG, PNG):", type=["jpg", "png"], key="vision_image_uploader")

        if uploaded_image:
            st.image(uploaded_image, caption="Imagem Carregada", use_column_width=True)
            if st.button("Analisar Imagem com IA", use_container_width=True):
                with st.spinner("Analisando imagem com Visão Computacional..."):
                    analysis_results = analyze_image_with_ai(uploaded_image)
                st.success("Análise de visão concluída!")
                safe_rerun() # Rerun to display results

        if st.session_state.vision_analysis_result:
            st.markdown("<div class='vision-analysis'>", unsafe_allow_html=True)
            st.markdown("### Resultados da Análise de Visão:")
            st.markdown(f"**Descrição:** {st.session_state.vision_analysis_result['results']['description']}")
            st.markdown(f"**Palavras-chave:** {', '.join(st.session_state.vision_analysis_result['results']['keywords'])}")
            st.markdown(f"**Sentimento:** {st.session_state.vision_analysis_result['results']['sentiment']}")
            st.markdown(f"**Confiança:** {st.session_state.vision_analysis_result['results']['confidence']:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # Page: Messages (UPDATED for Email-based users)
    # -------------------------
    elif st.session_state.page == "messages":
        st.markdown("<div class='glass-box'>", unsafe_allow_html=True)
        st.subheader("📧 Central de Mensagens")
        st.write("Comunique-se com outros pesquisadores da plataforma.")

        tab1, tab2, tab3 = st.tabs(["Caixa de Entrada", "Enviadas", "Nova Mensagem"])

        all_msgs = get_user_messages(USERNAME, 'inbox') + get_user_messages(USERNAME, 'sent') # For reply logic
        all_users = load_users() # Load all users once

        with tab1:
            inbox_msgs = get_user_messages(USERNAME, 'inbox')
            if not inbox_msgs:
                st.info("Sua caixa de entrada está vazia.")
            else:
                for msg in inbox_msgs:
                    is_unread = USERNAME not in msg['read_by']
                    sender_name = all_users.get(msg['from'], {}).get('name', msg['from'])

                    status_icon = "✉️" if is_unread else "✅"
                    with st.expander(f"{status_icon} {msg['subject']} — De: {sender_name}", expanded=is_unread):
                        st.write(f"**Assunto:** {msg['subject']}")
                        st.write(f"**De:** {sender_name}")
                        st.write(f"**Data:** {datetime.fromisoformat(msg['ts']).strftime('%d/%m/%Y %H:%M')}")
                        st.markdown("---")
                        st.write(msg['body'])

                        if msg.get('attachment'):
                            att = msg['attachment']
                            if os.path.exists(att['path']):
                                with open(att['path'], 'rb') as f:
                                    st.download_button(
                                        f"📎 Baixar {att['name']}",
                                        data=f.read(),
                                        file_name=att['name'],
                                        mime="application/octet-stream"
                                    )

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if is_unread and st.button("✅ Marcar como lida", key=f"read_{msg['id']}"):
                                mark_message_read(msg['id'], USERNAME)
                                st.success("Mensagem marcada como lida!")
                                safe_rerun()
                        with col2:
                            if st.button("📧 Responder", key=f"reply_{msg['id']}"):
                                st.session_state.reply_message_id = msg['id']
                                st.session_state.compose_inline = True
                                safe_rerun()
                        with col3:
                            if st.button("🗑️ Excluir", key=f"delete_inbox_{msg['id']}"):
                                if delete_message(msg['id'], USERNAME):
                                    st.success("Mensagem excluída!")
                                    safe_rerun()
                                else:
                                    st.error("Erro ao excluir mensagem.")

        with tab2:
            sent_msgs = get_user_messages(USERNAME, 'sent')
            if not sent_msgs:
                st.info("Nenhuma mensagem enviada.")
            else:
                for msg in sent_msgs:
                    recipient_name = all_users.get(msg['to'], {}).get('name', msg['to'])

                    with st.expander(f"📤 {msg['subject']} — Para: {recipient_name}"):
                        st.write(f"**Assunto:** {msg['subject']}")
                        st.write(f"**Para:** {recipient_name}")
                        st.write(f"**Data:** {datetime.fromisoformat(msg['ts']).strftime('%d/%m/%Y %H:%M')}")
                        st.markdown("---")
                        st.write(msg['body'])

                        if st.button("🗑️ Excluir", key=f"delete_sent_{msg['id']}"):
                            if delete_message(msg['id'], USERNAME):
                                st.success("Mensagem excluída!")
                                safe_rerun()
                            else:
                                st.error("Erro ao excluir mensagem.")

        with tab3:
            st.subheader("✍️ Nova Mensagem")

            reply_to_msg = None
            if st.session_state.get('reply_message_id'):
                reply_to_msg = next((m for m in all_msgs if m['id'] == st.session_state.reply_message_id), None)

            with st.form("compose_message", clear_on_submit=True):
                users = load_users()

                user_options = {}
                for email, user_data in users.items():
                    if email != USERNAME:
                        user_options[user_data.get('name', email)] = email

                default_recipient_names = []
                if reply_to_msg:
                    sender_email_reply = reply_to_msg['from']
                    sender_name_reply = users.get(sender_email_reply, {}).get('name', sender_email_reply)
                    if sender_name_reply in user_options:
                        default_recipient_names.append(sender_name_reply)

                recipients_display = st.multiselect("Para:", options=sorted(list(user_options.keys())), default=default_recipient_names)

                subject = st.text_input("Assunto:",
                                        value=f"Re: {reply_to_msg['subject']}" if reply_to_msg else "")
                body = st.text_area("Mensagem:", height=200,
                                     value=f"\n\n---\nEm resposta à mensagem de {users.get(reply_to_msg['from'], {}).get('name', reply_to_msg['from'])}:\n> {reply_to_msg['body'][:500].replace(chr(10), chr(10)+'> ')}..." if reply_to_msg else "")

                attachment = st.file_uploader("Anexar arquivo", type=['pdf', 'docx', 'txt', 'jpg', 'png'])

                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("📤 Enviar Mensagem", use_container_width=True):
                        if not recipients_display:
                            st.error("Selecione pelo menos um destinatário.")
                        elif not subject:
                            st.error("Digite um assunto.")
                        elif not body:
                            st.error("Digite uma mensagem.")
                        else:
                            for recipient_name_selected in recipients_display:
                                recipient_email_actual = user_options[recipient_name_selected]
                                send_message(USERNAME, recipient_email_actual, subject, body, attachment)
                                st.success(f"Mensagem enviada para {recipient_name_selected}!")

                            if st.session_state.get('reply_message_id'):
                                st.session_state.reply_message_id = None
                            if st.session_state.get('compose_inline'):
                                st.session_state.compose_inline = False

                            time.sleep(1)
                            safe_rerun()

                with col2:
                    if st.form_submit_button("❌ Cancelar", type="secondary", use_container_width=True):
                        if st.session_state.get('reply_message_id'):
                            st.session_state.reply_message_id = None
                        if st.session_state.get('compose_inline'):
                            st.session_state.compose_inline = False
                        safe_rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # Page: configuracoes (UPDATED for Email-based users)
    # -------------------------
    elif st.session_state.page == "config":
        st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
        st.subheader("⚙️ Configurações e Personalização")

        # Configurações de aparência
        st.subheader("🎨 Aparência e Visual")

        col1, col2 = st.columns(2)

        with col1:
            font_scale = st.slider("Tamanho da fonte:",
                                    min_value=0.8,
                                    max_value=1.5,
                                    value=st.session_state.settings.get("font_scale", 1.0),
                                    step=0.1,
                                    help="Ajusta o tamanho geral do texto")

            node_font_size = st.slider("Tamanho da fonte nos mapas:",
                                        min_value=10,
                                        max_value=24,
                                        value=st.session_state.settings.get("node_font_size", 14),
                                        step=1,
                                        help="Tamanho do texto nos nós do mapa mental")

        with col2:
            plot_height = st.slider("Altura dos gráficos (px):",
                                    min_value=400,
                                    max_value=1200,
                                    value=st.session_state.settings.get("plot_height", 600),
                                    step=100,
                                    help="Altura padrão para visualizações de gráficos")

            node_opacity = st.slider("Opacidade dos nós:",
                                        min_value=0.3,
                                        max_value=1.0,
                                        value=st.session_state.settings.get("node_opacity", 0.8),
                                        step=0.1,
                                        help="Transparência dos elementos no mapa mental")

        if st.button("💾 Aplicar Configurações", use_container_width=True):
            st.session_state.settings.update({
                "font_scale": font_scale,
                "plot_height": plot_height,
                "node_opacity": node_opacity,
                "node_font_size": node_font_size
            })
            apply_global_styles(font_scale)
            save_user_state(USERNAME) # Save updated settings
            st.success("Configurações aplicadas! A página será recarregada.")
            time.sleep(1)
            safe_rerun()

        # Gerenciamento de dados
        st.subheader("📊 Gerenciamento de Dados")

        col3, col4 = st.columns(2)

        with col3:
            if st.button("🗑️ Limpar Todos os Dados do Usuário", type="secondary", use_container_width=True):
                if st.checkbox("CONFIRMAR: Esta ação não pode ser desfeita. Todos os seus dados (pesquisas, mensagens, configurações) serão perdidos."):
                    # Delete user's research files
                    for folder_name, researches in st.session_state.all_research_data.items():
                        for research_id, research_info in researches.items():
                            Path(research_info['df_path']).unlink(missing_ok=True)

                    # Clear user data from global users.json
                    users = load_users()
                    if USERNAME in users:
                        del users[USERNAME]
                        save_users(users)

                    # Clear user's messages
                    all_messages = load_messages()
                    all_messages = [msg for msg in all_messages if msg['from'] != USERNAME and msg['to'] != USERNAME]
                    save_messages(all_messages)

                    st.session_state.clear()
                    st.session_state.authenticated = False
                    st.session_state.page = "login"
                    st.success("Todos os seus dados foram removidos! Redirecionando para o login.")
                    time.sleep(2)
                    safe_rerun()

        with col4:
            import zipfile
            from io import BytesIO

            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add user state data
                state_data = {
                    "notes": st.session_state.get("notes", ""),
                    "favorites": st.session_state.get("favorites", []),
                    "settings": st.session_state.get("settings", {}),
                    "tutorial_completed": st.session_state.get("tutorial_completed", False),
                    "research_data": st.session_state.get("all_research_data", {}),
                    "connections": st.session_state.user_obj.get("connections", [])
                }
                zip_file.writestr("user_state.json", json.dumps(state_data, indent=2, ensure_ascii=False))

                # Add all research files
                for folder_name, researches in st.session_state.all_research_data.items():
                    for research_id, research_info in researches.items():
                        df_path = Path(research_info['df_path'])
                        if df_path.exists():
                            zip_file.write(df_path, f"research_files/{folder_name}/{research_info['original_filename']}")

            st.download_button(
                "📥 Exportar Backup Completo",
                data=zip_buffer.getvalue(),
                file_name=f"nugep_pqr_backup_{USERNAME}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip",
                use_container_width=True
            )

        # Informações do sistema
        st.subheader("ℹ️ Informações do Sistema")

        st.write(f"**Usuário (E-mail):** {USERNAME}")
        st.write(f"**Nome:** {USER_OBJ.get('name', 'Não informado')}")
        st.write(f"**Bolsa/Afiliação:** {USER_OBJ.get('scholarship', 'Não informada')}")
        created_at_str = USER_OBJ.get('created_at', 'Data não disponível')
        try:
            created_at_dt = datetime.fromisoformat(created_at_str)
            st.write(f"**Cadastrado em:** {created_at_dt.strftime('%d/%m/%Y %H:%M')}")
        except:
            st.write(f"**Cadastrado em:** {created_at_str}")

        st.write("**Estatísticas:**")
        st.write(f"- Favoritos salvos: {len(st.session_state.favorites)}")
        st.write(f"- Mensagens não lidas: {UNREAD_COUNT}")

        total_researches = sum(len(folder) for folder in st.session_state.all_research_data.values())
        st.write(f"- Total de pesquisas: {total_researches}")
        st.write(f"- Total de pastas: {len(st.session_state.all_research_data)}")

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # Placeholder for other pages (e.g., Mind Map, Flowchart, etc.)
    # -------------------------
    # If you want to add more pages, you'd add `elif st.session_state.page == "new_page_name":` here.
    # For now, we've focused on the core social and research management.

# -------------------------
# Finalização e salvamento automático
# -------------------------
if st.session_state.authenticated and st.session_state.autosave and st.session_state.user_email:
    try:
        save_user_state(st.session_state.user_email)
    except Exception as e:
        st.sidebar.error(f"Erro ao salvar estado automaticamente: {e}") # Display error in sidebar

# -------------------------
# Rodapé
# -------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#bfc6cc; font-size:0.9em; padding:10px;'>"
    "NUGEP-PQR — Sistema Avançado de Gestão de Pesquisa e Análise | "
    "IA Avançada • Visão Computacional • Mapa Mental Inteligente"
    "</div>",
    unsafe_allow_html=True
)
