
import os
import re
import io
import json
import time
import random
import string
import unicodedata
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from networkx.readwrite import json_graph
from fpdf import FPDF

# extras para futura expansão / ML
import numpy as np
try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    joblib = None

# -------------------------
# Config & safe helpers
# -------------------------
st.set_page_config(page_title="CONSUMIDIA", layout="wide", initial_sidebar_state="expanded")

# safe rerun helper
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
# CSS (same as original, kept compact)
# -------------------------
DEFAULT_CSS = r"""
:root{ --glass-bg: rgba(255,255,255,0.05); --glass-border: rgba(255,255,255,0.1); --glass-highlight: rgba(255,255,255,0.06); --accent: #8e44ad; --muted-text: #d6d9dc; --icon-color: #ffffff; }
.css-1d391kg { background: linear-gradient(180deg,#071428 0%, #031926 100%) !important; }
.glass-box{ background: var(--glass-bg); border: 1px solid var(--glass-border); border-radius: 16px; padding: 18px; box-shadow: 0 8px 32px rgba(4,9,20,0.5); backdrop-filter: blur(12px) saturate(1.2); }
.stButton>button, .stDownloadButton>button{ background: var(--glass-bg) !important; color: var(--muted-text) !important; border: 1px solid var(--glass-border) !important; padding: 8px 14px !important; border-radius: 12px !important; box-shadow: 0 4px 12px rgba(3,7,15,0.45) !important; backdrop-filter: blur(8px) saturate(1.1) !important; transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease; }
.consumidia-title{ font-weight:800; font-size:36px; background: linear-gradient(90deg, #8e44ad, #2979ff, #1abc9c, #ff8a00); -webkit-background-clip: text; background-clip: text; color:transparent; display:inline-block; }
"""

try:
    css_path = Path("style.css")
    if not css_path.exists():
        css_path.write_text(DEFAULT_CSS, encoding='utf-8')
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
except Exception:
    st.markdown(f"<style>{DEFAULT_CSS}</style>", unsafe_allow_html=True)

st.markdown("<div style='max-width:1100px;margin:18px auto 8px;text-align:center;'><h1 class='consumidia-title' style='font-size:40px;margin:0;line-height:1;'>CONSUMIDIA</h1></div>", unsafe_allow_html=True)

# -------------------------
# Supabase client (use st.secrets or env vars; NUNCA hardcode service_role in frontend)
# -------------------------
try:
    from supabase import create_client
except Exception:
    create_client = None

SUPABASE_URL = None
SUPABASE_KEY = None
if isinstance(st.secrets, dict):
    SUPABASE_URL = st.secrets.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
    SUPABASE_KEY = st.secrets.get("SUPABASE_ANON_KEY") or st.secrets.get("SUPABASE_KEY")

if not SUPABASE_URL:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
if not SUPABASE_KEY:
    SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")

_supabase = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client criado com sucesso!")
    except Exception as e:
        print(f"Erro ao criar cliente Supabase: {e}")
        _supabase = None
else:
    print("Supabase não configurado - usando fallback local")

# -------------------------
# Local users fallback (JSON)
# -------------------------
USERS_FILE = "users.json"

def load_users():
    # only used for local fallback; when supabase is present we rely on auth/profiles
    if _supabase:
        return None
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                obj = json.load(f)
                return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}

def save_users(users):
    if _supabase:
        return False
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def gen_password(length=8):
    choices = string.ascii_letters + string.digits
    return ''.join(random.choice(choices) for _ in range(length))

# -------------------------
# Util helpers
# -------------------------
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn').lower()

# -------------------------
# Favorites (session)
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
# Color palette
# -------------------------
PALETA = {"verde": "#00c853", "laranja": "#ff8a00", "amarelo": "#ffd600", "vermelho": "#ff3d00", "azul": "#2979ff", "roxo": "#8e44ad", "cinza": "#7f8c8d", "preto": "#000000", "turquesa": "#1abc9c"}

def normalize_color(name_or_hex: str):
    if not name_or_hex:
        return None
    s = str(name_or_hex).strip().lower()
    if s in PALETA:
        return PALETA[s]
    if re.match(r"^#([0-9a-f]{3}|[0-9a-f]{6})$", s):
        return s
    return s

# -------------------------
# Session defaults
# -------------------------
_defaults = {
    "authenticated": False, "username": None, "user_obj": None, "df": None,
    "G": nx.Graph(), "notes": "", "autosave": False, "page": "planilha",
    "restored_from_saved": False, "favorites": [], "reply_message_id": None,
    "anim_ts_login": 0.0
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------
# Per-user state helpers
# -------------------------
def user_state_file(username):
    return f"artemis_state_{username}.json"

def user_backup_dir(username):
    p = Path("backups") / username
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_state_for_user(username):
    path = user_state_file(username)
    backup_path = None
    try:
        if st.session_state.df is not None:
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            safe = re.sub(r"[^\w\-_.]", "_", st.session_state.get("uploaded_name") or "planilha")
            backup_filename = f"{safe}_{ts}.csv"
            backup_path = str((user_backup_dir(username) / backup_filename).resolve())
            st.session_state.df.to_csv(backup_path, index=False, encoding="utf-8")
    except Exception:
        backup_path = None
    data = {
        "graph": json_graph.node_link_data(st.session_state.G),
        "notes": st.session_state.notes,
        "uploaded_name": st.session_state.get("uploaded_name", None),
        "backup_csv": backup_path,
        "saved_at": datetime.utcnow().isoformat(),
        "favorites": st.session_state.get("favorites", [])
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path

def load_state_for_user(username, load_backup_csv=True):
    path = user_state_file(username)
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        try:
            meta = json.load(f)
        except Exception:
            return False
    try:
        st.session_state.G = json_graph.node_link_graph(meta.get("graph", {}))
    except Exception:
        st.session_state.G = nx.Graph()
    st.session_state.notes = meta.get("notes", "")
    st.session_state.uploaded_name = meta.get("uploaded_name", None)
    st.session_state.favorites = meta.get("favorites", [])
    backup_csv = meta.get("backup_csv")
    if load_backup_csv and backup_csv and os.path.exists(backup_csv):
        try:
            st.session_state.df = pd.read_csv(backup_csv, encoding="utf-8")
        except Exception:
            try:
                st.session_state.df = pd.read_csv(backup_csv, encoding="latin1")
            except Exception:
                st.session_state.df = None
    st.session_state.restored_from_saved = True
    return True

# -------------------------
# Robust spreadsheet reader
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

# -------------------------
# Graph creation + plotly 3D
# -------------------------
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
            k = f"Autor: {autor}"
            G.add_node(k, tipo="Autor", label=autor)
        if titulo:
            k = f"Título: {titulo}"
            G.add_node(k, tipo="Título", label=titulo)
        if ano:
            k = f"Ano: {ano}"
            G.add_node(k, tipo="Ano", label=ano)
        if tema:
            k = f"Tema: {tema}"
            G.add_node(k, tipo="Tema", label=tema)
        if autor and titulo:
            G.add_edge(f"Autor: {autor}", f"Título: {titulo}")
        if titulo and ano:
            G.add_edge(f"Título: {titulo}", f"Ano: {ano}")
        if titulo and tema:
            G.add_edge(f"Título: {titulo}", f"Tema: {tema}")
    return G


def graph_to_plotly_3d(G, show_labels=False, height=600):
    if len(G.nodes()) == 0:
        fig = go.Figure(); fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    pos = nx.spring_layout(G, dim=3, seed=42)
    degrees = dict(G.degree())
    deg_values = [degrees.get(n, 1) for n in G.nodes()]
    min_deg, max_deg = (min(deg_values), max(deg_values)) if deg_values else (1, 1)
    node_sizes = [8 + ((d - min_deg) / (max_deg - min_deg + 1e-6)) * 18 for d in deg_values]
    x_nodes = [pos[n][0] for n in G.nodes()]
    y_nodes = [pos[n][1] for n in G.nodes()]
    z_nodes = [pos[n][2] for n in G.nodes()]
    x_edges, y_edges, z_edges = [], [], []
    for e in G.edges():
        x_edges += [pos[e[0]][0], pos[e[1]][0], None]
        y_edges += [pos[e[0]][1], pos[e[1]][1], None]
        z_edges += [pos[e[0]][2], pos[e[1]][2], None]
    color_map = {"Autor": PALETA["verde"], "Título": PALETA["roxo"], "Ano": PALETA["azul"], "Tema": PALETA["laranja"]}
    node_colors = [color_map.get(G.nodes[n].get("tipo", ""), PALETA["vermelho"]) for n in G.nodes()]
    labels = [G.nodes[n].get("label", str(n)) for n in G.nodes()]
    hover = [f"<b>{G.nodes[n].get('label','')}</b><br>Tipo: {G.nodes[n].get('tipo','')}" for n in G.nodes()]
    edge_trace = go.Scatter3d(x=x_edges, y=y_edges, z=z_edges, mode="lines", line=dict(color="rgba(200,200,200,0.12)", width=1.2), hoverinfo="none")
    node_trace = go.Scatter3d(x=x_nodes, y=y_nodes, z=z_nodes, mode="markers+text" if show_labels else "markers",
                              marker=dict(size=node_sizes, color=node_colors, opacity=0.95, line=dict(width=1)), hovertext=hover, hoverinfo="text",
                              text=labels if show_labels else None, textposition="top center")
    legend_items = []
    for label, cor in [("Autor", PALETA["verde"]), ("Título", PALETA["roxo"]), ("Ano", PALETA["azul"]), ("Tema", PALETA["laranja"])]:
        legend_items.append(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(size=8, color=cor), name=label))
    fig = go.Figure(data=[edge_trace, node_trace] + legend_items)
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor="rgba(0,0,0,0)"), margin=dict(l=0, r=0, t=20, b=0), height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(font=dict(color="#d6d9dc"), orientation="h", y=1.05, x=0.02))
    fig.update_layout(scene_camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)))
    return fig

# -------------------------
# PDF with highlights
# -------------------------
def _safe_for_pdf(s: str):
    if s is None:
        return ""
    s2 = s.replace("—", "-").replace("–", "-")
    return s2.encode("latin-1", "replace").decode("latin-1")

def generate_pdf_with_highlights(texto, highlight_hex="#ffd600"):
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=12); pdf.add_page(); pdf.set_font("Arial", size=12)
    for linha in (texto or "").split("\n"):
        parts = re.split(r"(==.*?==)", linha)
        for part in parts:
            if not part:
                continue
            if part.startswith("==") and part.endswith("=="):
                inner = part[2:-2]
                inner_safe = _safe_for_pdf(inner)
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
                if part:
                    safe_part = _safe_for_pdf(part)
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
# ICONS + helpers
# -------------------------
ICON_SVGS = {
    "planilha": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><rect class="draw" x="2" y="3" width="20" height="18" rx="2"/></svg>',
    "anotacoes": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M3 21v-3a4 4 0 0 1 4-4h2l6-6 4 4-6 6" /></svg>',
    "mapa": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><polyline class="draw" points="3 6 9 3 15 6 21 3"/></svg>',
    "graficos": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><rect class="draw" x="3" y="10" width="4" height="10" rx="1"/></svg>',
    "save": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M5 21h14a1 1 0 0 0 1-1V7L16 3H8L4 7v13a1 1 0 0 0 1 1z"/></svg>',
    "logout": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/></svg>',
    "upload_file": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/></svg>',
    "download_backup": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><polyline class="draw" points="7 10 12 15 17 10"/></svg>',
    "login": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M12 2a5 5 0 0 1 5 5v3"/></svg>',
    "register": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><circle class="draw" cx="12" cy="8" r="3"/></svg>',
    "favoritos": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path class="draw" d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77z"/></svg>',
    "trash": '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg>'
}

def icon_html_svg(key, size=28, color=None):
    svg = ICON_SVGS.get(key, "")
    col = color or "var(--icon-color)"
    style = f"color:{col}; width:{size}px; height:{size}px; display:inline-block; vertical-align:middle;"
    return f'<span style="{style}">{svg}</span>'

# -------------------------
# Unified action button helper
# -------------------------
def action_button(label, icon_key, st_key, expanded_label=None, wide=False):
    c_icon, c_btn = st.columns([0.12, 0.88])
    with c_icon:
        st.markdown(f"<div style='margin-top:6px'>{icon_html_svg(icon_key, size=24)}</div>", unsafe_allow_html=True)
    with c_btn:
        clicked = st.button(expanded_label or label, key=st_key, use_container_width=True)
    return clicked

# -------------------------
# Supabase helpers for auth/profiles
# -------------------------
if _supabase:
    def supa_signup(email: str, password: str):
        try:
            out = _supabase.auth.sign_up({"email": email, "password": password})
            return out
        except Exception as e:
            return {"error": str(e)}

    def supa_signin(email: str, password: str):
        try:
            out = _supabase.auth.sign_in_with_password({"email": email, "password": password})
            return out
        except Exception as e:
            return {"error": str(e)}

    def supa_create_profile(user_id: str, full_name: str, role: str = "user"):
        try:
            res = _supabase.table("profiles").insert({"id": user_id, "full_name": full_name, "role": role}).execute()
            return res
        except Exception as e:
            return {"error": str(e)}
else:
    # stubs to avoid NameError
    def supa_signup(*a, **k): return {"error":"supabase_not_configured"}
    def supa_signin(*a, **k): return {"error":"supabase_not_configured"}
    def supa_create_profile(*a, **k): return {"error":"supabase_not_configured"}

# -------------------------
# Messages & attachments (supabase storage) with local fallback
# -------------------------
MESSAGES_FILE = "messages.json"
ATTACHMENTS_BUCKET = "user_files"
ATTACHMENTS_DIR = Path("user_files"); ATTACHMENTS_DIR.mkdir(exist_ok=True)

def _supabase_insert_message(entry):
    try:
        _supabase.table("messages").insert(entry).execute()
        return True
    except Exception:
        return False

def _supabase_update_message(message_id, updates: dict):
    try:
        _supabase.table("messages").update(updates).eq("id", message_id).execute()
        return True
    except Exception:
        return False

def _supabase_delete_message(message_id):
    try:
        _supabase.table("messages").delete().eq("id", message_id).execute()
        return True
    except Exception:
        return False

def _supabase_get_messages(filter_col=None, filter_val=None, box='inbox'):
    try:
        q = _supabase.table("messages").select("*").order("ts", desc=True)
        if filter_col and filter_val is not None:
            q = q.eq(filter_col, filter_val)
        res = q.execute()
        msgs = getattr(res, "data", None) or (res[0] if isinstance(res, (list, tuple)) and res else res)
        return msgs or []
    except Exception:
        return None

def _supabase_upload_file(filename, file_bytes):
    try:
        path = f"{int(time.time())}_{filename}"
        _supabase.storage.from_(ATTACHMENTS_BUCKET).upload(path, file_bytes, {"cacheControl":"3600","upsert":False})
        public = _supabase.storage.from_(ATTACHMENTS_BUCKET).get_public_url(path)
        public_url = None
        if isinstance(public, dict):
            public_url = public.get("publicURL") or public.get("public_url")
        elif hasattr(public, "get"):
            public_url = public.get("publicURL")
        else:
            public_url = getattr(public, "publicURL", None) or getattr(public, "public_url", None)
        return {"name": filename, "path": path, "url": public_url}
    except Exception:
        return None

def _supabase_remove_file(path):
    try:
        _supabase.storage.from_(ATTACHMENTS_BUCKET).remove([path])
        return True
    except Exception:
        return False

# Local message helpers
def _local_load_all_messages():
    if os.path.exists(MESSAGES_FILE):
        try:
            with open(MESSAGES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def _local_save_all_messages(msgs):
    with open(MESSAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(msgs, f, ensure_ascii=False, indent=2)

def _local_upload_attachment(sender, attachment_file):
    safe_filename = re.sub(r'[^\\w\\.\\-]', '_', attachment_file.name)
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
        msgs = _supabase_get_messages()
        if msgs is not None:
            return msgs
    return _local_load_all_messages()

def save_all_messages(msgs):
    if _supabase:
        try:
            for m in msgs:
                try:
                    _supabase_insert_message(m)
                except Exception:
                    pass
            return True
        except Exception:
            pass
    _local_save_all_messages(msgs)
    return True

def send_message(sender, recipient, subject, body, attachment_file=None):
    mid = f"m_{int(time.time())}_{random.randint(1000,9999)}"
    entry = {"id": mid, "from": sender, "to": recipient, "subject": subject or "(sem assunto)", "body": body, "ts": datetime.utcnow().isoformat(), "read": False, "attachment": None}
    if _supabase:
        if attachment_file:
            try:
                content_bytes = attachment_file.getbuffer()
                upload_meta = _supabase_upload_file(attachment_file.name, content_bytes)
                if upload_meta:
                    entry["attachment"] = upload_meta
            except Exception:
                entry["attachment"] = None
        ok = _supabase_insert_message(entry)
        if ok:
            return entry
    if attachment_file:
        entry["attachment"] = _local_upload_attachment(sender, attachment_file)
    msgs = _local_load_all_messages()
    msgs.append(entry)
    _local_save_all_messages(msgs)
    return entry

def get_user_messages(username, box_type='inbox'):
    if _supabase:
        try:
            if box_type == 'inbox':
                msgs = _supabase_get_messages(filter_col="to", filter_val=username)
            else:
                msgs = _supabase_get_messages(filter_col="from", filter_val=username)
            if msgs is not None:
                return sorted(msgs, key=lambda x: x.get("ts",""), reverse=True)
        except Exception:
            pass
    msgs = _local_load_all_messages()
    key = "to" if box_type == 'inbox' else "from"
    user_msgs = [m for m in msgs if m.get(key) == username]
    user_msgs.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return user_msgs

def mark_message_read(message_id, username):
    if _supabase:
        try:
            _supabase_update_message(message_id, {"read": True})
            return True
        except Exception:
            pass
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
    if _supabase:
        try:
            resp = _supabase.table("messages").select("*").eq("id", message_id).execute()
            msg = (getattr(resp, "data", None) or [])[0]
            if msg:
                if msg.get("to") == username or msg.get("from") == username:
                    if msg.get("attachment"):
                        path = msg["attachment"].get("path")
                        if path:
                            try:
                                _supabase_remove_file(path)
                            except Exception:
                                pass
                    _supabase_delete_message(message_id)
                    return True
        except Exception:
            pass
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
# Authentication UI & logic
# -------------------------
if "debug_auth" not in st.session_state:
    st.session_state.debug_auth = False

# Authentication flow
if not st.session_state.authenticated:
    st.markdown("<div class='glass-box auth' style='max-width:1100px;margin:0 auto; position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Acesso — Faça login ou cadastre-se")
    tabs = st.tabs(["Entrar", "Cadastrar"])

    with tabs[0]:
        login_user = st.text_input("Usuário / Email", key="ui_login_user")
        login_pass = st.text_input("Senha", type="password", key="ui_login_pass")
        if st.button("Entrar", "btn_login_main"):
            if _supabase:
                out = supa_signin(login_user.strip(), login_pass.strip())
                if isinstance(out, dict) and out.get("error"):
                    st.warning("Erro no login: " + out.get("error"))
                else:
                    # extrai user defensivamente
                    user = None
                    if isinstance(out, dict):
                        user = out.get("user") or (out.get("data") or {}).get("user")
                    elif hasattr(out, 'get'):
                        user = out.get('user')
                    if user and user.get("email"):
                        st.session_state.authenticated = True
                        st.session_state.username = user.get("email")
                        st.session_state.user_obj = {"id": user.get("id"), "email": user.get("email")}
                        st.success(f"Logado: {user.get('email')}")
                        safe_rerun()
                    else:
                        st.info("Login efetuado — verifique retorno: ")
                        st.write(out)
            else:
                lu = (login_user or "").strip(); lp = (login_pass or "").strip()
                users = load_users() or {}
                if not users:
                    # create default admin for local fallback
                    users = {"admin": {"name": "Administrador", "scholarship": "Admin", "password": "admin123", "created_at": datetime.utcnow().isoformat()}}
                    save_users(users)
                    st.warning("Nenhum usuário local encontrado. Usuário de emergência criado: 'admin' / 'admin123' (troque a senha).")
                if lu in users and users[lu].get("password") == lp:
                    st.session_state.authenticated = True
                    st.session_state.username = lu
                    st.session_state.user_obj = users[lu]
                    st.success("Login efetuado (local).")
                    safe_rerun()
                else:
                    st.warning("Usuário/Senha inválidos (local).")

    with tabs[1]:
        reg_name = st.text_input("Nome completo", key="ui_reg_name")
        reg_bolsa = st.selectbox("Tipo de bolsa", ["IC - Iniciação Científica", "BIA - Bolsa de Incentivo Acadêmico", "Extensão", "Doutorado"], key="ui_reg_bolsa")
        reg_user = st.text_input("Email (ou username para modo local)", key="ui_reg_user")
        if st.button("Cadastrar", "btn_register_main"):
            if _supabase:
                # Use a generated password then inform user to reset or use email flow
                pw = gen_password(12)
                out = supa_signup(reg_user.strip(), pw)
                if isinstance(out, dict) and out.get("error"):
                    st.error("Erro no cadastro: " + out.get("error"))
                else:
                    user = out.get("user") or (out.get("data") or {}).get("user") if isinstance(out, dict) else None
                    if user and user.get("id"):
                        r = supa_create_profile(user.get("id"), reg_name or user.get("email"), role="user")
                        st.success("Conta criada! Verifique seu email se a confirmação estiver habilitada.")
                        st.write("user id:", user.get("id"))
                    else:
                        st.info("Conta criada — verifique retorno:")
                        st.write(out)
            else:
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
                        save_users(users)
                        st.success(f"Usuário criado. Username: {new_user} — Senha gerada: {pwd}")
                        st.info("Anote a senha e troque depois")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# -------------------------
# Post-auth setup: context & UI
# -------------------------
USERNAME = st.session_state.username
users_local = load_users() or {}
USER_OBJ = st.session_state.user_obj or users_local.get(USERNAME, {})
USER_STATE = user_state_file(USERNAME if USERNAME else "anon")

if not st.session_state.restored_from_saved and os.path.exists(USER_STATE):
    try:
        load_state_for_user(USERNAME)
        st.success("Estado salvo do usuário restaurado automaticamente.")
    except Exception:
        pass

# Unread messages
MESSAGES_PATH = Path(MESSAGES_FILE)
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

# Top bar & navigation
st.markdown("<div class='glass-box' style='padding-top:10px; padding-bottom:10px;'><div class='specular'></div>", unsafe_allow_html=True)
top1, top2 = st.columns([0.6, 0.4])
with top1:
    st.markdown(f"<div style='color:var(--muted-text);font-weight:700; padding-top:8px;'>Usuário: {USER_OBJ.get('name','')} — {USER_OBJ.get('scholarship','')}</div>", unsafe_allow_html=True)
with top2:
    nav_right1, nav_right2, nav_right3 = st.columns([1,1,1])
    with nav_right1:
        st.session_state.autosave = st.checkbox("Auto-save", value=st.session_state.autosave, key="ui_autosave")
    with nav_right2:
        if st.button("💾 Salvar", key="btn_save_now", use_container_width=True):
            save_state_for_user(USERNAME)
            st.success("Progresso salvo.")
    with nav_right3:
        if st.button("🚪 Sair", key="btn_logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.user_obj = None
            st.session_state.reply_message_id = None
            for k in ("ui_login_user", "ui_login_pass", "ui_reg_user", "ui_reg_name"):
                if k in st.session_state:
                    try:
                        del st.session_state[k]
                    except Exception:
                        st.session_state[k] = ""
            safe_rerun()

st.markdown("<div style='margin-top:-20px'>", unsafe_allow_html=True)
nav_cols = st.columns(6)
nav_buttons = {
    "planilha": "📄 Planilha", "mapa": "🞠 Mapa", "anotacoes": "📝 Anotações",
    "graficos": "📊 Gráficos", "busca": "🔍 Busca", "mensagens": mens_label
}
for i, (page_key, page_label) in enumerate(nav_buttons.items()):
    with nav_cols[i]:
        if st.button(page_label, key=f"nav_{page_key}", use_container_width=True):
            st.session_state.page = page_key
            st.session_state.reply_message_id = None
            safe_rerun()
st.markdown("</div></div><hr>", unsafe_allow_html=True)

# -------------------------
# Page dispatcher
# -------------------------
if st.session_state.page == "planilha":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Planilha / Backup")
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("Restaurar estado salvo", key="btn_restore_state"):
            if load_state_for_user(USERNAME):
                st.success("Estado salvo restaurado (grafo + anotações).")
            else:
                st.info("Nenhum estado salvo encontrado.")
    with col2:
        meta = None
        if os.path.exists(USER_STATE):
            try:
                with open(USER_STATE, "r", encoding="utf-8") as f: meta = json.load(f)
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
                save_state_for_user(USERNAME)
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
                        if st.session_state.autosave: save_state_for_user(USERNAME)
                        safe_rerun()
            with right:
                del_n = st.selectbox("Excluir nó", [""] + list(st.session_state.G.nodes), key=f"del_{USERNAME}")
                if st.button("Excluir nó", key=f"btn_del_{USERNAME}"):
                    if del_n and del_n in st.session_state.G:
                        st.session_state.G.remove_node(del_n)
                        st.success(f"Nó '{del_n}' removido.")
                        if st.session_state.autosave: save_state_for_user(USERNAME)
                        safe_rerun()
                st.markdown("---")
                r_old = st.selectbox("Renomear: selecione nó", [""] + list(st.session_state.G.nodes), key=f"r_old_{USERNAME}")
                r_new = st.text_input("Novo nome", key=f"r_new_{USERNAME}")
                if st.button("Renomear", key=f"btn_ren_{USERNAME}"):
                    if r_old and r_new and r_old in st.session_state.G and r_new not in st.session_state.G:
                        nx.relabel_nodes(st.session_state.G, {r_old: r_new}, copy=False)
                        st.success(f"'{r_old}' → '{r_new}'")
                        if st.session_state.autosave: save_state_for_user(USERNAME)
                        safe_rerun()

    st.markdown("### Visualização 3D")
    try:
        fig = graph_to_plotly_3d(st.session_state.G, show_labels=False, height=700)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
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

elif st.session_state.page == "busca":
    st.markdown("<div class='glass-box' style='position:relative; padding:12px;'><div class='specular'></div>", unsafe_allow_html=True)
    tab_busca, tab_favoritos = st.tabs([f"\ud83d\udd0d Busca Inteligente", f"\u2b50 Favoritos ({len(get_session_favorites())})"])

    with tab_busca:
        st.header("Busca Inteligente")

        @st.cache_data(ttl=300)
        def collect_latest_backups():
            base = Path("backups")
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

        backups_df = collect_latest_backups()

        if backups_df is None:
            st.warning("Nenhum backup de usuário encontrado para a busca. Salve seu progresso para criar um.")
        else:
            all_cols = [c for c in backups_df.columns if c.lower() not in ['_artemis_username', 'ano']]
            col1, col2, col3 = st.columns([0.6, 0.25, 0.15])
            with col1:
                query = st.text_input("Termo de busca", key="ui_query_search", placeholder="Digite palavras-chave...")
            with col2:
                search_col = st.selectbox("Buscar em", options=all_cols)
            with col3:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                search_clicked = st.button("Buscar", use_container_width=True)

            if 'search_results' not in st.session_state:
                st.session_state.search_results = pd.DataFrame()

            if search_clicked:
                if query and search_col:
                    norm_query = normalize_text(query)
                    search_series = backups_df[search_col].astype(str).apply(normalize_text)
                    results = backups_df[search_series.str.contains(norm_query, na=False)]
                    st.session_state.search_results = results
                else:
                    st.session_state.search_results = pd.DataFrame()

            results_df = st.session_state.search_results
            if not results_df.empty:
                st.markdown(f"**{len(results_df)} resultado(s) encontrado(s).** Exibindo os 20 primeiros.")
                st.markdown("---")
                for idx, row in results_df.head(20).iterrows():
                    with st.container():
                        result_data = row.to_dict()
                        username_src = result_data.get('_artemis_username', 'N/A')
                        col_info, col_action = st.columns([0.8, 0.2])
                        with col_info:
                            user_icon_svg = icon_html_svg('register', size=18, color='var(--muted-text)')
                            st.markdown(f"""
                            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px; color: var(--muted-text);">
                                {user_icon_svg}
                                <span>Encontrado no trabalho de <strong style="color: #e1e3e6;">{username_src}</strong></span>
                            </div>
                            """, unsafe_allow_html=True)
                            display_data = {k:v for k,v in result_data.items() if k != '_artemis_username'}
                            for k, v in display_data.items():
                                st.markdown(f"**{str(k).capitalize()}:** {v}")
                        with col_action:
                            if action_button("Favoritar", "favoritos", f"fav_{idx}"):
                                if add_to_favorites(result_data):
                                    st.toast("Adicionado!", icon="⭐")
                                    save_state_for_user(USERNAME)
                                else:
                                    st.toast("Já está nos favoritos.")
            elif search_clicked:
                st.info("Nenhum resultado encontrado para a sua busca.")

    with tab_favoritos:
        st.header("Seus Resultados Salvos")
        favorites = get_session_favorites()
        if not favorites:
            st.info("Você ainda não favoritou nenhum resultado.")
        else:
            _, col_clear = st.columns([0.7, 0.3])
            with col_clear:
                if action_button("Limpar Todos", "trash", "clear_favs"):
                    clear_all_favorites()
                    save_state_for_user(USERNAME)
                    safe_rerun()
            st.markdown("---")
            sorted_favorites = sorted(favorites, key=lambda x: x['added_at'], reverse=True)
            for fav in sorted_favorites:
                with st.container():
                    col_info, col_action = st.columns([0.8, 0.2])
                    with col_info:
                        fav_data = fav['data'].copy()
                        source_user = fav_data.pop('_artemis_username', 'N/A')
                        user_icon_svg = icon_html_svg('register', size=18, color='var(--muted-text)')
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px; color: var(--muted-text);">
                            {user_icon_svg}
                            <span>Proveniente do trabalho de <strong style="color: #e1e3e6;">{source_user}</strong></span>
                        </div>
                        """, unsafe_allow_html=True)
                        for k, v in fav_data.items():
                            st.markdown(f"**{k.capitalize()}:** {v}")
                    with col_action:
                        if action_button("Remover", "trash", f"del_fav_{fav['id']}"):
                            remove_from_favorites(fav['id'])
                            save_state_for_user(USERNAME)
                            safe_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "mensagens":
    st.markdown("<div class='glass-box' style='position:relative; padding:12px;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("Central de Mensagens")

    inbox = get_user_messages(USERNAME, 'inbox')
    outbox = get_user_messages(USERNAME, 'outbox')
    tab_inbox, tab_compose, tab_sent = st.tabs([f"\ud83d\udce5 Caixa de Entrada ({sum(1 for m in inbox if not m.get('read'))})", "✍️ Escrever Nova", f"\ud83d\udce4 Enviadas ({len(outbox)})"])

    with tab_inbox:
        if not inbox:
            st.info("Sua caixa de entrada está vazia.")
        else:
            reply_message_id = st.session_state.get('reply_message_id')
            for m in inbox:
                m_id = m.get('id')
                is_read = m.get("read", False)
                expander_label = f"{'✅' if is_read else '🔵'} De: **{m.get('from')}** | Assunto: **{m.get('subject')}**"
                with st.expander(expander_label, expanded=(reply_message_id == m_id)):
                    st.markdown(f"**Recebido em:** `{m.get('ts')}`")
                    st.markdown("---")
                    st.markdown(m.get("body"))

                    if not is_read:
                        mark_message_read(m_id, USERNAME)

                    if m.get("attachment"):
                        att = m["attachment"]
                        st.markdown("---")
                        if att.get("url"):
                            st.markdown(f"[⬇️ Baixar Anexo: {att.get('name')}]({att.get('url')})")
                        else:
                            localp = att.get("path")
                            try:
                                if localp and os.path.exists(localp):
                                    with open(localp, "rb") as fp:
                                        st.download_button(label=f"⬇️ Baixar Anexo: {att.get('name')}", data=fp, file_name=att.get('name'), key=f"dl_{m_id}")
                                else:
                                    st.warning("O anexo não foi encontrado.")
                            except Exception:
                                st.warning("Erro ao disponibilizar o anexo.")
                    st.markdown("<br>", unsafe_allow_html=True)

                    if reply_message_id == m_id:
                        st.markdown("---")
                        st.subheader("Responder")
                        with st.form(key=f"reply_form_{m_id}", clear_on_submit=True):
                            original_body = m.get('body', '')
                            quoted_text = f"\n\n---\nEm {m.get('ts')}, {m.get('from')} escreveu:\n> " + "\n> ".join(original_body.split('\n'))
                            reply_body = st.text_area("Mensagem:", value=quoted_text, height=150, key=f"reply_body_{m_id}")
                            reply_attachment = st.file_uploader("Anexar arquivo:", key=f"reply_attach_{m_id}")
                            c1_form, c2_form = st.columns(2)
                            with c1_form:
                                if st.form_submit_button("✉️ Enviar Resposta", use_container_width=True):
                                    send_message(sender=USERNAME, recipient=m.get('from'), subject=f"Re: {m.get('subject')}", body=reply_body, attachment_file=reply_attachment)
                                    st.session_state.reply_message_id = None
                                    st.toast("Resposta enviada!")
                                    safe_rerun()
                            with c2_form:
                                if st.form_submit_button("Cancelar", use_container_width=True):
                                    st.session_state.reply_message_id = None
                                    safe_rerun()
                    else:
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("Responder", key=f"reply_{m_id}", use_container_width=True):
                                st.session_state.reply_message_id = m_id
                                safe_rerun()
                        with c2:
                            if st.button("Apagar", key=f"del_inbox_{m_id}", use_container_width=True):
                                delete_message(m_id, USERNAME)
                                if st.session_state.reply_message_id == m_id:
                                    st.session_state.reply_message_id = None
                                st.toast("Mensagem apagada.")
                                safe_rerun()

    with tab_compose:
        with st.form(key="compose_form", clear_on_submit=True):
            users_dict = load_users() or {}
            all_usernames = [u for u in users_dict.keys() if u != USERNAME]
            if not all_usernames:
                st.info("Nenhum outro usuário cadastrado — digite o username do destinatário manualmente.")
                to_user = st.text_input("Para (username):", key="compose_manual_to")
            else:
                to_user = st.selectbox("Para:", options=["(escolha)"] + all_usernames, index=0, key="compose_select_to")
                if to_user == "(escolha)":
                    to_user = None
                if st.checkbox("Enviar para username específico (digitar manualmente)", key="compose_manual_toggle"):
                    manu = st.text_input("Username (manual):", key="compose_manual_to2")
                    if manu:
                        to_user = manu.strip()
            subj = st.text_input("Assunto:")
            body = st.text_area("Mensagem:", height=200)
            attachment = st.file_uploader("Anexar arquivo:", key="compose_attachment")
            submitted = st.form_submit_button("✉️ Enviar Mensagem", use_container_width=True)
            if submitted:
                if not to_user:
                    st.error("Destinatário inválido. Informe um username existente ou digite manualmente.")
                else:
                    if to_user not in users_dict and _supabase is None:
                        st.warning(f"Usuário '{to_user}' não encontrado. Verifique o username.")
                    else:
                        send_message(USERNAME, to_user, subj, body, attachment_file=attachment)
                        st.success(f"Mensagem enviada para {to_user}.")
                        if st.session_state.autosave:
                            save_state_for_user(USERNAME)
                        safe_rerun()

    with tab_sent:
        sent = get_user_messages(USERNAME, 'outbox')
        if not sent:
            st.info("Nenhuma mensagem enviada ainda.")
        else:
            for m in sent:
                with st.expander(f"Para: {m.get('to')} | {m.get('subject')}"):
                    st.markdown(m.get('body'))
    st.markdown("</div>", unsafe_allow_html=True)  
