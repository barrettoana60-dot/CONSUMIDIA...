import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import os
from datetime import datetime, timedelta
import hashlib
import base64
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from supabase import create_client
import uuid
import seaborn as sns
from collections import Counter
import re

# ==================== CONFIGURAÇÃO INICIAL ====================
st.set_page_config(
    page_title="Folksonomia Digital | Museus Interativos",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎨"
)

# ==================== SUPABASE CONFIGURATION ====================
SUPABASE_URL = "https://irxyfzfvvdszfkkjothq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlyeHlmemZ2dmRzemZra2pvdGhxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDY5MjE2NzMsImV4cCI6MjA2MjQ5NzY3M30.vcEG3PUVG_X_PdML_JHqygAjcumfvrcAAteEYF5msHo"

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==================== CSS MODERNO E ANIMADO ====================
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Playfair+Display:wght@400;700&display=swap');

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .main-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }

    .floating-button {
        display: inline-block;
        padding: 15px 30px;
        margin: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        border: none;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

    .floating-button:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
    }

    .obra-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        margin: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.5s cubic-bezier(0.23, 1, 0.320, 1);
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }

    .obra-card:hover {
        transform: translateY(-20px) rotateX(5deg) rotateY(5deg);
        box-shadow: 0 20px 50px rgba(0,0,0,0.2);
    }

    .obra-card img {
        transition: transform 0.8s ease;
        border-radius: 15px;
    }

    .obra-card:hover img {
        transform: scale(1.1) rotate(2deg);
    }

    .gradient-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        font-family: 'Playfair Display', serif;
        text-align: center;
        margin: 30px 0;
        animation: fadeInDown 1s ease;
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        color: white;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0;
    }

    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 12px 30px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        color: white;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    @media (max-width: 768px) {
        .gradient-title {
            font-size: 2rem;
        }
        .metric-value {
            font-size: 1.8rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== FUNÇÕES AUXILIARES ====================

def check_and_init_admin():
    """Inicializa administrador padrão se não existir"""
    try:
        response = supabase_client.table('admin').select('*').execute()
        if not response.data:
            hashed_password = hashlib.sha256("admin123".encode()).hexdigest()
            supabase_client.table('admin').insert({
                "username": "admin",
                "password": hashed_password
            }).execute()
    except Exception as e:
        st.error(f"Erro ao verificar admin: {e}")

def generate_user_id():
    """Gera ID único para usuário"""
    return base64.b64encode(os.urandom(12)).decode('ascii')

def upload_image_to_storage(file):
    """Faz upload de imagem para Supabase Storage"""
    try:
        if not hasattr(file, 'name') or not file.name:
            st.error("Arquivo sem nome válido")
            return None

        file_name = file.name.lower()
        if '.' in file_name:
            file_ext = file_name.split('.')[-1]
        else:
            st.error("Arquivo sem extensão")
            return None

        if file_ext not in ['jpg', 'jpeg', 'png']:
            st.error(f"Formato não permitido: {file_ext}")
            return None

        unique_filename = f"{uuid.uuid4()}.{file_ext}"

        supabase_client.storage.from_('obras-imagens').upload(
            unique_filename,
            file.getvalue(),
            file_options={"contentType": f"image/{file_ext.replace('jpg', 'jpeg')}"}
        )

        image_url = supabase_client.storage.from_('obras-imagens').get_public_url(unique_filename)
        return image_url
    except Exception as e:
        st.error(f"Erro no upload: {str(e)}")
        return None

@st.cache_data(ttl=5, show_spinner=False)
def load_obras():
    """Carrega obras do banco de dados"""
    try:
        response = supabase_client.table('obras').select('*').execute()
        if response.data:
            return response.data
        else:
            obras = [{
                "id": 1, 
                "titulo": "Guernica", 
                "artista": "Pablo Picasso", 
                "ano": "1937",
                "imagem": "https://upload.wikimedia.org/wikipedia/en/7/74/PicassoGuernica.jpg"
            }]
            supabase_client.table('obras').insert(obras).execute()
            return obras
    except Exception as e:
        st.error(f"Erro ao carregar obras: {e}")
        return []

def save_user_answers(user_id, answers):
    """Salva respostas do questionário"""
    try:
        new_row = {
            "user_id": user_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "q1": answers["q1"],
            "q2": answers["q2"],
            "q3": answers["q3"]
        }
        supabase_client.table('users').insert(new_row).execute()
        return True
    except Exception as e:
        st.error(f"Erro ao salvar respostas: {e}")
        return False

def save_tag(user_id, obra_id, tag):
    """Salva tag associada a uma obra"""
    try:
        new_row = {
            "user_id": user_id,
            "obra_id": obra_id,
            "tag": tag.lower().strip(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        supabase_client.table('tags').insert(new_row).execute()
        return True
    except Exception as e:
        st.error(f"Erro ao salvar tag: {e}")
        return False

def get_tags_for_obra(obra_id):
    """Obtém tags de uma obra específica"""
    try:
        response = supabase_client.table('tags').select('tag').eq('obra_id', obra_id).execute()
        if response.data:
            tags_df = pd.DataFrame(response.data)
            tag_counts = tags_df['tag'].value_counts().reset_index()
            tag_counts.columns = ["tag", "count"]
            return tag_counts
        return pd.DataFrame(columns=["tag", "count"])
    except Exception as e:
        st.error(f"Erro ao obter tags: {e}")
        return pd.DataFrame(columns=["tag", "count"])

def check_admin_credentials(username, password):
    """Verifica credenciais do administrador"""
    try:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        response = supabase_client.table('admin').select('*')\
            .eq('username', username)\
            .eq('password', hashed_password)\
            .execute()
        return len(response.data) > 0
    except Exception as e:
        st.error(f"Erro ao verificar credenciais: {e}")
        return False

# ==================== ANÁLISES AVANÇADAS ====================

def calculate_tag_diversity(tags_df):
    """Calcula diversidade de tags usando índice de Shannon"""
    if tags_df.empty:
        return 0

    tag_counts = tags_df['tag'].value_counts()
    proportions = tag_counts / tag_counts.sum()
    shannon_index = -sum(proportions * np.log(proportions))
    return shannon_index

def get_tag_growth_rate(tags_df):
    """Calcula taxa de crescimento de tags"""
    if tags_df.empty or 'timestamp' not in tags_df.columns:
        return None

    tags_df['date'] = pd.to_datetime(tags_df['timestamp']).dt.date
    daily_counts = tags_df.groupby('date').size().reset_index(name='count')

    if len(daily_counts) < 2:
        return None

    daily_counts['growth_rate'] = daily_counts['count'].pct_change() * 100
    return daily_counts

def analyze_user_engagement(users_df, tags_df):
    """Análise de engajamento dos usuários"""
    if users_df.empty or tags_df.empty:
        return None

    tags_per_user = tags_df.groupby('user_id').size().reset_index(name='tag_count')

    engagement_stats = {
        'avg_tags_per_user': tags_per_user['tag_count'].mean(),
        'median_tags_per_user': tags_per_user['tag_count'].median(),
        'max_tags_per_user': tags_per_user['tag_count'].max(),
        'total_active_users': len(tags_per_user),
        'total_registered_users': len(users_df)
    }

    return engagement_stats

def get_top_contributors(tags_df, top_n=10):
    """Identifica principais contribuidores"""
    if tags_df.empty:
        return pd.DataFrame()

    contributors = tags_df.groupby('user_id').agg({
        'tag': 'count',
        'timestamp': 'min'
    }).reset_index()

    contributors.columns = ['user_id', 'total_tags', 'first_contribution']
    contributors = contributors.sort_values('total_tags', ascending=False).head(top_n)

    return contributors

def analyze_tag_patterns(tags_df):
    """Analisa padrões nas tags"""
    if tags_df.empty:
        return None

    patterns = {
        'avg_tag_length': tags_df['tag'].str.len().mean(),
        'single_word_tags': sum(tags_df['tag'].str.split().str.len() == 1),
        'multi_word_tags': sum(tags_df['tag'].str.split().str.len() > 1),
        'numeric_tags': sum(tags_df['tag'].str.contains(r'\d', regex=True)),
        'special_char_tags': sum(tags_df['tag'].str.contains(r'[^a-zA-Z0-9\s]', regex=True))
    }

    return patterns

# ==================== VISUALIZAÇÕES AVANÇADAS ====================

def create_interactive_tag_frequency(tags_df):
    """Gráfico interativo de frequência de tags"""
    if tags_df.empty:
        return None

    all_tags = tags_df["tag"].value_counts().reset_index()
    all_tags.columns = ["tag", "count"]
    top_tags = all_tags.head(20)

    fig = px.bar(
        top_tags, 
        x='count', 
        y='tag',
        orientation='h',
        title='Top 20 Tags Mais Frequentes',
        labels={'count': 'Frequência', 'tag': 'Tag'},
        color='count',
        color_continuous_scale='Viridis',
        text='count'
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'},
        font=dict(family="Poppins, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')

    return fig

def create_tag_timeline(tags_df):
    """Linha do tempo interativa de tags"""
    if tags_df.empty or 'timestamp' not in tags_df.columns:
        return None

    tags_df['date'] = pd.to_datetime(tags_df['timestamp']).dt.date
    timeline = tags_df.groupby('date').size().reset_index(name='count')

    timeline['moving_avg'] = timeline['count'].rolling(window=3, min_periods=1).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timeline['date'],
        y=timeline['count'],
        mode='lines+markers',
        name='Tags por Dia',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#764ba2')
    ))

    fig.add_trace(go.Scatter(
        x=timeline['date'],
        y=timeline['moving_avg'],
        mode='lines',
        name='Média Móvel (3 dias)',
        line=dict(color='#e73c7e', width=2, dash='dash')
    ))

    fig.update_layout(
        title='Evolução Temporal das Tags',
        xaxis_title='Data',
        yaxis_title='Número de Tags',
        hovermode='x unified',
        height=500,
        font=dict(family="Poppins, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig

def create_heatmap_tags_by_obra(tags_df, obras):
    """Mapa de calor: tags por obra"""
    if tags_df.empty:
        return None

    obra_tags = tags_df.groupby('obra_id').size().reset_index(name='count')

    obra_info = pd.DataFrame(obras)
    merged = obra_info.merge(obra_tags, left_on='id', right_on='obra_id', how='left')
    merged['count'] = merged['count'].fillna(0)

    fig = px.bar(
        merged,
        x='titulo',
        y='count',
        title='Distribuição de Tags por Obra',
        labels={'titulo': 'Obra', 'count': 'Número de Tags'},
        color='count',
        color_continuous_scale='Plasma',
        text='count'
    )

    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        font=dict(family="Poppins, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')

    return fig

def create_sunburst_chart(tags_df, obras):
    """Gráfico sunburst de tags por obra"""
    if tags_df.empty:
        return None

    obra_info = pd.DataFrame(obras)
    merged = tags_df.merge(obra_info[['id', 'titulo']], left_on='obra_id', right_on='id', how='left')

    sunburst_data = merged.groupby(['titulo', 'tag']).size().reset_index(name='count')

    fig = px.sunburst(
        sunburst_data,
        path=['titulo', 'tag'],
        values='count',
        title='Hierarquia de Tags por Obra',
        color='count',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        height=700,
        font=dict(family="Poppins, sans-serif"),
    )

    return fig

def create_wordcloud_plotly(tags_df):
    """Nuvem de palavras interativa"""
    if tags_df.empty:
        return None

    tag_counts = tags_df["tag"].value_counts().to_dict()
    wc = WordCloud(
        width=1200, 
        height=600, 
        background_color="white",
        colormap='viridis',
        relative_scaling=0.5,
        min_font_size=10
    ).generate_from_frequencies(tag_counts)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)

    return fig

def create_engagement_funnel(users_df, tags_df):
    """Funil de engajamento de usuários"""
    total_registered = len(users_df) if not users_df.empty else 0
    total_active = len(tags_df['user_id'].unique()) if not tags_df.empty else 0
    total_tags = len(tags_df) if not tags_df.empty else 0

    if not tags_df.empty:
        multi_contrib = len(tags_df.groupby('user_id').filter(lambda x: len(x) > 1)['user_id'].unique())
    else:
        multi_contrib = 0

    fig = go.Figure(go.Funnel(
        y=['Usuários Registrados', 'Usuários Ativos', 'Múltiplas Contribuições', 'Total de Tags'],
        x=[total_registered, total_active, multi_contrib, total_tags],
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=["#667eea", "#764ba2", "#e73c7e", "#23d5ab"]),
    ))

    fig.update_layout(
        title='Funil de Engajamento de Usuários',
        height=500,
        font=dict(family="Poppins, sans-serif"),
    )

    return fig

def create_tag_network(tags_df):
    """Rede de co-ocorrência simplificada"""
    if tags_df.empty:
        return None

    from itertools import combinations

    user_tags = tags_df.groupby('user_id')['tag'].apply(list).values
    cooccurrence = Counter()

    for tags in user_tags:
        if len(tags) > 1:
            for pair in combinations(sorted(set(tags)), 2):
                cooccurrence[pair] += 1

    if not cooccurrence:
        return None

    edges = [(pair[0], pair[1], count) for pair, count in cooccurrence.most_common(15)]

    nodes = set()
    for edge in edges:
        nodes.add(edge[0])
        nodes.add(edge[1])

    import math
    node_list = list(nodes)
    n = len(node_list)
    positions = {}

    for i, node in enumerate(node_list):
        angle = 2 * math.pi * i / n
        positions[node] = (math.cos(angle), math.sin(angle))

    edge_traces = []
    for edge in edges:
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]

        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=edge[2]/max([e[2] for e in edges])*10, color='rgba(125,125,125,0.3)'),
            hoverinfo='none',
            showlegend=False
        ))

    node_x = [positions[node][0] for node in node_list]
    node_y = [positions[node][1] for node in node_list]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(size=20, color='#667eea', line=dict(width=2, color='white')),
        text=node_list,
        textposition="top center",
        textfont=dict(size=10, family="Poppins"),
        hoverinfo='text',
        showlegend=False
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title='Rede de Co-ocorrência de Tags',
        showlegend=False,
        hovermode='closest',
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    return fig

# ==================== INTERFACE PRINCIPAL ====================

def main():
    load_custom_css()

    try:
        check_and_init_admin()
    except Exception as e:
        st.error(f"Erro ao verificar admin: {e}")

    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = generate_user_id()
    if 'step' not in st.session_state:
        st.session_state['step'] = 'intro'
    if 'answers' not in st.session_state:
        st.session_state['answers'] = {}
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "Início"

    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: white; font-family: Playfair Display;'>🎨 Folksonomia</h1>
            <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Museus Interativos</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        pages = ["🏠 Início", "🖼️ Explorar Obras", "📊 Área Administrativa"]
        page_mapping = {
            "🏠 Início": "Início",
            "🖼️ Explorar Obras": "Explorar Obras",
            "📊 Área Administrativa": "Área Administrativa"
        }

        selected_page = st.radio("Navegação", pages, label_visibility="collapsed")

        page = page_mapping[selected_page]

        if page != st.session_state.get('current_page'):
            st.session_state['current_page'] = page
            st.rerun()

        st.markdown("---")

        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-top: 20px;'>
            <p style='color: white; margin: 0; font-size: 0.8rem;'>ID do Usuário:</p>
            <p style='color: rgba(255,255,255,0.7); margin: 5px 0 0 0; font-size: 0.7rem; word-break: break-all;'>{st.session_state['user_id'][:12]}...</p>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state['current_page'] == "Início":
        show_intro()
    elif st.session_state['current_page'] == "Explorar Obras":
        show_obras()
    elif st.session_state['current_page'] == "Área Administrativa":
        show_admin()

# ==================== PÁGINA INICIAL ====================

def show_intro():
    st.markdown("<div class='gradient-title'>Projeto de Folksonomia em Museus</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; max-width: 800px; margin: 0 auto 40px auto; color: white; font-size: 1.1rem; line-height: 1.8;'>
        Bem-vindo à nossa plataforma interativa de catalogação colaborativa! 
        Explore obras de arte e contribua com suas próprias tags.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state['step'] == 'intro':
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)

        st.markdown("""
        <h2 style='color: white; text-align: center; margin-bottom: 30px;'>📋 Questionário Inicial</h2>
        """, unsafe_allow_html=True)

        with st.form("intro_form"):
            col1, col2 = st.columns([1, 1])

            with col1:
                q1 = st.selectbox(
                    "Qual é o seu nível de familiaridade com museus?",
                    ["Nunca visito museus", "Visito raramente", "Visito ocasionalmente", "Visito frequentemente"]
                )

                q2 = st.selectbox(
                    "Você já ouviu falar sobre documentação museológica?",
                    ["Nunca ouvi falar", "Já ouvi, mas não sei o que é", "Tenho uma ideia básica", "Conheço bem o tema"]
                )

            with col2:
                q3 = st.text_area(
                    "O que você entende por 'tags' ou etiquetas digitais aplicadas a acervo?",
                    max_chars=500,
                    height=200
                )

            st.markdown("<br>", unsafe_allow_html=True)

            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
            with col_btn2:
                submit = st.form_submit_button("✨ Enviar Respostas", use_container_width=True)

            if submit:
                if not q3.strip():
                    st.error("Por favor, responda todas as perguntas!")
                else:
                    st.session_state['answers'] = {"q1": q1, "q2": q2, "q3": q3}
                    save_user_answers(st.session_state['user_id'], st.session_state['answers'])
                    st.session_state['step'] = 'completed'
                    st.success("✅ Respostas enviadas com sucesso!")
                    st.balloons()
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.success("✅ Questionário concluído com sucesso!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class='obra-card' style='text-align: center;'>
                <h3 style='color: #667eea;'>🖼️ Explorar</h3>
                <p>Descubra obras incríveis</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class='obra-card' style='text-align: center;'>
                <h3 style='color: #764ba2;'>🏷️ Contribuir</h3>
                <p>Adicione suas tags</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class='obra-card' style='text-align: center;'>
                <h3 style='color: #e73c7e;'>📊 Analisar</h3>
                <p>Veja estatísticas</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            if st.button("🎨 Começar a Explorar", use_container_width=True):
                st.session_state['current_page'] = "Explorar Obras"
                st.rerun()

# ==================== PÁGINA DE OBRAS ====================

def show_obras():
    st.markdown("<div class='gradient-title'>Galeria de Obras Interativa</div>", unsafe_allow_html=True)

    if st.session_state['step'] == 'intro':
        st.warning("⚠️ Complete o questionário inicial antes de explorar.")
        if st.button("📋 Ir para o Questionário"):
            st.session_state['current_page'] = "Início"
            st.rerun()
        return

    obras = load_obras()

    if not obras:
        st.info("Nenhuma obra cadastrada.")
        return

    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 1])

    with col_filter1:
        search_term = st.text_input("🔍 Buscar obra", "")

    with col_filter2:
        sort_by = st.selectbox("Ordenar por:", ["Título", "Artista", "Ano"])

    with col_filter3:
        view_mode = st.selectbox("Visualização:", ["Grid", "Lista"])

    st.markdown("</div>", unsafe_allow_html=True)

    filtered_obras = obras
    if search_term:
        filtered_obras = [
            obra for obra in obras 
            if search_term.lower() in obra['titulo'].lower() or 
               search_term.lower() in obra['artista'].lower()
        ]

    if sort_by == "Título":
        filtered_obras = sorted(filtered_obras, key=lambda x: x['titulo'])
    elif sort_by == "Artista":
        filtered_obras = sorted(filtered_obras, key=lambda x: x['artista'])
    elif sort_by == "Ano":
        filtered_obras = sorted(filtered_obras, key=lambda x: x['ano'])

    st.markdown(f"""
    <div style='text-align: center; color: white; margin: 20px 0;'>
        <h3>Mostrando {len(filtered_obras)} obra(s)</h3>
    </div>
    """, unsafe_allow_html=True)

    if view_mode == "Grid":
        cols = st.columns(3)
        for i, obra in enumerate(filtered_obras):
            with cols[i % 3]:
                st.markdown(f"""
                <div class='obra-card'>
                    <img src='{obra['imagem']}' style='width: 100%; border-radius: 15px; margin-bottom: 15px;' />
                    <h3 style='color: #667eea; margin: 10px 0;'>{obra['titulo']}</h3>
                    <p style='color: #666; margin: 5px 0;'><strong>{obra['artista']}</strong></p>
                    <p style='color: #999; margin: 5px 0;'>{obra['ano']}</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"🏷️ Adicionar Tag", key=f"btn_{obra['id']}", use_container_width=True):
                    st.session_state['selected_obra'] = obra
                    st.rerun()

                if 'selected_obra' in st.session_state and st.session_state['selected_obra']['id'] == obra['id']:
                    with st.form(f"tag_form_{obra['id']}"):
                        tag = st.text_input("Digite sua tag:", key=f"tag_input_{obra['id']}")

                        col_submit1, col_submit2 = st.columns(2)
                        with col_submit1:
                            submitted = st.form_submit_button("✅ Enviar", use_container_width=True)
                        with col_submit2:
                            cancel = st.form_submit_button("❌ Cancelar", use_container_width=True)

                        if submitted and tag:
                            save_tag(st.session_state['user_id'], obra['id'], tag)
                            st.success(f"Tag '{tag}' adicionada! 🎉")
                            st.balloons()
                            del st.session_state['selected_obra']
                            st.rerun()

                        if cancel:
                            del st.session_state['selected_obra']
                            st.rerun()

                    tags = get_tags_for_obra(obra['id'])
                    if not tags.empty:
                        st.markdown("**🏆 Tags Populares:**")
                        for _, row in tags.head(5).iterrows():
                            st.markdown(f"""
                            <div style='display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        color: white; padding: 5px 15px; border-radius: 20px; margin: 3px; font-size: 0.85rem;'>
                                {row['tag']} ({row['count']})
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Seja o primeiro! 🌟")

    else:
        for obra in filtered_obras:
            with st.container():
                col_img, col_info = st.columns([1, 2])

                with col_img:
                    st.image(obra['imagem'], use_container_width=True)

                with col_info:
                    st.markdown(f"### {obra['titulo']}")
                    st.markdown(f"**Artista:** {obra['artista']}")
                    st.markdown(f"**Ano:** {obra['ano']}")

                    if st.button(f"🏷️ Adicionar Tag", key=f"btn_list_{obra['id']}"):
                        st.session_state['selected_obra'] = obra
                        st.rerun()

                    tags = get_tags_for_obra(obra['id'])
                    if not tags.empty:
                        st.markdown("**Tags Populares:**")
                        tag_html = ""
                        for _, row in tags.head(10).iterrows():
                            tag_html += f"""
                            <span style='display: inline-block; background: #667eea; color: white; 
                                         padding: 3px 10px; border-radius: 15px; margin: 2px; font-size: 0.8rem;'>
                                {row['tag']} ({row['count']})
                            </span>
                            """
                        st.markdown(tag_html, unsafe_allow_html=True)

                st.markdown("---")

# ==================== ÁREA ADMINISTRATIVA ====================

def show_admin():
    st.markdown("<div class='gradient-title'>Área Administrativa</div>", unsafe_allow_html=True)

    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False

    if not st.session_state['admin_logged_in']:
        st.markdown("<div class='main-container' style='max-width: 500px; margin: 50px auto;'>", unsafe_allow_html=True)

        st.markdown("""
        <h2 style='color: white; text-align: center; margin-bottom: 30px;'>
            🔐 Login Administrativo
        </h2>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("👤 Usuário:", placeholder="Digite seu usuário")
            password = st.text_input("🔑 Senha:", type="password", placeholder="Digite sua senha")

            st.markdown("<br>", unsafe_allow_html=True)

            submitted = st.form_submit_button("🚀 Entrar", use_container_width=True)

            if submitted:
                if check_admin_credentials(username, password):
                    st.session_state['admin_logged_in'] = True
                    st.session_state['admin_username'] = username
                    st.success("Login realizado! 🎉")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Credenciais inválidas.")

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("ℹ️ Informações"):
            st.info("**Credenciais padrão:**\n- Usuário: `admin`\n- Senha: `admin123`")

    else:
        st.markdown(f"""
        <div style='text-align: right; color: white; margin-bottom: 20px;'>
            Bem-vindo, <strong>{st.session_state.get('admin_username', 'Admin')}</strong>! 👋
        </div>
        """, unsafe_allow_html=True)

        admin_tabs = st.tabs([
            "📊 Dashboard Analytics",
            "🖼️ Gerenciar Obras",
            "👥 Gerenciar Admins",
            "⚙️ Configurações"
        ])

        with admin_tabs[0]:
            show_analytics_dashboard()

        with admin_tabs[1]:
            show_manage_obras()

        with admin_tabs[2]:
            show_manage_admins()

        with admin_tabs[3]:
            show_settings()

        st.markdown("---")
        col_logout1, col_logout2, col_logout3 = st.columns([1, 1, 1])
        with col_logout2:
            if st.button("🚪 Sair", use_container_width=True):
                st.session_state['admin_logged_in'] = False
                if 'admin_username' in st.session_state:
                    del st.session_state['admin_username']
                st.rerun()

def show_analytics_dashboard():
    """Dashboard completo de analytics"""

    st.markdown("""
    <h2 style='color: white; margin-bottom: 30px;'>
        📊 Dashboard de Análise Avançada
    </h2>
    """, unsafe_allow_html=True)

    try:
        tags_response = supabase_client.table('tags').select('*').execute()
        users_response = supabase_client.table('users').select('*').execute()
        obras = load_obras()

        tags_df = pd.DataFrame(tags_response.data) if tags_response.data else pd.DataFrame()
        users_df = pd.DataFrame(users_response.data) if users_response.data else pd.DataFrame()

    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return

    st.markdown("### 📈 Métricas Principais")

    col1, col2, col3, col4 = st.columns(4)

    total_users = len(users_df['user_id'].unique()) if not users_df.empty else 0
    total_tags = len(tags_df) if not tags_df.empty else 0
    unique_tags = len(tags_df['tag'].unique()) if not tags_df.empty else 0
    total_obras = len(obras)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total de Usuários</div>
            <div class='metric-value'>{total_users}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total de Tags</div>
            <div class='metric-value'>{total_tags}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Tags Únicas</div>
            <div class='metric-value'>{unique_tags}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Obras Cadastradas</div>
            <div class='metric-value'>{total_obras}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    if tags_df.empty:
        st.info("📭 Ainda não há dados suficientes para análise.")
        return

    viz_tabs = st.tabs([
        "📊 Visão Geral",
        "🏷️ Análise de Tags",
        "👥 Engajamento",
        "🔍 Insights Avançados",
        "📥 Exportar Dados"
    ])

    with viz_tabs[0]:
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("#### 📊 Top 20 Tags")
            fig_freq = create_interactive_tag_frequency(tags_df)
            if fig_freq:
                st.plotly_chart(fig_freq, use_container_width=True)

        with col_chart2:
            st.markdown("#### 📈 Evolução Temporal")
            fig_timeline = create_tag_timeline(tags_df)
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown("---")

        col_chart3, col_chart4 = st.columns(2)

        with col_chart3:
            st.markdown("#### 🎨 Distribuição por Obra")
            fig_heatmap = create_heatmap_tags_by_obra(tags_df, obras)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)

        with col_chart4:
            st.markdown("#### 🔄 Funil de Engajamento")
            fig_funnel = create_engagement_funnel(users_df, tags_df)
            if fig_funnel:
                st.plotly_chart(fig_funnel, use_container_width=True)

    with viz_tabs[1]:
        st.markdown("### 🏷️ Análise Detalhada de Tags")

        col_wc, col_net = st.columns([1, 1])

        with col_wc:
            st.markdown("#### ☁️ Nuvem de Palavras")
            fig_wc = create_wordcloud_plotly(tags_df)
            if fig_wc:
                st.pyplot(fig_wc)

        with col_net:
            st.markdown("#### 🕸️ Rede de Co-ocorrência")
            fig_network = create_tag_network(tags_df)
            if fig_network:
                st.plotly_chart(fig_network, use_container_width=True)

        st.markdown("---")

        st.markdown("#### 🌅 Hierarquia de Tags")
        fig_sunburst = create_sunburst_chart(tags_df, obras)
        if fig_sunburst:
            st.plotly_chart(fig_sunburst, use_container_width=True)

        st.markdown("---")

        st.markdown("#### 📋 Padrões de Tags")
        patterns = analyze_tag_patterns(tags_df)
        if patterns:
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)

            with col_p1:
                st.metric("Comprimento Médio", f"{patterns['avg_tag_length']:.1f} chars")
            with col_p2:
                st.metric("Tags Simples", patterns['single_word_tags'])
            with col_p3:
                st.metric("Tags Compostas", patterns['multi_word_tags'])
            with col_p4:
                diversity = calculate_tag_diversity(tags_df)
                st.metric("Diversidade", f"{diversity:.2f}")

    with viz_tabs[2]:
        st.markdown("### 👥 Análise de Engajamento")

        engagement = analyze_user_engagement(users_df, tags_df)

        if engagement:
            col_e1, col_e2, col_e3, col_e4 = st.columns(4)

            with col_e1:
                st.metric("Média Tags/User", f"{engagement['avg_tags_per_user']:.1f}")
            with col_e2:
                st.metric("Mediana Tags/User", f"{engagement['median_tags_per_user']:.1f}")
            with col_e3:
                st.metric("Máx Tags/User", engagement['max_tags_per_user'])
            with col_e4:
                st.metric("Usuários Ativos", engagement['total_active_users'])

        st.markdown("---")
        st.markdown("#### 🏆 Top Contribuidores")
        contributors = get_top_contributors(tags_df, 10)
        if not contributors.empty:
            st.dataframe(contributors, use_container_width=True)

    with viz_tabs[3]:
        st.markdown("### 🔍 Insights Avançados")

        growth = get_tag_growth_rate(tags_df)
        if growth is not None and len(growth) > 1:
            st.markdown("#### 📈 Taxa de Crescimento")
            fig_growth = px.line(
                growth,
                x='date',
                y='growth_rate',
                title='Taxa de Crescimento Diário (%)',
                labels={'date': 'Data', 'growth_rate': 'Crescimento (%)'}
            )
            st.plotly_chart(fig_growth, use_container_width=True)

        st.markdown("#### 📊 Estatísticas Gerais")
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            if not tags_df.empty:
                avg_tags_per_obra = len(tags_df) / len(obras) if obras else 0
                st.metric("Média Tags/Obra", f"{avg_tags_per_obra:.1f}")

        with col_s2:
            if not tags_df.empty:
                most_tagged_obra = tags_df['obra_id'].mode()[0] if not tags_df['obra_id'].mode().empty else None
                if most_tagged_obra:
                    obra_info = next((o for o in obras if o['id'] == most_tagged_obra), None)
                    if obra_info:
                        st.metric("Obra Mais Tagada", obra_info['titulo'])

        with col_s3:
            if not tags_df.empty:
                most_common_tag = tags_df['tag'].mode()[0] if not tags_df['tag'].mode().empty else "N/A"
                st.metric("Tag Mais Comum", most_common_tag)

    with viz_tabs[4]:
        st.markdown("### 📥 Exportar Dados")

        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            if not tags_df.empty:
                csv_tags = tags_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download Tags (CSV)",
                    data=csv_tags,
                    file_name=f'tags_data_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv',
                    use_container_width=True
                )

        with col_exp2:
            if not users_df.empty:
                csv_users = users_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download Usuários (CSV)",
                    data=csv_users,
                    file_name=f'users_data_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv',
                    use_container_width=True
                )

def show_manage_obras():
    """Gerenciamento de obras"""
    st.markdown("### 🖼️ Gerenciar Obras")

    obras = load_obras()

    if obras:
        obras_df = pd.DataFrame(obras)
        st.subheader("Obras Existentes")
        st.dataframe(obras_df[["id", "titulo", "artista", "ano"]], use_container_width=True)
    else:
        st.write("Nenhuma obra cadastrada.")

    st.markdown("---")
    st.subheader("➕ Adicionar Nova Obra")

    with st.form("adicionar_obra"):
        novo_titulo = st.text_input("Título da Obra:")
        novo_artista = st.text_input("Artista:")
        novo_ano = st.text_input("Ano:")

        imagem_opcao = st.radio("Fonte da Imagem:", ["URL", "Upload"])
        imagem_path = ""

        if imagem_opcao == "URL":
            imagem_path = st.text_input("URL da Imagem:")
        else:
            st.write("**Formatos: JPG, JPEG, PNG**")
            uploaded_file = st.file_uploader("Carregar Imagem", accept_multiple_files=False)
            if uploaded_file is not None:
                try:
                    st.image(uploaded_file, caption="Preview", width=300)
                except Exception:
                    st.error("Erro ao exibir preview")

        submit_obra = st.form_submit_button("Adicionar Obra")

        if submit_obra:
            if not novo_titulo or not novo_artista:
                st.error("Preencha título e artista!")
            elif imagem_opcao == "URL" and not imagem_path:
                st.error("Informe a URL!")
            elif imagem_opcao == "Upload" and uploaded_file is None:
                st.error("Faça o upload!")
            else:
                if imagem_opcao == "Upload" and uploaded_file is not None:
                    file_name = uploaded_file.name.lower()
                    if not (file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png')):
                        st.error("Use apenas JPG, JPEG ou PNG")
                        st.stop()

                    with st.spinner("Fazendo upload..."):
                        imagem_path = upload_image_to_storage(uploaded_file)
                        if not imagem_path:
                            st.error("Falha no upload")
                            st.stop()

                novo_id = 1
                if obras:
                    ids = [obra["id"] for obra in obras]
                    novo_id = max(ids) + 1

                try:
                    nova_obra = {
                        "id": novo_id,
                        "titulo": novo_titulo,
                        "artista": novo_artista,
                        "ano": novo_ano,
                        "imagem": imagem_path
                    }
                    supabase_client.table('obras').insert(nova_obra).execute()
                    st.cache_data.clear()
                    st.success(f"Obra '{novo_titulo}' adicionada!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")

    st.markdown("---")
    st.subheader("❌ Excluir Obra")

    with st.form("excluir_obra"):
        if obras:
            obra_para_excluir = st.selectbox(
                "Selecione a obra:",
                [""] + [f"{obra['id']}: {obra['titulo']} - {obra['artista']}" for obra in obras]
            )
            submit_exclusao = st.form_submit_button("Excluir Obra")

            if submit_exclusao and obra_para_excluir:
                try:
                    obra_id = int(obra_para_excluir.split(":")[0])
                    tags_response = supabase_client.table('tags').select('*').eq('obra_id', obra_id).execute()

                    if tags_response.data:
                        st.warning(f"Esta obra possui {len(tags_response.data)} tags. Exclua as tags primeiro.")
                    else:
                        supabase_client.table('obras').delete().eq('id', obra_id).execute()
                        st.cache_data.clear()
                        st.success("Obra excluída!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Erro: {e}")
        else:
            st.write("Nenhuma obra para excluir.")

def show_manage_admins():
    """Gerenciamento de administradores"""
    st.subheader("👥 Gerenciar Administradores")
