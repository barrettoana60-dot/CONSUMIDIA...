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
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURAÇÃO INICIAL ====================

# Configuração do Supabase
SUPABASE_URL = "https://irxyfzfvvdszfkkjothq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlyeHlmemZ2dmRzemZra2pvdGhxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDY5MjE2NzMsImV4cCI6MjA2MjQ5NzY3M30.vcEG3PUVG_X_PdML_JHqygAjcumfvrcAAteEYF5msHo"

# Inicializar cliente Supabase
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configuração da página com tema escuro moderno
st.set_page_config(
    page_title="Folksonomia Cultural | Museu Digital",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Sistema Inteligente de Folksonomia para Museus v2.0"
    }
)

# ==================== ESTILOS CSS MODERNOS ====================

def load_custom_css():
    st.markdown("""
    <style>
    /* Importar fontes modernas */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Roboto:wght@300;400;500&display=swap');

    /* Reset e configuração base */
    * {
        font-family: 'Poppins', sans-serif;
    }

    /* Fundo gradiente animado */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Container principal com glassmorphism */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    /* Barra lateral moderna */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        border-right: 2px solid rgba(255, 255, 255, 0.1);
    }

    section[data-testid="stSidebar"] > div {
        background: transparent;
    }

    /* Títulos com efeito neon */
    h1, h2, h3 {
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }

    /* Botões flutuantes com hover effect */
    .floating-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 30px;
        border-radius: 50px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }

    .floating-button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.6);
    }

    .floating-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }

    .floating-button:hover::before {
        left: 100%;
    }

    /* Cards de obras com efeito 3D */
    .obra-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

    .obra-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        opacity: 0;
        transition: opacity 0.3s;
    }

    .obra-card:hover {
        transform: translateY(-15px) rotateX(5deg) rotateY(5deg);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4);
    }

    .obra-card:hover::before {
        opacity: 1;
    }

    .obra-card img {
        transition: transform 0.5s ease;
        border-radius: 15px;
    }

    .obra-card:hover img {
        transform: scale(1.1) rotate(2deg);
    }

    /* Métricas animadas */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        animation: slideInUp 0.5s ease;
    }

    @keyframes slideInUp {
        from {
            transform: translateY(30px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5);
    }

    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 10px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Gráficos com bordas suaves */
    .plotly-graph-div {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* Inputs personalizados */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 12px;
        transition: all 0.3s ease;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }

    /* Tabs modernos */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.5);
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.2);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }

    /* Animação de loading personalizada */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }

    /* Tags badges */
    .tag-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }

    .tag-badge:hover {
        transform: scale(1.1);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.5);
    }

    /* Efeito de partículas no fundo */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }

    /* Botão de navegação sidebar */
    .css-1d391kg, .css-1avcm0n {
        color: white !important;
    }

    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px;
        padding: 15px;
        animation: slideInRight 0.5s ease;
    }

    @keyframes slideInRight {
        from {
            transform: translateX(100px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    /* Expander moderno */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 10px;
        padding: 15px;
        font-weight: 600;
    }

    /* Data table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Download buttons */
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    /* Header navigation buttons */
    .nav-button {
        display: inline-block;
        margin: 0 10px;
        padding: 12px 25px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 25px;
        font-weight: 600;
        color: #667eea;
        text-decoration: none;
        transition: all 0.3s ease;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }

    .nav-button:hover {
        transform: translateY(-8px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        background: white;
    }

    /* Scrollbar personalizada */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== FUNÇÕES DO SUPABASE ====================

def check_and_init_admin():
    """Inicializa admin padrão se não existir"""
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

def upload_image_to_storage(file):
    """Upload de imagem para Supabase Storage"""
    try:
        if not hasattr(file, 'name') or not file.name:
            st.error("O arquivo enviado não possui um nome válido.")
            return None

        file_name = file.name.lower()
        if '.' in file_name:
            file_ext = file_name.split('.')[-1]
        else:
            st.error("O arquivo não possui uma extensão.")
            return None

        if file_ext not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            st.error(f"Formato não permitido: {file_ext}. Use jpg, jpeg, png, gif ou webp.")
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
            obras = [
                {"id": 1, "titulo": "Guernica", "artista": "Pablo Picasso", "ano": "1937",
                 "imagem": "https://upload.wikimedia.org/wikipedia/en/7/74/PicassoGuernica.jpg"},
            ]
            supabase_client.table('obras').insert(obras).execute()
            return obras
    except Exception as e:
        st.error(f"Erro ao carregar obras: {e}")
        return []

def generate_user_id():
    """Gera ID único para usuário"""
    return base64.b64encode(os.urandom(12)).decode('ascii')

def save_user_answers(user_id, answers):
    """Salva respostas do questionário"""
    try:
        new_row = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
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
            "timestamp": datetime.now().isoformat()
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
    """Verifica credenciais do admin"""
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

# ==================== ANÁLISES AVANÇADAS COM IA ====================

def analyze_tags_sentiment(tags_df):
    """Análise de sentimento e padrões nas tags usando clustering"""
    if tags_df.empty:
        return None

    try:
        # Criar features TF-IDF das tags
        vectorizer = TfidfVectorizer(max_features=50)
        tag_list = tags_df['tag'].tolist()

        if len(tag_list) < 3:
            return {"clusters": [], "insights": "Dados insuficientes para análise"}

        # Clustering K-means
        X = vectorizer.fit_transform(tag_list)
        n_clusters = min(5, len(tag_list) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)

        # Agrupar tags por cluster
        tags_df['cluster'] = clusters
        cluster_groups = {}

        for i in range(n_clusters):
            cluster_tags = tags_df[tags_df['cluster'] == i]['tag'].value_counts().head(5)
            cluster_groups[f"Grupo {i+1}"] = cluster_tags.to_dict()

        # Insights automáticos
        insights = []
        total_tags = len(tags_df)
        unique_tags = len(tags_df['tag'].unique())

        insights.append(f"📊 Diversidade: {unique_tags} tags únicas de {total_tags} totais ({(unique_tags/total_tags)*100:.1f}%)")

        most_common = tags_df['tag'].value_counts().iloc[0]
        insights.append(f"🏆 Tag mais popular: '{tags_df['tag'].value_counts().index[0]}' ({most_common} usos)")

        # Análise temporal se houver timestamp
        if 'timestamp' in tags_df.columns:
            tags_df['date'] = pd.to_datetime(tags_df['timestamp']).dt.date
            recent_tags = tags_df[tags_df['date'] >= (datetime.now().date() - timedelta(days=7))]
            if len(recent_tags) > 0:
                insights.append(f"📈 Crescimento: {len(recent_tags)} tags na última semana")

        return {
            "clusters": cluster_groups,
            "insights": insights,
            "diversity_score": (unique_tags/total_tags)*100
        }
    except Exception as e:
        return {"clusters": [], "insights": [f"Erro na análise: {str(e)}"]}

def generate_advanced_visualizations(tags_df, users_df):
    """Gera visualizações avançadas com Plotly"""
    visualizations = {}

    if not tags_df.empty:
        # 1. Treemap de tags hierárquico
        tag_counts = tags_df['tag'].value_counts().reset_index()
        tag_counts.columns = ['tag', 'count']
        tag_counts = tag_counts.head(20)

        fig_treemap = px.treemap(
            tag_counts,
            path=['tag'],
            values='count',
            title='Distribuição Hierárquica de Tags',
            color='count',
            color_continuous_scale='Viridis'
        )
        fig_treemap.update_layout(
            template='plotly_white',
            font=dict(family='Poppins', size=12),
            height=500
        )
        visualizations['treemap'] = fig_treemap

        # 2. Sunburst chart
        fig_sunburst = px.sunburst(
            tag_counts,
            path=['tag'],
            values='count',
            title='Visualização Radial de Tags',
            color='count',
            color_continuous_scale='RdYlBu'
        )
        fig_sunburst.update_layout(
            template='plotly_white',
            font=dict(family='Poppins', size=12),
            height=500
        )
        visualizations['sunburst'] = fig_sunburst

        # 3. Gráfico de evolução temporal interativo
        if 'timestamp' in tags_df.columns:
            tags_df['date'] = pd.to_datetime(tags_df['timestamp'])
            tags_df['hour'] = tags_df['date'].dt.hour
            tags_df['day_of_week'] = tags_df['date'].dt.day_name()

            # Heatmap de atividade por hora e dia
            activity_pivot = tags_df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')

            fig_heatmap = px.density_heatmap(
                activity_pivot,
                x='hour',
                y='day_of_week',
                z='count',
                title='Mapa de Calor: Atividade por Hora e Dia',
                color_continuous_scale='Plasma',
                labels={'hour': 'Hora do Dia', 'day_of_week': 'Dia da Semana', 'count': 'Atividade'}
            )
            fig_heatmap.update_layout(
                template='plotly_white',
                font=dict(family='Poppins', size=12),
                height=400
            )
            visualizations['heatmap'] = fig_heatmap

            # Timeline de tags acumuladas
            tags_timeline = tags_df.groupby(tags_df['date'].dt.date).size().cumsum().reset_index()
            tags_timeline.columns = ['date', 'cumulative_tags']

            fig_timeline = px.area(
                tags_timeline,
                x='date',
                y='cumulative_tags',
                title='Crescimento Acumulado de Tags',
                labels={'date': 'Data', 'cumulative_tags': 'Tags Acumuladas'}
            )
            fig_timeline.update_traces(
                fillcolor='rgba(102, 126, 234, 0.3)',
                line=dict(color='rgb(102, 126, 234)', width=3)
            )
            fig_timeline.update_layout(
                template='plotly_white',
                font=dict(family='Poppins', size=12),
                height=400
            )
            visualizations['timeline'] = fig_timeline

        # 4. Network graph de co-ocorrência de tags
        # Análise de quais tags aparecem juntas para as mesmas obras
        obra_tags = tags_df.groupby('obra_id')['tag'].apply(list).reset_index()

        # 5. Top tags por obra - Gráfico de barras agrupadas
        if 'obra_id' in tags_df.columns:
            top_tags_by_obra = tags_df.groupby(['obra_id', 'tag']).size().reset_index(name='count')
            top_tags_by_obra = top_tags_by_obra.sort_values('count', ascending=False).head(30)

            fig_tags_obras = px.bar(
                top_tags_by_obra,
                x='tag',
                y='count',
                color='obra_id',
                title='Top Tags por Obra',
                labels={'tag': 'Tag', 'count': 'Frequência', 'obra_id': 'Obra ID'},
                barmode='group'
            )
            fig_tags_obras.update_layout(
                template='plotly_white',
                font=dict(family='Poppins', size=12),
                height=500,
                xaxis_tickangle=-45
            )
            visualizations['tags_by_obra'] = fig_tags_obras

    # 6. Análise de usuários
    if not users_df.empty:
        # Distribuição de respostas Q1
        if 'q1' in users_df.columns:
            q1_counts = users_df['q1'].value_counts().reset_index()
            q1_counts.columns = ['response', 'count']

            fig_q1 = px.pie(
                q1_counts,
                values='count',
                names='response',
                title='Familiaridade com Museus',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig_q1.update_layout(
                template='plotly_white',
                font=dict(family='Poppins', size=12),
                height=400
            )
            visualizations['q1_distribution'] = fig_q1

        # Distribuição de respostas Q2
        if 'q2' in users_df.columns:
            q2_counts = users_df['q2'].value_counts().reset_index()
            q2_counts.columns = ['response', 'count']

            fig_q2 = px.bar(
                q2_counts,
                x='response',
                y='count',
                title='Conhecimento sobre Documentação Museológica',
                color='count',
                color_continuous_scale='Viridis'
            )
            fig_q2.update_layout(
                template='plotly_white',
                font=dict(family='Poppins', size=12),
                height=400,
                xaxis_tickangle=-45
            )
            visualizations['q2_distribution'] = fig_q2

    return visualizations

def create_dashboard_metrics(tags_df, users_df, obras):
    """Cria métricas avançadas para o dashboard"""
    metrics = {}

    # Métricas básicas
    metrics['total_users'] = len(users_df['user_id'].unique()) if not users_df.empty else 0
    metrics['total_tags'] = len(tags_df) if not tags_df.empty else 0
    metrics['unique_tags'] = len(tags_df['tag'].unique()) if not tags_df.empty else 0
    metrics['total_obras'] = len(obras)

    # Métricas calculadas
    if metrics['total_tags'] > 0:
        metrics['avg_tags_per_user'] = metrics['total_tags'] / max(metrics['total_users'], 1)
        metrics['avg_tags_per_obra'] = metrics['total_tags'] / max(metrics['total_obras'], 1)
        metrics['diversity_index'] = (metrics['unique_tags'] / metrics['total_tags']) * 100
    else:
        metrics['avg_tags_per_user'] = 0
        metrics['avg_tags_per_obra'] = 0
        metrics['diversity_index'] = 0

    # Tendências temporais
    if not tags_df.empty and 'timestamp' in tags_df.columns:
        tags_df['date'] = pd.to_datetime(tags_df['timestamp'])

        # Tags das últimas 24h
        last_24h = tags_df[tags_df['date'] >= (datetime.now() - timedelta(hours=24))]
        metrics['tags_last_24h'] = len(last_24h)

        # Tags da última semana
        last_week = tags_df[tags_df['date'] >= (datetime.now() - timedelta(days=7))]
        metrics['tags_last_week'] = len(last_week)

        # Taxa de crescimento
        if len(tags_df) > 1:
            tags_df_sorted = tags_df.sort_values('date')
            first_half = len(tags_df_sorted[:len(tags_df_sorted)//2])
            second_half = len(tags_df_sorted[len(tags_df_sorted)//2:])
            if first_half > 0:
                metrics['growth_rate'] = ((second_half - first_half) / first_half) * 100
            else:
                metrics['growth_rate'] = 0
        else:
            metrics['growth_rate'] = 0
    else:
        metrics['tags_last_24h'] = 0
        metrics['tags_last_week'] = 0
        metrics['growth_rate'] = 0

    # Engajamento
    if not users_df.empty and not tags_df.empty:
        users_with_tags = tags_df['user_id'].unique()
        metrics['engagement_rate'] = (len(users_with_tags) / max(len(users_df), 1)) * 100
    else:
        metrics['engagement_rate'] = 0

    return metrics

# ==================== COMPONENTES DE INTERFACE ====================

def render_floating_header():
    """Renderiza header flutuante com botões animados"""
    st.markdown("""
    <div style='text-align: center; padding: 20px 0; margin-bottom: 30px;'>
        <h1 style='font-size: 3.5rem; margin-bottom: 10px; animation: slideInDown 0.8s ease;'>
            🎨 Folksonomia Cultural
        </h1>
        <p style='font-size: 1.2rem; color: #666; font-weight: 300; animation: fadeIn 1.2s ease;'>
            Sistema Inteligente de Catalogação Colaborativa para Museus
        </p>
    </div>

    <style>
    @keyframes slideInDown {
        from {
            transform: translateY(-50px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

def render_metric_card(title, value, delta=None, icon="📊"):
    """Renderiza card de métrica animado"""
    delta_html = ""
    if delta is not None:
        color = "green" if delta >= 0 else "red"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f"<div style='font-size: 0.9rem; color: {color}; margin-top: 5px;'>{arrow} {abs(delta):.1f}%</div>"

    return f"""
    <div class='metric-card'>
        <div style='font-size: 2rem; margin-bottom: 10px;'>{icon}</div>
        <div class='metric-label'>{title}</div>
        <div class='metric-value'>{value}</div>
        {delta_html}
    </div>
    """

def render_obra_card(obra, index):
    """Renderiza card de obra com animação 3D"""
    return f"""
    <div class='obra-card' style='animation-delay: {index * 0.1}s;'>
        <img src='{obra["imagem"]}' style='width: 100%; height: 250px; object-fit: cover;' />
        <h3 style='margin-top: 15px; font-size: 1.3rem; color: #333;'>{obra["titulo"]}</h3>
        <p style='color: #666; font-size: 0.95rem; margin: 5px 0;'>{obra["artista"]}</p>
        <p style='color: #999; font-size: 0.85rem;'>{obra["ano"]}</p>
    </div>
    """

# ==================== PÁGINAS PRINCIPAIS ====================

def show_intro():
    """Página inicial com questionário"""
    render_floating_header()

    st.markdown("""
    <div style='background: rgba(255,255,255,0.9); padding: 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); animation: fadeIn 1s ease;'>
        <h2 style='color: #667eea; margin-bottom: 20px;'>🌟 Bem-vindo ao Projeto!</h2>
        <p style='font-size: 1.1rem; line-height: 1.8; color: #555;'>
            Estamos revolucionando a forma como o público interage com acervos museológicos através da 
            <strong>folksonomia</strong> - um sistema de classificação colaborativa onde VOCÊ é protagonista!
        </p>
        <p style='font-size: 1.1rem; line-height: 1.8; color: #555; margin-top: 15px;'>
            Sua participação nos ajuda a entender como diferentes pessoas percebem e categorizam obras de arte,
            criando uma rede de conhecimento coletivo e democrático.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state['step'] == 'intro':
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 15px; color: white; margin: 20px 0;
                    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);'>
            <h3 style='color: white; margin-bottom: 15px;'>📋 Questionário Inicial</h3>
            <p style='color: rgba(255,255,255,0.9);'>
                Antes de começar, nos ajude a conhecer você melhor! São apenas 3 perguntas rápidas.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("intro_form", clear_on_submit=True):
            col1, col2 = st.columns([1, 1])

            with col1:
                q1 = st.selectbox(
                    "1️⃣ Qual é o seu nível de familiaridade com museus?",
                    ["Nunca visito museus", "Visito raramente", "Visito ocasionalmente", "Visito frequentemente"],
                    help="Isso nos ajuda a entender o perfil dos participantes"
                )

                q2 = st.selectbox(
                    "2️⃣ Você já ouviu falar sobre documentação museológica?",
                    ["Nunca ouvi falar", "Já ouvi, mas não sei o que é", "Tenho uma ideia básica", "Conheço bem o tema"],
                    help="Queremos saber seu conhecimento prévio sobre o tema"
                )

            with col2:
                q3 = st.text_area(
                    "3️⃣ O que você entende por 'tags' ou etiquetas digitais aplicadas a acervos?",
                    max_chars=500,
                    height=150,
                    help="Compartilhe sua compreensão sobre tags - não há resposta certa ou errada!",
                    placeholder="Ex: Tags são palavras-chave que ajudam a organizar e encontrar conteúdo..."
                )

            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                submit = st.form_submit_button("🚀 Começar Jornada", use_container_width=True)

            if submit:
                if not q3.strip():
                    st.error("❌ Por favor, responda a questão 3 antes de continuar.")
                else:
                    st.session_state['answers'] = {"q1": q1, "q2": q2, "q3": q3}
                    save_user_answers(st.session_state['user_id'], st.session_state['answers'])
                    st.session_state['step'] = 'completed'
                    st.success("✅ Respostas salvas com sucesso!")
                    st.balloons()
                    st.rerun()
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #52c234 0%, #061700 100%); 
                    padding: 30px; border-radius: 20px; color: white; text-align: center;
                    box-shadow: 0 10px 30px rgba(82, 194, 52, 0.3);
                    animation: pulse 2s ease infinite;'>
            <h2 style='color: white; font-size: 2rem; margin-bottom: 15px;'>🎉 Obrigado por participar!</h2>
            <p style='font-size: 1.2rem; color: rgba(255,255,255,0.95);'>
                Suas respostas foram registradas. Agora você pode explorar as obras e adicionar suas tags!
            </p>
            <p style='font-size: 1rem; color: rgba(255,255,255,0.8); margin-top: 15px;'>
                Use o menu lateral para navegar até "Explorar Obras" 👈
            </p>
        </div>
        """, unsafe_allow_html=True)

def show_obras():
    """Página de exploração de obras com animações"""
    render_floating_header()

    if st.session_state['step'] == 'intro':
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 30px; border-radius: 20px; color: white; text-align: center;
                    box-shadow: 0 10px 30px rgba(240, 147, 251, 0.4);'>
            <h2 style='color: white;'>⚠️ Atenção!</h2>
            <p style='font-size: 1.2rem; margin: 20px 0;'>
                Por favor, complete o questionário inicial antes de explorar as obras.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("📝 Responder Questionário", use_container_width=True):
                st.session_state['current_page'] = "Início"
                st.rerun()
        return

    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h2 style='color: #667eea; font-size: 2.5rem;'>🖼️ Galeria de Obras</h2>
        <p style='font-size: 1.1rem; color: #666;'>
            Explore as obras abaixo e adicione tags que representem sua interpretação pessoal
        </p>
    </div>
    """, unsafe_allow_html=True)

    obras = load_obras()

    # Filtro de pesquisa
    search = st.text_input("🔍 Buscar obras por título ou artista:", placeholder="Digite para filtrar...")

    if search:
        obras = [o for o in obras if search.lower() in o['titulo'].lower() or search.lower() in o['artista'].lower()]

    # Grid de obras com animações
    cols_per_row = 3
    for i in range(0, len(obras), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(obras):
                obra = obras[i + j]
                with col:
                    # Container com animação
                    st.markdown(render_obra_card(obra, i + j), unsafe_allow_html=True)

                    # Botão de seleção estilizado
                    if st.button(f"🏷️ Adicionar Tag", key=f"btn_{obra['id']}", use_container_width=True):
                        st.session_state['selected_obra'] = obra
                        st.rerun()

                    # Se obra selecionada, mostrar formulário
                    if 'selected_obra' in st.session_state and st.session_state['selected_obra']['id'] == obra['id']:
                        with st.expander("✨ Adicionar sua tag", expanded=True):
                            with st.form(f"tag_form_{obra['id']}"):
                                tag = st.text_input(
                                    "Digite uma palavra ou frase que descreva esta obra:",
                                    placeholder="Ex: melancolia, guerra, cubismo...",
                                    help="Seja criativo! Não há resposta errada."
                                )

                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    submitted = st.form_submit_button("💾 Salvar Tag", use_container_width=True)
                                with col2:
                                    cancel = st.form_submit_button("❌ Cancelar", use_container_width=True)

                                if cancel:
                                    if 'selected_obra' in st.session_state:
                                        del st.session_state['selected_obra']
                                    st.rerun()

                                if submitted and tag:
                                    save_tag(st.session_state['user_id'], obra['id'], tag)
                                    st.success(f"✅ Tag '{tag}' adicionada com sucesso!")
                                    st.cache_data.clear()
                                    if 'selected_obra' in st.session_state:
                                        del st.session_state['selected_obra']
                                    st.balloons()
                                    st.rerun()

                        # Mostrar tags populares
                        tags = get_tags_for_obra(obra['id'])
                        if not tags.empty:
                            st.markdown("**🔥 Tags Populares:**")
                            tags_html = ""
                            for _, row in tags.head(10).iterrows():
                                tags_html += f"<span class='tag-badge'>{row['tag']} ({row['count']})</span>"
                            st.markdown(tags_html, unsafe_allow_html=True)
                        else:
                            st.info("🎯 Seja o primeiro a adicionar uma tag para esta obra!")

def show_admin():
    """Área administrativa com dashboard completo"""
    render_floating_header()

    # Login
    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False

    if not st.session_state['admin_logged_in']:
        st.markdown("""
        <div style='max-width: 400px; margin: 50px auto; background: white; 
                    padding: 40px; border-radius: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.1);'>
            <h2 style='text-align: center; color: #667eea; margin-bottom: 30px;'>🔐 Login Administrativo</h2>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("login_form"):
                username = st.text_input("👤 Usuário:", placeholder="Digite seu usuário")
                password = st.text_input("🔑 Senha:", type="password", placeholder="Digite sua senha")

                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("🚀 Entrar", use_container_width=True)

                if submitted:
                    if check_admin_credentials(username, password):
                        st.session_state['admin_logged_in'] = True
                        st.success("✅ Login realizado com sucesso!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Credenciais inválidas. Tente novamente.")
    else:
        # Dashboard Administrativo
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 15px; color: white; margin-bottom: 30px;
                    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);'>
            <h2 style='color: white; margin: 0;'>📊 Dashboard Administrativo</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 10px 0 0 0;'>
                Análise completa e inteligente dos dados coletados
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Criar abas principais
        admin_tabs = st.tabs([
            "📊 Dashboard Inteligente",
            "🎨 Gerenciar Obras", 
            "👥 Gerenciar Administradores",
            "📥 Exportar Dados"
        ])

        # ==================== TAB 1: DASHBOARD INTELIGENTE ====================
        with admin_tabs[0]:
            try:
                # Carregar dados
                tags_response = supabase_client.table('tags').select('*').execute()
                users_response = supabase_client.table('users').select('*').execute()
                obras = load_obras()

                tags_df = pd.DataFrame(tags_response.data) if tags_response.data else pd.DataFrame()
                users_df = pd.DataFrame(users_response.data) if users_response.data else pd.DataFrame()

                # Calcular métricas
                metrics = create_dashboard_metrics(tags_df, users_df, obras)

                # Mostrar métricas principais
                st.markdown("### 📈 Métricas Principais")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(
                        render_metric_card("Total de Usuários", metrics['total_users'], icon="👥"),
                        unsafe_allow_html=True
                    )

                with col2:
                    st.markdown(
                        render_metric_card("Total de Tags", metrics['total_tags'], 
                                         delta=metrics['growth_rate'], icon="🏷️"),
                        unsafe_allow_html=True
                    )

                with col3:
                    st.markdown(
                        render_metric_card("Tags Únicas", metrics['unique_tags'], icon="✨"),
                        unsafe_allow_html=True
                    )

                with col4:
                    st.markdown(
                        render_metric_card("Obras Cadastradas", metrics['total_obras'], icon="🖼️"),
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Segunda linha de métricas
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(
                        render_metric_card("Média Tags/Usuário", 
                                         f"{metrics['avg_tags_per_user']:.1f}", icon="📊"),
                        unsafe_allow_html=True
                    )

                with col2:
                    st.markdown(
                        render_metric_card("Média Tags/Obra", 
                                         f"{metrics['avg_tags_per_obra']:.1f}", icon="🎨"),
                        unsafe_allow_html=True
                    )

                with col3:
                    st.markdown(
                        render_metric_card("Índice Diversidade", 
                                         f"{metrics['diversity_index']:.1f}%", icon="🌈"),
                        unsafe_allow_html=True
                    )

                with col4:
                    st.markdown(
                        render_metric_card("Taxa Engajamento", 
                                         f"{metrics['engagement_rate']:.1f}%", icon="🎯"),
                        unsafe_allow_html=True
                    )

                st.markdown("<br><br>", unsafe_allow_html=True)

                # Análise Inteligente com IA
                if not tags_df.empty:
                    st.markdown("### 🤖 Análise Inteligente com IA")

                    with st.spinner("🔍 Analisando padrões nos dados..."):
                        analysis = analyze_tags_sentiment(tags_df)

                    if analysis:
                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                        padding: 20px; border-radius: 15px; color: white;
                                        box-shadow: 0 8px 20px rgba(240, 147, 251, 0.3);'>
                                <h4 style='color: white; margin-bottom: 15px;'>💡 Insights Automáticos</h4>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown("<br>", unsafe_allow_html=True)

                            if isinstance(analysis['insights'], list):
                                for insight in analysis['insights']:
                                    st.markdown(f"""
                                    <div style='background: white; padding: 15px; border-radius: 10px; 
                                                margin-bottom: 10px; border-left: 4px solid #667eea;
                                                box-shadow: 0 3px 10px rgba(0,0,0,0.1);'>
                                        {insight}
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info(analysis['insights'])

                        with col2:
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 20px; border-radius: 15px; color: white;
                                        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);'>
                                <h4 style='color: white; margin-bottom: 15px;'>🔬 Clusters Identificados</h4>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown("<br>", unsafe_allow_html=True)

                            if analysis['clusters']:
                                for cluster_name, cluster_tags in analysis['clusters'].items():
                                    with st.expander(f"📂 {cluster_name}", expanded=True):
                                        tags_html = ""
                                        for tag, count in list(cluster_tags.items())[:5]:
                                            tags_html += f"<span class='tag-badge'>{tag} ({count})</span>"
                                        st.markdown(tags_html, unsafe_allow_html=True)

                    st.markdown("<br><br>", unsafe_allow_html=True)

                    # Visualizações Avançadas
                    st.markdown("### 📊 Visualizações Avançadas")

                    viz = generate_advanced_visualizations(tags_df, users_df)

                    # Criar sub-tabs para visualizações
                    viz_tabs = st.tabs([
                        "🌳 Treemap",
                        "☀️ Sunburst", 
                        "🔥 Mapa de Calor",
                        "📈 Timeline",
                        "📊 Por Obra",
                        "👥 Usuários"# Projeto Folksonomia em Museus - Versão Ultra Moderna 🎨

Vou criar uma versão completamente reformulada com interface moderna, animações e dashboards avançados!

