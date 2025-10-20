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

# optional ML libs
try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.manifold import TSNE
except Exception:
    joblib = None
    TfidfVectorizer = None
    cosine_similarity = None
    KMeans = None
    LatentDirichletAllocation = None
    TSNE = None

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
# Base CSS MODERNIZADO
# -------------------------
BASE_CSS = r"""
:root{ 
    --glass-bg-dark: rgba(255,255,255,0.03); 
    --muted-text-dark:#bfc6cc;
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --modern-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
}
body { transition: background-color .25s ease, color .25s ease; }
.glass-box{ border-radius:16px; padding:20px; box-shadow: var(--modern-shadow); backdrop-filter: blur(10px); }
.card, .msg-card { border-radius:12px; padding:16px; margin-bottom:12px; transition: transform 0.2s ease, box-shadow 0.2s ease; }
.card:hover, .msg-card:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5); }
.avatar{width:48px;height:48px;border-radius:12px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;margin-right:12px}
.small-muted{font-size:13px;}
.card-title{font-weight:700;font-size:16px}
.card-mark{ background: linear-gradient(135deg, #FFD166 0%, #FFA62E 100%); padding: 2px 6px; border-radius:6px; color: #000; font-weight:600; }
/* Estilos para botões interativos */
.stButton>button, .stDownloadButton>button {
    transition: transform 0.1s ease, opacity 0.1s ease, background-color 0.15s ease !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}
.stButton>button:active, .stDownloadButton>button:active {
    transform: scale(0.97);
    opacity: 0.8;
}
/* Estilos para o mapa mental MODERNIZADO */
.node-editor { 
    background: rgba(255,255,255,0.05); 
    border-radius: 16px; 
    padding: 20px; 
    margin: 12px 0; 
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}
.node-preview {
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    border-left: 6px solid #6c5ce7;
}
.mindmap-3d {
    background: linear-gradient(145deg, #1a2a6c, #b21f1f, #fdbb2d);
    border-radius: 20px;
    padding: 20px;
    margin: 12px 0;
    border: 3px solid #4ECDC4;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
}
.flowchart-box {
    background: rgba(255,255,255,0.1);
    border: 3px solid #FF6B6B;
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    backdrop-filter: blur(10px);
}
.three-d-effect {
    background: linear-gradient(145deg, #1a2a6c, #b21f1f);
    border-radius: 20px;
    padding: 25px;
    margin: 15px 0;
    border: 3px solid #FECA57;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    backdrop-filter: blur(15px);
}
.modern-node {
    border-radius: 16px !important;
    border: 2px solid rgba(255,255,255,0.3) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
    backdrop-filter: blur(10px) !important;
}
/* Configurações de fonte */
.font-config {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 20px;
    margin: 12px 0;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}
/* Animações suaves */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in {
    animation: fadeIn 0.5s ease-in-out;
}
"""

DEFAULT_CSS = r"""
.css-1d391kg { background: linear-gradient(180deg,#071428 0%, #031926 100%) !important; }
.glass-box{ background: rgba(14, 25, 42, 0.8); border:1px solid rgba(42, 59, 82, 0.6); box-shadow:var(--modern-shadow); }
.stButton>button, .stDownloadButton>button{ background: var(--primary-gradient) !important; color:#ffffff !important; border:none !important; padding:12px 20px !important; border-radius:12px !important; font-weight:600 !important; }
.stButton>button:hover, .stDownloadButton>button:hover {
    background: var(--secondary-gradient) !important;
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
}
.card, .msg-card { background: rgba(14, 25, 42, 0.8); border-radius:16px; padding:16px; margin-bottom:12px; border:1px solid rgba(42, 59, 82, 0.6); backdrop-filter: blur(10px); }
.avatar{color:#fff;background:var(--primary-gradient)}
.small-muted{color:#bfc6cc;}
.card-title{color:#fff}
"""

st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)
st.markdown(f"<style>{DEFAULT_CSS}</style>", unsafe_allow_html=True)

# TÍTULO - centralizado (geral) e estilo padronizado
st.markdown("<div style='text-align:center; padding-top:8px; padding-bottom:6px;'><h1 style='margin:0;color:#ffffff;background:var(--primary-gradient);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>NUGEP-PQR</h1></div>", unsafe_allow_html=True)
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
# AI Helper Functions - SUPER MELHORADA
# -------------------------
class DataAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.insights = []
        self.user_profile = self._analyze_user_profile()
    
    def _analyze_user_profile(self):
        """Analisa o perfil do usuário baseado nos dados"""
        profile = {
            "research_focus": [],
            "collaboration_level": "baixo",
            "geographic_scope": "local",
            "temporal_pattern": "estável",
            "data_quality": "boa"
        }
        
        # Análise de foco de pesquisa
        text_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
        all_text = ""
        for col in text_columns[:3]:
            all_text += " " + self.df[col].fillna('').astype(str).str.cat(sep=' ')
        
        if len(all_text) > 100:
            words = re.findall(r'\b[a-zà-ú]{5,}\b', all_text.lower())
            from collections import Counter
            word_freq = Counter([w for w in words if w not in PORTUGUESE_STOP_WORDS])
            profile["research_focus"] = [word for word, count in word_freq.most_common(5)]
        
        return profile
    
    def generate_comprehensive_analysis(self):
        """Gera uma análise completa e inteligente dos dados"""
        analysis = ""
        
        # Análise básica
        analysis += self._basic_analysis()
        analysis += self._author_analysis()
        analysis += self._temporal_analysis()
        analysis += self._thematic_analysis()
        analysis += self._collaboration_analysis()
        analysis += self._geographic_analysis()
        analysis += self._trend_analysis()
        analysis += self._personalized_recommendations()
        
        return analysis
    
    def _basic_analysis(self):
        """Análise básica dos dados"""
        text = "### 📊 Visão Geral Inteligente\n\n"
        text += f"- **Total de registros**: {len(self.df)} "
        if len(self.df) < 20:
            text += "🔬 (Base em desenvolvimento - ideal para explorar direções iniciais)\n"
        elif len(self.df) < 50:
            text += "📈 (Base consolidada - permite análises confiáveis)\n"
        else:
            text += "🚀 (Base robusta - excelente para análises complexas)\n"
        
        text += f"- **Colunas disponíveis**: {', '.join(self.df.columns.tolist())}\n"
        
        # Estatísticas por tipo de dado
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        text += f"- **Colunas numéricas**: {len(numeric_cols)} "
        if len(numeric_cols) > 3:
            text += "📐 (Boa diversidade numérica)\n"
        else:
            text += "📝 (Foco em dados textuais)\n"
        
        text += f"- **Colunas de texto**: {len(text_cols)}\n\n"
        
        # Análise de completude
        completeness = self._calculate_completeness()
        text += f"- **Qualidade dos dados**: {completeness['score']}/10 "
        if completeness['score'] >= 8:
            text += "🎯 (Excelente qualidade)\n"
        elif completeness['score'] >= 6:
            text += "✅ (Boa qualidade)\n"
        else:
            text += "⚠️ (Recomendo melhorar a completude)\n"
        
        text += f"- **Dados ausentes**: {completeness['missing_percent']:.1f}%\n\n"
        
        return text
    
    def _calculate_completeness(self):
        """Calcula a qualidade e completude dos dados"""
        total_cells = self.df.size
        missing_cells = self.df.isna().sum().sum()
        missing_percent = (missing_cells / total_cells) * 100
        
        # Pontuação baseada na completude
        score = max(0, 10 - (missing_percent / 10))
        
        return {
            "score": round(score, 1),
            "missing_percent": missing_percent,
            "total_cells": total_cells,
            "missing_cells": missing_cells
        }
    
    def _author_analysis(self):
        """Análise de autores e colaborações - SUPER MELHORADA"""
        text = "### 👥 Análise de Rede de Autores\n\n"
        
        # BUSCA INTELIGENTE POR COLUNAS DE AUTORES
        author_col = None
        possible_author_cols = []
        
        for col in self.df.columns:
            col_lower = col.lower()
            # Adicionar mais palavras-chave e verificar conteúdo
            if any(keyword in col_lower for keyword in ['autor', 'author', 'pesquisador', 'escritor', 'writer', 'nome']):
                possible_author_cols.append(col)
                
                # Verificar se a coluna tem dados que parecem nomes
                sample_data = self.df[col].dropna().head(5)
                if len(sample_data) > 0:
                    # Verificar se contém vírgulas, pontos e vírgulas (indicando múltiplos autores)
                    has_multiple_authors = any(';' in str(val) or ',' in str(val) or ' e ' in str(val).lower() for val in sample_data)
                    if has_multiple_authors or any(len(str(val).split()) >= 2 for val in sample_data):
                        author_col = col
                        break
        
        # Se não encontrou, usar a primeira possível
        if not author_col and possible_author_cols:
            author_col = possible_author_cols[0]
        
        if not author_col:
            return "❌ **Análise de Autores**: Nenhuma coluna de autores identificada. **Sugestão**: Adicione uma coluna 'autores' para análise de colaborações.\n\n"
        
        text += f"**Coluna analisada**: '{author_col}'\n\n"
        
        # PROCESSAMENTO AVANÇADO DOS AUTORES
        all_authors = []
        authors_found = 0
        collaboration_network = []
        
        for authors_str in self.df[author_col].dropna():
            if isinstance(authors_str, str) and authors_str.strip():
                authors_found += 1
                # Múltiplas estratégias de parsing
                authors = re.split(r'[;,]|\be\b|\band\b|&', authors_str)
                author_list = []
                for author in authors:
                    author_clean = author.strip()
                    if (author_clean and len(author_clean) > 2 and 
                        author_clean.lower() not in ['', 'e', 'and', 'et', 'de', 'da', 'do', 'dos', 'das'] and
                        not author_clean.isdigit() and
                        not author_clean.replace('.', '').isdigit()):
                        author_list.append(author_clean)
                        all_authors.append(author_clean)
                
                # Adicionar à rede de colaboração
                if len(author_list) > 1:
                    collaboration_network.append(author_list)
        
        if all_authors:
            author_counts = pd.Series(all_authors).value_counts()
            total_unique_authors = len(author_counts)
            
            text += "**🔍 Perfil de Colaboração**:\n"
            text += f"- **Autores únicos**: {total_unique_authors}\n"
            text += f"- **Trabalhos com autores**: {authors_found}\n"
            
            # Análise de colaboração avançada
            solo_works = 0
            team_works = 0
            
            for authors_str in self.df[author_col].dropna():
                if isinstance(authors_str, str):
                    authors = re.split(r'[;,]|\be\b|\band\b|&', authors_str)
                    num_authors = len([a for a in authors if a.strip() and len(a.strip()) > 2])
                    if num_authors == 1:
                        solo_works += 1
                    elif num_authors > 1:
                        team_works += 1
            
            collaboration_rate = (team_works / authors_found) * 100 if authors_found > 0 else 0
            
            text += f"- **Trabalhos individuais**: {solo_works}\n"
            text += f"- **Trabalhos em equipe**: {team_works}\n"
            text += f"- **Taxa de colaboração**: {collaboration_rate:.1f}%\n\n"
            
            # Análise de rede
            if collaboration_network:
                text += "**🤝 Rede de Colaboração**:\n"
                # Encontrar colaborações frequentes
                from collections import Counter
                collaboration_pairs = []
                for team in collaboration_network:
                    if len(team) >= 2:
                        for i in range(len(team)):
                            for j in range(i+1, len(team)):
                                collaboration_pairs.append(tuple(sorted([team[i], team[j]])))
                
                if collaboration_pairs:
                    pair_counts = Counter(collaboration_pairs)
                    top_pairs = pair_counts.most_common(3)
                    text += "Parcerias mais frequentes:\n"
                    for (author1, author2), count in top_pairs:
                        text += f"  • **{author1}** + **{author2}**: {count} colaboração(ões)\n"
                    text += "\n"
            
            # Autores mais produtivos
            text += "**🏆 Top Autores Mais Produtivos**:\n"
            for i, (author, count) in enumerate(author_counts.head(6).items(), 1):
                productivity = "🔴" if count >= 10 else "🟡" if count >= 5 else "🟢"
                text += f"{i}. {productivity} **{author}**: {count} publicação(ões)\n"
            
            # Recomendações baseadas na análise
            text += "\n**💡 Insights e Sugestões**:\n"
            if collaboration_rate < 30:
                text += "• **Oportunidade**: Considere aumentar colaborações para ampliar impacto\n"
            elif collaboration_rate > 70:
                text += "• **Força**: Excelente rede colaborativa estabelecida\n"
            
            if total_unique_authors < 5:
                text += "• **Diversidade**: Busque colaborar com novos pesquisadores\n"
            
            text += f"\n**Total de nomes extraídos**: {len(all_authors)}\n\n"
            
        else:
            text += f"⚠️ **Autores**: Coluna '{author_col}' encontrada mas não foi possível extrair autores válidos\n\n"
            text += f"**💡 Dica**: Verifique o formato: 'Autor1; Autor2; Autor3' ou 'Autor1, Autor2, Autor3'\n\n"
        
        return text
    
    def _temporal_analysis(self):
        """Análise temporal dos dados - SUPER MELHORADA"""
        text = "### 📈 Análise Temporal e Evolução\n\n"
        
        # Buscar coluna de ano de forma mais abrangente
        year_col = None
        year_data_found = False
        
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['ano', 'year', 'data', 'date', 'publication', 'publicado']):
                year_col = col
                year_data_found = True
                break
        
        if not year_data_found:
            # Tentar encontrar colunas numéricas que possam ser anos
            for col in self.df.select_dtypes(include=[np.number]).columns:
                sample_data = self.df[col].dropna().head(10)
                if len(sample_data) > 0:
                    # Verificar se os valores são anos (entre 1900 e ano atual)
                    current_year = datetime.now().year
                    if all(1900 <= val <= current_year for val in sample_data if pd.notnull(val)):
                        year_col = col
                        year_data_found = True
                        text += f"⚠️ **Atenção**: Usando coluna '{col}' para análise temporal (detecção automática)\n\n"
                        break
        
        if not year_col:
            return "❌ **Análise Temporal**: Nenhuma coluna de anos identificada. **Sugestão**: Adicione uma coluna 'ano' para análise de tendências temporais.\n\n"
            
        try:
            years = pd.to_numeric(self.df[year_col], errors='coerce').dropna()
        except:
            years = pd.Series(dtype=float)
        
        if len(years) > 0:
            min_year = int(years.min())
            max_year = int(years.max())
            year_range = max_year - min_year
            current_year = datetime.now().year
            
            text += f"- **Período analisado**: {min_year} - {max_year} ({year_range} anos)\n"
            
            # Análise de distribuição
            year_counts = years.value_counts().sort_index()
            
            # Ano mais frequente
            if not year_counts.empty:
                most_frequent_year = int(year_counts.index[0])
                most_frequent_count = int(year_counts.iloc[0])
                text += f"- **Ano com mais publicações**: {most_frequent_year} ({most_frequent_count} publicações)\n"
            
            # Análise de tendência
            if len(year_counts) > 3:
                recent_5_years = [y for y in year_counts.index if y >= current_year - 5]
                older_years = [y for y in year_counts.index if y < current_year - 5]
                
                recent_production = sum(year_counts[y] for y in recent_5_years)
                older_production = sum(year_counts[y] for y in older_years)
                
                text += f"- **Produção recente (últimos 5 anos)**: {recent_production} trabalhos\n"
                text += f"- **Produção anterior**: {older_production} trabalhos\n"
                
                # Tendência
                if recent_production > older_production and len(recent_5_years) > 0:
                    growth_rate = ((recent_production / len(recent_5_years)) / (older_production / max(1, len(older_years))) - 1) * 100
                    if growth_rate > 20:
                        text += "- **📈 Tendência**: Crescimento significativo na produção recente\n"
                    elif growth_rate > 0:
                        text += "- **↗️ Tendência**: Leve crescimento na produção\n"
                    else:
                        text += "- **➡️ Tendência**: Produção estável\n"
                else:
                    text += "- **🔍 Tendência**: Produção mais concentrada no passado\n"
            
            # Distribuição por década para períodos longos
            if year_range > 20:
                decades = (years // 10) * 10
                decade_counts = decades.value_counts().sort_index()
                if len(decade_counts) > 1:
                    text += "\n**🕰️ Distribuição por Década**:\n"
                    for decade, count in decade_counts.head(6).items():
                        percentage = (count / len(years)) * 100
                        text += f"- **{int(decade)}s**: {int(count)} publicação(ões) ({percentage:.1f}%)\n"
            
            text += f"\n**Total de registros com anos**: {len(years)}\n\n"
            
            # Sugestões baseadas na análise temporal
            text += "**💡 Insights Temporais**:\n"
            if max_year >= current_year - 2:
                text += "• **Atualidade**: Sua pesquisa inclui trabalhos recentes - excelente!\n"
            elif max_year <= current_year - 5:
                text += "• **Atualização**: Considere incluir trabalhos mais recentes\n"
            
            if year_range < 5:
                text += "• **Ampliação**: Explore períodos temporais mais extensos\n"
                
        else:
            text += f"⚠️ **Anos**: Coluna '{year_col}' encontrada mas sem dados numéricos válidos\n\n"
        
        return text
    
    def _thematic_analysis(self):
        """Análise temática dos dados - SUPER MELHORADA"""
        text = "### 🔍 Análise Temática e Conceitual\n\n"
        
        # Combinar texto de todas as colunas relevantes
        texto_completo = ""
        text_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
        
        for col in text_cols[:5]:  # Analisar até 5 colunas de texto
            col_text = self.df[col].fillna('').astype(str).str.cat(sep=' ')
            if len(col_text) > 100:  # Só adiciona se tiver conteúdo significativo
                texto_completo += " " + col_text
        
        if not texto_completo.strip():
            return "❌ **Análise Temática**: Não há texto suficiente para análise. **Sugestão**: Inclua colunas com títulos, resumos ou palavras-chave.\n\n"
        
        # Extrair temas com análise avançada
        palavras = re.findall(r'\b[a-zà-ú]{4,}\b', texto_completo.lower())
        stop_words = set(PORTUGUESE_STOP_WORDS)
        palavras_filtradas = [p for p in palavras if p not in stop_words and len(p) > 3]
        
        if palavras_filtradas:
            from collections import Counter
            contador = Counter(palavras_filtradas)
            temas_comuns = contador.most_common(15)
            
            text += f"**📊 Estatísticas Textuais**:\n"
            text += f"- **Total de palavras analisadas**: {len(palavras):,}\n"
            text += f"- **Palavras únicas significativas**: {len(contador)}\n"
            text += f"- **Texto analisado**: {len(texto_completo):,} caracteres\n\n"
            
            text += "**🎯 Tópicos Mais Frequentes**:\n"
            for i, (tema, count) in enumerate(temas_comuns[:10], 1):
                importancia = "🔥" if count >= 10 else "⭐" if count >= 5 else "🔹"
                text += f"{i}. {importancia} **{tema}**: {count} ocorrências\n"
            
            # Análise de bigramas (conceitos compostos)
            if len(palavras_filtradas) > 20:
                bigramas = []
                for i in range(len(palavras_filtradas)-1):
                    bigrama = f"{palavras_filtradas[i]} {palavras_filtradas[i+1]}"
                    bigramas.append(bigrama)
                
                contador_bigramas = Counter(bigramas)
                bigramas_comuns = contador_bigramas.most_common(8)
                
                if bigramas_comuns:
                    text += "\n**🔗 Conceitos Relacionados (Bigramas)**:\n"
                    for bigrama, count in bigramas_comuns[:6]:
                        text += f"• **{bigrama}** ({count} vezes)\n"
            
            # Temas emergentes (menos frequentes mas significativos)
            temas_emergentes = [tema for tema, count in temas_comuns[5:12] if count >= 3]
            if temas_emergentes:
                text += f"\n**💡 Temas Emergentes**: {', '.join(temas_emergentes[:4])}\n"
            
            # Análise de diversidade temática
            diversidade = len(contador) / len(palavras) * 1000  # Índice de diversidade
            text += f"\n**🌐 Diversidade Temática**: {diversidade:.1f} "
            if diversidade > 50:
                text += "(Alta diversidade de tópicos)\n"
            elif diversidade > 30:
                text += "(Boa variedade temática)\n"
            else:
                text += "(Foco temático concentrado)\n"
                
        else:
            text += "⚠️ **Temas**: Não foi possível identificar palavras-chave frequentes\n\n"
        
        return text
    
    def _collaboration_analysis(self):
        """Análise de colaborações e redes - SUPER MELHORADA"""
        text = "### 🤝 Análise Avançada de Colaborações\n\n"
        
        author_col = None
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['autor', 'author']):
                author_col = col
                break
        
        if author_col:
            coautorias = 0
            total_trabalhos = len(self.df[author_col].dropna())
            autores_por_trabalho = []
            
            for authors_str in self.df[author_col].dropna():
                if isinstance(authors_str, str):
                    authors = re.split(r'[;,]|\be\b|\band\b|&', authors_str)
                    num_autores = len([a for a in authors if a.strip() and len(a.strip()) > 2])
                    autores_por_trabalho.append(num_autores)
                    
                    if num_autores > 1:
                        coautorias += 1
            
            if total_trabalhos > 0:
                taxa_colaboracao = (coautorias/total_trabalhos)*100
                media_autores = np.mean(autores_por_trabalho) if autores_por_trabalho else 0
                
                text += f"**📈 Métricas de Colaboração**:\n"
                text += f"- **Trabalhos em colaboração**: {coautorias}\n"
                text += f"- **Taxa de colaboração**: {taxa_colaboracao:.1f}%\n"
                text += f"- **Média de autores por trabalho**: {media_autores:.1f}\n"
                text += f"- **Total de trabalhos analisados**: {total_trabalhos}\n\n"
                
                # Análise de perfil colaborativo
                text += "**🔍 Perfil de Colaboração**:\n"
                if taxa_colaboracao > 60:
                    text += "- **🏆 Colaborador Intenso**: Alta taxa de trabalhos em equipe\n"
                    text += "- **💡 Força**: Excelente rede de colaboração estabelecida\n"
                elif taxa_colaboracao > 30:
                    text += "- **🤝 Colaborador Moderado**: Bom equilíbrio entre trabalho individual e em grupo\n"
                    text += "- **🚀 Oportunidade**: Pode expandir ainda mais as colaborações\n"
                else:
                    text += "- **🔬 Pesquisador Independente**: Predominância de trabalho individual\n"
                    text += "- **💡 Sugestão**: Considere oportunidades de colaboração\n"
                
                # Análise do tamanho das equipes
                team_sizes = Counter(autores_por_trabalho)
                text += f"\n**👥 Distribuição de Tamanho de Equipes**:\n"
                for size, count in team_sizes.most_common():
                    text += f"- **{size} autor(es)**: {count} trabalho(s)\n"
                    
            else:
                text += "⚠️ **Colaboração**: Sem dados de autores para análise\n"
        else:
            text += "❌ **Colaboração**: Nenhuma coluna de autores identificada\n"
        
        text += "\n"
        return text
    
    def _geographic_analysis(self):
        """Análise geográfica dos dados - SUPER MELHORADA"""
        text = "### 🌎 Análise Geográfica e de Distribuição\n\n"
        
        # Buscar coluna de país de forma mais abrangente
        country_col = None
        country_data_found = False
        
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['país', 'pais', 'country', 'local', 'location', 'nacionalidade', 'região', 'cidade', 'city']):
                country_col = col
                country_data_found = True
                break
        
        if not country_data_found:
            # Tentar encontrar dados de país em colunas de texto
            for col in self.df.select_dtypes(include=['object']).columns:
                sample_data = self.df[col].dropna().head(10)
                if len(sample_data) > 0:
                    # Verificar se contém nomes de países conhecidos
                    common_countries = ['brasil', 'brazil', 'eua', 'usa', 'portugal', 'espanha', 'frança', 'argentina', 'méxico', 'colômbia']
                    has_countries = any(any(country in str(val).lower() for country in common_countries) for val in sample_data)
                    if has_countries:
                        country_col = col
                        country_data_found = True
                        text += f"⚠️ **Atenção**: Usando coluna '{col}' para análise geográfica (detecção automática)\n\n"
                        break
        
        if not country_col:
            return "❌ **Análise Geográfica**: Nenhuma coluna de países identificada. **Sugestão**: Adicione uma coluna 'país' para análise de distribuição geográfica.\n\n"
            
        countries = self.df[country_col].dropna()
        
        if len(countries) > 0:
            country_counts = countries.value_counts()
            total_countries = len(countries)
            unique_countries = len(country_counts)
            
            text += f"**🌐 Alcance Geográfico**:\n"
            text += f"- **Regiões/países únicos**: {unique_countries}\n"
            text += f"- **Total de registros com localização**: {total_countries}\n"
            text += f"- **Cobertura geográfica**: {(total_countries/len(self.df))*100:.1f}% dos registros\n\n"
            
            text += "**🗺️ Distribuição Geográfica**:\n"
            for country, count in country_counts.head(8).items():
                percentual = (count / total_countries) * 100
                emoji = "🇧🇷" if 'brasil' in str(country).lower() or 'brazil' in str(country).lower() else "🌍"
                text += f"{emoji} **{country}**: {count} ({percentual:.1f}%)\n"
            
            # Análise de diversidade geográfica
            diversity_index = (unique_countries / total_countries) * 100
            
            text += f"\n**📊 Diversidade Geográfica**: {diversity_index:.1f}%\n"
            
            # Classificação do escopo geográfico
            if unique_countries == 1:
                text += "- **🎯 Foco**: Pesquisa concentrada em uma única região\n"
                text += "- **💡 Sugestão**: Considere expandir para estudos comparativos internacionais\n"
            elif unique_countries <= 3:
                text += "- **📍 Escopo Regional**: Pesquisa com foco em poucas regiões\n"
                text += "- **🚀 Oportunidade**: Boa base para expansão geográfica\n"
            elif unique_countries <= 8:
                text += "- **🌎 Escopo Multinacional**: Pesquisa com boa diversidade geográfica\n"
                text += "- **🏆 Força**: Abrangência internacional consolidada\n"
            else:
                text += "- **🚀 Escopo Global**: Pesquisa com excelente abrangência internacional\n"
                text += "- **🎯 Destaque**: Diversidade geográfica como diferencial\n"
            
        else:
            text += f"⚠️ **Países**: Coluna '{country_col}' encontrada mas sem dados válidos\n\n"
        
        return text
    
    def _trend_analysis(self):
        """Análise de tendências e insights - SUGESTÕES INTELIGENTES REAIS"""
        text = "### 💡 Análise Estratégica e Recomendações\n\n"
        
        insights = []
        sugestoes_inteligentes = []
        
        # ANÁLISE INTELIGENTE BASEADA NOS DADOS REAIS
        total_registros = len(self.df)
        
        # 1. Análise de completude estratégica
        colunas_essenciais = ['autor', 'ano', 'título', 'resumo', 'palavras_chave']
        colunas_presentes = [col for col in colunas_essenciais 
                            if any(col in col_name.lower() for col_name in self.df.columns)]
        completude = len(colunas_presentes) / len(colunas_essenciais) * 100
        
        if completude < 50:
            sugestoes_inteligentes.append("📋 **Estrutura de Dados**: Adicione colunas básicas (autor, ano, título) para análises mais ricas")
        elif completude < 80:
            sugestoes_inteligentes.append("📊 **Estrutura Boa**: Considere adicionar resumos e palavras-chave para análises avançadas")
        else:
            sugestoes_inteligentes.append("🎯 **Estrutura Excelente**: Todos os elementos essenciais estão presentes - ótimo para análises complexas")
        
        # 2. Análise temporal estratégica
        year_col = next((col for col in self.df.columns if 'ano' in col.lower() or 'year' in col.lower()), None)
        if year_col:
            try:
                anos = pd.to_numeric(self.df[year_col], errors='coerce').dropna()
                if len(anos) > 0:
                    range_anos = int(anos.max()) - int(anos.min())
                    if range_anos < 3:
                        sugestoes_inteligentes.append("⏳ **Perspectiva Temporal**: Dados concentrados em poucos anos - busque maior variedade temporal para análise de tendências")
                    elif range_anos > 10:
                        sugestoes_inteligentes.append("📈 **Força Temporal**: Período extenso permite análise robusta de evolução - explore tendências de longo prazo")
                    elif range_anos > 5:
                        sugestoes_inteligentes.append("🕰️ **Boa Janela Temporal**: Período adequado para análise de médio prazo")
            except:
                pass
        
        # 3. Análise de rede de autores estratégica
        author_col = next((col for col in self.df.columns if any(kw in col.lower() for kw in ['autor', 'author'])), None)
        if author_col:
            autores_unicos = set()
            colaboracoes = 0
            
            for authors_str in self.df[author_col].dropna():
                if isinstance(authors_str, str):
                    authors = re.split(r'[;,]', authors_str)
                    autores_validos = [a.strip() for a in authors if a.strip() and len(a.strip()) > 2]
                    autores_unicos.update(autores_validos)
                    
                    if len(autores_validos) > 1:
                        colaboracoes += 1
            
            if len(autores_unicos) < 5:
                sugestoes_inteligentes.append("👥 **Rede de Pesquisa**: Pouca diversidade de pesquisadores - considere expandir colaborações")
            elif len(autores_unicos) > 20:
                sugestoes_inteligentes.append("🤝 **Rede Colaborativa Forte**: Excelente diversidade de autores - explore análises de rede")
            elif len(autores_unicos) > 10:
                sugestoes_inteligentes.append("🔗 **Boa Rede**: Diversidade adequada de colaboradores")
            
            if colaboracoes > 0:
                taxa_colab = (colaboracoes / len(self.df[author_col].dropna())) * 100
                if taxa_colab < 30:
                    sugestoes_inteligentes.append("💡 **Oportunidade de Colaboração**: Baixa taxa de coautoria - busque parcerias para ampliar impacto")
        
        # 4. Análise de temas e lacunas
        texto_completo = ""
        for col in self.df.select_dtypes(include=['object']).columns[:3]:
            texto_completo += " " + self.df[col].fillna('').astype(str).str.cat(sep=' ')
        
        if len(texto_completo) > 1000:
            palavras = re.findall(r'\b[a-zà-ú]{5,}\b', texto_completo.lower())
            from collections import Counter
            contagem = Counter(palavras)
            temas_comuns = [pal for pal, cnt in contagem.most_common(10) 
                           if pal not in PORTUGUESE_STOP_WORDS and cnt > 2]
            
            if temas_comuns:
                sugestoes_inteligentes.append(f"🔍 **Foco Temático Principal**: {', '.join(temas_comuns[:3])}")
                
                # Identificar possíveis lacunas
                temas_emergentes = ['blockchain', 'sustentabilidade', 'inteligência artificial', 'transformação digital', 'inovação aberta']
                lacunas_identificadas = [tema for tema in temas_emergentes if tema not in ' '.join(temas_comuns).lower()]
                if lacunas_identificadas:
                    sugestoes_inteligentes.append(f"🌱 **Temas Emergentes para Explorar**: {', '.join(lacunas_identificadas[:2])}")
        
        # 5. Estratégias baseadas no tamanho da base
        if total_registros < 15:
            sugestoes_inteligentes.extend([
                "📥 **Expansão de Dados**: Mínimo 20-30 registros para análises estatísticas confiáveis",
                "🔎 **Busca Estratégica**: Use a busca integrada para encontrar trabalhos relacionados",
                "🎯 **Foco Qualitativo**: Ideal para estudos de caso e análises qualitativas profundas"
            ])
        elif total_registros < 50:
            sugestoes_inteligentes.extend([
                "📊 **Análises Quantitativas Básicas**: Explore correlações e distribuições",
                "🗺️ **Organização Conceitual**: Use o mapa mental para estruturar relações entre conceits",
                "🤖 **Análise de IA**: Consulte o assistente para insights específicos"
            ])
        else:
            sugestoes_inteligentes.extend([
                "📈 **Análises Avançadas**: Dados suficientes para machine learning e redes complexas",
                "🌐 **Exploração de Redes**: Identifique clusters e padrões de colaboração",
                "📋 **Metanálise**: Considere análise comparativa entre diferentes períodos/regiões"
            ])
        
        # 6. Análise de qualidade dos dados
        missing_data = self.df.isna().sum().sum()
        missing_percent = (missing_data / self.df.size) * 100
        
        if missing_percent > 20:
            sugestoes_inteligentes.append("🔄 **Qualidade de Dados**: Alta taxa de dados ausentes - considere completar informações faltantes")
        elif missing_percent > 10:
            sugestoes_inteligentes.append("✅ **Qualidade Adequada**: Boa completude dos dados")
        else:
            sugestoes_inteligentes.append("🏆 **Excelente Qualidade**: Dados muito completos - ideal para análises complexas")
        
        # Formatar resposta
        text += "**🎯 Recomendações Estratégicas Baseadas na Sua Pesquisa:**\n\n"
        for i, sugestao in enumerate(sugestoes_inteligentes, 1):
            text += f"{i}. {sugestao}\n"
        
        text += f"\n**📋 Resumo da Base de Dados:**\n"
        text += f"• **Registros**: {total_registros} "
        if total_registros < 20:
            text += "(Base em desenvolvimento)\n"
        elif total_registros < 50:
            text += "(Base consolidada)\n"
        else:
            text += "(Base robusta)\n"
            
        text += f"• **Completude**: {completude:.1f}%\n"
        if author_col:
            text += f"• **Coluna autores**: '{author_col}'\n"
        if year_col:
            text += f"• **Coluna anos**: '{year_col}'\n"
        text += f"• **Dados ausentes**: {missing_percent:.1f}%\n"
        
        return text

    def _personalized_recommendations(self):
        """Recomendações personalizadas baseadas no perfil do usuário"""
        text = "### 🎯 Plano de Ação Personalizado\n\n"
        
        recommendations = []
        
        # Análise do perfil de pesquisa
        if len(self.df) < 20:
            recommendations.extend([
                "**Fase Inicial**: Comece expandindo sua base para 30+ registros para análises mais confiáveis",
                "**Busca Estratégica**: Use a aba de recomendações para encontrar trabalhos fundamentais na sua área",
                "**Mapeamento Conceitual**: Utilize o mapa mental para organizar os conceitos centrais da sua pesquisa"
            ])
        
        # Análise de colaboração
        author_col = next((col for col in self.df.columns if any(kw in col.lower() for kw in ['autor', 'author'])), None)
        if author_col:
            autores_unicos = set()
            for authors_str in self.df[author_col].dropna():
                if isinstance(authors_str, str):
                    authors = re.split(r'[;,]', authors_str)
                    for author in authors:
                        if author.strip():
                            autores_unicos.add(author.strip())
            
            if len(autores_unicos) < 5:
                recommendations.append("**Rede de Colaboração**: Busque colaborar com 2-3 novos pesquisadores na sua próxima pesquisa")
        
        # Análise temporal
        year_col = next((col for col in self.df.columns if 'ano' in col.lower() or 'year' in col.lower()), None)
        if year_col:
            try:
                anos = pd.to_numeric(self.df[year_col], errors='coerce').dropna()
                if len(anos) > 0:
                    max_year = anos.max()
                    current_year = datetime.now().year
                    if max_year < current_year - 3:
                        recommendations.append("**Atualização Temporal**: Inclua trabalhos dos últimos 2-3 anos para manter a relevância")
            except:
                pass
        
        # Recomendações gerais baseadas no tamanho
        if len(self.df) >= 30:
            recommendations.extend([
                "**Análise Avançada**: Use a análise de IA para explorar correlações e padrões complexos",
                "**Visualização**: Crie gráficos de rede para visualizar colaborações entre autores",
                "**Metodologia**: Considere aplicar análise de conteúdo aos resumos dos trabalhos"
            ])
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                text += f"{i}. {rec}\n"
        else:
            text += "✅ Sua pesquisa está bem estruturada! Continue explorando as funcionalidades da plataforma para insights adicionais.\n"
        
        text += "\n**🚀 Próximos Passos Imediatos**:\n"
        text += "• Use o assistente de IA para perguntas específicas sobre sua planilha\n"
        text += "• Explore a aba de recomendações para trabalhos relacionados\n"
        text += "• Organize seus insights no mapa mental interativo\n"
        text += "• Consulte a análise gráfica para visualizações dos dados\n"
        
        return text

# -------------------------
# SISTEMA DE IA SUPER INTELIGENTE E PERSONALIZADO
# -------------------------
def get_ai_assistant_response(question, context, user_data=None):
    """Assistente de IA SUPER INTELIGENTE - Respostas precisas e personalizadas"""
    
    question_lower = question.lower().strip()
    df = context.df
    
    # Análise do contexto da pergunta
    question_type = _classify_question(question_lower)
    user_profile = context.user_profile if hasattr(context, 'user_profile') else {}
    
    # Resposta baseada no tipo de pergunta e perfil do usuário
    if question_type == "authors":
        return _analyze_authors_advanced(df, question_lower, user_profile)
    elif question_type == "geography":
        return _analyze_geography_advanced(df, question_lower, user_profile)
    elif question_type == "temporal":
        return _analyze_temporal_advanced(df, question_lower, user_profile)
    elif question_type == "thematic":
        return _analyze_themes_advanced(df, question_lower, user_profile)
    elif question_type == "collaboration":
        return _analyze_collaborations_advanced(df, question_lower, user_profile)
    elif question_type == "statistics":
        return _analyze_statistics_advanced(df, question_lower, user_profile)
    elif question_type == "trends":
        return _analyze_trends_advanced(df, question_lower, user_profile)
    elif question_type == "suggestions":
        return _provide_personalized_suggestions(df, question_lower, user_profile)
    elif question_type == "quality":
        return _analyze_data_quality_advanced(df, question_lower, user_profile)
    else:
        return _comprehensive_analysis_response(df, question, user_profile)

def _classify_question(question):
    """Classifica o tipo de pergunta para resposta mais precisa"""
    question = question.lower()
    
    if any(word in question for word in ['autor', 'autores', 'pesquisador', 'quem escreveu', 'quem publicou']):
        return "authors"
    elif any(word in question for word in ['país', 'países', 'geográfica', 'geografia', 'local', 'região', 'onde']):
        return "geography"
    elif any(word in question for word in ['ano', 'anos', 'temporal', 'evolução', 'cronologia', 'quando', 'período']):
        return "temporal"
    elif any(word in question for word in ['tema', 'temas', 'conceito', 'palavras', 'frequentes', 'termos', 'assuntos']):
        return "thematic"
    elif any(word in question for word in ['colaboração', 'colaborações', 'coautoria', 'parceria', 'trabalho conjunto']):
        return "collaboration"
    elif any(word in question for word in ['estatística', 'estatísticas', 'números', 'quantidade', 'total', 'quantos']):
        return "statistics"
    elif any(word in question for word in ['tendência', 'tendências', 'futuro', 'emergente', 'novo', 'recente']):
        return "trends"
    elif any(word in question for word in ['sugestão', 'sugestões', 'recomendação', 'recomendações', 'o que fazer', 'próximo passo']):
        return "suggestions"
    elif any(word in question for word in ['qualidade', 'completude', 'dados faltantes', 'melhorar']):
        return "quality"
    else:
        return "comprehensive"

def _analyze_authors_advanced(df, question, user_profile):
    """Análise avançada de autores com recomendações personalizadas"""
    author_col = next((col for col in df.columns if any(kw in col.lower() for kw in ['autor', 'author'])), None)
    
    if not author_col:
        return "**❌ Análise de Autores**: Não encontrei coluna de autores na sua planilha.\n\n**💡 Sugestão Prática**: Adicione uma coluna 'autores' no formato 'Nome1; Nome2; Nome3' para habilitar análises de colaboração."

    autores_contagem = {}
    colaboracoes = 0
    autores_por_trabalho = []
    network_connections = []
    
    for authors_str in df[author_col].dropna():
        if isinstance(authors_str, str):
            autores = re.split(r'[;,]|\be\b|\band\b|&', authors_str)
            autores_lista = [a.strip() for a in autores if a.strip() and len(a.strip()) > 2]
            
            autores_por_trabalho.append(len(autores_lista))
            
            if len(autores_lista) > 1:
                colaboracoes += 1
                # Registrar conexões para análise de rede
                for i in range(len(autores_lista)):
                    for j in range(i+1, len(autores_lista)):
                        network_connections.append((autores_lista[i], autores_lista[j]))
            
            for autor in autores_lista:
                autores_contagem[autor] = autores_contagem.get(autor, 0) + 1
    
    if not autores_contagem:
        return "**⚠️ Autores**: Coluna encontrada mas não consegui extrair nomes válidos.\n\n**🔧 Correção Rápida**: Verifique se os nomes estão no formato 'Sobrenome, Nome' ou 'Nome Sobrenome' separados por ponto e vírgula."

    autores_ordenados = sorted(autores_contagem.items(), key=lambda x: x[1], reverse=True)
    total_autores = len(autores_contagem)
    total_trabalhos = len(df[author_col].dropna())
    media_autores = np.mean(autores_por_trabalho) if autores_por_trabalho else 0
    taxa_colaboracao = (colaboracoes / total_trabalhos) * 100 if total_trabalhos > 0 else 0

    resposta = "**👥 ANÁLISE AVANÇADA DE AUTORES**\n\n"
    
    # Métricas principais
    resposta += f"**📊 Métricas Principais**:\n"
    resposta += f"• **Autores únicos**: {total_autores}\n"
    resposta += f"• **Trabalhos analisados**: {total_trabalhos}\n"
    resposta += f"• **Média de autores/trabalho**: {media_autores:.1f}\n"
    resposta += f"• **Taxa de colaboração**: {taxa_colaboracao:.1f}%\n\n"

    # Top autores
    resposta += "**🏆 Top 8 Autores Mais Produtivos**:\n"
    for i, (autor, count) in enumerate(autores_ordenados[:8], 1):
        produtividade = "🔥" if count >= 10 else "⭐" if count >= 5 else "🔹"
        resposta += f"{i}. {produtividade} **{autor}** - {count} publicação(ões)\n"
    
    # Análise de rede se houver colaborações
    if network_connections:
        from collections import Counter
        pair_counts = Counter(network_connections)
        top_pairs = pair_counts.most_common(3)
        
        resposta += "\n**🤝 Parcerias Mais Fortes**:\n"
        for (autor1, autor2), count in top_pairs:
            resposta += f"• **{autor1}** + **{autor2}**: {count} colaboração(ões)\n"

    # Análise personalizada baseada no perfil
    resposta += "\n**💡 ANÁLISE ESTRATÉGICA**:\n"
    
    if taxa_colaboracao < 25:
        resposta += "• **🎯 Oportunidade**: Baixa colaboração - considere parcerias para ampliar impacto\n"
        resposta += "• **🚀 Ação Sugerida**: Busque 2-3 colaboradores para próximos projetos\n"
    elif taxa_colaboracao > 60:
        resposta += "• **🏆 Força**: Alta colaboração - excelente rede estabelecida\n"
        resposta += "• **📈 Próximo Passo**: Aproveite para análises de rede avançadas\n"
    else:
        resposta += "• **✅ Situação**: Colaboração equilibrada - mantenha a diversidade\n"

    if media_autores < 1.5:
        resposta += "• **🔍 Padrão**: Predominância de trabalhos individuais\n"
    elif media_autores > 3:
        resposta += "• **👥 Característica**: Equipes grandes - ideal para projetos complexos\n"

    # Recomendações específicas baseadas na pergunta
    if 'produt' in question:
        resposta += f"\n**📈 Autor Mais Produtivo**: {autores_ordenados[0][0]} com {autores_ordenados[0][1]} publicações\n"
    
    if 'colabor' in question and network_connections:
        resposta += f"\n**🔗 Rede de Colaboração**: {len(network_connections)} conexões identificadas\n"

    return resposta

def _analyze_geography_advanced(df, question, user_profile):
    """Análise geográfica avançada com insights estratégicos"""
    country_col = next((col for col in df.columns if any(kw in col.lower() for kw in ['país', 'pais', 'country', 'local'])), None)
    
    if not country_col:
        return "**❌ Análise Geográfica**: Não encontrei coluna de países/regiões.\n\n**💡 Sugestão**: Adicione uma coluna 'país' ou 'localização' para análises de distribuição geográfica."

    paises = df[country_col].dropna()
    if paises.empty:
        return "**⚠️ Geografia**: Coluna encontrada mas sem dados válidos.\n\n**🔧 Ação**: Verifique se a coluna contém nomes de países ou regiões."

    contagem_paises = paises.value_counts()
    total_paises = len(contagem_paises)
    total_registros = len(paises)
    cobertura = (total_registros / len(df)) * 100

    resposta = "**🌎 ANÁLISE GEOGRÁFICA AVANÇADA**\n\n"
    
    resposta += f"**📈 Métricas de Distribuição**:\n"
    resposta += f"• **Países/regiões únicos**: {total_paises}\n"
    resposta += f"• **Registros com localização**: {total_registros}\n"
    resposta += f"• **Cobertura geográfica**: {cobertura:.1f}% dos dados\n\n"

    resposta += "**🗺️ Distribuição por Região**:\n"
    for pais, count in contagem_paises.head(8).items():
        percentual = (count / total_registros) * 100
        emoji = _get_country_emoji(pais)
        resposta += f"{emoji} **{pais}**: {count} registro(s) ({percentual:.1f}%)\n"

    # Análise de diversidade
    diversidade = (total_paises / total_registros) * 100
    
    resposta += f"\n**🌐 Diversidade Geográfica**: {diversidade:.1f}%\n"

    # Análise estratégica
    resposta += "\n**💡 ANÁLISE ESTRATÉGICA**:\n"
    
    if total_paises == 1:
        resposta += "• **🎯 Foco Geográfico**: Pesquisa concentrada em uma única região\n"
        resposta += "• **🚀 Oportunidade**: Considere estudos comparativos internacionais\n"
    elif total_paises <= 3:
        resposta += "• **📍 Escopo Regional**: Foco em poucas regiões específicas\n"
        resposta += "• **📈 Sugestão**: Expanda para 2-3 novas regiões geográficas\n"
    elif total_paises <= 8:
        resposta += "• **🌎 Escopo Multinacional**: Boa diversidade geográfica\n"
        resposta += "• **🏆 Força**: Abrangência como diferencial competitivo\n"
    else:
        resposta += "• **🚀 Escopo Global**: Excelente cobertura internacional\n"
        resposta += "• **🎯 Destaque**: Diversidade geográfica como vantagem estratégica\n"

    if cobertura < 70:
        resposta += f"• **⚠️ Atenção**: {100-cobertura:.1f}% dos registros sem localização\n"

    # Insights específicos baseados na pergunta
    if 'brasil' in question or 'brazil' in question:
        brasil_count = contagem_paises.get('Brasil', contagem_paises.get('Brazil', 0))
        if brasil_count > 0:
            resposta += f"\n**🇧🇷 Foco no Brasil**: {brasil_count} trabalhos ({brasil_count/total_registros*100:.1f}% do total)"

    return resposta

def _get_country_emoji(country_name):
    """Retorna emoji baseado no nome do país"""
    country_str = str(country_name).lower()
    emoji_map = {
        'brasil': '🇧🇷', 'brazil': '🇧🇷',
        'eua': '🇺🇸', 'usa': '🇺🇸', 'estados unidos': '🇺🇸',
        'portugal': '🇵🇹', 'espanha': '🇪🇸', 'spanha': '🇪🇸',
        'frança': '🇫🇷', 'franca': '🇫🇷', 'france': '🇫🇷',
        'alemanha': '🇩🇪', 'germany': '🇩🇪',
        'itália': '🇮🇹', 'italia': '🇮🇹', 'italy': '🇮🇹',
        'china': '🇨🇳', 'japão': '🇯🇵', 'japao': '🇯🇵', 'japan': '🇯🇵'
    }
    
    for key, emoji in emoji_map.items():
        if key in country_str:
            return emoji
    
    return '🌍'

def _analyze_temporal_advanced(df, question, user_profile):
    """Análise temporal avançada com tendências"""
    year_col = next((col for col in df.columns if any(kw in col.lower() for kw in ['ano', 'year', 'data'])), None)
    
    if not year_col:
        return "**❌ Análise Temporal**: Não encontrei coluna temporal.\n\n**💡 Sugestão**: Adicione uma coluna 'ano' para análise de evolução e tendências."

    try:
        anos = pd.to_numeric(df[year_col], errors='coerce').dropna()
        if anos.empty:
            return "**⚠️ Anos**: Coluna encontrada mas sem valores numéricos válidos.\n\n**🔧 Correção**: Certifique-se de que a coluna contém apenas anos (ex: 2020, 2021, 2022)."
        
        min_ano = int(anos.min())
        max_ano = int(anos.max())
        periodo = max_ano - min_ano
        anos_unicos = len(anos.unique())
        current_year = datetime.now().year
        
        resposta = "**📈 ANÁLISE TEMPORAL AVANÇADA**\n\n"
        
        resposta += f"**📊 Período Analisado**:\n"
        resposta += f"• **Período**: {min_ano} - {max_ano} ({periodo} anos)\n"
        resposta += f"• **Anos com registros**: {anos_unicos}\n"
        resposta += f"• **Total de registros**: {len(anos)}\n\n"

        contagem_por_ano = anos.value_counts().sort_index()
        
        # Análise de tendência
        if len(contagem_por_ano) > 3:
            recent_threshold = current_year - 3
            anos_recentes = contagem_por_ano[contagem_por_ano.index >= recent_threshold]
            anos_anteriores = contagem_por_ano[contagem_por_ano.index < recent_threshold]
            
            if len(anos_recentes) > 0 and len(anos_anteriores) > 0:
                media_recente = anos_recentes.mean()
                media_anterior = anos_anteriores.mean()
                
                if media_anterior > 0:
                    crescimento = ((media_recente - media_anterior) / media_anterior) * 100
                    
                    resposta += "**📈 Análise de Tendência**:\n"
                    resposta += f"• **Média recente (últimos 3 anos)**: {media_recente:.1f} trabalhos/ano\n"
                    resposta += f"• **Média anterior**: {media_anterior:.1f} trabalhos/ano\n"
                    resposta += f"• **Taxa de crescimento**: {crescimento:+.1f}%\n\n"
                    
                    if crescimento > 20:
                        resposta += "• **🚀 Tendência**: Crescimento significativo na produção recente\n"
                    elif crescimento > 5:
                        resposta += "• **↗️ Tendência**: Leve crescimento na produção\n"
                    elif crescimento > -5:
                        resposta += "• **➡️ Tendência**: Produção estável\n"
                    else:
                        resposta += "• **📉 Tendência**: Redução na produção recente\n"

        # Ano mais produtivo
        if not contagem_por_ano.empty:
            ano_mais_produtivo = contagem_por_ano.idxmax()
            producao_pico = contagem_por_ano.max()
            resposta += f"**🏆 Ano Mais Produtivo**: {int(ano_mais_produtivo)} ({producao_pico} publicações)\n\n"

        # Análise estratégica
        resposta += "**💡 INSIGHTS TEMPORAIS**:\n"
        
        if max_ano >= current_year - 1:
            resposta += "• **✅ Atualidade**: Inclui trabalhos muito recentes - excelente!\n"
        elif max_ano <= current_year - 5:
            resposta += "• **⚠️ Atenção**: Últimos trabalhos têm mais de 5 anos - busque atualizar\n"
        
        if periodo < 3:
            resposta += "• **🔍 Janela Temporal**: Período curto - ideal para análise pontual\n"
        elif periodo > 10:
            resposta += "• **📊 Força Temporal**: Período extenso - permite análise de longo prazo\n"

        return resposta
        
    except Exception as e:
        return f"**❌ Erro na análise temporal**: {str(e)}\n\n**💡 Dica**: Verifique se a coluna de anos contém apenas números (ex: 2023, 2024)."

def _analyze_themes_advanced(df, question, user_profile):
    """Análise temática avançada"""
    texto_analise = ""
    colunas_texto = [col for col in df.columns if df[col].dtype == 'object']
    
    for col in colunas_texto[:5]:
        texto_analise += " " + df[col].fillna('').astype(str).str.cat(sep=' ')
    
    if len(texto_analise.strip()) < 100:
        return "**❌ Análise Temática**: Não há texto suficiente para análise temática.\n\n**💡 Sugestão**: Inclua colunas com títulos, resumos ou descrições."

    palavras = re.findall(r'\b[a-zà-ú]{4,}\b', texto_analise.lower())
    palavras_filtradas = [p for p in palavras if p not in PORTUGUESE_STOP_WORDS and len(p) > 3]
    
    if not palavras_filtradas:
        return "**🔍 Temas**: Texto analisado mas não identifiquei palavras-chave significativas."

    from collections import Counter
    contador = Counter(palavras_filtradas)
    temas_comuns = contador.most_common(15)
    
    resposta = "**🔤 ANÁLISE TEMÁTICA AVANÇADA**\n\n"
    resposta += f"**📊 Estatísticas Textuais**:\n"
    resposta += f"• **Total de palavras únicas**: {len(contador)}\n"
    resposta += f"• **Texto analisado**: {len(texto_analise):,} caracteres\n\n"
    
    resposta += "**🎯 Palavras-chave Mais Frequentes:**\n"
    for i, (tema, count) in enumerate(temas_comuns[:12], 1):
        importancia = "🔥" if count >= 10 else "⭐" if count >= 5 else "🔹"
        resposta += f"{i}. {importancia} **{tema}** - {count} ocorrências\n"
    
    # Análise de conceitos relacionados
    if len(palavras_filtradas) > 10:
        bigramas = []
        for i in range(len(palavras_filtradas)-1):
            bigrama = f"{palavras_filtradas[i]} {palavras_filtradas[i+1]}"
            bigramas.append(bigrama)
        
        contador_bigramas = Counter(bigramas)
        bigramas_comuns = contador_bigramas.most_common(8)
        
        if bigramas_comuns:
            resposta += "\n**🔗 Conceitos Relacionados (Bigramas):**\n"
            for bigrama, count in bigramas_comuns[:6]:
                resposta += f"• **{bigrama}** ({count})\n"
    
    # Temas emergentes
    temas_emergentes = [tema for tema, count in temas_comuns[8:15] if count >= 2]
    if temas_emergentes:
        resposta += f"\n**💡 Temas Emergentes**: {', '.join(temas_emergentes[:5])}"
    
    # Análise estratégica
    resposta += "\n\n**💡 INSIGHTS TEMÁTICOS**:\n"
    if len(contador) > 50:
        resposta += "• **🌐 Diversidade**: Amplo espectro temático - boa variedade de tópicos\n"
    elif len(contador) > 20:
        resposta += "• **🎯 Foco**: Temas bem definidos - pesquisa concentrada\n"
    else:
        resposta += "• **🔍 Especialização**: Foco muito específico - considere ampliar horizontes\n"

    return resposta

def _analyze_collaborations_advanced(df, question, user_profile):
    """Análise de colaborações avançada"""
    author_col = next((col for col in df.columns if any(kw in col.lower() for kw in ['autor', 'author'])), None)
    
    if not author_col:
        return "**❌ Colaborações**: Não encontrei dados de autores para análise.\n\n**💡 Sugestão**: Adicione uma coluna 'autores' para habilitar análises de colaboração."

    colaboracoes = 0
    total_trabalhos = 0
    autores_por_trabalho = []
    network_data = []
    
    for authors_str in df[author_col].dropna():
        if isinstance(authors_str, str):
            total_trabalhos += 1
            autores = re.split(r'[;,]', authors_str)
            num_autores = len([a for a in autores if a.strip()])
            autores_por_trabalho.append(num_autores)
            
            if num_autores > 1:
                colaboracoes += 1
                # Coletar dados para análise de rede
                autores_validos = [a.strip() for a in autores if a.strip()]
                network_data.append(autores_validos)
    
    if total_trabalhos == 0:
        return "**⚠️ Colaborações**: Sem dados válidos para análise."

    taxa_colaboracao = (colaboracoes / total_trabalhos) * 100
    media_autores = np.mean(autores_por_trabalho)
    
    resposta = "**🤝 ANÁLISE DE COLABORAÇÕES AVANÇADA**\n\n"
    resposta += f"**📊 Métricas de Colaboração**:\n"
    resposta += f"• **Total de trabalhos analisados**: {total_trabalhos}\n"
    resposta += f"• **Trabalhos em colaboração**: {colaboracoes}\n"
    resposta += f"• **Taxa de colaboração**: {taxa_colaboracao:.1f}%\n"
    resposta += f"• **Média de autores por trabalho**: {media_autores:.1f}\n\n"
    
    # Análise de perfil colaborativo
    resposta += "**🔍 Perfil de Colaboração**:\n"
    if taxa_colaboracao > 60:
        resposta += "• **🏆 Colaborador Intenso**: Alta taxa de trabalhos em equipe\n"
        resposta += "• **💡 Força**: Excelente rede de colaboração estabelecida\n"
        resposta += "• **🚀 Próximo Passo**: Explore análises de rede complexas\n"
    elif taxa_colaboracao > 30:
        resposta += "• **🤝 Colaborador Moderado**: Bom equilíbrio entre trabalho individual e em grupo\n"
        resposta += "• **📈 Oportunidade**: Pode expandir ainda mais as colaborações\n"
        resposta += "• **💡 Sugestão**: Identifique pesquisadores complementares\n"
    else:
        resposta += "• **🔬 Pesquisador Independente**: Predominância de trabalho individual\n"
        resposta += "• **🎯 Oportunidade**: Considere oportunidades de colaboração\n"
        resposta += "• **🚀 Ação**: Busque 1-2 colaborações para próximos projetos\n"

    # Análise de tamanho de equipes
    team_sizes = Counter(autores_por_trabalho)
    resposta += f"\n**👥 Distribuição de Tamanho de Equipes**:\n"
    for size, count in team_sizes.most_common():
        percentage = (count / total_trabalhos) * 100
        resposta += f"- **{size} autor(es)**: {count} trabalho(s) ({percentage:.1f}%)\n"

    return resposta

def _analyze_statistics_advanced(df, question, user_profile):
    """Análise estatística avançada"""
    total_registros = len(df)
    total_colunas = len(df.columns)
    
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    colunas_texto = df.select_dtypes(include=['object']).columns.tolist()
    
    resposta = "**📊 ESTATÍSTICAS AVANÇADAS**\n\n"
    resposta += f"**📈 Visão Geral**:\n"
    resposta += f"• **Total de registros**: {total_registros}\n"
    resposta += f"• **Total de colunas**: {total_colunas}\n"
    resposta += f"• **Colunas numéricas**: {len(colunas_numericas)}\n"
    resposta += f"• **Colunas de texto**: {len(colunas_texto)}\n\n"
    
    # Análise de completude
    colunas_principais = ['autor', 'ano', 'título', 'resumo']
    colunas_presentes = []
    
    for col in colunas_principais:
        if any(col in col_name.lower() for col_name in df.columns):
            colunas_presentes.append(col)
    
    completude = (len(colunas_presentes) / len(colunas_principais)) * 100
    resposta += f"**📋 Completude dos Metadados**:\n"
    resposta += f"• **Pontuação**: {completude:.1f}%\n"
    resposta += f"• **Metadados presentes**: {', '.join(colunas_presentes) if colunas_presentes else 'Nenhum'}\n\n"
    
    # Análise de qualidade
    missing_percent = (df.isna().sum().sum() / df.size) * 100
    resposta += f"**✅ Qualidade dos Dados**:\n"
    resposta += f"• **Dados ausentes**: {missing_percent:.1f}%\n"
    
    if missing_percent < 10:
        resposta += "• **🏆 Excelente**: Dados muito completos\n"
    elif missing_percent < 20:
        resposta += "• **✅ Bom**: Boa qualidade geral\n"
    else:
        resposta += "• **⚠️ Atenção**: Considere completar dados faltantes\n"
    
    # Tamanho da base
    resposta += f"\n**🚀 Capacidade Analítica**:\n"
    if total_registros < 20:
        resposta += "• **🔬 Fase Inicial**: Ideal para análises qualitativas e exploratórias\n"
    elif total_registros < 50:
        resposta += "• **📈 Fase Intermediária**: Boa para análises estatísticas básicas\n"
    elif total_registros < 100:
        resposta += "• **📊 Fase Avançada**: Adequada para análises complexas\n"
    else:
        resposta += "• **🚀 Fase Expert**: Excelente para machine learning e análises preditivas\n"

    return resposta

def _analyze_trends_advanced(df, question, user_profile):
    """Análise de tendências avançada"""
    year_col = next((col for col in df.columns if any(kw in col.lower() for kw in ['ano', 'year'])), None)
    
    resposta = "**🚀 ANÁLISE DE TENDÊNCIAS E OPORTUNIDADES**\n\n"
    
    if year_col:
        try:
            anos = pd.to_numeric(df[year_col], errors='coerce').dropna()
            if len(anos) > 5:
                anos_recentes = anos[anos >= anos.max() - 5]
                if len(anos_recentes) > 0:
                    resposta += f"**📊 Foco Temporal Recente**:\n"
                    resposta += f"• **Produção nos últimos 5 anos**: {len(anos_recentes)} trabalhos\n"
                    resposta += f"• **Percentual da produção total**: {(len(anos_recentes)/len(anos))*100:.1f}%\n\n"
        
        except:
            pass
    
    # Análise de temas emergentes
    texto_analise = ""
    for col in df.select_dtypes(include=['object']).columns[:3]:
        texto_analise += " " + df[col].fillna('').astype(str).str.cat(sep=' ')
    
    if len(texto_analise) > 500:
        palavras = re.findall(r'\b[a-zà-ú]{5,}\b', texto_analise.lower())
        palavras_filtradas = [p for p in palavras if p not in PORTUGUESE_STOP_WORDS]
        
        from collections import Counter
        contador = Counter(palavras_filtradas)
        temas_tendencia = [pal for pal, cnt in contador.most_common(15) if cnt >= 2]
        
        if temas_tendencia:
            resposta += "**🔍 Temas em Destaque:**\n"
            for i, tema in enumerate(temas_tendencia[:8], 1):
                resposta += f"{i}. {tema}\n"
    
    # Identificação de oportunidades
    resposta += "\n**💡 OPORTUNIDADES IDENTIFICADAS**:\n"
    
    # Verificar se há dados de colaboração internacional
    country_col = next((col for col in df.columns if any(kw in col.lower() for kw in ['país', 'pais', 'country'])), None)
    if country_col:
        paises = df[country_col].dropna()
        if len(paises) > 0:
            unique_countries = len(paises.value_counts())
            if unique_countries <= 2:
                resposta += "• **🌎 Expansão Internacional**: Oportunidade para colaborações internacionais\n"
    
    # Verificar diversidade temática
    if len(texto_analise) > 1000:
        palavras_unicas = len(set(re.findall(r'\b[a-zà-ú]{5,}\b', texto_analise.lower())))
        if palavras_unicas < 30:
            resposta += "• **🔍 Ampliação Temática**: Considere explorar novos tópicos relacionados\n"
    
    resposta += "\n**🎯 PRÓXIMOS PASSOS SUGERIDOS**:\n"
    resposta += "- Use a aba 'Análise' para gráficos detalhados das tendências\n"
    resposta += "- Explore o mapa mental para conectar conceitos emergentes\n"
    resposta += "- Consulte as recomendações para trabalhos relacionados\n"
    resposta += "- Use o assistente de IA para análises específicas\n"

    return resposta

def _provide_personalized_suggestions(df, question, user_profile):
    """Sugestões personalizadas baseadas na análise completa"""
    total = len(df)
    
    resposta = "**💡 RECOMENDAÇÕES PERSONALIZADAS**\n\n"
    
    # Análise da base
    if total < 20:
        resposta += "**🎯 FASE INICIAL - Estratégias Recomendadas:**\n"
        resposta += "• **Expansão de Dados**: Alcance 30+ registros para análises estatísticas confiáveis\n"
        resposta += "• **Busca Direcionada**: Use termos específicos na aba de recomendações\n"
        resposta += "• **Mapeamento Conceitual**: Organize ideias no mapa mental para clareza\n"
        resposta += "• **Foco Qualitativo**: Aprofunde em estudos de caso com os dados atuais\n\n"
    
    elif total < 50:
        resposta += "**📈 FASE DE CONSOLIDAÇÃO - Próximos Passos:**\n"
        resposta += "• **Análise de Correlação**: Explore relações entre variáveis na aba de gráficos\n"
        resposta += "• **Refinamento Temático**: Use a análise de IA para identificar subtemas\n"
        resposta += "• **Expansão Colaborativa**: Busque parcerias com base no perfil de autores\n"
        resposta += "• **Visualização Avançada**: Crie gráficos de rede para colaborações\n\n"
    
    else:
        resposta += "**🚀 FASE AVANÇADA - Oportunidades:**\n"
        resposta += "• **Análise Preditiva**: Explore padrões com machine learning\n"
        resposta += "• **Metanálise**: Compare diferentes períodos/regiões\n"
        resposta += "• **Análise de Rede Completa**: Mapeie toda a rede de colaboração\n"
        resposta += "• **Identificação de Lacunas**: Use IA para sugerir novas direções\n\n"

    # Análise de qualidade
    missing_percent = (df.isna().sum().sum() / df.size) * 100
    if missing_percent > 15:
        resposta += f"**🔄 MELHORIA DE QUALIDADE**:\n"
        resposta += f"• **Dados Ausentes**: {missing_percent:.1f}% - complete informações faltantes\n"
        resposta += "• **Padronização**: Verifique formatos de datas e nomes\n"
        resposta += "• **Metadados**: Adicione resumos e palavras-chave quando possível\n\n"

    # Sugestões baseadas no perfil
    if 'próximo' in question or 'futuro' in question:
        resposta += "**🔮 DIREÇÕES FUTURAS SUGERIDAS:**\n"
        resposta += "• **Exploração de Novos Temas**: Use a análise temática para identificar áreas emergentes\n"
        resposta += "• **Colaborações Estratégicas**: Identifique pesquisadores complementares\n"
        resposta += "• **Análise Comparativa**: Compare sua pesquisa com tendências globais\n"
        resposta += "• **Síntese de Conhecimento**: Use o mapa mental para integrar descobertas\n"

    resposta += "\n**🛠️ FERRAMENTAS RECOMENDADAS AGORA:**\n"
    resposta += "• **Assistente de IA**: Para perguntas específicas sobre seus dados\n"
    resposta += "• **Mapa Mental**: Para organizar e conectar conceitos\n"
    resposta += "• **Análise Gráfica**: Para visualizações e correlações\n"
    resposta += "• **Sistema de Busca**: Para encontrar referências relacionadas\n"

    return resposta

def _analyze_data_quality_advanced(df, question, user_profile):
    """Análise da qualidade dos dados avançada"""
    total = len(df)
    colunas = df.columns.tolist()
    
    resposta = "**📋 ANÁLISE DE QUALIDADE DE DADOS**\n\n"
    resposta += f"**📊 Visão Geral**:\n"
    resposta += f"• **Total de registros**: {total}\n"
    resposta += f"• **Colunas disponíveis**: {len(colunas)}\n\n"
    
    # Análise de completude
    resposta += "**✅ Completude por Coluna**:\n"
    for col in df.columns[:8]:  # Mostrar até 8 colunas
        na_count = df[col].isna().sum()
        preenchimento = ((total - na_count) / total) * 100
        status = "🟢" if preenchimento > 90 else "🟡" if preenchimento > 70 else "🔴"
        resposta += f"• {status} **{col}**: {preenchimento:.1f}% preenchido\n"
    
    # Metadados essenciais
    essenciais = ['autor', 'ano', 'título']
    presentes = [col for col in essenciais if any(col in col_name.lower() for col_name in df.columns)]
    
    resposta += f"\n**🎯 Metadados Essenciais**: {len(presentes)} de {len(essenciais)} presentes\n"
    
    if len(presentes) < len(essenciais):
        resposta += "**💡 Sugestão**: Considere adicionar colunas para:\n"
        for essencial in essenciais:
            if essencial not in presentes:
                resposta += f"  - **{essencial}**: Informações básicas da pesquisa\n"
    
    # Análise de qualidade geral
    missing_total = df.isna().sum().sum()
    missing_percent = (missing_total / df.size) * 100
    
    resposta += f"\n**📈 Qualidade Geral dos Dados**:\n"
    resposta += f"• **Dados ausentes**: {missing_percent:.1f}%\n"
    
    if missing_percent < 10:
        resposta += "• **🏆 Excelente**: Dados muito completos\n"
    elif missing_percent < 20:
        resposta += "• **✅ Bom**: Qualidade adequada para análises\n"
    elif missing_percent < 40:
        resposta += "• **⚠️ Atenção**: Considere completar dados faltantes\n"
    else:
        resposta += "• **🔴 Crítico**: Muitos dados ausentes - priorize completude\n"

    return resposta

def _comprehensive_analysis_response(df, original_question, user_profile):
    """Resposta abrangente para perguntas complexas"""
    total = len(df)
    
    return f"""**🤖 ASSISTENTE DE IA NUGEP-PQR - ANÁLISE INTELIGENTE**

**Sua pergunta**: "{original_question}"

**📊 Contexto da Sua Pesquisa**:
• **Base de dados**: {total} registros
• **Perfil identificado**: {', '.join(user_profile.get('research_focus', ['Em análise']))[:50]}
• **Prontidão para análise**: {'🔴 Básica' if total < 20 else '🟡 Intermediária' if total < 50 else '🟢 Avançada'}

**🎯 TIPOS DE ANÁLISE DISPONÍVEIS**:

**🔍 ANÁLISES ESPECÍFICAS**:
• "Quais são os autores mais relevantes e suas colaborações?"
• "Como está distribuída minha pesquisa geograficamente?"  
• "Qual a evolução temporal dos meus trabalhos?"
• "Quais os conceitos e temas mais frequentes?"
• "Como melhorar minhas colaborações?"

**📈 ANÁLISES ESTRATÉGICAS**:
• "Analise os padrões e tendências na minha pesquisa"
• "Identifique oportunidades de desenvolvimento" 
• "Sugira próximos passos baseados nos meus dados"
• "Como posso ampliar o impacto da minha pesquisa?"

**💡 RECOMENDAÇÕES PERSONALIZADAS**:
• "Quais são minhas principais lacunas e oportunidades?"
• "Como posso melhorar a qualidade dos meus dados?"
• "Quais ferramentas devo usar agora?"

**🚀 PARA UMA RESPOSTA MAIS PRECISA**:
Faça perguntas mais específicas sobre sua planilha! Por exemplo:
• "Analise minha rede de colaboração"
• "Quais temas emergentes identificar?"
• "Sugira melhorias na estrutura dos dados"

**💬 Estou aqui para ajudar a extrair insights valiosos da sua pesquisa!**"""

# -------------------------
# Miro-like Mind Map Components - SUPER MODERNIZADO
# -------------------------
class ModernMindMap:
    def __init__(self):
        self.node_types = {
            "ideia": {"color": "#4ECDC4", "icon": "💡", "shape": "dot", "gradient": "linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%)"},
            "tarefa": {"color": "#45B7D1", "icon": "✅", "shape": "square", "gradient": "linear-gradient(135deg, #45B7D1 0%, #3A92A8 100%)"},
            "pergunta": {"color": "#96CEB4", "icon": "❓", "shape": "diamond", "gradient": "linear-gradient(135deg, #96CEB4 0%, #7DA895 100%)"},
            "recurso": {"color": "#FECA57", "icon": "📚", "shape": "triangle", "gradient": "linear-gradient(135deg, #FECA57 0%, #D4A847 100%)"},
            "objetivo": {"color": "#FF6B6B", "icon": "🎯", "shape": "star", "gradient": "linear-gradient(135deg, #FF6B6B 0%, #D45959 100%)"},
            "nota": {"color": "#A29BFE", "icon": "📝", "shape": "circle", "gradient": "linear-gradient(135deg, #A29BFE 0%, #857FD4 100%)"},
            "destaque": {"color": "#FF9FF3", "icon": "⭐", "shape": "hexagon", "gradient": "linear-gradient(135deg, #FF9FF3 0%, #D485CF 100%)"},
            "problema": {"color": "#54A0FF", "icon": "🔍", "shape": "triangle", "gradient": "linear-gradient(135deg, #54A0FF 0%, #4585D4 100%)"}
        }
    
    def create_node(self, node_id, label, node_type="ideia", description="", x=0, y=0):
        """Cria uma ideia no estilo moderno"""
        node_data = self.node_types.get(node_type, self.node_types["ideia"])
        return {
            "id": node_id,
            "label": f"{node_data['icon']} {label}",
            "type": node_type,
            "description": description,
            "color": node_data["color"],
            "gradient": node_data["gradient"],
            "shape": node_data["shape"],
            "x": x,
            "y": y,
            "font": {"color": "#FFFFFF", "size": 16, "face": "Arial", "strokeWidth": 2, "strokeColor": "#000000"},
            "size": 25,
            "borderWidth": 2,
            "borderColor": "#FFFFFF",
            "shadow": True,
            "mass": 1.5
        }
    
    def _calculate_smart_position(self, existing_nodes, selected_node_id):
        """Calcula posição inteligente para novo nó com distribuição otimizada"""
        if not existing_nodes:
            return 500, 400  # Posição central se não há nós
        
        # Se há nó selecionado, posicionar em anel ao redor
        if selected_node_id:
            selected_node = next((n for n in existing_nodes if n["id"] == selected_node_id), None)
            if selected_node:
                # Posicionar em um raio de 200px do nó selecionado
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(150, 250)
                x = selected_node.get("x", 500) + distance * math.cos(angle)
                y = selected_node.get("y", 400) + distance * math.sin(angle)
                return x, y
        
        # Distribuição em espiral para melhor visualização
        center_x, center_y = 500, 400
        occupied_positions = [(n.get("x", 0), n.get("y", 0)) for n in existing_nodes]
        
        for radius in range(200, 1001, 120):  # De 200 a 1000 pixels
            for angle in range(0, 360, 30):  # A cada 30 graus
                rad = math.radians(angle)
                x = center_x + radius * math.cos(rad)
                y = center_y + radius * math.sin(rad)
                
                # Verificar se está suficientemente longe de outros nós
                too_close = any(
                    math.sqrt((x - ox)**2 + (y - oy)**2) < 180 
                    for ox, oy in occupied_positions
                )
                
                if not too_close:
                    return x, y
        
        # Fallback: posição aleatória com verificação
        for _ in range(20):
            x = random.randint(100, 900)
            y = random.randint(100, 700)
            too_close = any(math.sqrt((x - ox)**2 + (y - oy)**2) < 150 for ox, oy in occupied_positions)
            if not too_close:
                return x, y
        
        return random.randint(200, 800), random.randint(150, 650)
    
    def generate_layout(self, nodes, edges, layout_type="hierarchical"):
        """Gera layout automático moderno para as ideias"""
        if layout_type == "hierarchical":
            return self._hierarchical_layout(nodes, edges)
        elif layout_type == "radial":
            return self._radial_layout(nodes, edges)
        elif layout_type == "circular":
            return self._circular_layout(nodes, edges)
        else:
            return self._force_directed_layout(nodes, edges)
    
    def _hierarchical_layout(self, nodes, edges):
        """Layout hierárquico moderno"""
        if not nodes:
            return nodes
            
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node["id"])
        for edge in edges:
            G.add_edge(edge["source"], edge["target"])
        
        try:
            # Layout mais estável e visualmente agradável
            pos = nx.spring_layout(G, k=3, iterations=100, scale=2, seed=42)
            
            # Aplicar posições com suavização
            for node in nodes:
                if node["id"] in pos:
                    new_x = pos[node["id"]][0] * 800 + 400
                    new_y = pos[node["id"]][1] * 600 + 300
                    
                    # Transição suave se já tinha posição
                    if "x" in node and "y" in node:
                        node["x"] = (node["x"] + new_x) / 2
                        node["y"] = (node["y"] + new_y) / 2
                    else:
                        node["x"] = new_x
                        node["y"] = new_y
        except:
            # Fallback organizado
            for i, node in enumerate(nodes):
                if "x" not in node or "y" not in node:
                    node["x"] = 400 + (i % 4) * 200
                    node["y"] = 300 + (i // 4) * 150
        
        return nodes

    def _radial_layout(self, nodes, edges):
        """Layout radial moderno"""
        center_x, center_y = 500, 400
        
        if len(nodes) == 1:
            nodes[0]["x"] = center_x
            nodes[0]["y"] = center_y
            return nodes
        
        radius = min(400, 200 + len(nodes) * 30)  # Raio adaptável
        
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / len(nodes)
            node["x"] = center_x + radius * np.cos(angle)
            node["y"] = center_y + radius * np.sin(angle)
        
        return nodes

    def _circular_layout(self, nodes, edges):
        """Layout circular para visualização limpa"""
        center_x, center_y = 500, 400
        radius = 300
        
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / len(nodes)
            node["x"] = center_x + radius * np.cos(angle)
            node["y"] = center_y + radius * np.sin(angle)
        
        return nodes

    def _force_directed_layout(self, nodes, edges):
        """Layout de força direcionada moderno e estável"""
        if not nodes:
            return nodes
        
        G = nx.Graph()
        
        # Adicionar nós mantendo referências
        for node in nodes:
            G.add_node(node["id"])
        
        # Adicionar arestas com pesos
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], weight=2)
        
        try:
            # CONFIGURAÇÃO OTIMIZADA PARA ESTABILIDADE E VISUAL
            pos_existente = {}
            for node in nodes:
                if "x" in node and "y" in node:
                    pos_existente[node["id"]] = [node["x"], node["y"]]
            
            # Parâmetros para layout moderno
            k = 4  # Distância ideal entre nós
            iterations = 100  # Mais iterações para melhor estabilidade
            scale = 2.5  # Escala adequada
            
            if pos_existente:
                pos = nx.spring_layout(G, pos=pos_existente, k=k, iterations=iterations, 
                                     scale=scale, seed=42)
            else:
                pos = nx.spring_layout(G, k=k, iterations=iterations, scale=scale, seed=42)
            
            # Aplicar novas posições
            for node in nodes:
                if node["id"] in pos:
                    new_x = pos[node["id"]][0] * 800 + 400
                    new_y = pos[node["id"]][1] * 600 + 300
                    
                    # Manter alguma estabilidade para nós existentes
                    if "x" in node and "y" in node:
                        node["x"] = (node["x"] * 0.3 + new_x * 0.7)
                        node["y"] = (node["y"] * 0.3 + new_y * 0.7)
                    else:
                        node["x"] = new_x
                        node["y"] = new_y
                        
        except Exception as e:
            print(f"Layout automático falhou: {e}")
            # Fallback: grid organizado
            for i, node in enumerate(nodes):
                if "x" not in node or "y" not in node:
                    node["x"] = 400 + (i % 5) * 180
                    node["y"] = 300 + (i // 5) * 120
        
        return nodes

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
                inner = part[2:-2].replace("—", "-").replace("—", "-").encode("latin-1", "replace").decode("latin-1")
                hexv = (highlight_hex or "#ffd600").lstrip("#")
                if len(hexv) == 3: hexv = ''.join([c*2 for c in hexv])
                try: r, g, b = int(hexv[0:2], 16), int(hexv[2:4], 16), int(hexv[4:6], 16)
                except Exception: r, g, b = (255, 214, 0)
                pdf.set_fill_color(r, g, b)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 6, txt=inner, border=0, fill=True)
            else:
                safe_part = part.replace("—", "-").replace("—", "-").encode("latin-1", "replace").decode("latin-1")
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
    "ia_response": None,
    "modern_mind_map": None,
    "settings": {
        "plot_height": 720, "font_scale": 1.0, "node_opacity": 1.0,
        "font_size": 14,
        "node_font_size": 14,
        "modern_ui": True,
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

# -------------------------
# Navigation atualizada com favoritos
# -------------------------
nav_buttons = {
    "planilha": "📄 Planilha", 
    "recomendacoes": "💡 Recomendações", 
    "mapa": "🗺️ Mapa Mental",
    "anotacoes": "📝 Anotações", 
    "graficos": "📊 Análise", 
    "busca": "🔍 Busca",
    "favoritos": "⭐ Favoritos",
    "mensagens": f"✉️ Mensagens ({UNREAD_COUNT})" if UNREAD_COUNT > 0 else "✉️ Mensagens",
    "config": "⚙️ Configurações"
}

st.markdown("<div class='glass-box' style='padding-top:10px; padding-bottom:10px;'><div class='specular'></div>", unsafe_allow_html=True)
top1, top2 = st.columns([0.6, 0.4])
with top1:
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
        * **🗺️ Mapa Mental**: Visualize e edite mapas mentais e fluxogramas interativos para organizar ideias.
        * **📝 Anotações**: Um bloco de notas para destacar texto com `==sinais de igual==` e exportar como PDF.
        * **📊 Análise**: Gere gráficos e análises inteligentes a partir da sua planilha.
        * **🔍 Busca**: Pesquise em todas as planilhas carregadas na plataforma.
        * **⭐ Favoritos**: Acesse rapidamente seus artigos favoritos salvos.
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
# Page: recomendacoes
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

    # recommendation onboarding or refine
    if not st.session_state.recommendation_onboarding_complete:
        if df_total.empty:
            st.warning("Ainda não há dados suficientes para gerar recomendações automaticamente.")
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
                    if det.get('similarity'):
                        st.metric("Similaridade", f"{det['similarity']:.2f}")
                    
                    if det.get('_artemis_username'):
                        st.write(f"Fonte: {det['_artemis_username']}")

                col_btn1, col_btn2, col_btn3 = st.columns(3)
                with col_btn1:
                    if st.button("⭐ Adicionar aos Favoritos", use_container_width=True, key=f"fav_detail_rec_{vi}_{USERNAME}"):
                        if add_to_favorites(det): st.toast("Adicionado aos favoritos!", icon="⭐")
                        else: st.toast("Este artigo já está nos favoritos.")
                with col_btn2:
                    if st.button("📝 Ver Anotações", use_container_width=True, key=f"notes_rec_{vi}_{USERNAME}"):
                        st.session_state.page = "anotacoes"
                        safe_rerun()
                with col_btn3:
                    if st.button("🔍 Buscar Similares", use_container_width=True, key=f"similar_rec_{vi}_{USERNAME}"):
                        st.session_state.page = "busca"
                        st.session_state.search_input_main = det.get('título', '').split()[0] if det.get('título') else ''
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
# PÁGINA: FAVORITOS - IMPLEMENTAÇÃO COMPLETA
# -------------------------
elif st.session_state.page == "favoritos":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("⭐ Seus Favoritos Salvos")
    
    favorites = get_session_favorites()
    
    if not favorites:
        st.info("""
        ## 🌟 Nenhum favorito ainda!
        
        **Como adicionar favoritos:**
        - Use a aba **🔍 Busca** para encontrar artigos interessantes
        - Clique em **⭐ Favoritar** nos resultados
        - Ou adicione das **💡 Recomendações**
        
        Seus favoritos aparecerão aqui para acesso rápido!
        """)
        
        # Botões de ação rápida
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Ir para Busca", use_container_width=True):
                st.session_state.page = "busca"
                safe_rerun()
        with col2:
            if st.button("💡 Ver Recomendações", use_container_width=True):
                st.session_state.page = "recomendacoes"
                safe_rerun()
                
    else:
        # Estatísticas rápidas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Favoritos", len(favorites))
        with col2:
            sources = [fav['data'].get('_artemis_username', 'Desconhecido') for fav in favorites]
            unique_sources = len(set(sources))
            st.metric("Fontes Diferentes", unique_sources)
        with col3:
            # Contar por tipo de fonte
            local_sources = len([s for s in sources if s != 'web'])
            web_sources = len([s for s in sources if s == 'web'])
            st.metric("Locais/Web", f"{local_sources}/{web_sources}")
        with col4:
            if st.button("🗑️ Limpar Todos", use_container_width=True, type="secondary"):
                if st.checkbox("Confirmar limpeza de TODOS os favoritos?"):
                    clear_all_favorites()
                    st.success("Todos os favoritos foram removidos!")
                    safe_rerun()
        
        st.markdown("---")
        
        # Filtros e organização
        col_search, col_sort, col_filter = st.columns([2, 1, 1])
        with col_search:
            search_fav = st.text_input("🔍 Buscar nos favoritos:", 
                                     placeholder="Digite título, autor, tema...",
                                     key="search_favorites")
        with col_sort:
            sort_option = st.selectbox("Ordenar por:", 
                                     ["Data de adição", "Título A-Z", "Título Z-A", "Fonte"],
                                     key="sort_favorites")
        with col_filter:
            filter_source = st.selectbox("Filtrar por fonte:", 
                                       ["Todas", "Local", "Web"],
                                       key="filter_favorites")
        
        # Aplicar filtros
        filtered_favorites = favorites.copy()
        
        # Filtro de busca
        if search_fav:
            filtered_favorites = [
                fav for fav in filtered_favorites 
                if search_fav.lower() in str(fav['data'].get('título', '')).lower() or
                   search_fav.lower() in str(fav['data'].get('autor', '')).lower() or
                   search_fav.lower() in str(fav['data'].get('_artemis_username', '')).lower()
            ]
        
        # Filtro por fonte
        if filter_source == "Local":
            filtered_favorites = [fav for fav in filtered_favorites if fav['data'].get('_artemis_username') != 'web']
        elif filter_source == "Web":
            filtered_favorites = [fav for fav in filtered_favorites if fav['data'].get('_artemis_username') == 'web']
        
        # Ordenação
        if sort_option == "Data de adição":
            filtered_favorites.sort(key=lambda x: x['added_at'], reverse=True)
        elif sort_option == "Título A-Z":
            filtered_favorites.sort(key=lambda x: str(x['data'].get('título', '')).lower())
        elif sort_option == "Título Z-A":
            filtered_favorites.sort(key=lambda x: str(x['data'].get('título', '')).lower(), reverse=True)
        elif sort_option == "Fonte":
            filtered_favorites.sort(key=lambda x: str(x['data'].get('_artemis_username', '')).lower())
        
        # Visualização detalhada
        if 'fav_detail_view' in st.session_state and st.session_state.fav_detail_view:
            fav_id = st.session_state.fav_detail_view
            favorite = next((fav for fav in filtered_favorites if fav['id'] == fav_id), None)
            
            if favorite:
                fav_data = favorite['data']
                fav_data = enrich_article_metadata(fav_data)
                
                st.markdown("## 📄 Detalhes do Favorito")
                
                if st.button("⬅️ Voltar para lista", key="back_from_detail"):
                    st.session_state.fav_detail_view = None
                    safe_rerun()
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {escape_html(fav_data.get('título','— Sem título —'))}")
                    st.markdown(f"**Autor(es):** {escape_html(fav_data.get('autor','— Não informado —'))}")
                    st.markdown(f"**Ano:** {escape_html(str(fav_data.get('ano', fav_data.get('year','— —'))))}")
                    
                    if fav_data.get('doi'):
                        doi_link = f"https://doi.org/{fav_data.get('doi')}"
                        st.markdown(f"**DOI:** [{fav_data.get('doi')}]({doi_link})")
                    elif fav_data.get('url'):
                        st.markdown(f"**Link:** [{fav_data.get('url')}]({fav_data.get('url')})")
                    
                    if fav_data.get('similarity'):
                        st.markdown(f"**Similaridade:** {fav_data['similarity']:.3f}")
                    
                    st.markdown("---")
                    st.markdown("**Resumo**")
                    st.markdown(escape_html(fav_data.get('resumo', 'Resumo não disponível.')))
                
                with col2:
                    st.markdown("**Ações**")
                    if st.button("📝 Ir para Anotações", use_container_width=True, key="fav_to_notes"):
                        st.session_state.page = "anotacoes"
                        safe_rerun()
                    
                    if st.button("🔍 Buscar Similares", use_container_width=True, key="fav_similar"):
                        st.session_state.page = "busca"
                        st.session_state.search_input_main = fav_data.get('título', '').split()[0] if fav_data.get('título') else ''
                        safe_rerun()
                    
                    if st.button("🗑️ Remover Favorito", use_container_width=True, type="secondary", key="remove_fav_detail"):
                        remove_from_favorites(fav_id)
                        st.session_state.fav_detail_view = None
                        st.success("Favorito removido!")
                        safe_rerun()
            
            else:
                st.error("Favorito não encontrado.")
                if st.button("⬅️ Voltar para lista"):
                    st.session_state.fav_detail_view = None
                    safe_rerun()
        
        else:
            # Lista de favoritos
            st.markdown(f"**📊 Mostrando {len(filtered_favorites)} de {len(favorites)} favoritos**")
            
            if not filtered_favorites:
                st.info("Nenhum favorito encontrado com os filtros atuais.")
            
            for fav in filtered_favorites:
                fav_data = fav['data']
                user_src = fav_data.get('_artemis_username', 'N/A')
                added_date = datetime.fromisoformat(fav['added_at']).strftime('%d/%m/%Y %H:%M')
                
                # Obter nome do usuário se for local
                all_users = load_users()
                user_name = all_users.get(user_src, {}).get('name', user_src) if user_src != 'web' else 'Web (Crossref)'
                
                with st.container():
                    col1, col2, col3 = st.columns([4, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="card">
                            <div class="card-title">{escape_html(fav_data.get('título', '(Sem título)'))}</div>
                            <div class="small-muted">
                                <strong>Autor(es):</strong> {escape_html(fav_data.get('autor', 'Não informado'))}<br>
                                <strong>Fonte:</strong> {escape_html(user_name)} • 
                                <strong>Adicionado:</strong> {added_date}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button("👁️", key=f"view_{fav['id']}", help="Ver detalhes", use_container_width=True):
                            st.session_state.fav_detail_view = fav['id']
                            safe_rerun()
                    
                    with col3:
                        if st.button("🗑️", key=f"del_{fav['id']}", help="Remover", use_container_width=True):
                            remove_from_favorites(fav['id'])
                            st.success("Favorito removido!")
                            safe_rerun()
                    
                    st.markdown("---")
            
            # Opções de exportação
            if filtered_favorites:
                st.markdown("### 💾 Exportar Favoritos")
                col_exp1, col_exp2 = st.columns(2)
                
                with col_exp1:
                    # Exportar como CSV
                    export_data = []
                    for fav in filtered_favorites:
                        fav_data = fav['data']
                        export_data.append({
                            'Título': fav_data.get('título', ''),
                            'Autores': fav_data.get('autor', ''),
                            'Ano': fav_data.get('ano', ''),
                            'DOI': fav_data.get('doi', ''),
                            'URL': fav_data.get('url', ''),
                            'Fonte': fav_data.get('_artemis_username', ''),
                            'Data_Adicao': fav['added_at']
                        })
                    
                    if export_data:
                        df_export = pd.DataFrame(export_data)
                        csv_data = df_export.to_csv(index=False, encoding='utf-8')
                        st.download_button(
                            "📊 Exportar CSV",
                            data=csv_data,
                            file_name=f"favoritos_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                with col_exp2:
                    # Exportar como JSON
                    json_data = json.dumps(filtered_favorites, indent=2, ensure_ascii=False)
                    st.download_button(
                        "📋 Exportar JSON",
                        data=json_data,
                        file_name=f"favoritos_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# PÁGINA MAPA MENTAL ATUALIZADA
# -------------------------
elif st.session_state.page == "mapa":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("🗺️ Mapa Mental Interativo Moderno")
    st.info("💡 **Crie, conecte e visualize suas ideias** - Interface moderna com visualização 3D avançada!")
    
    # Inicializar sistema de mapa mental MODERNO
    if 'modern_mind_map' not in st.session_state:
        st.session_state.modern_mind_map = ModernMindMap()
        st.session_state.miro_nodes = []
        st.session_state.miro_edges = []
        st.session_state.miro_selected_node = None
        st.session_state.miro_layout = "force"
        st.session_state.visualization_3d = True
    
    # Sidebar principal MODERNIZADA
    with st.sidebar:
        st.header("🎨 Controles do Mapa")
        
        # Modo de visualização
        visualization_mode = st.selectbox("🎭 Modo de Visualização:", 
                                        ["Mapa 3D Moderno", "Fluxograma", "Rede Colaborativa", "Mapa Radial"])
        
        # Criar nova ideia - INTERFACE MODERNIZADA
        with st.expander("🚀 Nova Ideia", expanded=True):
            with st.form("create_modern_node", clear_on_submit=True):
                node_label = st.text_input("Título da ideia:", placeholder="Ex: Pesquisa Qualitativa", key="new_node_label")
                node_type = st.selectbox("Tipo:", options=list(st.session_state.modern_mind_map.node_types.keys()), 
                                       format_func=lambda x: f"{st.session_state.modern_mind_map.node_types[x]['icon']} {x.title()}", 
                                       key="new_node_type")
                node_desc = st.text_area("Descrição:", placeholder="Detalhes, insights ou ações relacionadas...", 
                                       height=100, key="new_node_desc")
                
                # BOTÃO DE ENVIO CORRETO - usando st.form_submit_button
                submit_btn = st.form_submit_button("🎯 Criar Ideia", use_container_width=True)
                
                if submit_btn and node_label:
                    node_id = f"node_{int(time.time())}_{random.randint(1000,9999)}"
                    
                    # Posicionamento inteligente
                    x, y = st.session_state.modern_mind_map._calculate_smart_position(
                        st.session_state.miro_nodes, 
                        st.session_state.miro_selected_node
                    )
                    
                    new_node = st.session_state.modern_mind_map.create_node(
                        node_id, node_label, node_type, node_desc, x, y
                    )
                    st.session_state.miro_nodes.append(new_node)
                    st.session_state.miro_selected_node = node_id
                    st.success("✨ Ideia criada com sucesso!")
                    safe_rerun()
        
        # Conectar ideias
        with st.expander("🔗 Conectar Ideias", expanded=False):
            if len(st.session_state.miro_nodes) >= 2:
                nodes_list = [(node["id"], node["label"]) for node in st.session_state.miro_nodes]
                with st.form("connect_nodes"):
                    source_options = {f"{label}": node_id for node_id, label in nodes_list}
                    target_options = {f"{label}": node_id for node_id, label in nodes_list}
                    
                    source_label = st.selectbox("De:", options=list(source_options.keys()), key="connect_source")
                    target_label = st.selectbox("Para:", options=[k for k in target_options.keys() if k != source_label], key="connect_target")
                    
                    if st.form_submit_button("🔗 Conectar", use_container_width=True):
                        source_id = source_options[source_label]
                        target_id = target_options[target_label]
                        
                        existing = any(e["source"] == source_id and e["target"] == target_id for e in st.session_state.miro_edges)
                        if not existing:
                            st.session_state.miro_edges.append({
                                "source": source_id,
                                "target": target_id,
                                "label": "conecta a"
                            })
                            st.success("Conexão criada!")
                            safe_rerun()
                        else:
                            st.warning("Essas ideias já estão conectadas.")
            else:
                st.info("Precisa de pelo menos 2 ideias para conectar")
        
        # Gerenciamento avançado
        with st.expander("🔧 Ferramentas Avançadas", expanded=False):
            st.session_state.miro_layout = st.selectbox("Layout Automático:", 
                                                      ["force", "hierarchical", "radial", "circular"])
            
            if st.button("🔄 Reorganizar Mapa", use_container_width=True):
                st.session_state.miro_nodes = st.session_state.modern_mind_map.generate_layout(
                    st.session_state.miro_nodes, st.session_state.miro_edges, st.session_state.miro_layout
                )
                st.success("Mapa reorganizado!")
                safe_rerun()
            
            # Exportar/Importar
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                if st.button("💾 Exportar", use_container_width=True):
                    map_data = {
                        "nodes": st.session_state.miro_nodes,
                        "edges": st.session_state.miro_edges,
                        "layout": st.session_state.miro_layout
                    }
                    st.download_button(
                        "⬇️ Baixar Mapa",
                        data=json.dumps(map_data, indent=2),
                        file_name=f"mapa_mental_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        mime="application/json"
                    )
            with col_exp2:
                uploaded_map = st.file_uploader("Importar", type=["json"], key="map_uploader")
                if uploaded_map:
                    try:
                        map_data = json.load(uploaded_map)
                        st.session_state.miro_nodes = map_data.get("nodes", [])
                        st.session_state.miro_edges = map_data.get("edges", [])
                        st.success("Mapa importado!")
                        safe_rerun()
                    except:
                        st.error("Erro ao importar mapa")
    
    # Área principal do mapa MODERNIZADA
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"🎨 {visualization_mode}")
        
        if st.session_state.miro_nodes:
            # Configurações visuais baseadas no modo
            if visualization_mode == "Mapa 3D Moderno":
                st.markdown('<div class="three-d-effect">', unsafe_allow_html=True)
                node_size = 35
                font_size = 16
                physics_enabled = True
                hierarchical_enabled = False
                
                # Aplicar efeitos 3D modernos
                for node in st.session_state.miro_nodes:
                    node["size"] = node_size
                    node["font"] = {"size": font_size, "color": "#FFFFFF", "strokeWidth": 2, "strokeColor": "#000000"}
                    node["borderWidth"] = 3
                    node["borderColor"] = "#FFFFFF"
                    node["shadow"] = True
                    node["mass"] = 2.0
            
            elif visualization_mode == "Fluxograma":
                st.markdown('<div class="flowchart-box">', unsafe_allow_html=True)
                node_size = 30
                font_size = 14
                physics_enabled = False
                hierarchical_enabled = True
                
            else:  # Modos padrão
                node_size = 28
                font_size = 14
                physics_enabled = True
                hierarchical_enabled = False

            # Preparar nós e arestas
            nodes_for_viz = []
            for node in st.session_state.miro_nodes:
                nodes_for_viz.append(
                    Node(
                        id=node["id"],
                        label=node["label"],
                        size=node.get("size", node_size),
                        color=node["color"],
                        shape=node.get("shape", "dot"),
                        font=node.get("font", {"color": "#FFFFFF", "size": font_size}),
                        x=node.get("x", 0),
                        y=node.get("y", 0),
                        borderWidth=node.get("borderWidth", 2),
                        borderColor=node.get("borderColor", "#FFFFFF"),
                        shadow=node.get("shadow", True)
                    )
                )

            edges_for_viz = []
            for edge in st.session_state.miro_edges:
                edges_for_viz.append(
                    Edge(
                        source=edge["source"],
                        target=edge["target"],
                        label=edge.get("label", ""),
                        color="#rgba(255,255,255,0.7)",
                        width=3,
                        dashes=False
                    )
                )

            # Configuração moderna
            config_params = {
                "width": 800,
                "height": 600,
                "directed": True,
                "physics": physics_enabled,
                "hierarchical": hierarchical_enabled,
            }

            if hierarchical_enabled:
                config_params["hierarchical"] = {
                    "enabled": True,
                    "levelSeparation": 200,
                    "nodeSpacing": 150,
                    "treeSpacing": 250,
                    "direction": "UD"
                }

            config = Config(**config_params)

            # Renderizar o gráfico
            try:
                return_value = agraph(nodes=nodes_for_viz, edges=edges_for_viz, config=config)

                if return_value:
                    st.session_state.miro_selected_node = return_value

            except Exception as e:
                st.error(f"Erro ao renderizar o mapa: {e}")
                st.info("💡 Tente reorganizar o mapa ou reduzir o número de ideias")

            # Fechar divs de estilo
            if visualization_mode == "Mapa 3D Moderno":
                st.markdown('</div>', unsafe_allow_html=True)
            elif visualization_mode == "Fluxograma":
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("""
            ## 🌟 Comece seu mapa mental!
            
            **Para criar seu primeiro mapa:**
            1. Use o painel à esquerda para criar ideias
            2. Conecte ideias relacionadas
            3. Explore diferentes visualizações
            4. Organize seu pensamento de forma visual
            
            **💡 Dica**: Comece com uma ideia central e expanda a partir dela!
            """)

    with col2:
        st.subheader("📋 Gerenciamento")
        
        # Estatísticas rápidas
        if st.session_state.miro_nodes:
            st.metric("Total de Ideias", len(st.session_state.miro_nodes))
            st.metric("Conexões", len(st.session_state.miro_edges))
            
            # Nó selecionado
            if st.session_state.miro_selected_node:
                selected_node = next((n for n in st.session_state.miro_nodes 
                                    if n["id"] == st.session_state.miro_selected_node), None)
                if selected_node:
                    st.markdown(f"**🎯 Selecionado:** {selected_node['label']}")
            
            # Lista de ideias com ações rápidas
            st.markdown("### 💡 Suas Ideias")
            for node in st.session_state.miro_nodes[:8]:  # Mostrar primeiras 8
                is_selected = st.session_state.miro_selected_node == node["id"]
                emoji = "🟢" if is_selected else "⚪"
                
                col_btn1, col_btn2 = st.columns([3, 1])
                with col_btn1:
                    if st.button(f"{emoji} {node['label'][:20]}...", 
                               key=f"sel_{node['id']}", 
                               use_container_width=True):
                        st.session_state.miro_selected_node = node["id"]
                        safe_rerun()
                with col_btn2:
                    if st.button("🗑️", key=f"quick_del_{node['id']}", 
                               help="Excluir", use_container_width=True):
                        st.session_state.miro_nodes = [n for n in st.session_state.miro_nodes if n['id'] != node['id']]
                        st.session_state.miro_edges = [e for e in st.session_state.miro_edges 
                                                     if e['source'] != node['id'] and e['target'] != node['id']]
                        if st.session_state.miro_selected_node == node['id']:
                            st.session_state.miro_selected_node = None
                        st.success("Ideia removida!")
                        safe_rerun()
        
        # Ações em lote
        with st.expander("⚡ Ações Rápidas", expanded=False):
            if st.button("🎨 Aplicar Cores Automáticas", use_container_width=True):
                for node in st.session_state.miro_nodes:
                    if node["type"] in st.session_state.modern_mind_map.node_types:
                        node_data = st.session_state.modern_mind_map.node_types[node["type"]]
                        node["color"] = node_data["color"]
                st.success("Cores atualizadas!")
                safe_rerun()
            
            if st.button("🧹 Limpar Mapa", type="secondary", use_container_width=True):
                if st.checkbox("Confirmar limpeza total do mapa?"):
                    st.session_state.miro_nodes = []
                    st.session_state.miro_edges = []
                    st.session_state.miro_selected_node = None
                    st.success("Mapa limpo!")
                    safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: anotacoes
# -------------------------
elif st.session_state.page == "anotacoes":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("📝 Anotações Pessoais")
    
    notes_content = st.text_area("Escreva suas anotações aqui (use ==destaque== para realçar texto):", 
                                value=st.session_state.notes, 
                                height=400,
                                key="notes_editor")
    
    if notes_content != st.session_state.notes:
        st.session_state.notes = notes_content
        if st.session_state.autosave:
            save_user_state_minimal(USER_STATE)
            st.toast("Anotações salvas automaticamente.", icon="💾")
    
    # Visualização com highlights
    st.subheader("📄 Visualização com Destaques")
    if notes_content:
        highlighted_html = re.sub(r'==(.*?)==', r'<mark class="card-mark">\1</mark>', escape_html(notes_content))
        st.markdown(f'<div class="card">{highlighted_html.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
    else:
        st.info("Nenhuma anotação ainda. Comece a escrever acima!")
    
    # Exportação
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 Salvar Anotações", use_container_width=True):
            if save_user_state_minimal(USER_STATE):
                st.success("Anotações salvas!")
            else:
                st.error("Erro ao salvar.")
    
    with col2:
        if st.button("📄 Exportar PDF", use_container_width=True):
            if notes_content:
                pdf_bytes = generate_pdf_with_highlights(notes_content)
                st.download_button(
                    "⬇️ Baixar PDF",
                    data=pdf_bytes,
                    file_name=f"anotacoes_{USERNAME}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.warning("Nenhuma anotação para exportar.")
    
    with col3:
        if st.button("🗑️ Limpar Tudo", type="secondary", use_container_width=True):
            if st.checkbox("Confirmar limpeza de todas as anotações?"):
                st.session_state.notes = ""
                save_user_state_minimal(USER_STATE)
                st.success("Anotações limpas!")
                safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: graficos - COM BOTÃO DA IA CORRETO
# -------------------------
elif st.session_state.page == "graficos":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("📊 Análise e Visualização de Dados")
    
    if st.session_state.df is None or st.session_state.df.empty:
        st.info("📁 Carregue uma planilha na aba 'Planilha' para ver análises e gráficos.")
    else:
        df = st.session_state.df
        
        # Análise inteligente automática
        st.subheader("🤖 Análise Inteligente dos Dados")
        if st.button("🔍 Gerar Análise Completa da Planilha", use_container_width=True):
            with st.spinner("Analisando dados... Isso pode levar alguns segundos"):
                analyzer = DataAnalyzer(df)
                analysis = analyzer.generate_comprehensive_analysis()
                st.markdown(analysis)
        
        st.markdown("---")
        
        # Assistente de IA SUPER INTELIGENTE
        st.subheader("💬 Converse com a IA sobre seus dados")
        ia_col1, ia_col2 = st.columns([4, 1])
        with ia_col1:
            question = st.text_input(
                "Faça uma pergunta sobre a planilha:", 
                placeholder="Ex: Quais são os autores mais produtivos?",
                key="ia_question_input",
                label_visibility="collapsed"
            )
        with ia_col2:
            ask_button = st.button("Perguntar à IA", key="ia_ask_button", use_container_width=True)

        if ask_button and question:
            with st.spinner("A IA está pensando..."):
                analyzer = DataAnalyzer(df)
                response = get_ai_assistant_response(question, analyzer)
                st.session_state.ia_response = response
        elif ask_button and not question:
            st.session_state.ia_response = None
            st.warning("Por favor, digite uma pergunta.")

        if st.session_state.ia_response:
            st.markdown(st.session_state.ia_response)
        
        st.markdown("---")
        
        # Visualizações gráficas
        st.subheader("📈 Visualizações Gráficas")
        
        chart_type = st.selectbox("Escolha o tipo de gráfico:", 
                                ["Barras", "Linhas", "Pizza"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("Eixo X (ou Categoria para Pizza):", options=df.columns.tolist())
        
        with col2:
            if chart_type in ["Barras", "Linhas"]:
                y_axis = st.selectbox("Eixo Y (Valores):", options=[None] + df.columns.tolist())
            else:
                y_axis = None
        
        try:
            if chart_type == "Barras":
                if y_axis: # Se o usuário selecionou um eixo Y, agregue os dados
                    if df[y_axis].dtype in ['int64', 'float64']:
                        grouped_df = df.groupby(x_axis)[y_axis].sum().reset_index().sort_values(by=y_axis, ascending=False).head(20)
                        fig = px.bar(grouped_df, x=x_axis, y=y_axis, title=f"Soma de '{y_axis}' por '{x_axis}'")
                    else:
                        st.warning(f"Para agregar, o Eixo Y ('{y_axis}') deve ser numérico.")
                        fig = None
                else: # Se não, faça uma contagem de frequência no eixo X
                    value_counts = df[x_axis].value_counts().head(20)
                    fig = px.bar(value_counts, x=value_counts.index, y=value_counts.values, 
                               title=f"Contagem de Ocorrências em '{x_axis}'", labels={'x': x_axis, 'y': 'Contagem'})
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Linhas":
                if y_axis and df[y_axis].dtype in ['int64', 'float64']:
                    # Tenta ordenar o eixo X se for numérico (como ano) ou data
                    df_sorted = df.sort_values(by=x_axis)
                    fig = px.line(df_sorted, x=x_axis, y=y_axis, title=f"'{y_axis}' ao longo de '{x_axis}'")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Para gráfico de linhas, o eixo Y deve ser uma coluna numérica.")
            
            elif chart_type == "Pizza":
                value_counts = df[x_axis].value_counts().head(10)
                fig = px.pie(values=value_counts.values, names=value_counts.index, 
                           title=f"Distribuição de '{x_axis}'")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Erro ao gerar gráfico: {e}")
            st.info("Tente selecionar diferentes colunas ou tipos de gráfico.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: busca - COM FILTROS MELHORADOS
# -------------------------
elif st.session_state.page == "busca":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("🔍 Busca Avançada")
    
    try:
        with st.spinner("Carregando dados de todos os usuários..."):
            df_total = collect_latest_backups()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        df_total = pd.DataFrame()

    if df_total.empty:
        st.info("Ainda não há dados disponíveis na plataforma para busca.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Digite o termo para buscar:", 
                                         placeholder="Digite palavras-chave, autores, temas...",
                                         key="search_input_main")
        with col2:
            search_scope = st.selectbox("Buscar em:", 
                                        ["Todas as colunas", "Título", "Autor", "País", "Tema"], 
                                        key="search_scope_selector")

        if st.button("🔍 Executar Busca", use_container_width=True):
            if search_query:
                with st.spinner("Buscando..."):
                    results = df_total.copy()
                    query = search_query.strip()
                    
                    if search_scope == "Todas as colunas":
                        mask = results.astype(str).apply(lambda x: x.str.contains(query, case=False, na=False)).any(axis=1)
                        results = results[mask]
                    else:
                        col_map = {
                            "Título": ['título', 'titulo', 'title'],
                            "Autor": ['autor', 'autores', 'author'],
                            "País": ['país', 'pais', 'country'],
                            "Tema": ['tema', 'temas', 'keywords', 'resumo', 'abstract']
                        }
                        target_cols = col_map.get(search_scope, [])
                        existing_cols = [col for col in target_cols if col in results.columns]
                        
                        if not existing_cols:
                            st.warning(f"Nenhuma coluna correspondente a '{search_scope}' encontrada. Buscando em todas as colunas.")
                            mask = results.astype(str).apply(lambda x: x.str.contains(query, case=False, na=False)).any(axis=1)
                            results = results[mask]
                        else:
                            mask = pd.Series(False, index=results.index)
                            for col in existing_cols:
                                mask |= results[col].astype(str).str.contains(query, case=False, na=False)
                            results = results[mask]
                
                st.session_state.search_results = results
                st.session_state.search_page = 1
                
                if results.empty:
                    st.info("Nenhum resultado encontrado para sua busca.")
                else:
                    st.success(f"Encontrados {len(results)} resultados!")
            else:
                st.warning("Por favor, digite um termo de busca.")

        results_df = st.session_state.get('search_results', pd.DataFrame())
        
        if not results_df.empty:
            if st.session_state.get("search_view_index") is not None:
                # Visualização detalhada de um resultado
                vi = st.session_state.search_view_index
                if 0 <= vi < len(results_df):
                    det = results_df.iloc[vi].to_dict()
                    det = enrich_article_metadata(det)

                    st.markdown("### 📄 Detalhes do Resultado")
                    if st.button("⬅️ Voltar para resultados", key=f"search_back_{USERNAME}"):
                        st.session_state.search_view_index = None
                        safe_rerun()

                    st.markdown(f"**{escape_html(det.get('título','— Sem título —'))}**")
                    st.markdown(f"_Autor(es):_ {escape_html(det.get('autor','— —'))}")
                    st.markdown(f"_Ano:_ {escape_html(str(det.get('ano','— —')))}")
                    
                    if det.get('doi'):
                        doi_link = f"https://doi.org/{det.get('doi')}"
                        st.markdown(f"_DOI:_ [{det.get('doi')}]({doi_link})")
                    
                    st.markdown("---")
                    st.markdown(escape_html(det.get('resumo','Resumo não disponível.')))
                    
                    col_btn1, col_btn2, col_btn3 = st.columns(3)
                    with col_btn1:
                        if st.button("⭐ Adicionar aos Favoritos", use_container_width=True, key=f"fav_search_{vi}_{USERNAME}"):
                            if add_to_favorites(det):
                                st.toast("Adicionado aos favoritos!", icon="⭐")
                            else:
                                st.toast("Este artigo já está nos favoritos.")
                    with col_btn2:
                        if st.button("📝 Ver Anotações", use_container_width=True, key=f"notes_search_{vi}_{USERNAME}"):
                            st.session_state.page = "anotacoes"
                            safe_rerun()
                    with col_btn3:
                        if st.button("💡 Ver Recomendações", use_container_width=True, key=f"rec_search_{vi}_{USERNAME}"):
                            st.session_state.page = "recomendacoes"
                            st.session_state.palavra_chave_recomendacao = det.get('título', '').split()[0] if det.get('título') else ''
                            safe_rerun()

            else:
                # Lista de resultados
                per_page = 8
                total = len(results_df)
                max_pages = max(1, (total + per_page - 1) // per_page)
                page = max(1, min(st.session_state.get("search_page", 1), max_pages))
                start, end = (page - 1) * per_page, min(page * per_page, total)
                page_df = results_df.iloc[start:end]

                st.markdown(f"**📊 {total}** resultado(s) encontrado(s) — exibindo {start+1} a {end}.")

                # Obter todos os nomes de usuários de uma vez
                all_users = load_users()

                for idx, row in page_df.iterrows():
                    user_src_cpf = row.get("_artemis_username", "N/A")
                    user_src_name = all_users.get(user_src_cpf, {}).get('name', user_src_cpf)

                    title = str(row.get('título') or row.get('titulo') or '(Sem título)')
                    author_snippet = str(row.get('autor') or "")[:100]
                    year = row.get('ano') or ""
                    
                    # Destacar termos de busca
                    query_for_highlight = st.session_state.get("search_input_main", "")
                    if query_for_highlight:
                        title = highlight_search_terms(title, query_for_highlight)
                        author_snippet = highlight_search_terms(author_snippet, query_for_highlight)
                    else:
                        title = escape_html(title)
                        author_snippet = escape_html(author_snippet)

                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">{title}</div>
                        <div class="small-muted">{author_snippet}</div>
                        <div class="small-muted">Ano: {escape_html(str(year))} • <b>Fonte: {escape_html(user_src_name)}</b></div>
                    </div>""", unsafe_allow_html=True)

                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        if st.button("⭐ Favoritar", key=f"fav_{idx}_{USERNAME}", use_container_width=True):
                            if add_to_favorites(row.to_dict()):
                                st.toast("Adicionado aos favoritos!", icon="⭐")
                            else:
                                st.toast("Já está nos favoritos.")
                    with b_col2:
                        if st.button("🔎 Ver detalhes", key=f"view_{idx}_{USERNAME}", use_container_width=True):
                            st.session_state.search_view_index = idx
                            safe_rerun()
                    st.markdown("<hr style='margin-top:8px; margin-bottom:8px; border-color:#233447'>", unsafe_allow_html=True)
                
                # Navegação de páginas
                if max_pages > 1:
                    p1, p2, p3 = st.columns([1, 1, 1])
                    with p1:
                        if st.button("◀ Anterior", key=f"search_prev_{USERNAME}", disabled=(page <= 1), use_container_width=True):
                            st.session_state.search_page -= 1
                            safe_rerun()
                    with p2:
                        st.markdown(f"<div style='text-align:center; padding-top:8px'><b>Página {page} / {max_pages}</b></div>", unsafe_allow_html=True)
                    with p3:
                        if st.button("Próxima ▶", key=f"search_next_{USERNAME}", disabled=(page >= max_pages), use_container_width=True):
                            st.session_state.search_page += 1
                            safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: mensagens
# -------------------------
elif st.session_state.page == "mensagens":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("✉️ Sistema de Mensagens")

    # Abas para caixa de entrada, enviadas e nova mensagem
    tab1, tab2, tab3 = st.tabs(["📥 Caixa de Entrada", "📤 Enviadas", "📝 Nova Mensagem"])

    with tab1:
        inbox_msgs = get_user_messages(USERNAME, 'inbox')
        if not inbox_msgs:
            st.info("Nenhuma mensagem na caixa de entrada.")
        else:
            st.write(f"**{UNREAD_COUNT} mensagem(s) não lida(s)**" if UNREAD_COUNT > 0 else "Todas as mensagens lidas.")
            
            all_users = load_users()
            for msg in inbox_msgs:
                is_unread = not msg.get('read', False)
                unread_indicator = "🔵" if is_unread else "⚪"
                
                sender_name = all_users.get(msg['from'], {}).get('name', msg['from'])
                
                with st.expander(f"{unread_indicator} {msg['subject']} — De: {sender_name}", expanded=is_unread):
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
            all_users = load_users()
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
            for username, user_data in users.items():
                if username != USERNAME:
                    user_options[f"{user_data.get('name', username)} ({format_cpf_display(username)})"] = username
            
            # Pre-selecionar destinatário se for uma resposta
            default_recipient = []
            if reply_to_msg:
                sender_cpf = reply_to_msg['from']
                sender_name = users.get(sender_cpf, {}).get('name', sender_cpf)
                sender_display = f"{sender_name} ({format_cpf_display(sender_cpf)})"
                if sender_display in user_options:
                    default_recipient.append(sender_display)

            recipients = st.multiselect("Para:", options=sorted(list(user_options.keys())), default=default_recipient)
            subject = st.text_input("Assunto:", 
                                  value=f"Re: {reply_to_msg['subject']}" if reply_to_msg else "")
            body = st.text_area("Mensagem:", height=200,
                              value=f"\n\n---\nEm resposta à mensagem de {users.get(reply_to_msg['from'], {}).get('name', reply_to_msg['from'])}:\n> {reply_to_msg['body'][:500].replace(chr(10), chr(10)+'> ')}..." if reply_to_msg else "")
            
            attachment = st.file_uploader("Anexar arquivo", type=['pdf', 'docx', 'txt', 'jpg', 'png'])
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("📤 Enviar Mensagem", use_container_width=True):
                    if not recipients:
                        st.error("Selecione pelo menos um destinatário.")
                    elif not subject:
                        st.error("Digite um assunto.")
                    elif not body:
                        st.error("Digite uma mensagem.")
                    else:
                        for recipient_display in recipients:
                            recipient_username = user_options[recipient_display]
                            send_message(USERNAME, recipient_username, subject, body, attachment)
                            st.success(f"Mensagem enviada para {recipient_display.split('(')[0].strip()}!")
                        
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
# Page: configuracoes
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
        save_user_state_minimal(USER_STATE)
        st.success("Configurações aplicadas! A página será recarregada.")
        time.sleep(1)
        safe_rerun()

    # Gerenciamento de dados
    st.subheader("📊 Gerenciamento de Dados")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🗑️ Limpar Todos os Dados", type="secondary", use_container_width=True):
            if st.checkbox("CONFIRMAR: Esta ação não pode ser desfeita. Todos os seus dados serão perdidos."):
                for key in list(st.session_state.keys()):
                    if key not in ['authenticated', 'username', 'user_obj']:
                        del st.session_state[key]
                
                if USER_STATE.exists():
                    USER_STATE.unlink()
                
                st.success("Todos os dados locais foram removidos!")
                time.sleep(2)
                safe_rerun()
    
    with col4:
        import zipfile
        from io import BytesIO
        
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            state_data = {
                "notes": st.session_state.get("notes", ""),
                "favorites": st.session_state.get("favorites", []),
                "settings": st.session_state.get("settings", {}),
                "tutorial_completed": st.session_state.get("tutorial_completed", False)
            }
            zip_file.writestr("user_state.json", json.dumps(state_data, indent=2))
            
            backup_path = st.session_state.get("last_backup_path")
            if backup_path and Path(backup_path).exists():
                zip_file.write(backup_path, f"planilha_backup_{Path(backup_path).name}")
        
        st.download_button(
            "📥 Exportar Backup Completo",
            data=zip_buffer.getvalue(),
            file_name=f"nugep_pqr_backup_{USERNAME}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip",
            use_container_width=True
        )

    # Informações do sistema
    st.subheader("ℹ️ Informações do Sistema")
    
    st.write(f"**Usuário (CPF):** {format_cpf_display(USERNAME)}")
    st.write(f"**Nome:** {USER_OBJ.get('name', 'Não informado')}")
    st.write(f"**Bolsa:** {USER_OBJ.get('scholarship', 'Não informada')}")
    created_at_str = USER_OBJ.get('created_at', 'Data não disponível')
    try:
        created_at_dt = datetime.fromisoformat(created_at_str)
        st.write(f"**Cadastrado em:** {created_at_dt.strftime('%d/%m/%Y %H:%M')}")
    except:
        st.write(f"**Cadastrado em:** {created_at_str}")
    
    st.write("**Estatísticas:**")
    st.write(f"- Favoritos salvos: {len(get_session_favorites())}")
    st.write(f"- Mensagens não lidas: {UNREAD_COUNT}")
    st.write(f"- Planilha carregada: {'Sim' if st.session_state.df is not None else 'Não'}")
    
    if st.session_state.df is not None:
        st.write(f"- Registros na planilha: {len(st.session_state.df)}")
        st.write(f"- Colunas na planilha: {len(st.session_state.df.columns)}")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Finalização e salvamento automático
# -------------------------
if st.session_state.autosave and st.session_state.get('notes') is not None:
    try:
        save_user_state_minimal(USER_STATE)
    except Exception:
        pass

# -------------------------
# Rodapé
# -------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#bfc6cc; font-size:0.9em; padding:10px;'>"
    "NUGEP-PQR — Sistema de Gestão de Pesquisa e Análise | "
    "Desenvolvido para pesquisadores e bolsistas"
    "</div>", 
    unsafe_allow_html=True
)
