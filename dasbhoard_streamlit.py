Vou implementar as melhorias solicitadas na análise da planilha e no mapa mental. Aqui está o código completo atualizado:

```python
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
.stButton>button, .stDownloadButton>button {
    transition: transform 0.1s ease, opacity 0.1s ease, background-color 0.15s ease !important;
}
.stButton>button:active, .stDownloadButton>button:active {
    transform: scale(0.97);
    opacity: 0.8;
}
.node-editor { 
    background: rgba(255,255,255,0.05); 
    border-radius: 10px; 
    padding: 15px; 
    margin: 10px 0; 
    border: 1px solid rgba(255,255,255,0.1);
}
.node-preview {
    background: rgba(255,255,255,0.02);
    border-radius: 8px;
    padding: 10px;
    margin: 5px 0;
    border-left: 4px solid #6c5ce7;
}
.mindmap-3d {
    background: linear-gradient(145deg, #1a2a6c, #b21f1f, #fdbb2d);
    border-radius: 15px;
    padding: 15px;
    margin: 10px 0;
    border: 2px solid #4ECDC4;
}
.flowchart-box {
    background: rgba(255,255,255,0.1);
    border: 2px solid #FF6B6B;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
}
.three-d-effect {
    background: linear-gradient(145deg, #1a2a6c, #b21f1f);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    border: 2px solid #FECA57;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.font-config {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid rgba(255,255,255,0.1);
}
.ai-response {
    background: linear-gradient(135deg, #1a2a6c, #2c3e50);
    border-radius: 12px;
    padding: 20px;
    margin: 15px 0;
    border-left: 5px solid #4ECDC4;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.vision-analysis {
    background: linear-gradient(135deg, #2c3e50, #34495e);
    border-radius: 12px;
    padding: 20px;
    margin: 15px 0;
    border-left: 5px solid #e74c3c;
}
"""

DEFAULT_CSS = r"""
.css-1d391kg { background: linear-gradient(180deg,#071428 0%, #031926 100%) !important; }
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
# ANÁLISE INTELIGENTE MELHORADA - PLANILHA COMO UM TODO
# -------------------------
class DataAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.insights = []
        self._detect_columns()
    
    def _detect_columns(self):
        """Detecção inteligente de colunas - MAIS ROBUSTA"""
        self.column_types = {
            'authors': [],
            'years': [], 
            'titles': [],
            'abstracts': [],
            'countries': [],
            'keywords': [],
            'dates': [],
            'institutions': [],
            'journals': [],
            'urls': []
        }
        
        for col in self.df.columns:
            col_lower = col.lower()
            col_data = self.df[col].dropna()
            
            # Detecção de autores
            if any(keyword in col_lower for keyword in ['autor', 'author', 'pesquisador', 'escritor', 'writer']):
                self.column_types['authors'].append(col)
            
            # Detecção de anos
            elif any(keyword in col_lower for keyword in ['ano', 'year', 'publication']):
                self.column_types['years'].append(col)
            
            # Detecção de títulos
            elif any(keyword in col_lower for keyword in ['título', 'titulo', 'title']):
                self.column_types['titles'].append(col)
            
            # Detecção de resumos
            elif any(keyword in col_lower for keyword in ['resumo', 'abstract', 'summary']):
                self.column_types['abstracts'].append(col)
            
            # Detecção de países
            elif any(keyword in col_lower for keyword in ['país', 'pais', 'country', 'nacionalidade', 'local']):
                self.column_types['countries'].append(col)
            
            # Detecção de datas
            elif any(keyword in col_lower for keyword in ['data', 'date']):
                self.column_types['dates'].append(col)
            
            # Detecção de palavras-chave
            elif any(keyword in col_lower for keyword in ['palavra', 'keyword', 'tema', 'assunto']):
                self.column_types['keywords'].append(col)
            
            # Detecção de instituições
            elif any(keyword in col_lower for keyword in ['instituição', 'instituicao', 'universidade', 'faculdade', 'institution']):
                self.column_types['institutions'].append(col)
            
            # Detecção de periódicos
            elif any(keyword in col_lower for keyword in ['periódico', 'periodico', 'journal', 'revista']):
                self.column_types['journals'].append(col)
            
            # Detecção de URLs
            elif any(keyword in col_lower for keyword in ['url', 'link', 'doi', 'website']):
                self.column_types['urls'].append(col)
    
    def generate_comprehensive_analysis(self):
        """Gera uma análise completa e inteligente dos dados"""
        analysis = "## 🧠 ANÁLISE INTELIGENTE COMPLETA DA PLANILHA\n\n"
        
        # Resumo executivo
        analysis += self._executive_summary()
        analysis += self._data_quality_analysis()
        analysis += self._author_network_analysis()
        analysis += self._temporal_analysis_advanced()
        analysis += self._thematic_analysis_advanced()
        analysis += self._geographic_analysis_advanced()
        analysis += self._collaboration_analysis_advanced()
        analysis += self._institutional_analysis()
        analysis += self._trends_and_patterns()
        analysis += self._recommendations()
        
        return analysis
    
    def _executive_summary(self):
        """Resumo executivo inteligente"""
        text = "### 📋 RESUMO EXECUTIVO\n\n"
        
        total_records = len(self.df)
        total_columns = len(self.df.columns)
        
        text += f"**📊 Dimensões da Base**: {total_records} registros × {total_columns} colunas\n\n"
        
        # Análise de completude
        completeness = {}
        for col in self.df.columns:
            non_null = self.df[col].notna().sum()
            completeness[col] = (non_null / total_records) * 100
        
        high_completeness = sum(1 for comp in completeness.values() if comp > 80)
        medium_completeness = sum(1 for comp in completeness.values() if 50 <= comp <= 80)
        
        text += f"**✅ Qualidade dos Dados**:\n"
        text += f"- {high_completeness} colunas com alta completude (>80%)\n"
        text += f"- {medium_completeness} colunas com completude moderada (50-80%)\n"
        text += f"- {total_columns - high_completeness - medium_completeness} colunas com baixa completude\n\n"
        
        # Principais achados
        if self.column_types['authors']:
            authors_col = self.column_types['authors'][0]
            unique_authors = self._extract_unique_authors(authors_col)
            if unique_authors:
                text += f"**👥 Autores Únicos**: {len(unique_authors)} pesquisadores identificados\n"
        
        if self.column_types['years']:
            years_col = self.column_types['years'][0]
            years_data = pd.to_numeric(self.df[years_col], errors='coerce').dropna()
            if len(years_data) > 0:
                year_range = f"{int(years_data.min())}-{int(years_data.max())}"
                text += f"**📅 Período**: {year_range} ({len(years_data)} anos de dados)\n"
        
        if self.column_types['countries']:
            countries_col = self.column_types['countries'][0]
            unique_countries = self.df[countries_col].nunique()
            text += f"**🌎 Países/Regiões**: {unique_countries} localizações distintas\n"
        
        return text + "\n"
    
    def _data_quality_analysis(self):
        """Análise detalhada da qualidade dos dados"""
        text = "### 🔍 ANÁLISE DE QUALIDADE DOS DADOS\n\n"
        
        issues = []
        recommendations = []
        
        # Verificar dados duplicados
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"❌ **{duplicates} registros duplicados** encontrados")
            recommendations.append("💡 **Remova registros duplicados** para melhorar a qualidade da análise")
        
        # Verificar valores nulos
        null_counts = self.df.isnull().sum()
        high_null_cols = null_counts[null_counts > len(self.df) * 0.5]  # >50% nulos
        if len(high_null_cols) > 0:
            issues.append(f"⚠️ **{len(high_null_cols)} colunas** com mais de 50% de dados faltantes")
            recommendations.append("💡 **Considere remover ou imputar** colunas com muitos dados faltantes")
        
        # Verificar colunas essenciais
        essential_cols_present = []
        if self.column_types['authors']: essential_cols_present.append("autores")
        if self.column_types['years']: essential_cols_present.append("anos")
        if self.column_types['titles']: essential_cols_present.append("títulos")
        
        text += f"**Colunas Essenciais Identificadas**: {', '.join(essential_cols_present) if essential_cols_present else 'Nenhuma'}\n\n"
        
        if issues:
            text += "**Problemas Identificados**:\n"
            for issue in issues:
                text += f"- {issue}\n"
            text += "\n"
        
        if recommendations:
            text += "**Recomendações de Melhoria**:\n"
            for rec in recommendations:
                text += f"- {rec}\n"
            text += "\n"
        
        return text
    
    def _author_network_analysis(self):
        """Análise avançada de rede de autores"""
        text = "### 👥 ANÁLISE DE REDE DE AUTORES\n\n"
        
        if not self.column_types['authors']:
            return text + "❌ Nenhuma coluna de autores identificada para análise\n\n"
        
        authors_col = self.column_types['authors'][0]
        authors_network = self._build_authors_network(authors_col)
        
        if not authors_network['unique_authors']:
            return text + "⚠️ Dados de autores encontrados mas não foi possível extrair nomes válidos\n\n"
        
        text += f"**📈 Estatísticas da Rede**:\n"
        text += f"- **Autores únicos**: {len(authors_network['unique_authors'])}\n"
        text += f"- **Trabalhos em colaboração**: {authors_network['collaborations']}\n"
        text += f"- **Taxa de colaboração**: {authors_network['collaboration_rate']:.1f}%\n"
        text += f"- **Autor mais produtivo**: {authors_network['most_prolific_author']} ({authors_network['most_prolific_count']} trabalhos)\n\n"
        
        # Análise de centralidade
        if authors_network['collaborations'] > 0:
            text += "**🎯 Autores Centrais na Rede**:\n"
            for author, degree in authors_network['central_authors'][:5]:
                text += f"- **{author}**: {degree} conexões\n"
            text += "\n"
        
        return text
    
    def _build_authors_network(self, authors_col):
        """Constrói rede de colaboração entre autores"""
        unique_authors = set()
        author_works = {}
        collaborations = 0
        
        for authors_str in self.df[authors_col].dropna():
            if isinstance(authors_str, str):
                authors_list = self._parse_authors(authors_str)
                unique_authors.update(authors_list)
                
                # Contar trabalhos por autor
                for author in authors_list:
                    author_works[author] = author_works.get(author, 0) + 1
                
                # Verificar colaboração
                if len(authors_list) > 1:
                    collaborations += 1
        
        # Encontrar autor mais produtivo
        most_prolific = max(author_works.items(), key=lambda x: x[1]) if author_works else (None, 0)
        
        # Calcular taxa de colaboração
        total_works = len(self.df[authors_col].dropna())
        collaboration_rate = (collaborations / total_works * 100) if total_works > 0 else 0
        
        # Autores centrais (baseado na produtividade)
        central_authors = sorted(author_works.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'unique_authors': unique_authors,
            'collaborations': collaborations,
            'collaboration_rate': collaboration_rate,
            'most_prolific_author': most_prolific[0],
            'most_prolific_count': most_prolific[1],
            'central_authors': central_authors
        }
    
    def _temporal_analysis_advanced(self):
        """Análise temporal avançada"""
        text = "### 📈 ANÁLISE TEMPORAL AVANÇADA\n\n"
        
        if not self.column_types['years']:
            return text + "❌ Nenhuma coluna temporal identificada\n\n"
        
        years_col = self.column_types['years'][0]
        try:
            years = pd.to_numeric(self.df[years_col], errors='coerce').dropna()
            
            if len(years) == 0:
                return text + "⚠️ Coluna temporal encontrada mas sem valores numéricos válidos\n\n"
            
            # Análise básica
            min_year, max_year = int(years.min()), int(years.max())
            year_range = max_year - min_year
            
            text += f"**🕰️ Período Analisado**: {min_year} - {max_year} ({year_range} anos)\n"
            text += f"**📊 Registros com anos**: {len(years)}/{len(self.df)} ({len(years)/len(self.df)*100:.1f}%)\n\n"
            
            # Análise de tendência
            year_counts = years.value_counts().sort_index()
            
            if len(year_counts) > 3:
                # Calcular crescimento
                recent_5_years = [y for y in years if y >= max_year - 5]
                older_years = [y for y in years if y < max_year - 5]
                
                recent_avg = len(recent_5_years) / 5 if recent_5_years else 0
                older_avg = len(older_years) / max(1, (max_year - 5 - min_year)) if older_years else 0
                
                text += "**📈 Tendência de Produção**:\n"
                if recent_avg > older_avg * 1.3:
                    text += "- 🚀 **Crescimento Acelerado** na produção recente\n"
                elif recent_avg > older_avg:
                    text += "- 📈 **Crescimento Moderado**\n"
                elif recent_avg < older_avg * 0.7:
                    text += "- 📉 **Declínio** na produção recente\n"
                else:
                    text += "- ➡️ **Estabilidade** na produção\n"
                
                # Ano mais produtivo
                best_year = year_counts.idxmax()
                best_count = year_counts.max()
                text += f"- 🏆 **Ano de Pico**: {int(best_year)} ({best_count} publicações)\n\n"
            
            # Distribuição por década
            if year_range > 15:
                decades = (years // 10) * 10
                decade_counts = decades.value_counts().sort_index()
                text += "**🗓️ Distribuição por Década**:\n"
                for decade, count in decade_counts.items():
                    text += f"- {int(decade)}s: {int(count)} publicação(ões)\n"
                text += "\n"
            
            return text
            
        except Exception as e:
            return text + f"❌ Erro na análise temporal: {str(e)}\n\n"
    
    def _thematic_analysis_advanced(self):
        """Análise temática avançada"""
        text = "### 🔍 ANÁLISE TEMÁTICA AVANÇADA\n\n"
        
        # Combinar texto de colunas relevantes
        text_columns = []
        if self.column_types['titles']:
            text_columns.extend(self.column_types['titles'])
        if self.column_types['abstracts']:
            text_columns.extend(self.column_types['abstracts'])
        if self.column_types['keywords']:
            text_columns.extend(self.column_types['keywords'])
        
        if not text_columns:
            return text + "❌ Nenhuma coluna de texto identificada para análise temática\n\n"
        
        # Combinar texto
        combined_text = ""
        for col in text_columns[:3]:  # Limitar a 3 colunas para performance
            col_text = self.df[col].fillna('').astype(str).str.cat(sep=' ')
            if len(col_text.strip()) > 100:
                combined_text += " " + col_text
        
        if len(combined_text.strip()) < 500:
            return text + "⚠️ Texto insuficiente para análise temática robusta\n\n"
        
        # Análise de frequência de termos
        palavras = re.findall(r'\b[a-zà-ú]{4,}\b', combined_text.lower())
        stop_words = set(PORTUGUESE_STOP_WORDS)
        palavras_filtradas = [p for p in palavras if p not in stop_words and len(p) > 3]
        
        if palavras_filtradas:
            term_freq = pd.Series(palavras_filtradas).value_counts().head(15)
            
            text += "**📊 Termos Mais Frequentes**:\n"
            for term, freq in term_freq.head(10).items():
                text += f"- **{term}**: {freq} ocorrências\n"
            text += "\n"
            
            # Bigramas (combinações de 2 palavras)
            bigrams = []
            words = palavras_filtradas
            for i in range(len(words)-1):
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) > 8:  # Filtrar bigramas muito curtos
                    bigrams.append(bigram)
            
            if bigrams:
                bigram_freq = pd.Series(bigrams).value_counts().head(8)
                text += "**🔗 Principais Bigramas**:\n"
                for bigram, freq in bigram_freq.items():
                    text += f"- **{bigram}**: {freq} ocorrências\n"
                text += "\n"
        
        return text
    
    def _geographic_analysis_advanced(self):
        """Análise geográfica avançada"""
        text = "### 🌎 ANÁLISE GEOGRÁFICA AVANÇADA\n\n"
        
        if not self.column_types['countries']:
            return text + "❌ Nenhuma coluna geográfica identificada\n\n"
        
        countries_col = self.column_types['countries'][0]
        countries = self.df[countries_col].dropna()
        
        if len(countries) == 0:
            return text + "⚠️ Coluna geográfica encontrada mas sem dados válidos\n\n"
        
        country_counts = countries.value_counts()
        
        text += f"**🗺️ Distribuição Geográfica** ({len(country_counts)} países/regiões):\n"
        for country, count in country_counts.head(10).items():
            percentage = (count / len(countries)) * 100
            text += f"- **{country}**: {count} ({percentage:.1f}%)\n"
        
        # Diversidade geográfica
        diversity_index = (len(country_counts) / len(countries)) * 100
        text += f"\n**🌍 Diversidade Geográfica**: {diversity_index:.1f}%\n"
        
        if len(country_counts) == 1:
            text += "- 🎯 **Foco Geográfico**: Concentrado em uma única região\n"
        elif len(country_counts) <= 3:
            text += "- 🎯 **Foco Geográfico**: Poucos países/regiões\n"
        elif len(country_counts) <= 8:
            text += "- 🌐 **Foco Geográfico**: Boa diversidade internacional\n"
        else:
            text += "- 🌐 **Foco Geográfico**: Excelente abrangência internacional\n"
        
        text += "\n"
        return text
    
    def _collaboration_analysis_advanced(self):
        """Análise de colaboração avançada"""
        text = "### 🤝 ANÁLISE DE COLABORAÇÃO AVANÇADA\n\n"
        
        if not self.column_types['authors']:
            return text + "❌ Dados de autores necessários para análise de colaboração\n\n"
        
        authors_col = self.column_types['authors'][0]
        collaboration_data = self._analyze_collaboration_patterns(authors_col)
        
        text += f"**📊 Estatísticas de Colaboração**:\n"
        text += f"- Trabalhos individuais: {collaboration_data['individual_works']}\n"
        text += f"- Trabalhos em colaboração: {collaboration_data['collaborative_works']}\n"
        text += f"- Taxa de colaboração: {collaboration_data['collaboration_rate']:.1f}%\n\n"
        
        if collaboration_data['top_collaborators']:
            text += "**👥 Maiores Colaboradores**:\n"
            for author, collab_count in collaboration_data['top_collaborators'][:5]:
                text += f"- **{author}**: {collab_count} colaborações\n"
            text += "\n"
        
        # Análise de padrões
        if collaboration_data['collaboration_rate'] > 60:
            text += "🎯 **Padrão**: Alta colaboração - rede de pesquisa muito ativa\n"
        elif collaboration_data['collaboration_rate'] > 30:
            text += "🎯 **Padrão**: Boa colaboração - trabalho em equipe presente\n"
        else:
            text += "💡 **Oportunidade**: Espaço para aumentar colaborações\n"
        
        text += "\n"
        return text
    
    def _institutional_analysis(self):
        """Análise institucional"""
        text = "### 🏛️ ANÁLISE INSTITUCIONAL\n\n"
        
        if not self.column_types['institutions']:
            return text + "❌ Nenhuma coluna institucional identificada\n\n"
        
        institutions_col = self.column_types['institutions'][0]
        institutions = self.df[institutions_col].dropna()
        
        if len(institutions) == 0:
            return text + "⚠️ Coluna institucional encontrada mas sem dados válidos\n\n"
        
        institution_counts = institutions.value_counts().head(10)
        
        text += "**🏫 Instituições Mais Produtivas**:\n"
        for institution, count in institution_counts.items():
            text += f"- **{institution}**: {count} publicações\n"
        
        text += f"\n**Total de instituições únicas**: {institutions.nunique()}\n\n"
        
        return text
    
    def _analyze_collaboration_patterns(self, authors_col):
        """Analisa padrões de colaboração"""
        individual_works = 0
        collaborative_works = 0
        author_collaborations = {}
        
        for authors_str in self.df[authors_col].dropna():
            if isinstance(authors_str, str):
                authors_list = self._parse_authors(authors_str)
                
                if len(authors_list) == 1:
                    individual_works += 1
                else:
                    collaborative_works += 1
                    
                    # Contar colaborações por autor
                    for author in authors_list:
                        author_collaborations[author] = author_collaborations.get(author, 0) + 1
        
        total_works = individual_works + collaborative_works
        collaboration_rate = (collaborative_works / total_works * 100) if total_works > 0 else 0
        
        top_collaborators = sorted(author_collaborations.items(), key=lambda x: x[1], reverse=True)[:8]
        
        return {
            'individual_works': individual_works,
            'collaborative_works': collaborative_works,
            'collaboration_rate': collaboration_rate,
            'top_collaborators': top_collaborators
        }
    
    def _trends_and_patterns(self):
        """Identifica tendências e padrões nos dados"""
        text = "### 🔮 TENDÊNCIAS E PADRÕES IDENTIFICADOS\n\n"
        
        patterns = []
        
        # Padrão de colaboração
        if self.column_types['authors']:
            authors_col = self.column_types['authors'][0]
            collab_data = self._analyze_collaboration_patterns(authors_col)
            if collab_data['collaboration_rate'] > 50:
                patterns.append("🤝 **Alta colaboração** entre pesquisadores")
            else:
                patterns.append("👤 **Produção individual** predominante")
        
        # Padrão temporal
        if self.column_types['years']:
            years_col = self.column_types['years'][0]
            years = pd.to_numeric(self.df[years_col], errors='coerce').dropna()
            if len(years) > 5:
                recent = years[years >= datetime.now().year - 5]
                if len(recent) > len(years) * 0.6:
                    patterns.append("🚀 **Produção recente** intensa")
        
        # Padrão geográfico
        if self.column_types['countries']:
            countries_col = self.column_types['countries'][0]
            unique_countries = self.df[countries_col].nunique()
            if unique_countries == 1:
                patterns.append("🎯 **Foco geográfico** concentrado")
            elif unique_countries > 5:
                patterns.append("🌐 **Abrangência internacional** significativa")
        
        if patterns:
            for pattern in patterns:
                text += f"- {pattern}\n"
        else:
            text += "ℹ️ Padrões serão identificados com mais dados\n"
        
        text += "\n"
        return text
    
    def _recommendations(self):
        """Recomendações inteligentes baseadas nos dados"""
        text = "### 💡 RECOMENDAÇÕES INTELIGENTES\n\n"
        
        recommendations = []
        
        # Recomendações baseadas no tamanho da base
        total_records = len(self.df)
        if total_records < 20:
            recommendations.extend([
                "📥 **Amplie a base** com mais registros para análises estatísticas confiáveis",
                "🔍 **Use a busca integrada** para encontrar trabalhos relacionados",
                "📊 **Foque em análises descritivas** básicas por enquanto"
            ])
        elif total_records < 50:
            recommendations.extend([
                "📈 **Explore tendências temporais** com os dados atuais",
                "🤝 **Analise redes de colaboração** entre autores",
                "🗺️ **Use o mapa mental** para organizar conceitos principais"
            ])
        else:
            recommendations.extend([
                "🔬 **Realize análises avançadas** de clusters e padrões",
                "🌐 **Explore redes complexas** de coautoria",
                "📚 **Use o sistema de recomendação** para descobrir novos artigos"
            ])
        
        # Recomendações específicas baseadas nos dados
        if not self.column_types['authors']:
            recommendations.append("👥 **Adicione dados de autores** para análise de redes de colaboração")
        
        if not self.column_types['years']:
            recommendations.append("📅 **Inclua informações temporais** para análise de tendências")
        
        if len(self.df.columns) < 5:
            recommendations.append("📋 **Enriqueça a base** com mais metadados (resumos, palavras-chave, etc)")
        
        for i, rec in enumerate(recommendations, 1):
            text += f"{i}. {rec}\n"
        
        text += "\n"
        return text
    
    def _parse_authors(self, authors_str):
        """Parser robusto de autores"""
        if not isinstance(authors_str, str):
            return []
        
        # Múltiplas estratégias de parsing
        separators = [';', ',', ' e ', ' and ', '&']
        
        for sep in separators:
            if sep in authors_str:
                authors = [a.strip() for a in authors_str.split(sep) if a.strip()]
                if len(authors) > 1:
                    # Filtrar autores válidos
                    valid_authors = []
                    for author in authors:
                        if (len(author) > 2 and 
                            not author.isdigit() and
                            author.lower() not in ['', 'e', 'and', 'et', 'de', 'da', 'do']):
                            valid_authors.append(author)
                    return valid_authors
        
        # Se não encontrou separadores, retorna o autor único se for válido
        author_clean = authors_str.strip()
        if (len(author_clean) > 2 and 
            not author_clean.isdigit() and
            author_clean.lower() not in ['', 'e', 'and', 'et']):
            return [author_clean]
        
        return []
    
    def _extract_unique_authors(self, authors_col):
        """Extrai autores únicos de forma robusta"""
        unique_authors = set()
        
        for authors_str in self.df[authors_col].dropna():
            authors_list = self._parse_authors(authors_str)
            unique_authors.update(authors_list)
        
        return unique_authors

# -------------------------
# MAPA MENTAL SIMPLIFICADO - SEM 3D
# -------------------------
class AdvancedMindMap:
    def __init__(self):
        self.node_types = {
            "ideia": {"color": "#4ECDC4", "icon": "💡", "shape": "dot", "size": 25},
            "tarefa": {"color": "#45B7D1", "icon": "✅", "shape": "square", "size": 30},
            "pergunta": {"color": "#96CEB4", "icon": "❓", "shape": "diamond", "size": 28},
            "recurso": {"color": "#FECA57", "icon": "📚", "shape": "triangle", "size": 32},
            "objetivo": {"color": "#FF6B6B", "icon": "🎯", "shape": "star", "size": 35},
            "nota": {"color": "#A29BFE", "icon": "📝", "shape": "circle", "size": 22},
            "problema": {"color": "#FF9FF3", "icon": "⚠️", "shape": "hexagon", "size": 33},
            "solucao": {"color": "#54A0FF", "icon": "🔧", "shape": "database", "size": 31}
        }
        
        self.connection_types = {
            "relacionado": {"color": "#74B9FF", "style": "solid"},
            "depende": {"color": "#FF6B6B", "style": "dashed"},
            "contem": {"color": "#00D2D3", "style": "solid"},
            "leva_a": {"color": "#F368E0", "style": "arrow"},
            "contradiz": {"color": "#FF9F43", "style": "dotted"},
            "suporta": {"color": "#10AC84", "style": "solid"}
        }
    
    def create_node(self, node_id, label, node_type="ideia", description="", x=None, y=None):
        """Cria um nó com posicionamento inteligente"""
        node_data = self.node_types.get(node_type, self.node_types["ideia"])
        
        if x is None or y is None:
            x, y = self._calculate_smart_position()
        
        return {
            "id": node_id,
            "label": f"{node_data['icon']} {label}",
            "type": node_type,
            "description": description,
            "color": node_data["color"],
            "shape": node_data["shape"],
            "size": node_data["size"],
            "x": x,
            "y": y,
            "font": {"color": "#FFFFFF", "size": 14, "face": "Arial"},
            "created_at": datetime.now().isoformat()
        }
    
    def _calculate_smart_position(self, existing_nodes=None):
        """Calcula posição inteligente para novo nó"""
        if not existing_nodes:
            return random.randint(300, 700), random.randint(200, 500)
        
        # Buscar área menos congestionada
        center_x, center_y = 500, 350
        for radius in range(100, 801, 100):
            for angle in range(0, 360, 30):
                rad = math.radians(angle)
                x = center_x + radius * math.cos(rad)
                y = center_y + radius * math.sin(rad)
                
                # Verificar se está longe o suficiente de outros nós
                too_close = any(
                    math.sqrt((x - n.get("x", 0))**2 + (y - n.get("y", 0))**2) < 120 
                    for n in existing_nodes
                )
                
                if not too_close:
                    return x, y
        
        # Fallback: posição aleatória
        return random.randint(200, 800), random.randint(150, 550)
    
    def create_connection(self, source_id, target_id, connection_type="relacionado", label=""):
        """Cria uma conexão entre nós"""
        conn_data = self.connection_types.get(connection_type, self.connection_types["relacionado"])
        
        return {
            "source": source_id,
            "target": target_id,
            "type": connection_type,
            "label": label or connection_type,
            "color": conn_data["color"],
            "style": conn_data["style"],
            "width": 3
        }
    
    def auto_layout(self, nodes, edges, layout_type="hierarchical"):
        """Layout automático inteligente SEM 3D"""
        if not nodes:
            return nodes
        
        G = nx.Graph()
        
        # Adicionar nós
        for node in nodes:
            G.add_node(node["id"])
        
        # Adicionar arestas
        for edge in edges:
            G.add_edge(edge["source"], edge["target"])
        
        try:
            if layout_type == "hierarchical":
                pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
            elif layout_type == "radial":
                pos = nx.circular_layout(G)
            elif layout_type == "force":
                pos = nx.spring_layout(G, k=3, iterations=150, seed=42)
            elif layout_type == "circular":
                pos = nx.circular_layout(G)
            else:
                pos = nx.random_layout(G, seed=42)
            
            # Aplicar posições
            for node in nodes:
                if node["id"] in pos:
                    node["x"] = pos[node["id"]][0] * 800 + 400
                    node["y"] = pos[node["id"]][1] * 600 + 300
                    
        except Exception as e:
            print(f"Layout automático falhou: {e}")
            # Fallback para grid
            for i, node in enumerate(nodes):
                node["x"] = 400 + (i % 4) * 200
                node["y"] = 300 + (i // 4) * 150
        
        return nodes
    
    def export_mindmap(self, nodes, edges, format_type="json"):
        """Exporta o mapa mental"""
        if format_type == "json":
            return json.dumps({
                "nodes": nodes,
                "edges": edges,
                "exported_at": datetime.now().isoformat(),
                "version": "2.0"
            }, indent=2)
        
        elif format_type == "text":
            text = "MAPEAMENTO MENTAL - EXPORTAÇÃO\n"
            text += "=" * 40 + "\n\n"
            
            for node in nodes:
                text += f"● {node['label']}\n"
                if node.get('description'):
                    text += f"  Descrição: {node['description']}\n"
                
                # Conexões deste nó
                connections = []
                for edge in edges:
                    if edge['source'] == node['id']:
                        target_node = next((n for n in nodes if n['id'] == edge['target']), None)
                        if target_node:
                            connections.append(f"→ {target_node['label']} ({edge['type']})")
                    elif edge['target'] == node['id']:
                        source_node = next((n for n in nodes if n['id'] == edge['source']), None)
                        if source_node:
                            connections.append(f"← {source_node['label']} ({edge['type']})")
                
                if connections:
                    text += "  Conexões: " + ", ".join(connections[:3])
                    if len(connections) > 3:
                        text += f" ... (+{len(connections)-3} mais)"
                    text += "\n"
                
                text += "\n"
            
            return text
        
        return ""

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
    # CORREÇÃO: Remover CPF dos dados salvos
    clean_data = {k: v for k, v in result_data.items() if k != '_artemis_username'}
    favorite_item = {"id": f"{int(time.time())}_{random.randint(1000,9999)}", "data": clean_data, "added_at": datetime.utcnow().isoformat()}
    temp_data_to_check = {k: v for k, v in clean_data.items() if k not in ['similarity']}
    existing_contents = [json.dumps({k: v for k, v in fav["data"].items() if k not in ['similarity']}, sort_keys=True) for fav in favorites]
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
    "settings": {
        "plot_height": 720, "font_scale": 1.0, "node_opacity": 1.0,
        "font_size": 14,
        "node_font_size": 14,
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
# Inicializar componentes avançados
# -------------------------
advanced_mindmap = AdvancedMindMap()

# -------------------------
# Authentication UI (login & register)
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
nav_buttons = {"planilha": "📄 Planilha", "recomendacoes": "💡 Recomendações", "favoritos": "⭐ Favoritos", "mapa": "🗺️ Mapa Mental",
               "anotacoes": "📝 Anotações", "graficos": "📊 Análise", "busca": "🔍 Busca",
               "mensagens": f"✉️ Mensagens ({UNREAD_COUNT})" if UNREAD_COUNT > 0 else "✉️ Mensagens", "config": "⚙️ Configurações"}
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
        * **⭐ Favoritos**: Acesse todos os seus artigos favoritados em um só lugar.
        * **🗺️ Mapa Mental**: Visualize e edite mapas mentais e fluxogramas interativos para organizar ideias.
        * **📝 Anotações**: Um bloco de notas para destacar texto com `==sinais de igual==` e exportar como PDF.
        * **📊 Análise**: Gere gráficos e análises inteligentes a partir da sua planilha.
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
                all_users = load_users()
                user_name = all_users.get(user_src, {}).get('name', user_src)
                initials = "".join([p[0] for p in str(user_name).split()[:2]]).upper() or "U"
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
                            <div class="small-muted">De <strong>{escape_html(user_name)}</strong> • {escape_html(author_snippet)}</div>
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
# NOVA PÁGINA: FAVORITOS
# -------------------------
elif st.session_state.page == "favoritos":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("⭐ Seus Artigos Favoritos")
    
    favorites = get_session_favorites()
    
    if not favorites:
        st.info("🌟 Você ainda não tem favoritos. Adicione artigos interessantes das abas 'Recomendações' ou 'Busca'!")
        st.markdown("""
        **💡 Como adicionar favoritos:**
        - Na aba **Recomendações**: Clique em "⭐ Favoritar" em qualquer artigo
        - Na aba **Busca**: Clique em "⭐ Favoritar" nos resultados da busca
        - Os favoritos ficam salvos mesmo depois de sair do sistema
        """)
    else:
        st.success(f"📚 Você tem {len(favorites)} artigo(s) favoritado(s)!")
        
        # Opções de organização
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            sort_option = st.selectbox("Ordenar por:", 
                                     ["Mais recentes", "Mais antigos", "Título (A-Z)", "Título (Z-A)"],
                                     key="favorites_sort")
        with col2:
            filter_source = st.selectbox("Filtrar por fonte:", 
                                       ["Todas", "Recomendações", "Busca", "Web"],
                                       key="favorites_filter")
        with col3:
            if st.button("🗑️ Limpar Todos", type="secondary", use_container_width=True):
                if st.checkbox("Confirmar limpeza de TODOS os favoritos?"):
                    clear_all_favorites()
                    st.success("Todos os favoritos foram removidos!")
                    safe_rerun()
        
        # Aplicar filtros e ordenação
        filtered_favorites = favorites.copy()
        
        # Filtro por fonte
        if filter_source != "Todas":
            if filter_source == "Recomendações":
                filtered_favorites = [f for f in filtered_favorites if f["data"].get("similarity")]
            elif filter_source == "Busca":
                filtered_favorites = [f for f in filtered_favorites if not f["data"].get("similarity") and not f["data"].get("_tema_origem")]
            elif filter_source == "Web":
                filtered_favorites = [f for f in filtered_favorites if f["data"].get("_tema_origem")]
        
        # Ordenação
        if sort_option == "Mais recentes":
            filtered_favorites.sort(key=lambda x: x['added_at'], reverse=True)
        elif sort_option == "Mais antigos":
            filtered_favorites.sort(key=lambda x: x['added_at'])
        elif sort_option == "Título (A-Z)":
            filtered_favorites.sort(key=lambda x: x['data'].get('título', '').lower())
        elif sort_option == "Título (Z-A)":
            filtered_favorites.sort(key=lambda x: x['data'].get('título', '').lower(), reverse=True)
        
        # Exibir favoritos
        for fav in filtered_favorites:
            fav_data = fav['data']
            
            # Determinar tipo de fonte
            source_type = "🔍 Busca"
            if fav_data.get("similarity"):
                source_type = "💡 Recomendações"
            elif fav_data.get("_tema_origem"):
                source_type = "🌐 Web"
            
            st.markdown(f"""
            <div class="card">
                <div class="card-title">{escape_html(fav_data.get('título', '(Sem título)'))}</div>
                <div class="small-muted">
                    {source_type} • Adicionado em {datetime.fromisoformat(fav['added_at']).strftime('%d/%m/%Y %H:%M')}
                </div>
                <div class="small-muted">
                    {escape_html(fav_data.get('autor', 'Autor não informado'))} • {escape_html(str(fav_data.get('ano', 'Ano não informado')))}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("📖 Ver Detalhes", key=f"view_fav_{fav['id']}", use_container_width=True):
                    st.session_state.fav_detail = fav_data
            with col2:
                if st.button("📝 Anotações", key=f"notes_fav_{fav['id']}", use_container_width=True):
                    st.session_state.page = "anotacoes"
                    safe_rerun()
            with col3:
                if st.button("❌ Remover", key=f"remove_fav_{fav['id']}", use_container_width=True):
                    remove_from_favorites(fav['id'])
                    st.success("Favorito removido!")
                    safe_rerun()
            
            st.markdown("---")
        
        # Visualização de detalhes
        if 'fav_detail' in st.session_state and st.session_state.fav_detail:
            det_fav = st.session_state.pop("fav_detail")
            det_fav = enrich_article_metadata(det_fav)
            
            st.markdown("## 📄 Detalhes do Favorito")
            st.markdown(f"**{escape_html(det_fav.get('título','— Sem título —'))}**")
            st.markdown(f"**Autor(es):** {escape_html(det_fav.get('autor','— —'))}")
            st.markdown(f"**Ano:** {escape_html(str(det_fav.get('ano','— —')))}")
            
            if det_fav.get('doi'):
                doi_link = f"https://doi.org/{det_fav.get('doi')}"
                st.markdown(f"**DOI:** [{det_fav.get('doi')}]({doi_link})")
            
            st.markdown("---")
            st.markdown("**Resumo**")
            st.markdown(escape_html(det_fav.get('resumo','Resumo não disponível.')))
            
            if st.button("⬅️ Voltar para lista de favoritos"):
                safe_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: mapa mental - SIMPLIFICADO SEM 3D
# -------------------------
elif st.session_state.page == "mapa":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("🗺️ Mapa Mental Interativo")
    st.info("💡 **Crie, conecte e visualize suas ideias** - Sistema simplificado e intuitivo!")
    
    # Inicializar sistema de mapa mental
    if 'advanced_mindmap_nodes' not in st.session_state:
        st.session_state.advanced_mindmap_nodes = []
        st.session_state.advanced_mindmap_edges = []
        st.session_state.advanced_mindmap_selected_node = None
        st.session_state.advanced_mindmap_layout = "hierarchical"
    
    # Sidebar simplificada
    with st.sidebar:
        st.header("🎨 Controles do Mapa")
        
        # Criar nova ideia
        with st.expander("➕ Nova Ideia", expanded=True):
            with st.form("create_advanced_node", clear_on_submit=True):
                node_label = st.text_input("Título da ideia:", placeholder="Ex: Pesquisa Qualitativa")
                node_type = st.selectbox("Tipo:", options=list(advanced_mindmap.node_types.keys()))
                node_desc = st.text_area("Descrição:", placeholder="Detalhes sobre esta ideia...", height=100)
                
                if st.form_submit_button("🎯 Adicionar Ideia", use_container_width=True):
                    if node_label:
                        node_id = f"node_{int(time.time())}_{random.randint(1000,9999)}"
                        new_node = advanced_mindmap.create_node(node_id, node_label, node_type, node_desc)
                        st.session_state.advanced_mindmap_nodes.append(new_node)
                        st.session_state.advanced_mindmap_selected_node = node_id
                        st.success("Ideia criada!")
                        safe_rerun()
        
        # Layout simplificado - SEM 3D
        with st.expander("🔄 Organização", expanded=False):
            layout_options = {
                "Hierárquico": "hierarchical",
                "Força": "force", 
                "Circular": "circular",
                "Radial": "radial"
            }
            
            selected_layout = st.selectbox("Layout:", options=list(layout_options.keys()))
            st.session_state.advanced_mindmap_layout = layout_options[selected_layout]
            
            if st.button("🔄 Reorganizar Mapa", use_container_width=True):
                st.session_state.advanced_mindmap_nodes = advanced_mindmap.auto_layout(
                    st.session_state.advanced_mindmap_nodes, 
                    st.session_state.advanced_mindmap_edges, 
                    st.session_state.advanced_mindmap_layout
                )
                st.success("Mapa reorganizado!")
                safe_rerun()
        
        # Exportação
        with st.expander("💾 Exportar", expanded=False):
            if st.button("📥 Exportar JSON", use_container_width=True):
                export_data = advanced_mindmap.export_mindmap(
                    st.session_state.advanced_mindmap_nodes,
                    st.session_state.advanced_mindmap_edges,
                    "json"
                )
                st.download_button(
                    "💾 Baixar JSON",
                    data=export_data,
                    file_name=f"mapa_mental_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
    
    # Área principal do mapa - VISUALIZAÇÃO SIMPLIFICADA
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎨 Visualização do Mapa")
        
        if st.session_state.advanced_mindmap_nodes:
            # Preparar nós e arestas
            nodes_for_viz = []
            for node in st.session_state.advanced_mindmap_nodes:
                node_data = advanced_mindmap.node_types.get(node["type"], advanced_mindmap.node_types["ideia"])
                
                nodes_for_viz.append(
                    Node(
                        id=node["id"],
                        label=node["label"],
                        size=node.get("size", node_data["size"]),
                        color=node["color"],
                        shape=node.get("shape", node_data["shape"]),
                        font={"color": "#FFFFFF", "size": 14},
                        x=node.get("x", 0),
                        y=node.get("y", 0)
                    )
                )

            edges_for_viz = []
            for edge in st.session_state.advanced_mindmap_edges:
                edges_for_viz.append(
                    Edge(
                        source=edge["source"],
                        target=edge["target"],
                        label=edge.get("label", ""),
                        color=edge["color"],
                        width=3
                    )
                )

            # Configuração SIMPLIFICADA - SEM 3D
            config = Config(
                width=800,
                height=600,
                directed=True,
                physics=True,
                hierarchical=False if st.session_state.advanced_mindmap_layout != "hierarchical" else {
                    "enabled": True,
                    "levelSeparation": 150,
                    "nodeSpacing": 100,
                    "treeSpacing": 200,
                    "direction": "UD"
                }
            )

            try:
                return_value = agraph(nodes=nodes_for_viz, edges=edges_for_viz, config=config)
                if return_value:
                    st.session_state.advanced_mindmap_selected_node = return_value
            except Exception as e:
                st.error(f"Erro ao renderizar o mapa: {e}")

        else:
            st.info("🌟 **Comece criando sua primeira ideia!** Use o painel à esquerda para adicionar ideias.")
            st.markdown("""
            **💡 Dicas para uso eficiente:**
            - Use **diferentes tipos** de nós para categorizar suas ideias
            - **Conecte ideias** com diferentes tipos de relações
            - Experimente **diferentes layouts** para visualizações alternativas
            - **Exporte** seu mapa para backup ou compartilhamento
            """)

    with col2:
        st.subheader("📋 Lista de Ideias")
        
        # Lista de ideias existentes
        if st.session_state.advanced_mindmap_nodes:
            st.write(f"**{len(st.session_state.advanced_mindmap_nodes)} ideias no mapa:**")
            
            for node in st.session_state.advanced_mindmap_nodes:
                is_selected = st.session_state.advanced_mindmap_selected_node == node["id"]
                node_icon = advanced_mindmap.node_types[node["type"]]["icon"]
                status_icon = "🟢" if is_selected else "⚪"
                
                with st.expander(f"{status_icon} {node_icon} {node['label']}", expanded=is_selected):
                    st.write(f"**Tipo:** {node['type']}")
                    if node.get('description'):
                        st.write(f"**Descrição:** {node['description']}")
                    
                    # Mostrar conexões
                    connections = []
                    for edge in st.session_state.advanced_mindmap_edges:
                        if edge['source'] == node['id']:
                            target_node = next((n for n in st.session_state.advanced_mindmap_nodes if n['id'] == edge['target']), None)
                            if target_node:
                                connections.append(f"→ {target_node['label']} ({edge['type']})")
                        elif edge['target'] == node['id']:
                            source_node = next((n for n in st.session_state.advanced_mindmap_nodes if n['id'] == edge['source']), None)
                            if source_node:
                                connections.append(f"← {source_node['label']} ({edge['type']})")
                    
                    if connections:
                        st.write("**Conexões:**")
                        for conn in connections[:5]:
                            st.write(f"• {conn}")
                        if len(connections) > 5:
                            st.write(f"... e mais {len(connections) - 5} conexões")
                    else:
                        st.write("_Sem conexões ainda_")
                    
                    # Botões de ação
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("✏️ Editar", key=f"edit_adv_{node['id']}", use_container_width=True):
                            st.session_state.editing_advanced_node = node['id']
                            safe_rerun()
                    
                    with col_btn2:
                        if st.button("🗑️ Excluir", key=f"delete_adv_{node['id']}", use_container_width=True):
                            st.session_state.advanced_mindmap_nodes = [n for n in st.session_state.advanced_mindmap_nodes if n['id'] != node['id']]
                            st.session_state.advanced_mindmap_edges = [e for e in st.session_state.advanced_mindmap_edges if e['source'] != node['id'] and e['target'] != node['id']]
                            if st.session_state.advanced_mindmap_selected_node == node['id']:
                                st.session_state.advanced_mindmap_selected_node = None
                            st.success("Ideia removida!")
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
# Page: graficos - COM ANÁLISE INTELIGENTE MELHORADA
# -------------------------
elif st.session_state.page == "graficos":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("📊 Análise e Visualização de Dados Avançada")
    
    if st.session_state.df is None or st.session_state.df.empty:
        st.info("📁 Carregue uma planilha na aba 'Planilha' para ver análises e gráficos.")
    else:
        df = st.session_state.df
        
        # ANÁLISE INTELIGENTE AUTOMÁTICA - USANDO A NOVA CLASSE
        st.subheader("🤖 Análise Inteligente da Planilha")
        
        if st.button("🔍 Gerar Análise Completa da Planilha", use_container_width=True):
            with st.spinner("Analisando dados de forma inteligente... Isso pode levar alguns segundos"):
                analyzer = DataAnalyzer(df)
                analysis = analyzer.generate_comprehensive_analysis()
                st.markdown(analysis)
        
        st.markdown("---")
        
        # VISUALIZAÇÕES GRÁFICAS INTELIGENTES
        st.subheader("📈 Visualizações Gráficas Inteligentes")
        
        # Detecção automática de tipos de dados
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Sugestões automáticas baseadas nos dados
        if numeric_cols and categorical_cols:
            st.info("🎯 **Sugestões automáticas baseadas em seus dados:**")
            
            col_sug1, col_sug2, col_sug3 = st.columns(3)
            
            with col_sug1:
                if st.button(f"📊 {categorical_cols[0]} vs {numeric_cols[0] if numeric_cols else 'contagem'}", use_container_width=True):
                    st.session_state.auto_chart = {'x': categorical_cols[0], 'y': numeric_cols[0] if numeric_cols else None, 'type': 'bar'}
            
            with col_sug2:
                if len(numeric_cols) >= 2 and st.button(f"📈 {numeric_cols[0]} vs {numeric_cols[1]}", use_container_width=True):
                    st.session_state.auto_chart = {'x': numeric_cols[0], 'y': numeric_cols[1], 'type': 'line'}
            
            with col_sug3:
                if st.button(f"🥧 Distribuição de {categorical_cols[0]}", use_container_width=True):
                    st.session_state.auto_chart = {'x': categorical_cols[0], 'y': None, 'type': 'pie'}
        
        chart_type = st.selectbox("Escolha o tipo de gráfico:", 
                                ["Barras", "Linhas", "Pizza", "Histograma", "Dispersão"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Seleção inteligente de eixo X
            x_options = categorical_cols + numeric_cols
            x_axis = st.selectbox("Eixo X (Categoria):", options=x_options)
        
        with col2:
            if chart_type in ["Barras", "Linhas", "Dispersão"]:
                y_options = [None] + numeric_cols
                y_axis = st.selectbox("Eixo Y (Valores):", options=y_options)
            else:
                y_axis = None
        
        # Configurações avançadas
        with st.expander("⚙️ Configurações Avançadas"):
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                top_n = st.slider("Top N categorias:", min_value=5, max_value=50, value=15)
                opacity = st.slider("Opacidade:", min_value=0.3, max_value=1.0, value=0.8)
            with col_adv2:
                color_theme = st.selectbox("Tema de cores:", options=["Viridis", "Plasma", "Inferno", "Magma", "Cividis"])
                show_grid = st.checkbox("Mostrar grade", value=True)
        
        try:
            # Gráficos inteligentes com detecção automática
            if chart_type == "Barras":
                if y_axis: # Se o usuário selecionou um eixo Y, agregue os dados
                    if df[y_axis].dtype in ['int64', 'float64']:
                        # Agrupar e ordenar
                        grouped_df = df.groupby(x_axis)[y_axis].sum().reset_index()
                        grouped_df = grouped_df.sort_values(by=y_axis, ascending=False).head(top_n)
                        
                        fig = px.bar(grouped_df, x=x_axis, y=y_axis, 
                                   title=f"Soma de '{y_axis}' por '{x_axis}'",
                                   color=y_axis, color_continuous_scale=color_theme.lower())
                        fig.update_traces(opacity=opacity)
                    else:
                        st.warning(f"Para agregar, o Eixo Y ('{y_axis}') deve ser numérico.")
                        fig = None
                else: # Se não, faça uma contagem de frequência no eixo X
                    value_counts = df[x_axis].value_counts().head(top_n)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title=f"Contagem de Ocorrências em '{x_axis}'", 
                               labels={'x': x_axis, 'y': 'Contagem'},
                               color=value_counts.values, color_continuous_scale=color_theme.lower())
                    fig.update_traces(opacity=opacity)
                
                if fig:
                    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
                    if show_grid:
                        fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Linhas":
                if y_axis and df[y_axis].dtype in ['int64', 'float64']:
                    # Verificar se o eixo X é temporal
                    if df[x_axis].dtype in ['int64', 'float64'] or any(keyword in x_axis.lower() for keyword in ['ano', 'year', 'data']):
                        df_sorted = df.sort_values(by=x_axis)
                        fig = px.line(df_sorted, x=x_axis, y=y_axis, 
                                    title=f"'{y_axis}' ao longo de '{x_axis}'",
                                    markers=True)
                        fig.update_traces(line=dict(width=3), opacity=opacity)
                    else:
                        # Agrupar por categoria
                        grouped_df = df.groupby(x_axis)[y_axis].mean().reset_index()
                        fig = px.line(grouped_df, x=x_axis, y=y_axis,
                                    title=f"Média de '{y_axis}' por '{x_axis}'",
                                    markers=True)
                        fig.update_traces(line=dict(width=3), opacity=opacity)
                    
                    if show_grid:
                        fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Para gráfico de linhas, o eixo Y deve ser uma coluna numérica.")
            
            elif chart_type == "Pizza":
                value_counts = df[x_axis].value_counts().head(top_n)
                fig = px.pie(values=value_counts.values, names=value_counts.index, 
                           title=f"Distribuição de '{x_axis}'",
                           color_discrete_sequence=px.colors.sequential.Viridis)
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Histograma":
                if x_axis in numeric_cols:
                    fig = px.histogram(df, x=x_axis, 
                                     title=f"Distribuição de '{x_axis}'",
                                     nbins=20, opacity=opacity)
                    if show_grid:
                        fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Para histograma, selecione uma coluna numérica.")
            
            elif chart_type == "Dispersão":
                if x_axis in numeric_cols and y_axis in numeric_cols:
                    fig = px.scatter(df, x=x_axis, y=y_axis,
                                   title=f"Relação entre '{x_axis}' e '{y_axis}'",
                                   opacity=opacity,
                                   trendline="lowess")
                    if show_grid:
                        fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Para gráfico de dispersão, ambas as colunas devem ser numéricas.")
        
        except Exception as e:
            st.error(f"Erro ao gerar gráfico: {e}")
            st.info("Tente selecionar diferentes colunas ou tipos de gráfico.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: busca
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
                    
                    col_btn1, col_btn2 = st.columns(2)
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

            else:
                # Lista de resultados
                per_page = 8
                total = len(results_df)
                max_pages = max(1, (total + per_page - 1) // per_page)
                page = max(1, min(st.session_state.get("search_page", 1), max_pages))
                start, end = (page - 1) * per_page, min(page * per_page, total)
                page_df = results_df.iloc[start:end]

                st.markdown(f"**📊 {total}** resultado(s) encontrado(s) — exibindo {start+1} a {end}.")

                # CORREÇÃO: Mostrar nome em vez de CPF
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
                
                # CORREÇÃO: Mostrar apenas nome, não CPF
                sender_name = all_users.get(msg['from'], {}).get('name', msg['from'])
                
                with st.expander(f"{unread_indicator} {msg['subject']} — De: {sender_name}", expanded=is_unread):
                    st.write(f"**Assunto:** {msg['subject']}")
                    st.write(f"**De:** {sender_name}")  # CORREÇÃO: Mostrar nome
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
                # CORREÇÃO: Mostrar apenas nome, não CPF
                recipient_name = all_users.get(msg['to'], {}).get('name', msg['to'])
                
                with st.expander(f"📤 {msg['subject']} — Para: {recipient_name}"):  # CORREÇÃO: Mostrar nome
                    st.write(f"**Assunto:** {msg['subject']}")
                    st.write(f"**Para:** {recipient_name}")  # CORREÇÃO: Mostrar nome
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
            
            # CORREÇÃO CRÍTICA: Mostrar apenas nomes, não CPFs
            user_options = {}
            for username, user_data in users.items():
                if username != USERNAME:
                    # Mostrar apenas o nome do usuário
                    user_options[user_data.get('name', username)] = username
            
            # Pre-selecionar destinatário se for uma resposta
            default_recipient = []
            if reply_to_msg:
                sender_cpf = reply_to_msg['from']
                sender_name = users.get(sender_cpf, {}).get('name', sender_cpf)
                # CORREÇÃO: Usar apenas o nome para seleção
                if sender_name in user_options:
                    default_recipient.append(sender_name)

            # CORREÇÃO: Multiselect mostra apenas nomes
            recipients = st.multiselect("Para:", options=sorted(list(user_options.keys())), default=default_recipient)
            
            subject = st.text_input("Assunto:", 
                                  value=f"Re: {reply_to_msg['subject']}" if reply_to_msg else "")
            body = st.text_area("Mensagem:", height=200,
                              value=f"\n\n---\nEm resposta à mensagem de {users.get(reply_to_msg['from'], {}).get('name', reply_to_msg['from'])}:\n> {reply_to_msg['body'][:500].replace(chr(10), chr(10)+'> ')}..." 
                                                            value=f"\n\n---\nEm resposta à mensagem de {users.get(reply_to_msg['from'], {}).get('name', reply_to_msg['from'])}:\n> {reply_to_msg['body'][:500].replace(chr(10), chr(10)+'> ')}..." if reply_to_msg else "")
            
            attachment_file = st.file_uploader("Anexar arquivo", type=['pdf', 'docx', 'txt', 'jpg', 'png'])
            
            if st.form_submit_button("📤 Enviar Mensagem", use_container_width=True):
                if not recipients:
                    st.error("Selecione pelo menos um destinatário.")
                elif not subject.strip():
                    st.error("O assunto é obrigatório.")
                elif not body.strip():
                    st.error("A mensagem não pode estar vazia.")
                else:
                    # Converter nomes de volta para CPFs para envio
                    recipient_cpfs = []
                    for recipient_name in recipients:
                        # Encontrar o CPF correspondente ao nome
                        for cpf, user_data in users.items():
                            if user_data.get('name') == recipient_name:
                                recipient_cpfs.append(cpf)
                                break
                    
                    if not recipient_cpfs:
                        st.error("Nenhum destinatário válido encontrado.")
                    else:
                        # Enviar mensagem para cada destinatário
                        success_count = 0
                        for recipient_cpf in recipient_cpfs:
                            sent_msg = send_message(USERNAME, recipient_cpf, subject, body, attachment_file)
                            if sent_msg:
                                success_count += 1
                        
                        if success_count > 0:
                            st.success(f"Mensagem enviada para {success_count} destinatário(s)!")
                            if st.session_state.get('reply_message_id'):
                                st.session_state.reply_message_id = None
                            safe_rerun()
                        else:
                            st.error("Erro ao enviar a mensagem.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: config
# -------------------------
elif st.session_state.page == "config":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("⚙️ Configurações do Sistema")
    
    # Configurações de aparência
    with st.expander("🎨 Configurações de Aparência", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            font_scale = st.slider(
                "Escala de Fonte:",
                min_value=0.8,
                max_value=1.5,
                value=st.session_state.settings.get("font_scale", 1.0),
                step=0.1,
                key="config_font_scale"
            )
            
            plot_height = st.slider(
                "Altura dos Gráficos:",
                min_value=400,
                max_value=1000,
                value=st.session_state.settings.get("plot_height", 720),
                step=50,
                key="config_plot_height"
            )
        
        with col2:
            node_opacity = st.slider(
                "Opacidade dos Nós (Mapa Mental):",
                min_value=0.3,
                max_value=1.0,
                value=st.session_state.settings.get("node_opacity", 1.0),
                step=0.1,
                key="config_node_opacity"
            )
            
            font_size = st.slider(
                "Tamanho da Fonte Base:",
                min_value=10,
                max_value=20,
                value=st.session_state.settings.get("font_size", 14),
                step=1,
                key="config_font_size"
            )
    
    # Configurações de funcionalidades
    with st.expander("🔧 Configurações de Funcionalidades", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            auto_save = st.checkbox(
                "Auto-save automático",
                value=st.session_state.autosave,
                key="config_auto_save"
            )
            
            enable_animations = st.checkbox(
                "Habilitar animações",
                value=True,
                key="config_animations"
            )
        
        with col2:
            show_tutorial = st.checkbox(
                "Mostrar tutorial inicial",
                value=not st.session_state.tutorial_completed,
                key="config_show_tutorial"
            )
            
            enable_notifications = st.checkbox(
                "Notificações de novas mensagens",
                value=True,
                key="config_notifications"
            )
    
    # Configurações de dados
    with st.expander("📊 Configurações de Dados", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            backup_retention = st.selectbox(
                "Retenção de Backups:",
                options=["7 dias", "30 dias", "90 dias", "1 ano", "Manter todos"],
                index=1,
                key="config_backup_retention"
            )
            
            data_export_format = st.selectbox(
                "Formato de Exportação:",
                options=["CSV", "Excel", "JSON"],
                index=0,
                key="config_export_format"
            )
        
        with col2:
            search_results_per_page = st.slider(
                "Resultados por página (Busca):",
                min_value=5,
                max_value=20,
                value=8,
                step=1,
                key="config_search_results"
            )
            
            recommendation_results_per_page = st.slider(
                "Resultados por página (Recomendações):",
                min_value=5,
                max_value=15,
                value=5,
                step=1,
                key="config_rec_results"
            )
    
    # Ações do sistema
    with st.expander("⚠️ Ações do Sistema", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Aplicar Configurações", use_container_width=True):
                # Aplicar configurações de aparência
                st.session_state.settings["font_scale"] = font_scale
                st.session_state.settings["plot_height"] = plot_height
                st.session_state.settings["node_opacity"] = node_opacity
                st.session_state.settings["font_size"] = font_size
                
                # Aplicar outras configurações
                st.session_state.autosave = auto_save
                if not show_tutorial:
                    st.session_state.tutorial_completed = True
                
                save_user_state_minimal(USER_STATE)
                st.success("Configurações aplicadas com sucesso!")
                safe_rerun()
        
        with col2:
            if st.button("🔄 Restaurar Padrões", use_container_width=True):
                if st.checkbox("Confirmar restauração das configurações padrão?"):
                    st.session_state.settings = _defaults["settings"].copy()
                    st.session_state.autosave = _defaults["autosave"]
                    save_user_state_minimal(USER_STATE)
                    st.success("Configurações restauradas para os padrões!")
                    safe_rerun()
        
        with col3:
            if st.button("🗑️ Limpar Cache", use_container_width=True):
                if st.checkbox("Confirmar limpeza do cache?"):
                    try:
                        # Limpar cache de dados
                        st.cache_data.clear()
                        st.success("Cache limpo com sucesso!")
                    except Exception as e:
                        st.error(f"Erro ao limpar cache: {e}")
    
    # Informações do sistema
    with st.expander("ℹ️ Informações do Sistema", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Informações do Usuário:**")
            st.write(f"- Nome: {USER_OBJ.get('name', 'N/A')}")
            st.write(f"- Tipo de Bolsa: {USER_OBJ.get('scholarship', 'N/A')}")
            st.write(f"- Data de Cadastro: {USER_OBJ.get('created_at', 'N/A')}")
            
            st.write("**Uso do Sistema:**")
            st.write(f"- Favoritos: {len(get_session_favorites())}")
            st.write(f"- Mensagens Não Lidas: {UNREAD_COUNT}")
            if st.session_state.df is not None:
                st.write(f"- Planilha Carregada: {st.session_state.uploaded_name or 'N/A'}")
        
        with col2:
            st.write("**Estatísticas do Sistema:**")
            
            # Contar usuários
            users = load_users()
            st.write(f"- Usuários Cadastrados: {len(users)}")
            
            # Contar mensagens totais
            all_messages = load_all_messages()
            user_messages = [m for m in all_messages if m.get('from') == USERNAME or m.get('to') == USERNAME]
            st.write(f"- Suas Mensagens: {len(user_messages)}")
            
            # Contar backups
            backup_count = 0
            user_backup_dir = BACKUPS_DIR / USERNAME
            if user_backup_dir.exists():
                backup_count = len(list(user_backup_dir.glob("*.csv")))
            st.write(f"- Seus Backups: {backup_count}")
            
            # Versão do sistema
            st.write(f"- Versão: NUGEP-PQR 2.0")

    # Exportação de dados
    with st.expander("💾 Exportação de Dados", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Exportar Meus Dados", use_container_width=True):
                export_data = {
                    "user_info": USER_OBJ,
                    "favorites": get_session_favorites(),
                    "notes": st.session_state.notes,
                    "settings": st.session_state.settings,
                    "exported_at": datetime.now().isoformat()
                }
                
                export_json = json.dumps(export_data, ensure_ascii=False, indent=2)
                st.download_button(
                    "⬇️ Baixar Dados do Usuário",
                    data=export_json,
                    file_name=f"nugep_pqr_dados_{USERNAME}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col2:
            if st.button("📊 Exportar Estatísticas", use_container_width=True):
                stats_data = {
                    "user_stats": {
                        "favorites_count": len(get_session_favorites()),
                        "unread_messages": UNREAD_COUNT,
                        "backup_count": backup_count,
                        "notes_length": len(st.session_state.notes)
                    },
                    "system_stats": {
                        "total_users": len(users),
                        "total_messages": len(all_messages),
                        "user_messages": len(user_messages)
                    },
                    "exported_at": datetime.now().isoformat()
                }
                
                stats_json = json.dumps(stats_data, ensure_ascii=False, indent=2)
                st.download_button(
                    "⬇️ Baixar Estatísticas",
                    data=stats_json,
                    file_name=f"nugep_pqr_stats_{USERNAME}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer and final touches
# -------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#bfc6cc; font-size:12px; padding:16px;'>"
    "NUGEP-PQR • Sistema de Gestão de Pesquisa • Desenvolvido para a comunidade acadêmica"
    "</div>",
    unsafe_allow_html=True
)

# Auto-save final se habilitado
if st.session_state.autosave and st.session_state.authenticated:
    try:
        save_user_state_minimal(USER_STATE)
    except Exception:
        pass
