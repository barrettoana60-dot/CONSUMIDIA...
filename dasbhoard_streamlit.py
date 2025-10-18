# app_nugep_pqr_full_final.py

# NUGEP-PQR — versão final com mapa mental estilo Miro e análise inteligente com IA

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
/* Estilos para o mapa mental */
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
# AI Helper Functions - MELHORADA
# -------------------------
class DataAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.insights = []
    
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
        
        return analysis
    
    def _basic_analysis(self):
        """Análise básica dos dados"""
        text = "### 📊 Visão Geral\n\n"
        text += f"- **Total de registros**: {len(self.df)}\n"
        text += f"- **Colunas disponíveis**: {', '.join(self.df.columns.tolist())}\n"
        
        # Estatísticas por tipo de dado
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        text += f"- **Colunas numéricas**: {len(numeric_cols)}\n"
        text += f"- **Colunas de texto**: {len(text_cols)}\n\n"
        
        return text
    
    def _author_analysis(self):
        """Análise de autores e colaborações - CORRIGIDA E MELHORADA"""
        text = "### 👥 Análise de Autores\n\n"
        
        # Buscar coluna de autores de forma mais abrangente
        author_col = None
        author_data_found = False
        
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['autor', 'author', 'pesquisador', 'escritor']):
                author_col = col
                author_data_found = True
                break
        
        if not author_data_found:
            # Tentar encontrar dados de autor em qualquer coluna
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    # Verificar se a coluna contém nomes (palavras com letras maiúsculas)
                    sample_data = self.df[col].dropna().head(10)
                    if len(sample_data) > 0:
                        has_names = any(any(word.istitle() for word in str(val).split()) for val in sample_data)
                        if has_names:
                            author_col = col
                            author_data_found = True
                            text += f"⚠️ **Atenção**: Usando coluna '{col}' para análise de autores (detecção automática)\n\n"
                            break
        
        if not author_col:
            return "❌ **Autores**: Nenhuma coluna de autores identificada na planilha\n\n"
            
        all_authors = []
        authors_found = 0
        
        for authors_str in self.df[author_col].dropna():
            if isinstance(authors_str, str) and authors_str.strip():
                authors_found += 1
                # Múltiplas estratégias de parsing
                authors = re.split(r'[;,]|\be\b|\band\b|&', authors_str)
                for author in authors:
                    author_clean = author.strip()
                    if (author_clean and len(author_clean) > 2 and 
                        author_clean.lower() not in ['', 'e', 'and', 'et', 'de', 'da', 'do'] and
                        not author_clean.isdigit()):
                        all_authors.append(author_clean)
        
        if all_authors:
            author_counts = pd.Series(all_authors).value_counts()
            text += "**Principais autores identificados:**\n"
            for author, count in author_counts.head(8).items():
                text += f"- **{author}**: {count} publicação(ões)\n"
            
            # Colaborações
            collaborations = 0
            for authors_str in self.df[author_col].dropna():
                if isinstance(authors_str, str) and len(re.split(r'[;,]|\be\b|\band\b|&', authors_str)) > 1:
                    collaborations += 1
            
            if collaborations > 0:
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
        """Análise temporal dos dados - CORRIGIDA E MELHORADA"""
        text = "### 📈 Análise Temporal\n\n"
        
        # Buscar coluna de ano de forma mais abrangente
        year_col = None
        year_data_found = False
        
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['ano', 'year', 'data', 'date', 'publication']):
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
            
            # Ano mais frequente
            year_counts = years.value_counts()
            if not year_counts.empty:
                most_frequent_year = int(year_counts.index[0])
                most_frequent_count = int(year_counts.iloc[0])
                text += f"- **Ano com mais publicações**: {most_frequent_year} ({most_frequent_count} publicações)\n"
            
            # Distribuição por década
            if year_range > 20:
                decades = (years // 10) * 10
                decade_counts = decades.value_counts().sort_index()
                if len(decade_counts) > 1:
                    text += "\n**Distribuição por década:**\n"
                    for decade, count in decade_counts.head(5).items():
                        text += f"- {int(decade)}s: {int(count)} publicação(ões)\n"
            
            # Tendência
            if len(years) > 5:
                recent_threshold = max_year - 5
                recent_years = years[years >= recent_threshold]
                older_years = years[years < recent_threshold]
                
                if len(recent_years) > 0 and len(older_years) > 0:
                    recent_avg = len(recent_years) / 5  # média por ano nos últimos 5 anos
                    older_avg = len(older_years) / max(1, (recent_threshold - min_year))  # média por ano no período anterior
                    
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
        """Análise temática dos dados"""
        text = "### 🔍 Análise Temática\n\n"
        
        # Combinar texto de todas as colunas relevantes
        texto_completo = ""
        text_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
        for col in text_cols[:4]:  # Aumentei para 4 colunas
            col_text = self.df[col].fillna('').astype(str).str.cat(sep=' ')
            if len(col_text) > 100:  # Só adiciona se tiver conteúdo significativo
                texto_completo += " " + col_text
        
        if not texto_completo.strip():
            return "❌ **Temas**: Não há texto suficiente para análise temática\n\n"
        
        # Extrair temas
        palavras = re.findall(r'\b[a-zà-ú]{4,}\b', texto_completo.lower())
        stop_words = set(PORTUGUESE_STOP_WORDS)
        palavras_filtradas = [p for p in palavras if p not in stop_words and len(p) > 3]
        
        if palavras_filtradas:
            temas = pd.Series(palavras_filtradas).value_counts().head(12)
            text += "**Palavras-chave mais frequentes:**\n"
            for i, (tema, count) in enumerate(temas.items(), 1):
                text += f"{i}. **{tema}**: {count} ocorrências\n"
            text += "\n"
        else:
            text += "⚠️ **Temas**: Não foi possível identificar palavras-chave frequentes\n\n"
        
        return text
    
    def _collaboration_analysis(self):
        """Análise de colaborações e redes"""
        text = "### 🤝 Análise de Colaborações\n\n"
        
        author_col = None
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['autor', 'author']):
                author_col = col
                break
        
        if author_col:
            coautorias = 0
            total_trabalhos = len(self.df[author_col].dropna())
            
            for authors_str in self.df[author_col].dropna():
                if isinstance(authors_str, str) and len(re.split(r'[;,]|\be\b|\band\b|&', authors_str)) > 1:
                    coautorias += 1
            
            if total_trabalhos > 0:
                taxa_colaboracao = (coautorias/total_trabalhos)*100
                text += f"- **Trabalhos em colaboração**: {coautorias}\n"
                text += f"- **Taxa de colaboração**: {taxa_colaboracao:.1f}%\n"
                
                if coautorias > 0:
                    if taxa_colaboracao > 60:
                        text += "- **Padrão**: Alta colaboração entre pesquisadores\n"
                    elif taxa_colaboracao > 30:
                        text += "- **Padrão**: Boa colaboração acadêmica\n"
                    else:
                        text += "- **Padrão**: Oportunidade para aumentar colaborações\n"
                else:
                    text += "- **Padrão**: Produção individual predominante\n"
            else:
                text += "⚠️ **Colaboração**: Sem dados de autores para análise\n"
            
            text += "\n"
        
        return text
    
    def _geographic_analysis(self):
        """Análise geográfica dos dados - CORRIGIDA E MELHORADA"""
        text = "### 🌎 Análise Geográfica\n\n"
        
        # Buscar coluna de país de forma mais abrangente
        country_col = None
        country_data_found = False
        
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['país', 'pais', 'country', 'local', 'location', 'nacionalidade', 'região']):
                country_col = col
                country_data_found = True
                break
        
        if not country_data_found:
            # Tentar encontrar dados de país em colunas de texto
            for col in self.df.select_dtypes(include=['object']).columns:
                sample_data = self.df[col].dropna().head(10)
                if len(sample_data) > 0:
                    # Verificar se contém nomes de países conhecidos
                    common_countries = ['brasil', 'brazil', 'eua', 'usa', 'portugal', 'espanha', 'frança']
                    has_countries = any(any(country in str(val).lower() for country in common_countries) for val in sample_data)
                    if has_countries:
                        country_col = col
                        country_data_found = True
                        text += f"⚠️ **Atenção**: Usando coluna '{col}' para análise geográfica (detecção automática)\n\n"
                        break
        
        if not country_col:
            return "❌ **Países**: Nenhuma coluna de países identificada na planilha\n\n"
            
        countries = self.df[country_col].dropna()
        
        if len(countries) > 0:
            country_counts = countries.value_counts()
            text += "**Países/regiões mais frequentes:**\n"
            for country, count in country_counts.head(8).items():
                text += f"- **{country}**: {count} publicação(ões)\n"
            
            # Diversidade geográfica
            unique_countries = len(country_counts)
            total_countries = len(countries)
            diversity_index = (unique_countries / total_countries) * 100
            
            text += f"\n- **Diversidade geográfica**: {diversity_index:.1f}%\n"
            text += f"- **Países/regiões únicos**: {unique_countries}\n"
            
            if unique_countries == 1:
                text += "- **Foco**: Pesquisa concentrada em uma única região\n"
            elif unique_countries <= 3:
                text += "- **Foco**: Pesquisa com foco regional\n"
            elif unique_countries <= 8:
                text += "- **Foco**: Pesquisa com boa diversidade geográfica\n"
            else:
                text += "- **Foco**: Pesquisa com excelente abrangência internacional\n"
            
            text += f"\n**Total de registros com localização**: {total_countries}\n\n"
        else:
            text += f"⚠️ **Países**: Coluna '{country_col}' encontrada mas sem dados válidos\n\n"
        
        return text
    
    def _trend_analysis(self):
        """Análise de tendências e insights - MELHORADA E SIMPLIFICADA"""
        text = "### 💡 Análise e Sugestões\n\n"
        
        insights = []
        sugestoes = []
        
        # Análise do tamanho da base
        if len(self.df) < 20:
            insights.append("Base de dados pequena")
            sugestoes.append("Adicione mais registros para análises mais confiáveis")
        elif len(self.df) < 50:
            insights.append("Base de dados em desenvolvimento")
            sugestoes.append("Continue expandindo sua coleção de referências")
        else:
            insights.append("Base de dados sólida")
            sugestoes.append("Excelente quantidade de dados para análises detalhadas")
        
        # Verificar completude dos dados
        has_authors = any(col.lower() in ['autor', 'author'] for col in self.df.columns)
        has_years = any(col.lower() in ['ano', 'year'] for col in self.df.columns)
        has_countries = any(col.lower() in ['país', 'pais', 'country'] for col in self.df.columns)
        
        metadados_completos = has_authors and has_years and has_countries
        
        if metadados_completos:
            insights.append("Metadados completos (autores, anos, países)")
            sugestoes.append("Todos os elementos essenciais para análise estão presentes")
        else:
            if not has_authors:
                sugestoes.append("Adicione informações de autores para análise de colaboração")
            if not has_years:
                sugestoes.append("Inclua dados de anos para análise temporal")
            if not has_countries:
                sugestoes.append("Adicione países para análise geográfica")
        
        # Gerar texto formatado
        if insights:
            text += "**Características identificadas:**\n"
            for insight in insights:
                text += f"• {insight}\n"
            text += "\n"
        
        if sugestoes:
            text += "**Sugestões para melhorar sua pesquisa:**\n"
            for i, sug in enumerate(sugestoes, 1):
                text += f"{i}. {sug}\n"
            text += "\n"
        
        # Recomendação final baseada nos dados
        if len(self.df) >= 30 and metadados_completos:
            text += "🎯 **Sua base permite:**\n- Análises estatísticas confiáveis\n- Estudos de rede de colaboração\n- Análise de tendências temporais\n- Mapeamento geográfico da pesquisa\n"
        elif len(self.df) >= 15:
            text += "🎯 **Próximos passos:**\n- Expanda gradualmente sua base\n- Complete os metadados faltantes\n- Defina focos temáticos específicos\n"
        else:
            text += "🎯 **Para começar:**\n- Foque em coletar mais dados\n- Estruture bem as colunas da planilha\n- Defina objetivos claros de pesquisa\n"
        
        return text

def get_ai_assistant_response(question, context):
    """Assistente de IA melhorado para responder perguntas sobre os dados"""
    
    question_lower = question.lower()
    
    # Análise de autores - MELHORADA
    if any(word in question_lower for word in ['autor', 'autores', 'pesquisador', 'escritor', 'quem escreveu']):
        author_col = None
        for col in context.df.columns:
            if any(keyword in col.lower() for keyword in ['autor', 'author']):
                author_col = col
                break
        
        if author_col:
            all_authors = []
            for authors_str in context.df[author_col].dropna():
                if isinstance(authors_str, str):
                    authors = re.split(r'[;,]|\be\b|\band\b|&', authors_str)
                    for author in authors:
                        author_clean = author.strip()
                        if author_clean and len(author_clean) > 2:
                            all_authors.append(author_clean)
            
            if all_authors:
                author_counts = pd.Series(all_authors).value_counts().head(5)
                response = "**👥 Autores Mais Relevantes:**\n\n"
                for author, count in author_counts.items():
                    response += f"• **{author}**: {count} publicação(ões)\n"
                
                return response
            else:
                return "**⚠️**: Encontrei a coluna de autores mas não consegui extrair nomes válidos. Verifique o formato dos dados."
        else:
            return "**❌**: Não encontrei coluna de autores. Adicione uma coluna 'autor' ou 'autores'."
    
    # Análise de países - MELHORADA
    elif any(word in question_lower for word in ['país', 'países', 'geografia', 'região', 'local']):
        country_col = None
        for col in context.df.columns:
            if any(keyword in col.lower() for keyword in ['país', 'pais', 'country']):
                country_col = col
                break
        
        if country_col:
            countries = context.df[country_col].dropna()
            if not countries.empty:
                country_counts = countries.value_counts()
                response = "**🌎 Distribuição Geográfica:**\n\n"
                
                for country, count in country_counts.head(6).items():
                    response += f"• **{country}**: {count} publicação(ões)\n"
                
                return response
            else:
                return "**⚠️**: Coluna de países encontrada mas sem dados válidos."
        else:
            return "**❌**: Não encontrei coluna de países. Adicione 'país' ou 'country' para análise geográfica."
    
    # Análise temporal - MELHORADA
    elif any(word in question_lower for word in ['ano', 'anos', 'temporal', 'evolução', 'cronologia']):
        year_col = None
        for col in context.df.columns:
            if any(keyword in col.lower() for keyword in ['ano', 'year']):
                year_col = col
                break
        
        if year_col:
            try:
                years = pd.to_numeric(context.df[year_col], errors='coerce').dropna()
            except:
                years = pd.Series(dtype=float)
            
            if len(years) > 0:
                min_year = int(years.min())
                max_year = int(years.max())
                
                response = f"**📅 Período Temporal:**\n\n"
                response += f"• **Período**: {min_year} - {max_year}\n"
                response += f"• **Intervalo**: {max_year - min_year} anos\n"
                
                year_counts = years.value_counts()
                if not year_counts.empty:
                    most_active = int(year_counts.index[0])
                    response += f"• **Ano mais ativo**: {most_active}\n"
                
                return response
            else:
                return "**⚠️**: Coluna de anos encontrada mas sem dados numéricos válidos."
        else:
            return "**❌**: Não encontrei coluna de anos. Adicione 'ano' ou 'year' para análise temporal."
    
    # Análise de temas
    elif any(word in question_lower for word in ['tema', 'temas', 'assunto', 'palavras-chave', 'termos']):
        texto_completo = ""
        text_cols = [col for col in context.df.columns if context.df[col].dtype == 'object']
        for col in text_cols[:3]:
            texto_completo += " " + context.df[col].fillna('').astype(str).str.cat(sep=' ')
        
        if texto_completo.strip():
            palavras = re.findall(r'\b[a-zà-ú]{4,}\b', texto_completo.lower())
            stop_words = set(PORTUGUESE_STOP_WORDS)
            palavras_filtradas = [p for p in palavras if p not in stop_words and len(p) > 3]
            
            if palavras_filtradas:
                temas = pd.Series(palavras_filtradas).value_counts().head(8)
                response = "**🔤 Palavras-chave Frequentes:**\n\n"
                
                for i, (tema, count) in enumerate(temas.items(), 1):
                    response += f"{i}. **{tema}** ({count})\n"
                
                return response
            else:
                return "**🔍**: Analisei o texto mas não identifiquei temas claros."
        else:
            return "**❌**: Não há texto suficiente para análise temática."
    
    # Sugestões gerais
    elif any(word in question_lower for word in ['sugestão', 'dica', 'o que fazer', 'próximo passo']):
        response = "**💡 Sugestões para Sua Pesquisa:**\n\n"
        
        response += "1. **Expanda sua base** com mais referências\n"
        response += "2. **Complete os metadados** (autores, anos, países)\n"
        response += "3. **Use a busca integrada** para encontrar trabalhos relacionados\n"
        response += "4. **Organize conceitos** no mapa mental\n"
        response += "5. **Explore colaborações** com outros pesquisadores\n"
        
        return response
    
    # Resposta padrão
    else:
        return f"""**🤖 Como posso ajudar?**

Posso analisar diferentes aspectos dos seus dados:

• **Autores**: "Quais são os autores mais relevantes?"
• **Países**: "Qual a distribuição geográfica?"
• **Anos**: "Como evoluiu a pesquisa ao longo do tempo?"
• **Temas**: "Quais são os conceitos mais frequentes?"
• **Sugestões**: "O que devo fazer em seguida?"

Faça uma pergunta específica para uma análise mais detalhada!"""

# -------------------------
# Miro-like Mind Map Components - ATUALIZADO E TRADUZIDO
# -------------------------
class MiroStyleMindMap:
    def __init__(self):
        self.node_types = {
            "ideia": {"color": "#4ECDC4", "icon": "💡", "shape": "dot"},
            "tarefa": {"color": "#45B7D1", "icon": "✅", "shape": "square"},
            "pergunta": {"color": "#96CEB4", "icon": "❓", "shape": "diamond"},
            "recurso": {"color": "#FECA57", "icon": "📚", "shape": "triangle"},
            "objetivo": {"color": "#FF6B6B", "icon": "🎯", "shape": "star"},
            "nota": {"color": "#A29BFE", "icon": "📝", "shape": "circle"}
        }
    
    def create_node(self, node_id, label, node_type="ideia", description="", x=0, y=0):
        """Cria uma ideia no estilo Miro"""
        node_data = self.node_types.get(node_type, self.node_types["ideia"])
        return {
            "id": node_id,
            "label": f"{node_data['icon']} {label}",
            "type": node_type,
            "description": description,
            "color": node_data["color"],
            "shape": node_data["shape"],
            "x": x,
            "y": y,
            "font": {"color": "#FFFFFF", "size": 14, "face": "Arial"},
            "size": 20
        }
    
    def generate_layout(self, nodes, edges, layout_type="hierarchical"):
        """Gera layout automático para as ideias"""
        if layout_type == "hierarchical":
            return self._hierarchical_layout(nodes, edges)
        elif layout_type == "radial":
            return self._radial_layout(nodes, edges)
        else:
            return self._force_directed_layout(nodes, edges)
    
    def _hierarchical_layout(self, nodes, edges):
        """Layout hierárquico (árvore)"""
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node["id"])
        for edge in edges:
            G.add_edge(edge["source"], edge["target"])
        
        try:
            pos = nx.spring_layout(G, k=2, iterations=50)
            for node in nodes:
                if node["id"] in pos:
                    node["x"] = pos[node["id"]][0] * 1000
                    node["y"] = pos[node["id"]][1] * 1000
        except:
            # Fallback layout
            for i, node in enumerate(nodes):
                node["x"] = (i % 3) * 300
                node["y"] = (i // 3) * 200
        
        return nodes

    def _radial_layout(self, nodes, edges):
        """Layout radial"""
        center_x, center_y = 500, 400
        radius = 300
        
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / len(nodes)
            node["x"] = center_x + radius * np.cos(angle)
            node["y"] = center_y + radius * np.sin(angle)
        
        return nodes

    def _force_directed_layout(self, nodes, edges):
        """Layout de força direcionada"""
        G = nx.Graph()
        for node in nodes:
            G.add_node(node["id"])
        for edge in edges:
            G.add_edge(edge["source"], edge["target"])
        
        try:
            pos = nx.spring_layout(G, k=1, iterations=100)
            for node in nodes:
                if node["id"] in pos:
                    node["x"] = pos[node["id"]][0] * 800
                    node["y"] = pos[node["id"]][1] * 600
        except:
            # Fallback random
            for node in nodes:
                node["x"] = random.randint(100, 900)
                node["y"] = random.randint(100, 700)
        
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
nav_buttons = {"planilha": "📄 Planilha", "recomendacoes": "💡 Recomendações", "mapa": "🗺️ Mapa Mental",
               "anotacoes": "📝 Anotações", "graficos": "📊 Análise", "busca": "🔍 Busca",
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
# Page: mapa mental - MELHORADO E CORRIGIDO
# -------------------------
elif st.session_state.page == "mapa":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("🗺️ Mapa Mental Interativo")
    st.info("💡 **Crie, conecte e visualize suas ideias** - Arraste as ideias e edite diretamente!")
    
    # Inicializar sistema de mapa mental
    if 'miro_map' not in st.session_state:
        st.session_state.miro_map = MiroStyleMindMap()
        st.session_state.miro_nodes = []
        st.session_state.miro_edges = []
        st.session_state.miro_selected_node = None
        st.session_state.miro_layout = "hierarchical"
    
    # Sidebar principal - TRADUZIDA
    with st.sidebar:
        st.header("🎨 Controles do Mapa")
        
        # Criar nova ideia - CORRIGIDO: limpa o formulário após criar
        with st.expander("➕ Nova Ideia", expanded=True):
            with st.form("create_miro_node", clear_on_submit=True):
                node_label = st.text_input("Título da ideia:", placeholder="Ex: Pesquisa Qualitativa", key="new_node_label")
                node_type = st.selectbox("Tipo:", options=list(st.session_state.miro_map.node_types.keys()), key="new_node_type")
                node_desc = st.text_area("Descrição:", placeholder="Detalhes sobre esta ideia...", height=100, key="new_node_desc")
                
                if st.form_submit_button("🎯 Adicionar Ideia", use_container_width=True):
                    if node_label:
                        node_id = f"node_{int(time.time())}_{random.randint(1000,9999)}"
                        new_node = st.session_state.miro_map.create_node(
                            node_id, node_label, node_type, node_desc
                        )
                        st.session_state.miro_nodes.append(new_node)
                        st.session_state.miro_selected_node = node_id
                        st.success("Ideia criada!")
                        safe_rerun()
        
        # Conectar ideias - MELHORADO
        with st.expander("🔗 Conectar Ideias", expanded=False):
            if len(st.session_state.miro_nodes) >= 2:
                nodes_list = [(node["id"], node["label"]) for node in st.session_state.miro_nodes]
                with st.form("connect_nodes"):
                    # Usar os títulos reais das ideias em vez dos IDs
                    source_options = {f"{label}": node_id for node_id, label in nodes_list}
                    target_options = {f"{label}": node_id for node_id, label in nodes_list}
                    
                    source_label = st.selectbox("De:", options=list(source_options.keys()), key="connect_source")
                    target_label = st.selectbox("Para:", options=[k for k in target_options.keys() if k != source_label], key="connect_target")
                    
                    if st.form_submit_button("🔗 Conectar", use_container_width=True):
                        source_id = source_options[source_label]
                        target_id = target_options[target_label]
                        
                        # Verificar se conexão já existe
                        existing = any(e["source"] == source_id and e["target"] == target_id 
                                     for e in st.session_state.miro_edges)
                        if not existing:
                            st.session_state.miro_edges.append({
                                "source": source_id,
                                "target": target_id,
                                "label": "conecta"
                            })
                            st.success("Conexão criada!")
                            safe_rerun()
                        else:
                            st.warning("Essas ideias já estão conectadas.")
            else:
                st.info("Precisa de pelo menos 2 ideias para conectar")
        
        # Configurações do mapa - ATUALIZADO
        with st.expander("👁️ Visualização", expanded=False):
            # Modos de visualização
            visualization_mode = st.selectbox(
                "Modo de Visualização:",
                options=["Mapa 3D", "Mapa 2D", "Fluxograma"],
                index=1,
                help="Escolha como visualizar seu mapa"
            )
            
            st.session_state.miro_layout = st.selectbox(
                "Organização Automática:",
                options=["hierarchical", "radial", "force"],
                help="Como organizar as ideias automaticamente"
            )
            
            if st.button("🔄 Reorganizar Mapa", use_container_width=True):
                st.session_state.miro_nodes = st.session_state.miro_map.generate_layout(
                    st.session_state.miro_nodes, 
                    st.session_state.miro_edges,
                    st.session_state.miro_layout
                )
                st.success("Mapa reorganizado!")
                safe_rerun()
            
            st.markdown("---")
            if st.button("🗑️ Limpar Mapa", type="secondary", use_container_width=True):
                if st.checkbox("Confirmar limpeza total do mapa?"):
                    st.session_state.miro_nodes = []
                    st.session_state.miro_edges = []
                    st.session_state.miro_selected_node = None
                    st.success("Mapa limpo!")
                    safe_rerun()
    
    # Área principal do mapa
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"🎨 {visualization_mode}")
        
        if st.session_state.miro_nodes:
            # Configurações baseadas no modo de visualização
            if visualization_mode == "Mapa 3D":
                # Estilo 3D - efeito visual melhorado
                st.markdown('<div class="three-d-effect">', unsafe_allow_html=True)
                st.info("🌐 **Modo 3D Ativo**: Efeito visual tridimensional com gradiente!")
                
                node_size = 30
                font_size = 16
                physics_enabled = True
                hierarchical_enabled = False
                
                # Aplicar efeitos 3D nos nós
                for node in st.session_state.miro_nodes:
                    # Intensificar cores para efeito 3D
                    if node["color"] == "#4ECDC4": node["color"] = "#00FFCC"  # Ideia - ciano brilhante
                    elif node["color"] == "#45B7D1": node["color"] = "#0099FF"  # Tarefa - azul brilhante
                    elif node["color"] == "#96CEB4": node["color"] = "#66FF99"  # Pergunta - verde brilhante
                    elif node["color"] == "#FECA57": node["color"] = "#FFCC00"  # Recurso - amarelo brilhante
                    elif node["color"] == "#FF6B6B": node["color"] = "#FF3366"  # Objetivo - rosa brilhante
                    elif node["color"] == "#A29BFE": node["color"] = "#9966FF"  # Nota - roxo brilhante
                    
            elif visualization_mode == "Fluxograma":
                # Estilo fluxograma - caixas retangulares
                st.markdown('<div class="flowchart-box">', unsafe_allow_html=True)
                st.info("📦 **Modo Fluxograma**: Use caixas para processos e decisões!")
                
                for node in st.session_state.miro_nodes:
                    node["shape"] = "square"  # Forçar formato quadrado
                node_size = 25
                font_size = 14
                physics_enabled = False
                hierarchical_enabled = True
                
            else:  # Mapa 2D
                node_size = 20
                font_size = 14
                physics_enabled = True
                hierarchical_enabled = True
            
            # Preparar ideias para visualização
            nodes_for_viz = []
            for node in st.session_state.miro_nodes:
                is_selected = node["id"] == st.session_state.miro_selected_node
                
                # Ajustar tamanho baseado no modo
                current_size = node_size + 5 if is_selected else node_size
                
                nodes_for_viz.append(Node(
                    id=node["id"],
                    label=node["label"],
                    color=node["color"],
                    size=current_size,
                    shape=node["shape"],
                    font={"color": "#FFFFFF", "size": font_size, "face": "Arial"},
                    x=node.get("x", 0),
                    y=node.get("y", 0)
                ))
            
            # Preparar conexões
            edges_for_viz = []
            for edge in st.session_state.miro_edges:
                edges_for_viz.append(Edge(
                    source=edge["source"],
                    target=edge["target"],
                    label=edge.get("label", ""),
                    color="#B0B0B0",
                    width=3 if visualization_mode == "Mapa 3D" else 2,
                    font={"size": 10, "color": "#bfc6cc"}
                ))
            
            # Configuração do gráfico
            config = Config(
                width="100%",
                height=700,
                directed=True,
                physics=physics_enabled,
                hierarchical=hierarchical_enabled,
                nodeHighlightBehavior=True,
                highlightColor="#F8F8F8",
                collapsible=True,
                node={"labelProperty": "label"},
                link={"labelProperty": "label", "renderLabel": True}
            )
            
            try:
                # Renderizar mapa interativo
                clicked_node = agraph(nodes=nodes_for_viz, edges=edges_for_viz, config=config)
                if clicked_node:
                    st.session_state.miro_selected_node = clicked_node
                    safe_rerun()
                    
                if visualization_mode in ["Mapa 3D", "Fluxograma"]:
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Erro na visualização: {e}")
                st.info("Tente reorganizar o mapa nas configurações.")
        else:
            # Tela inicial vazia
            st.markdown("""
            <div style='text-align: center; padding: 50px; background: rgba(255,255,255,0.02); border-radius: 10px;'>
                <h3 style='color: #bfc6cc;'>🎯 Seu Mapa de Ideias Vazio</h3>
                <p style='color: #8b9bab;'>Comece adicionando sua primeira ideia usando o painel lateral!</p>
                <div style='margin-top: 20px;'>
                    <div style='display: inline-block; margin: 10px; padding: 15px; background: rgba(78, 205, 196, 0.1); border-radius: 8px;'>
                        💡 Ideias
                    </div>
                    <div style='display: inline-block; margin: 10px; padding: 15px; background: rgba(69, 183, 209, 0.1); border-radius: 8px;'>
                        ✅ Tarefas
                    </div>
                    <div style='display: inline-block; margin: 10px; padding: 15px; background: rgba(150, 206, 180, 0.1); border-radius: 8px;'>
                        ❓ Perguntas
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("✏️ Editor de Ideia")
        
        selected_node_id = st.session_state.miro_selected_node
        if selected_node_id and st.session_state.miro_nodes:
            selected_node = next((n for n in st.session_state.miro_nodes if n["id"] == selected_node_id), None)
            
            if selected_node:
                with st.form(f"edit_miro_node_{selected_node_id}"):
                    st.write(f"**Editando:** {selected_node['label']}")
                    
                    # Extrair label sem emoji
                    current_label = selected_node['label']
                    if ' ' in current_label:
                        current_label = current_label.split(' ', 1)[1]
                    
                    new_label = st.text_input("Título:", value=current_label, key=f"label_{selected_node_id}")
                    new_type = st.selectbox("Tipo:", 
                                          options=list(st.session_state.miro_map.node_types.keys()),
                                          index=list(st.session_state.miro_map.node_types.keys()).index(selected_node["type"]),
                                          key=f"type_{selected_node_id}")
                    new_desc = st.text_area("Descrição:", 
                                          value=selected_node.get("description", ""), 
                                          height=120,
                                          key=f"desc_{selected_node_id}")
                    
                    col_edit1, col_edit2 = st.columns(2)
                    with col_edit1:
                        if st.form_submit_button("💾 Salvar", use_container_width=True):
                            if new_label.strip():
                                node_type_data = st.session_state.miro_map.node_types[new_type]
                                selected_node["label"] = f"{node_type_data['icon']} {new_label.strip()}"
                                selected_node["type"] = new_type
                                selected_node["description"] = new_desc
                                selected_node["color"] = node_type_data["color"]
                                selected_node["shape"] = node_type_data["shape"]
                                st.success("Ideia atualizada!")
                                safe_rerun()
                            else:
                                st.warning("O título não pode ser vazio.")
                    
                    with col_edit2:
                        if st.form_submit_button("🗑️ Excluir", type="secondary", use_container_width=True):
                            # Remover ideia e suas conexões
                            st.session_state.miro_nodes = [n for n in st.session_state.miro_nodes if n["id"] != selected_node_id]
                            st.session_state.miro_edges = [e for e in st.session_state.miro_edges 
                                                         if e["source"] != selected_node_id and e["target"] != selected_node_id]
                            st.session_state.miro_selected_node = None
                            st.success("Ideia excluída!")
                            safe_rerun()
                
                # Visualização da descrição
                if selected_node.get("description"):
                    st.markdown("---")
                    st.write("**Descrição atual:**")
                    st.markdown(f'<div class="node-preview">{selected_node["description"]}</div>', unsafe_allow_html=True)
                
                # Conexões desta ideia
                st.markdown("---")
                st.write("**🔗 Conexões:**")
                
                incoming = [e for e in st.session_state.miro_edges if e["target"] == selected_node_id]
                outgoing = [e for e in st.session_state.miro_edges if e["source"] == selected_node_id]
                
                if incoming:
                    st.write("**Recebe de:**")
                    for edge in incoming:
                        source_node = next((n for n in st.session_state.miro_nodes if n["id"] == edge["source"]), None)
                        if source_node:
                            st.write(f"• {source_node['label']}")
                
                if outgoing:
                    st.write("**Conecta para:**")
                    for edge in outgoing:
                        target_node = next((n for n in st.session_state.miro_nodes if n["id"] == edge["target"]), None)
                        if target_node:
                            st.write(f"• {target_node['label']}")
                
                if not incoming and not outgoing:
                    st.info("Esta ideia não possui conexões.")
        else:
            st.info("👆 Selecione uma ideia no mapa para editar.")
    
    # Ferramentas de exportação
    st.markdown("---")
    st.subheader("📤 Exportar Mapa")
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        if st.button("💾 Salvar como JSON", use_container_width=True):
            map_data = {
                "nodes": st.session_state.miro_nodes,
                "edges": st.session_state.miro_edges,
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "user": USERNAME,
                    "version": "miro_style_1.0"
                }
            }
            json_str = json.dumps(map_data, ensure_ascii=False, indent=2)
            st.download_button(
                "⬇️ Baixar JSON",
                data=json_str,
                file_name=f"mapa_ideias_{USERNAME}_{int(time.time())}.json",
                mime="application/json"
            )
    
    with exp_col2:
        if st.button("🖼️ Exportar como PNG", use_container_width=True):
            # Gerar visualização estática
            try:
                plt.figure(figsize=(16, 12))
                plt.title("Mapa de Ideias", fontsize=16, color='white', pad=20)
                plt.axis('off')
                
                # Salvar imagem temporária
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                           facecolor='#0E192A', edgecolor='none')
                plt.close()
                buf.seek(0)
                
                st.download_button(
                    "⬇️ Baixar PNG",
                    data=buf.getvalue(),
                    file_name=f"mapa_ideias_{USERNAME}_{int(time.time())}.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Erro ao gerar imagem: {e}")

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
# Page: graficos - ORGANIZADO COM ABAS
# -------------------------
elif st.session_state.page == "graficos":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("📊 Análise Inteligente com IA")
    
    if st.session_state.df is None:
        st.warning("Carregue uma planilha na página 'Planilha' para gerar análises.")
    else:
        df = st.session_state.df.copy()
        
        # Inicializar analisador de IA
        analyzer = DataAnalyzer(df)
        
        # Organização com abas
        tab_overview, tab_authors, tab_themes, tab_geo, tab_temp, tab_collab, tab_ai = st.tabs([
            "📊 Visão Geral", "👥 Autores", "🔤 Temas", "🌎 Geográfica", "📅 Temporal", "🤝 Colaboração", "🤖 IA"
        ])
        
        with tab_overview:
            # Análise completa automática
            analysis = analyzer.generate_comprehensive_analysis()
            st.markdown(analysis)
            
            # Estatísticas rápidas
            st.write("#### 📈 Estatísticas da Base")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total de Registros", len(df))
            with col2:
                st.metric("Colunas", len(df.columns))
            with col3:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                st.metric("Colunas Numéricas", len(numeric_cols))
            with col4:
                text_cols = df.select_dtypes(include=['object']).columns.tolist()
                st.metric("Colunas Texto", len(text_cols))
        
        with tab_authors:
            author_analysis = analyzer._author_analysis()
            st.markdown(author_analysis)
            
            # Gráfico de autores se disponível
            author_col = next((col for col in df.columns if any(keyword in col.lower() for keyword in ['autor', 'author'])), None)
            if author_col:
                all_authors = []
                for authors_str in df[author_col].dropna():
                    if isinstance(authors_str, str):
                        authors = re.split(r'[;,]|\be\b|\band\b|&', authors_str)
                        for author in authors:
                            author_clean = author.strip()
                            if author_clean and len(author_clean) > 2:
                                all_authors.append(author_clean)
                
                if all_authors:
                    author_counts = pd.Series(all_authors).value_counts().head(10)
                    if len(author_counts) > 0:
                        fig_auth = px.bar(
                            x=author_counts.values, 
                            y=author_counts.index,
                            orientation='h',
                            title="Principais Autores",
                            labels={'x': 'Número de Publicações', 'y': 'Autor'}
                        )
                        fig_auth.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)", 
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#d6d9dc"),
                            xaxis=dict(color="#d6d9dc"),
                            yaxis=dict(color="#d6d9dc")
                        )
                        st.plotly_chart(fig_auth, use_container_width=True)
        
        with tab_themes:
            thematic_analysis = analyzer._thematic_analysis()
            st.markdown(thematic_analysis)
            
            # Gráfico de temas
            texto_completo = ""
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            for col in text_cols[:3]:
                texto_completo += " " + df[col].fillna('').astype(str).str.cat(sep=' ')
            
            if texto_completo.strip():
                palavras = re.findall(r'\b[a-zà-ú]{4,}\b', texto_completo.lower())
                stop_words = set(PORTUGUESE_STOP_WORDS)
                palavras_filtradas = [p for p in palavras if p not in stop_words and len(p) > 3]
                
                if palavras_filtradas:
                    temas = pd.Series(palavras_filtradas).value_counts().head(15)
                    if len(temas) > 0:
                        fig_temas = px.bar(
                            x=temas.values,
                            y=temas.index,
                            orientation='h',
                            title="Palavras-chave Mais Frequentes",
                            labels={'x': 'Frequência', 'y': 'Palavra'}
                        )
                        fig_temas.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)", 
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#d6d9dc"),
                            xaxis=dict(color="#d6d9dc"),
                            yaxis=dict(color="#d6d9dc")
                        )
                        st.plotly_chart(fig_temas, use_container_width=True)
        
        with tab_geo:
            geographic_analysis = analyzer._geographic_analysis()
            st.markdown(geographic_analysis)
            
            # Gráfico geográfico
            country_col = next((col for col in df.columns if any(keyword in col.lower() for keyword in ['país', 'pais', 'country'])), None)
            if country_col:
                countries = df[country_col].dropna()
                if not countries.empty:
                    country_counts = countries.value_counts()
                    if len(country_counts) > 0:
                        fig_geo = px.pie(
                            values=country_counts.values, 
                            names=country_counts.index,
                            title="Distribuição Geográfica"
                        )
                        fig_geo.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)", 
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#d6d9dc")
                        )
                        st.plotly_chart(fig_geo, use_container_width=True)
        
        with tab_temp:
            temporal_analysis = analyzer._temporal_analysis()
            st.markdown(temporal_analysis)
            
            # Gráfico temporal
            year_col = next((col for col in df.columns if any(keyword in col.lower() for keyword in ['ano', 'year'])), None)
            if year_col:
                try:
                    years = pd.to_numeric(df[year_col], errors='coerce').dropna()
                except:
                    years = pd.Series(dtype=float)
                    
                if not years.empty and len(years.unique()) > 1:
                    year_counts = years.value_counts().sort_index()
                    fig_temp = px.line(
                        x=year_counts.index, 
                        y=year_counts.values,
                        title="Publicações por Ano",
                        labels={'x': 'Ano', 'y': 'Número de Publicações'}
                    )
                    fig_temp.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)", 
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#d6d9dc"),
                        xaxis=dict(color="#d6d9dc"),
                        yaxis=dict(color="#d6d9dc")
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)
        
        with tab_collab:
            collaboration_analysis = analyzer._collaboration_analysis()
            st.markdown(collaboration_analysis)
        
        with tab_ai:
            # Assistente de IA MELHORADO
            st.write("### 🤖 Assistente de IA - Análise Personalizada")
            
            col_ai1, col_ai2 = st.columns([3, 1])
            with col_ai1:
                ai_question = st.text_input(
                    "Pergunte sobre seus dados:",
                    placeholder="Ex: Quais são os autores mais relevantes? Quais temas aparecem juntos?",
                    key="ai_question"
                )
            with col_ai2:
               ai_ask = st.button("🔍 Analisar com IA", use_container_width=True)
            
            if ai_ask and ai_question:
                with st.spinner("🤔 Analisando seus dados..."):
                    time.sleep(1)  # Simular processamento
                    response = get_ai_assistant_response(ai_question, analyzer)
                    st.success("Análise concluída!")
                    st.markdown(f"**Resposta da IA:**")
                    st.markdown(response)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: busca
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
    users_map = load_users()

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
            
            # CORREÇÃO: Mostrar apenas o nome, sem CPF
            if origin_uid == "web":
                user_display_name = "Fonte: Web"
            else:
                user_obj = users_map.get(str(origin_uid), {})
                user_display_name = user_obj.get("name", "Usuário")  # Removido o CPF

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
                
                # CORREÇÃO: Mostrar apenas o nome, sem CPF
                if origin_user == "web":
                    origin_display = "Fonte: Web"
                else:
                    ou = users_map.get(str(origin_user), {})
                    origin_display = ou.get("name", "Usuário")  # Removido o CPF
                    
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
                    if det.get('similarity'):
                        st.metric("Similaridade", f"{det['similarity']:.2f}")
                    
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
# Page: mensagens - CORREÇÃO: REMOVER CPF DA EXIBIÇÃO
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
                
                # CORREÇÃO: Mostrar apenas o nome, sem CPF
                users_map = load_users()
                from_user = msg.get('from')
                from_display = users_map.get(from_user, {}).get('name', from_user)  # Removido CPF
                
                st.markdown(f"**De:** {escape_html(from_display)}")
                st.markdown(f"**Assunto:** {escape_html(msg.get('subject'))}")
                st.markdown(f"**Data:** {datetime.fromisoformat(msg.get('ts')).strftime('%d/%m/%Y %H:%M')}")
                st.markdown("---")
                st.markdown(f"<div style='white-space:pre-wrap; padding:10px; border-radius:8px; background:rgba(0,0,0,0.2);'>{escape_html(msg.get('body'))}</div>", unsafe_allow_html=True)
                
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
                    
                    # CORREÇÃO: Mostrar apenas o nome, sem CPF
                    users_map = load_users()
                    from_user = msg.get('from')
                    from_display = users_map.get(from_user, {}).get('name', from_user)  # Removido CPF
                    
                    st.markdown(f"**{read_marker}{escape_html(msg.get('subject', '(sem assunto)'))}**")
                    st.markdown(f"<span class='small-muted'>De: {escape_html(from_display)} em {datetime.fromisoformat(msg.get('ts')).strftime('%d/%m/%Y %H:%M')}</span>", unsafe_allow_html=True)
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
                # CORREÇÃO: Mostrar apenas o nome, sem CPF
                users_map = load_users()
                to_user = msg.get('to')
                to_display = users_map.get(to_user, {}).get('name', to_user)  # Removido CPF
                
                st.markdown(f"**{escape_html(msg.get('subject', '(sem assunto)'))}**")
                st.markdown(f"<span class='small-muted'>Para: {escape_html(to_display)} em {datetime.fromisoformat(msg.get('ts')).strftime('%d/%m/%Y %H:%M')}</span>", unsafe_allow_html=True)
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
                users_map = load_users()
                from_user = original_msg.get('from')
                from_display = users_map.get(from_user, {}).get('name', from_user)  # Removido CPF
                
                st.info(f"Respondendo a: {from_display}")
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
            users = load_users()
            # CORREÇÃO: Mostrar apenas nomes na lista de destinatários
            user_options = []
            user_mapping = {}
            
            for username, user_data in users.items():
                if username != USERNAME:
                    display_name = user_data.get('name', 'Usuário')  # Removido CPF
                    user_options.append(display_name)
                    user_mapping[display_name] = username
            
            if not user_options:
                st.warning("Nenhum outro usuário cadastrado — não é possível enviar mensagens.")
            else:
                # Selecionar pelo nome de exibição
                selected_display = st.selectbox(
                    "Para:", 
                    options=user_options,
                    index=user_options.index(default_to) if default_to in user_options else 0
                )
                
                # Obter o CPF real do usuário selecionado
                to_user = user_mapping.get(selected_display, "")
                
                subject = st.text_input("Assunto:", value=default_subj)
                body = st.text_area("Mensagem:", height=200, value=default_body)
                attachment = st.file_uploader("Anexo (opcional)", type=['pdf', 'txt', 'doc', 'docx', 'xls', 'xlsx'])
            
