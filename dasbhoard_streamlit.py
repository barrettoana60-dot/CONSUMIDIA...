
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
/* Configurações de fonte */
.font-config {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid rgba(255,255,255,0.1);
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
        """Análise de autores e colaborações - CORRIGIDA E FUNCIONANDO"""
        text = "### 👥 Análise de Autores\n\n"
        
        # BUSCA MAIS AGRESSIVA POR COLUNA DE AUTORES
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
                    has_multiple_authors = any(';' in str(val) or ',' in str(val) for val in sample_data)
                    if has_multiple_authors or any(len(str(val).split()) >= 2 for val in sample_data):
                        author_col = col
                        break
        
        # Se não encontrou, usar a primeira possível
        if not author_col and possible_author_cols:
            author_col = possible_author_cols[0]
        
        if not author_col:
            return "❌ **Autores**: Nenhuma coluna de autores identificada. Verifique se há colunas como 'autor', 'autores', 'author' na sua planilha.\n\n"
        
        text += f"**Coluna utilizada**: '{author_col}'\n\n"
        
        # PROCESSAMENTO MELHORADO DOS AUTORES
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
                        author_clean.lower() not in ['', 'e', 'and', 'et', 'de', 'da', 'do', 'dos', 'das'] and
                        not author_clean.isdigit() and
                        not author_clean.replace('.', '').isdigit()):
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
                text += f"{i}. **{tema}**: {count} palavras repetidas\n"
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
        """Análise de tendências e insights - SUGESTÕES INTELIGENTES REAIS"""
        text = "### 💡 Análise e Sugestões Inteligentes\n\n"
        
        insights = []
        sugestoes_inteligentes = []
        
        # ANÁLISE INTELIGENTE BASEADA NOS DADOS REAIS
        total_registros = len(self.df)
        
        # 1. Análise de completude
        colunas_principais = ['autor', 'ano', 'título', 'resumo']
        colunas_presentes = [col for col in colunas_principais 
                            if any(col in col_name.lower() for col_name in self.df.columns)]
        completude = len(colunas_presentes) / len(colunas_principais) * 100
        
        if completude < 50:
            sugestoes_inteligentes.append("📋 **Melhore a estrutura da planilha** - Adicione colunas básicas como autor, ano, título")
        elif completude < 80:
            sugestoes_inteligentes.append("📊 **Estrutura boa** - Considere adicionar mais metadados para análises avançadas")
        else:
            sugestoes_inteligentes.append("🎯 **Estrutura excelente** - Todos os elementos essenciais estão presentes")
        
        # 2. Análise temporal (se houver anos)
        year_col = next((col for col in self.df.columns if 'ano' in col.lower() or 'year' in col.lower()), None)
        if year_col:
            try:
                anos = pd.to_numeric(self.df[year_col], errors='coerce').dropna()
                if len(anos) > 0:
                    range_anos = int(anos.max()) - int(anos.min())
                    if range_anos < 3:
                        sugestoes_inteligentes.append("⏳ **Expanda o período** - Dados concentrados em poucos anos, busque maior variedade temporal")
                    elif range_anos > 10:
                        sugestoes_inteligentes.append("📈 **Analise tendências** - Período extenso permite análise de evolução temporal")
            except:
                pass
        
        # 3. Análise de diversidade de autores
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
                sugestoes_inteligentes.append("👥 **Amplie rede de autores** - Pouca diversidade de pesquisadores")
            elif len(autores_unicos) > 20:
                sugestoes_inteligentes.append("🤝 **Rede colaborativa forte** - Excelente diversidade de autores")
        
        # 4. Análise de temas emergentes
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
                sugestoes_inteligentes.append(f"🔍 **Foque em**: {', '.join(temas_comuns[:3])}")
        
        # 5. Sugestões baseadas no tamanho
        if total_registros < 15:
            sugestoes_inteligentes.extend([
                "📥 **Colete mais dados** - Mínimo 20 registros para análises confiáveis",
                "🔎 **Use busca integrada** - Encontre trabalhos relacionados na plataforma"
            ])
        elif total_registros < 50:
            sugestoes_inteligentes.extend([
                "📊 **Análises básicas possíveis** - Explore gráficos e estatísticas",
                "🗺️ **Organize conceitos** - Use o mapa mental para estruturar ideias"
            ])
        else:
            sugestoes_inteligentes.extend([
                "📈 **Análises avançadas** - Dados suficientes para ML e redes complexas",
                "🌐 **Explore colaborações** - Identifique redes de coautoria"
            ])
        
        # Formatar resposta
        text += "**Sugestões Inteligentes Baseadas na Sua Planilha:**\n\n"
        for i, sugestao in enumerate(sugestoes_inteligentes, 1):
            text += f"{i}. {sugestao}\n"
        
        text += f"\n**Resumo da Base:**\n"
        text += f"• Registros: {total_registros}\n"
        text += f"• Completude: {completude:.1f}%\n"
        if author_col:
            text += f"• Coluna autores: '{author_col}'\n"
        if year_col:
            text += f"• Coluna anos: '{year_col}'\n"
        
        return text

# -------------------------
# SISTEMA DE IA INTELIGENTE MELHORADO
# -------------------------
def get_ai_assistant_response(question, context):
    """Assistente de IA SUPER INTELIGENTE - Responde qualquer tipo de pergunta"""
    
    question_lower = question.lower().strip()
    df = context.df
    
    # PERGUNTAS SOBRE AUTORES
    if any(word in question_lower for word in ['autor', 'autores', 'pesquisador', 'escritor', 'quem escreveu', 'quem publicou']):
        return _analyze_authors(df, question_lower)
    
    # PERGUNTAS SOBRE PAÍSES/GEOGRAFIA
    elif any(word in question_lower for word in ['país', 'países', 'geográfica', 'geografia', 'distribuição', 'local', 'região', 'onde']):
        return _analyze_geography(df, question_lower)
    
    # PERGUNTAS SOBRE TEMPO/EVOLUÇÃO
    elif any(word in question_lower for word in ['ano', 'anos', 'temporal', 'evolução', 'cronologia', 'linha do tempo', 'como evoluiu', 'quando', 'período']):
        return _analyze_temporal(df, question_lower)
    
    # PERGUNTAS SOBRE TEMAS/CONCEITOS
    elif any(word in question_lower for word in ['tema', 'temas', 'conceito', 'conceitos', 'palavras', 'frequentes', 'termos', 'assuntos', 'palavras-chave', 'keywords']):
        return _analyze_themes(df, question_lower)
    
    # PERGUNTAS SOBRE COLABORAÇÕES
    elif any(word in question_lower for word in ['colaboração', 'colaborações', 'coautoria', 'parceria', 'trabalho conjunto', 'rede']):
        return _analyze_collaborations(df, question_lower)
    
    # PERGUNTAS SOBRE ESTATÍSTICAS GERAIS
    elif any(word in question_lower for word in ['estatística', 'estatísticas', 'números', 'quantidade', 'total', 'quantos', 'resumo', 'visão geral']):
        return _analyze_statistics(df, question_lower)
    
    # PERGUNTAS SOBRE TENDÊNCIAS
    elif any(word in question_lower for word in ['tendência', 'tendências', 'futuro', 'emergente', 'novo', 'recente']):
        return _analyze_trends(df, question_lower)
    
    # PERGUNTAS COMPLEXAS/ANÁLISE
    elif any(word in question_lower for word in ['análise', 'analisar', 'insight', 'interpretação', 'o que significa', 'significado']):
        return _analyze_complex_questions(df, question_lower)
    
    # SUGESTÕES
    elif any(word in question_lower for word in ['sugestão', 'sugestões', 'recomendação', 'recomendações', 'o que fazer', 'próximo passo', 'como melhorar']):
        return _provide_suggestions(df, question_lower)
    
    # PERGUNTAS SOBRE A BASE DE DADOS
    elif any(word in question_lower for word in ['dados', 'base de dados', 'planilha', 'dataset', 'qualidade']):
        return _analyze_data_quality(df, question_lower)
    
    # RESPOSTA PADRÃO PARA PERGUNTAS NÃO IDENTIFICADAS
    else:
        return _general_analysis_response(df, question)

def _analyze_authors(df, question):
    """Análise avançada de autores"""
    author_col = next((col for col in df.columns if any(kw in col.lower() for kw in ['autor', 'author'])), None)
    
    if not author_col:
        return "**❌ Autores**: Não encontrei coluna de autores na planilha."
    
    autores_contagem = {}
    colaboracoes = 0
    autores_por_trabalho = []
    
    for authors_str in df[author_col].dropna():
        if isinstance(authors_str, str):
            autores = re.split(r'[;,]|\be\b|\band\b|&', authors_str)
            autores_lista = [a.strip() for a in autores if a.strip() and len(a.strip()) > 2]
            
            autores_por_trabalho.append(len(autores_lista))
            
            if len(autores_lista) > 1:
                colaboracoes += 1
            
            for autor in autores_lista:
                autores_contagem[autor] = autores_contagem.get(autor, 0) + 1
    
    if not autores_contagem:
        return "**⚠️ Autores**: Coluna encontrada mas não consegui extrair nomes válidos."
    
    autores_ordenados = sorted(autores_contagem.items(), key=lambda x: x[1], reverse=True)
    total_autores = len(autores_contagem)
    total_trabalhos = len(df[author_col].dropna())
    media_autores = np.mean(autores_por_trabalho) if autores_por_trabalho else 0
    
    resposta = "**👥 ANÁLISE DE AUTORES**\n\n"
    resposta += f"**Total de autores únicos**: {total_autores}\n"
    resposta += f"**Total de trabalhos com autores**: {total_trabalhos}\n"
    resposta += f"**Média de autores por trabalho**: {media_autores:.1f}\n"
    resposta += f"**Taxa de colaboração**: {(colaboracoes/total_trabalhos)*100:.1f}%\n\n"
    
    resposta += "**Top 10 autores mais produtivos:**\n"
    for i, (autor, count) in enumerate(autores_ordenados[:10], 1):
        resposta += f"{i}. **{autor}** - {count} publicação(ões)\n"
    
    # Análises específicas baseadas na pergunta
    if 'relevante' in question:
        resposta += f"\n**💡 Autores mais relevantes**: {', '.join([autor for autor, _ in autores_ordenados[:5]])}"
    
    if 'colaboração' in question:
        resposta += f"\n**🤝 Colaborações**: {colaboracoes} trabalhos em coautoria"
    
    if 'produtivo' in question:
        resposta += f"\n**🏆 Autor mais produtivo**: {autores_ordenados[0][0]} com {autores_ordenados[0][1]} publicações"
    
    return resposta

def _analyze_geography(df, question):
    """Análise geográfica avançada"""
    country_col = next((col for col in df.columns if any(kw in col.lower() for kw in ['país', 'pais', 'country', 'local'])), None)
    
    if not country_col:
        return "**❌ Geografia**: Não encontrei coluna de países/regiões."
    
    paises = df[country_col].dropna()
    if paises.empty:
        return "**⚠️ Geografia**: Coluna encontrada mas sem dados válidos."
    
    contagem_paises = paises.value_counts()
    total_paises = len(contagem_paises)
    total_registros = len(paises)
    
    resposta = "**🌎 ANÁLISE GEOGRÁFICA**\n\n"
    resposta += f"**Total de países/regiões**: {total_paises}\n"
    resposta += f"**Total de registros com localização**: {total_registros}\n"
    resposta += f"**Diversidade geográfica**: {(total_paises/total_registros)*100:.1f}%\n\n"
    
    resposta += "**Distribuição geográfica:**\n"
    for pais, count in contagem_paises.head(10).items():
        percentual = (count / total_registros) * 100
        resposta += f"• **{pais}**: {count} ({percentual:.1f}%)\n"
    
    # Análises específicas
    if 'distribuição' in question:
        pais_principal = contagem_paises.index[0]
        percentual_principal = (contagem_paises.iloc[0] / total_registros) * 100
        resposta += f"\n**🎯 Foco principal**: {pais_principal} concentra {percentual_principal:.1f}% da produção"
    
    if 'diversidade' in question:
        if total_paises > 10:
            resposta += f"\n**🌐 Alta diversidade**: Pesquisa com abrangência internacional ({total_paises} regiões)"
        elif total_paises > 3:
            resposta += f"\n**🎯 Diversidade moderada**: {total_paises} regiões principais"
        else:
            resposta += f"\n**📍 Foco concentrado**: Pesquisa concentrada em {total_paises} região(ões)"
    
    return resposta

def _analyze_temporal(df, question):
    """Análise temporal avançada"""
    year_col = next((col for col in df.columns if any(kw in col.lower() for kw in ['ano', 'year', 'data'])), None)
    
    if not year_col:
        return "**❌ Anos**: Não encontrei coluna temporal."
    
    try:
        anos = pd.to_numeric(df[year_col], errors='coerce').dropna()
        if anos.empty:
            return "**⚠️ Anos**: Coluna encontrada mas sem valores numéricos válidos."
        
        min_ano = int(anos.min())
        max_ano = int(anos.max())
        periodo = max_ano - min_ano
        anos_unicos = len(anos.unique())
        
        resposta = "**📈 ANÁLISE TEMPORAL**\n\n"
        resposta += f"**Período analisado**: {min_ano} - {max_ano} ({periodo} anos)\n"
        resposta += f"**Anos com registros**: {anos_unicos}\n"
        resposta += f"**Total de registros temporais**: {len(anos)}\n\n"
        
        # Análise por década
        if periodo > 10:
            decadas = (anos // 10) * 10
            contagem_decadas = decadas.value_counts().sort_index()
            resposta += "**Distribuição por década:**\n"
            for decada, count in contagem_decadas.items():
                resposta += f"• **{int(decada)}s**: {int(count)} publicações\n"
        
        # Análise de tendência
        contagem_por_ano = anos.value_counts().sort_index()
        if len(contagem_por_ano) > 3:
            # Últimos 3 anos vs anteriores
            anos_recentes = contagem_por_ano.tail(3)
            anos_anteriores = contagem_por_ano.head(len(contagem_por_ano)-3)
            
            media_recente = anos_recentes.mean()
            media_anterior = anos_anteriores.mean() if len(anos_anteriores) > 0 else 0
            
            if media_recente > media_anterior * 1.3:
                tendencia = "📈 **CRESCENTE** - Produção em crescimento"
            elif media_recente < media_anterior * 0.7:
                tendencia = "📉 **DECRESCENTE** - Produção reduzindo"
            else:
                tendencia = "➡️ **ESTÁVEL** - Produção constante"
            
            resposta += f"\n**Tendência**: {tendencia}"
        
        # Ano mais produtivo
        if not contagem_por_ano.empty:
            ano_mais_produtivo = contagem_por_ano.idxmax()
            producao_pico = contagem_por_ano.max()
            resposta += f"\n**🏆 Ano mais produtivo**: {int(ano_mais_produtivo)} ({producao_pico} publicações)"
        
        return resposta
        
    except Exception as e:
        return f"**❌ Erro na análise temporal**: {str(e)}"

def _analyze_themes(df, question):
    """Análise temática avançada"""
    texto_analise = ""
    colunas_texto = [col for col in df.columns if df[col].dtype == 'object']
    
    for col in colunas_texto[:5]:  # Analisar até 5 colunas de texto
        texto_analise += " " + df[col].fillna('').astype(str).str.cat(sep=' ')
    
    if len(texto_analise.strip()) < 100:
        return "**❌ Temas**: Não há texto suficiente para análise temática."
    
    # Análise avançada de palavras-chave
    palavras = re.findall(r'\b[a-zà-ú]{4,}\b', texto_analise.lower())
    palavras_filtradas = [p for p in palavras if p not in PORTUGUESE_STOP_WORDS and len(p) > 3]
    
    if not palavras_filtradas:
        return "**🔍 Temas**: Texto analisado mas não identifiquei palavras-chave significativas."
    
    from collections import Counter
    contador = Counter(palavras_filtradas)
    temas_comuns = contador.most_common(15)
    
    resposta = "**🔤 ANÁLISE TEMÁTICA**\n\n"
    resposta += f"**Total de palavras únicas**: {len(contador)}\n"
    resposta += f"**Texto analisado**: {len(texto_analise):,} caracteres\n\n"
    
    resposta += "**Palavras-chave mais frequentes:**\n"
    for i, (tema, count) in enumerate(temas_comuns[:12], 1):
        resposta += f"{i}. **{tema}** - {count} palavras repetidas\n"
    
    # Análise de bigramas (palavras que aparecem juntas)
    if len(palavras_filtradas) > 10:
        bigramas = []
        for i in range(len(palavras_filtradas)-1):
            bigrama = f"{palavras_filtradas[i]} {palavras_filtradas[i+1]}"
            bigramas.append(bigrama)
        
        contador_bigramas = Counter(bigramas)
        bigramas_comuns = contador_bigramas.most_common(8)
        
        if bigramas_comuns:
            resposta += "\n**Conceitos relacionados (bigramas):**\n"
            for bigrama, count in bigramas_comuns[:6]:
                resposta += f"• **{bigrama}** ({count})\n"
    
    # Temas emergentes (palavras menos frequentes mas significativas)
    temas_emergentes = [tema for tema, count in temas_comuns[8:15] if count >= 2]
    if temas_emergentes:
        resposta += f"\n**💡 Temas emergentes**: {', '.join(temas_emergentes[:5])}"
    
    return resposta

def _analyze_collaborations(df, question):
    """Análise de colaborações"""
    author_col = next((col for col in df.columns if any(kw in col.lower() for kw in ['autor', 'author'])), None)
    
    if not author_col:
        return "**❌ Colaborações**: Não encontrei dados de autores para análise."
    
    colaboracoes = 0
    total_trabalhos = 0
    autores_por_trabalho = []
    
    for authors_str in df[author_col].dropna():
        if isinstance(authors_str, str):
            total_trabalhos += 1
            autores = re.split(r'[;,]', authors_str)
            num_autores = len([a for a in autores if a.strip()])
            autores_por_trabalho.append(num_autores)
            
            if num_autores > 1:
                colaboracoes += 1
    
    if total_trabalhos == 0:
        return "**⚠️ Colaborações**: Sem dados válidos para análise."
    
    taxa_colaboracao = (colaboracoes / total_trabalhos) * 100
    media_autores = np.mean(autores_por_trabalho)
    
    resposta = "**🤝 ANÁLISE DE COLABORAÇÕES**\n\n"
    resposta += f"**Total de trabalhos analisados**: {total_trabalhos}\n"
    resposta += f"**Trabalhos em colaboração**: {colaboracoes}\n"
    resposta += f"**Taxa de colaboração**: {taxa_colaboracao:.1f}%\n"
    resposta += f"**Média de autores por trabalho**: {media_autores:.1f}\n\n"
    
    # Classificação do nível de colaboração
    if taxa_colaboracao > 60:
        resposta += "**🎯 Alto nível de colaboração** - Pesquisa fortemente colaborativa"
    elif taxa_colaboracao > 30:
        resposta += "**🤝 Bom nível de colaboração** - Equilíbrio entre trabalho individual e em grupo"
    else:
        resposta += "**💡 Oportunidade para colaboração** - Predominância de trabalho individual"
    
    return resposta

def _analyze_statistics(df, question):
    """Análise estatística geral"""
    total_registros = len(df)
    total_colunas = len(df.columns)
    
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    colunas_texto = df.select_dtypes(include=['object']).columns.tolist()
    
    resposta = "**📊 ESTATÍSTICAS GERAIS**\n\n"
    resposta += f"**Total de registros**: {total_registros}\n"
    resposta += f"**Total de colunas**: {total_colunas}\n"
    resposta += f"**Colunas numéricas**: {len(colunas_numericas)}\n"
    resposta += f"**Colunas de texto**: {len(colunas_texto)}\n\n"
    
    # Completude dos dados
    colunas_principais = ['autor', 'ano', 'título', 'resumo']
    colunas_presentes = []
    
    for col in colunas_principais:
        if any(col in col_name.lower() for col_name in df.columns):
            colunas_presentes.append(col)
    
    completude = (len(colunas_presentes) / len(colunas_principais)) * 100
    resposta += f"**Completude dos metadados**: {completude:.1f}%\n"
    resposta += f"**Metadados presentes**: {', '.join(colunas_presentes) if colunas_presentes else 'Nenhum'}\n\n"
    
    # Tamanho da base
    if total_registros < 20:
        resposta += "**📈 Tamanho**: Base pequena - considere expandir para análises mais robustas"
    elif total_registros < 50:
        resposta += "**📈 Tamanho**: Base média - adequada para análises básicas"
    elif total_registros < 100:
        resposta += "**📈 Tamanho**: Base boa - permite análises detalhadas"
    else:
        resposta += "**📈 Tamanho**: Base excelente - ideal para análises complexas"
    
    return resposta

def _analyze_trends(df, question):
    """Análise de tendências"""
    year_col = next((col for col in df.columns if any(kw in col.lower() for kw in ['ano', 'year'])), None)
    
    resposta = "**🚀 ANÁLISE DE TENDÊNCIAS**\n\n"
    
    if year_col:
        try:
            anos = pd.to_numeric(df[year_col], errors='coerce').dropna()
            if len(anos) > 5:
                anos_recentes = anos[anos >= anos.max() - 5]
                if len(anos_recentes) > 0:
                    resposta += f"**Foco recente**: {len(anos_recentes)} publicações nos últimos 5 anos\n\n"
        
        except:
            pass
    
    # Análise de temas emergentes (simplificada)
    texto_analise = ""
    for col in df.select_dtypes(include=['object']).columns[:3]:
        texto_analise += " " + df[col].fillna('').astype(str).str.cat(sep=' ')
    
    if len(texto_analise) > 500:
        palavras = re.findall(r'\b[a-zà-ú]{5,}\b', texto_analise.lower())
        palavras_filtradas = [p for p in palavras if p not in PORTUGUESE_STOP_WORDS]
        
        from collections import Counter
        contador = Counter(palavras_filtradas)
        temas_tendencia = [pal for pal, cnt in contador.most_common(10) if cnt >= 3]
        
        if temas_tendencia:
            resposta += "**Temas em destaque:**\n"
            for tema in temas_tendencia[:5]:
                resposta += f"• {tema}\n"
    
    resposta += "\n**💡 Para análises mais profundas:**\n"
    resposta += "- Use a aba 'Análise' para gráficos detalhados\n"
    resposta += "- Explore o mapa mental para conexões entre conceitos\n"
    resposta += "- Consulte as recomendações para trabalhos relacionados"
    
    return resposta

def _analyze_complex_questions(df, question):
    """Resposta para perguntas complexas de análise"""
    resposta = "**🔍 ANÁLISE COMPLEXA**\n\n"
    
    # Análise multidimensional baseada na pergunta
    if any(word in question for word in ['correlação', 'relação', 'associação']):
        resposta += "Para análise de correlações entre variáveis, sugiro:\n"
        resposta += "1. Verifique se há colunas numéricas na sua planilha\n"
        resposta += "2. Use a aba 'Análise' para visualizar correlações\n"
        resposta += "3. Considere adicionar mais dados para análises estatísticas\n"
    
    elif any(word in question for word in ['padrão', 'padrões', 'comportamento']):
        resposta += "**Identificação de Padrões**:\n"
        resposta += "- Padrões temporais: Evolução ao longo dos anos\n"
        resposta += "- Padrões geográficos: Distribuição por regiões\n"
        resposta += "- Padrões de colaboração: Redes entre autores\n"
        resposta += "- Padrões temáticos: Frequência de conceitos\n\n"
        resposta += "**Use as ferramentas disponíveis**:\n"
        resposta += "• Gráficos na aba 'Análise'\n"
        resposta += "• Mapa mental para conexões conceituais\n"
        resposta += "• Busca inteligente para comparações"
    
    else:
        resposta += "**Análise Inteligente dos Seus Dados**:\n\n"
        
        # Análise rápida da base
        total = len(df)
        if total < 30:
            resposta += "📋 **Base em desenvolvimento** - Ideal para explorar direções iniciais\n"
        elif total < 100:
            resposta += "📊 **Base consolidada** - Permite análises confiáveis\n"
        else:
            resposta += "🚀 **Base robusta** - Excelente para análises complexas e dados estruturados\n"
        
        resposta += "\n**💡 Para análises específicas, pergunte sobre**:\n"
        resposta += "- Autores e colaborações\n"
        resposta += "- Distribuição temporal\n"
        resposta += "- Padrões geográficos\n"
        resposta += "- Temas e conceitos frequentes\n"
        resposta += "- Sugestões para expandir sua pesquisa"
    
    return resposta

def _provide_suggestions(df, question):
    """Sugestões inteligentes baseadas nos dados"""
    total = len(df)
    
    resposta = "**💡 SUGESTÕES INTELIGENTES**\n\n"
    
    # Sugestões baseadas no tamanho da base
    if total < 20:
        resposta += "**🎯 PRIORIDADE: Expansão de Dados**\n"
        resposta += "• Colete mais 20-30 registros para análises confiáveis\n"
        resposta += "• Use a busca integrada para encontrar trabalhos similares\n"
        resposta += "• Complete metadados essenciais (autores, anos, países)\n\n"
    
    elif total < 50:
        resposta += "**📊 Análises Recomendadas**:\n"
        resposta += "• Explore distribuição temporal e geográfica\n"
        resposta += "• Identifique autores e colaborações principais\n"
        resposta += "• Use o mapa mental para organizar conceitos\n\n"
    
    else:
        resposta += "**🚀 Análises Avançadas Possíveis**:\n"
        resposta += "• Redes de colaboração entre autores\n"
        resposta += "• Análise de tendências temporais\n"
        resposta += "• Mapeamento de temas emergentes\n"
        resposta += "• Correlações entre diferentes variáveis\n\n"
    
    # Sugestões específicas baseadas na pergunta
    if 'melhorar' in question or 'qualidade' in question:
        resposta += "**🔧 Melhoria da Qualidade**:\n"
        resposta += "• Verifique completude dos metadados\n"
        resposta += "• Padronize formatos (datas, autores)\n"
        resposta += "• Adicione resumos e palavras-chave\n"
    
    elif 'próximo' in question or 'futuro' in question:
        resposta += "**🔮 Direções Futuras**:\n"
        resposta += "• Identifique lacunas na literatura\n"
        resposta += "• Explore colaborações potenciais\n"
        resposta += "• Considere novas fontes de dados\n"
    
    resposta += "\n**🛠️ Ferramentas Recomendadas**:\n"
    resposta += "• Mapa Mental: Para organizar ideias\n"
    resposta += "• Análise IA: Para insights automáticos\n"
    resposta += "• Busca: Para encontrar referências\n"
    resposta += "• Recomendações: Para descoberta de conteúdo"
    
    return resposta

def _analyze_data_quality(df, question):
    """Análise da qualidade dos dados"""
    total = len(df)
    colunas = df.columns.tolist()
    
    resposta = "**📋 QUALIDADE DOS DADOS**\n\n"
    resposta += f"**Total de registros**: {total}\n"
    resposta += f"**Colunas disponíveis**: {len(colunas)}\n\n"
    
    # Análise de completude
    resposta += "**Completude por coluna**:\n"
    for col in df.columns[:8]:  # Mostrar até 8 colunas
        na_count = df[col].isna().sum()
        preenchimento = ((total - na_count) / total) * 100
        resposta += f"• **{col}**: {preenchimento:.1f}% preenchido\n"
    
    # Metadados essenciais
    essenciais = ['autor', 'ano', 'título']
    presentes = [col for col in essenciais if any(col in col_name.lower() for col_name in df.columns)]
    
    resposta += f"\n**Metadados essenciais**: {len(presentes)} de {len(essenciais)} presentes\n"
    
    if len(presentes) < len(essenciais):
        resposta += "**💡 Sugestão**: Considere adicionar colunas para autores, anos e títulos"
    
    return resposta

def _general_analysis_response(df, original_question):
    """Resposta geral para perguntas não categorizadas"""
    return f"""**🤖 ASSISTENTE INTELIGENTE NUGEP-PQR**

Não entendi completamente: "*{original_question}*"

**Posso ajudar com estas análises:**

📊 **PERGUNTAS ESPECÍFICAS:**
• "Quais são os autores mais relevantes?"
• "Qual a distribuição geográfica?"  
• "Como evoluiu a pesquisa ao longo do tempo?"
• "Quais são os conceitos mais frequentes?"
• "Quantas colaborações existem?"

🔍 **ANÁLISES COMPLEXAS:**
• "Analise os padrões de colaboração"
• "Mostre tendências temporais" 
• "Identifique temas emergentes"
• "Avalie a qualidade dos dados"

💡 **SUGESTÕES:**
• "O que devo fazer em seguida?"
• "Como posso melhorar minha pesquisa?"
• "Quais são as próximas etapas?"

**Sua base atual:** {len(df)} registros, {len(df.columns)} colunas

Faça uma pergunta mais específica sobre sua planilha!"""

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
    
    def _calculate_smart_position(self, existing_nodes, selected_node_id):
        """Calcula posição inteligente para novo nó"""
        if not existing_nodes:
            return 500, 400  # Posição central se não há nós
        
        # Se há nó selecionado, posicionar próximo a ele
        if selected_node_id:
            selected_node = next((n for n in existing_nodes if n["id"] == selected_node_id), None)
            if selected_node:
                # Posicionar em um raio de 150px do nó selecionado
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(100, 200)
                x = selected_node.get("x", 500) + distance * math.cos(angle)
                y = selected_node.get("y", 400) + distance * math.sin(angle)
                return x, y
        
        # Se não há nó selecionado, encontrar área menos congestionada
        occupied_positions = [(n.get("x", 0), n.get("y", 0)) for n in existing_nodes]
        
        # Tentar posições em espiral a partir do centro
        center_x, center_y = 500, 400
        for radius in range(200, 801, 100):  # De 200 a 800 pixels
            for angle in range(0, 360, 45):  # A cada 45 graus
                rad = math.radians(angle)
                x = center_x + radius * math.cos(rad)
                y = center_y + radius * math.sin(rad)
                
                # Verificar se está longe o suficiente de outros nós
                too_close = any(
                    math.sqrt((x - ox)**2 + (y - oy)**2) < 150 
                    for ox, oy in occupied_positions
                )
                
                if not too_close:
                    return x, y
        
        # Fallback: posição aleatória
        return random.randint(200, 800), random.randint(150, 650)
    
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
        """Layout de força direcionada - CORRIGIDO: estabilidade REAL melhorada"""
        if not nodes:
            return nodes
        
        G = nx.Graph()
        
        # Adicionar nós mantendo referências
        for node in nodes:
            G.add_node(node["id"])
        
        # Adicionar arestas
        for edge in edges:
            G.add_edge(edge["source"], edge["target"])
        
        try:
            # CONFIGURAÇÃO OTIMIZADA PARA ESTABILIDADE
            # Usar posições existentes como ponto de partida
            pos_existente = {}
            for node in nodes:
                if "x" in node and "y" in node:
                    pos_existente[node["id"]] = [node["x"], node["y"]]
            
            # Parâmetros para layout suave
            k = 3  # Distância ideal entre nós
            iterations = 50  # Menos iterações para mais velocidade
            scale = 2  # Escala moderada
            
            if pos_existente:
                # Se temos posições existentes, usar como seed
                pos = nx.spring_layout(G, pos=pos_existente, k=k, iterations=iterations, 
                                     scale=scale, seed=42)
            else:
                # Se não, começar do zero
                pos = nx.spring_layout(G, k=k, iterations=iterations, scale=scale, seed=42)
            
            # Aplicar novas posições suavemente
            for node in nodes:
                if node["id"] in pos:
                    # Se o nó já tinha posição, fazer transição suave
                    if "x" in node and "y" in node:
                        # Transição de 50% para manter estabilidade
                        node["x"] = (node["x"] + pos[node["id"]][0] * 800) / 2
                        node["y"] = (node["y"] + pos[node["id"]][1] * 600) / 2
                    else:
                        # Novo nó, posicionar normalmente
                        node["x"] = pos[node["id"]][0] * 800 + 400
                        node["y"] = pos[node["id"]][1] * 600 + 300
                        
        except Exception as e:
            print(f"Layout automático falhou: {e}")
            # Fallback: grid organizado que mantém a estabilidade
            for i, node in enumerate(nodes):
                if "x" not in node or "y" not in node:
                    # Só reposicionar nós sem posição
                    node["x"] = 400 + (i % 4) * 200
                    node["y"] = 300 + (i // 4) * 150
        
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
        "font_size": 14,  # NOVO: Tamanho da fonte
        "node_font_size": 14,  # NOVO: Tamanho da fonte dos nós
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
# Page: mapa mental - CORRIGIDO
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
    
    # Sidebar principal
    with st.sidebar:
        st.header("🎨 Controles do Mapa")
        
        # Criar nova ideia - CORREÇÃO: posicionamento seguro
        with st.expander("➕ Nova Ideia", expanded=True):
            with st.form("create_miro_node", clear_on_submit=True):
                node_label = st.text_input("Título da ideia:", placeholder="Ex: Pesquisa Qualitativa", key="new_node_label")
                node_type = st.selectbox("Tipo:", options=list(st.session_state.miro_map.node_types.keys()), key="new_node_type")
                node_desc = st.text_area("Descrição:", placeholder="Detalhes sobre esta ideia...", height=100, key="new_node_desc")
                
                if st.form_submit_button("🎯 Adicionar Ideia", use_container_width=True):
                    if node_label:
                        node_id = f"node_{int(time.time())}_{random.randint(1000,9999)}"
                        
                        # CORREÇÃO: Posicionamento seguro
                        x, y = 500, 400  # Posição central padrão
                        
                        # Se há nós existentes, posicionar de forma inteligente
                        if st.session_state.miro_nodes:
                            if st.session_state.miro_selected_node:
                                # Encontrar o nó selecionado
                                selected_node = next((n for n in st.session_state.miro_nodes 
                                                    if n["id"] == st.session_state.miro_selected_node), None)
                                if selected_node:
                                    # Posicionar próximo ao nó selecionado
                                    angle = random.uniform(0, 2 * math.pi)
                                    distance = random.uniform(100, 200)
                                    x = selected_node.get("x", 500) + distance * math.cos(angle)
                                    y = selected_node.get("y", 400) + distance * math.sin(angle)
                            else:
                                # Encontrar posição vazia
                                occupied_positions = [(n.get("x", 0), n.get("y", 0)) for n in st.session_state.miro_nodes]
                                for radius in range(200, 801, 100):
                                    for angle in range(0, 360, 45):
                                        rad = math.radians(angle)
                                        test_x = 500 + radius * math.cos(rad)
                                        test_y = 400 + radius * math.sin(rad)
                                        too_close = any(math.sqrt((test_x - ox)**2 + (test_y - oy)**2) < 150 for ox, oy in occupied_positions)
                                        if not too_close:
                                            x, y = test_x, test_y
                                            break
                                    else:
                                        continue
                                    break
                        
                        new_node = st.session_state.miro_map.create_node(
                            node_id, node_label, node_type, node_desc, x, y
                        )
                        st.session_state.miro_nodes.append(new_node)
                        st.session_state.miro_selected_node = node_id
                        st.success("Ideia criada!")
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
                                "label": "conecta"
                            })
                            st.success("Conexão criada!")
                            safe_rerun()
                        else:
                            st.warning("Essas ideias já estão conectadas.")
            else:
                st.info("Precisa de pelo menos 2 ideias para conectar")
        
        # Configurações do mapa
        with st.expander("👁️ Visualização", expanded=False):
            visualization_mode = st.selectbox("Modo de Visualização:", options=["Mapa 2D", "Mapa 3D", "Fluxograma"], index=0)
            
            st.session_state.miro_layout = st.selectbox("Organização Automática:", options=["hierarchical", "radial", "force"])
            
            if st.button("🔄 Reorganizar Mapa", use_container_width=True):
                st.session_state.miro_nodes = st.session_state.miro_map.generate_layout(
                    st.session_state.miro_nodes, st.session_state.miro_edges, st.session_state.miro_layout
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
                st.markdown('<div class="three-d-effect">', unsafe_allow_html=True)
                st.info("🌐 **Modo 3D Ativo**: Efeito visual tridimensional!")
                node_size = 30
                font_size = st.session_state.settings.get("node_font_size", 16)
                physics_enabled = True
                hierarchical_enabled = False
                
                # Aplicar efeitos 3D
                for node in st.session_state.miro_nodes:
                    if node["color"] == "#4ECDC4": node["color"] = "#00FFCC"
                    elif node["color"] == "#45B7D1": node["color"] = "#0099FF"
                    elif node["color"] == "#96CEB4": node["color"] = "#66FF99"
                    elif node["color"] == "#FECA57": node["color"] = "#FFCC00"
                    elif node["color"] == "#FF6B6B": node["color"] = "#FF3333"
                    elif node["color"] == "#A29BFE": node["color"] = "#9966FF"
                    node["size"] = node_size * 1.5
                    node["font"] = {"size": font_size, "color": "#FFFFFF"}

            elif visualization_mode == "Fluxograma":
                st.markdown('<div class="flowchart-box">', unsafe_allow_html=True)
                st.info("📋 **Modo Fluxograma**: Visualização estruturada!")
                node_size = 25
                font_size = st.session_state.settings.get("node_font_size", 14)
                physics_enabled = False
                hierarchical_enabled = True
                
                for node in st.session_state.miro_nodes:
                    node["shape"] = "square"
                    node["color"] = "#2E86AB"
                    node["size"] = node_size
                    node["font"] = {"size": font_size, "color": "#FFFFFF"}

            else:  # Mapa 2D padrão
                node_size = 25
                font_size = st.session_state.settings.get("node_font_size", 14)
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
                        y=node.get("y", 0)
                    )
                )

            edges_for_viz = []
            for edge in st.session_state.miro_edges:
                edges_for_viz.append(
                    Edge(
                        source=edge["source"],
                        target=edge["target"],
                        label=edge.get("label", ""),
                        color="#rgba(200,200,200,0.8)",
                        width=2
                    )
                )

            # Configuração do gráfico
            config = Config(
                width=800,
                height=600,
                directed=True,
                physics=physics_enabled,
                hierarchical=hierarchical_enabled,
                **({
                    "hierarchical": {
                        "enabled": hierarchical_enabled,
                        "levelSeparation": 150,
                        "nodeSpacing": 100,
                        "treeSpacing": 200,
                        "blockShifting": True,
                        "edgeMinimization": True,
                        "parentCentralization": True,
                        "direction": "UD",
                        "sortMethod": "hubsize"
                    }
                } if hierarchical_enabled else {})
            )

            # Renderizar o gráfico
            try:
                return_value = agraph(nodes=nodes_for_viz, edges=edges_for_viz, config=config)

                if return_value:
                    st.session_state.miro_selected_node = return_value
                    st.write(f"**Ideia selecionada:** {return_value}")

            except Exception as e:
                st.error(f"Erro ao renderizar o mapa: {e}")
                st.info("Tente reorganizar o mapa ou reduzir o número de ideias")

            # Fechar divs de estilo
            if visualization_mode == "Mapa 3D":
                st.markdown('</div>', unsafe_allow_html=True)
            elif visualization_mode == "Fluxograma":
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("🌟 **Comece criando sua primeira ideia!** Use o painel à esquerda para adicionar ideias e conectar conceitos.")

    with col2:
        st.subheader("📋 Ideias & Conexões")
        
        # Lista de ideias existentes
        if st.session_state.miro_nodes:
            st.write(f"**{len(st.session_state.miro_nodes)} ideias no mapa:**")
            
            for node in st.session_state.miro_nodes:
                is_selected = st.session_state.miro_selected_node == node["id"]
                emoji = "🔵" if not is_selected else "🟢"
                
                with st.expander(f"{emoji} {node['label']}", expanded=is_selected):
                    st.write(f"**Tipo:** {node['type']}")
                    if node.get('description'):
                        st.write(f"**Descrição:** {node['description']}")
                    
                    # Mostrar conexões
                    connections = []
                    for edge in st.session_state.miro_edges:
                        if edge['source'] == node['id']:
                            target_node = next((n for n in st.session_state.miro_nodes if n['id'] == edge['target']), None)
                            if target_node:
                                connections.append(f"→ {target_node['label']}")
                        elif edge['target'] == node['id']:
                            source_node = next((n for n in st.session_state.miro_nodes if n['id'] == edge['source']), None)
                            if source_node:
                                connections.append(f"← {source_node['label']}")
                    
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
                        if st.button("✏️ Editar", key=f"edit_{node['id']}", use_container_width=True):
                            st.session_state.editing_node = node['id']
                            safe_rerun()
                    
                    with col_btn2:
                        if st.button("🗑️ Excluir", key=f"delete_{node['id']}", use_container_width=True):
                            st.session_state.miro_nodes = [n for n in st.session_state.miro_nodes if n['id'] != node['id']]
                            st.session_state.miro_edges = [e for e in st.session_state.miro_edges if e['source'] != node['id'] and e['target'] != node['id']]
                            if st.session_state.miro_selected_node == node['id']:
                                st.session_state.miro_selected_node = None
                            st.success("Ideia removida!")
                            safe_rerun()
        
        # Editor de ideias
        if hasattr(st.session_state, 'editing_node'):
            editing_node_id = st.session_state.editing_node
            editing_node = next((n for n in st.session_state.miro_nodes if n['id'] == editing_node_id), None)
            
            if editing_node:
                st.markdown("---")
                st.subheader("✏️ Editando Ideia")
                
                with st.form(f"edit_node_{editing_node_id}"):
                    new_label = st.text_input("Título:", value=editing_node['label'].replace("💡 ", "").replace("✅ ", "").replace("❓ ", "").replace("📚 ", "").replace("🎯 ", "").replace("📝 ", ""), key=f"edit_label_{editing_node_id}")
                    new_type = st.selectbox("Tipo:", options=list(st.session_state.miro_map.node_types.keys()), index=list(st.session_state.miro_map.node_types.keys()).index(editing_node['type']), key=f"edit_type_{editing_node_id}")
                    new_desc = st.text_area("Descrição:", value=editing_node.get('description', ''), key=f"edit_desc_{editing_node_id}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("💾 Salvar", use_container_width=True):
                            editing_node['label'] = f"{st.session_state.miro_map.node_types[new_type]['icon']} {new_label}"
                            editing_node['type'] = new_type
                            editing_node['description'] = new_desc
                            editing_node['color'] = st.session_state.miro_map.node_types[new_type]['color']
                            editing_node['shape'] = st.session_state.miro_map.node_types[new_type]['shape']
                            del st.session_state.editing_node
                            st.success("Ideia atualizada!")
                            safe_rerun()
                    
                    with col2:
                        if st.form_submit_button("❌ Cancelar", use_container_width=True):
                            del st.session_state.editing_node
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
# Page: graficos - SIMPLIFICADO (APENAS 3 GRÁFICOS)
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
        if st.button("🔍 Gerar Análise Completa", use_container_width=True):
            with st.spinner("Analisando dados... Isso pode levar alguns segundos"):
                analyzer = DataAnalyzer(df)
                analysis = analyzer.generate_comprehensive_analysis()
                st.markdown(analysis)
        
        # Assistente de IA para perguntas
        st.subheader("💬 Assistente de IA - Faça uma Pergunta")
        question = st.text_input("Pergunte algo sobre seus dados:", 
                               placeholder="Ex: Quais são os autores mais relevantes? Como está a distribuição por anos?")
        
        if question:
            with st.spinner("Processando sua pergunta..."):
                analyzer = DataAnalyzer(df)
                response = get_ai_assistant_response(question, analyzer)
                st.markdown(response)
        
        # Visualizações gráficas SIMPLIFICADAS - APENAS 3 TIPOS
        st.subheader("📈 Visualizações Gráficas")
        
        # Seleção de tipo de gráfico - APENAS OS 3 SOLICITADOS
        chart_type = st.selectbox("Escolha o tipo de gráfico:", 
                                ["Barras", "Linhas", "Pizza"])
        
        # Configurações comuns
        col1, col2 = st.columns(2)
        
        with col1:
            # Eixo X
            x_axis = st.selectbox("Eixo X:", options=df.columns.tolist())
        
        with col2:
            # Eixo Y (para alguns gráficos)
            if chart_type in ["Barras", "Linhas"]:
                y_axis = st.selectbox("Eixo Y:", options=df.columns.tolist())
            else:
                y_axis = None
        
        # Gráficos específicos
        try:
            if chart_type == "Barras":
                if df[x_axis].dtype == 'object' and (y_axis and df[y_axis].dtype in ['int64', 'float64']):
                    # Gráfico de barras agrupado
                    fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} por {x_axis}")
                else:
                    # Contagem de valores categóricos
                    value_counts = df[x_axis].value_counts().head(15)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title=f"Distribuição de {x_axis}")
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Linhas":
                if y_axis and df[x_axis].dtype in ['datetime64[ns]', 'int64']:
                    fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} ao longo do {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Para gráfico de linhas, o eixo X deve ser numérico ou data.")
            
            elif chart_type == "Pizza":
                value_counts = df[x_axis].value_counts().head(8)
                fig = px.pie(values=value_counts.values, names=value_counts.index, 
                           title=f"Distribuição de {x_axis}")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Erro ao gerar gráfico: {e}")
            st.info("Tente selecionar diferentes colunas ou tipos de gráfico")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Page: busca - SEM FILTROS AVANÇADOS
# -------------------------
elif st.session_state.page == "busca":
    st.markdown("<div class='glass-box' style='position:relative;'><div class='specular'></div>", unsafe_allow_html=True)
    st.subheader("🔍 Busca Simples")
    
    try:
        with st.spinner("Carregando dados..."):
            df_total = collect_latest_backups()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        df_total = pd.DataFrame()

    if df_total.empty:
        st.info("Ainda não há dados disponíveis para busca.")
    else:
        # Interface de busca SIMPLIFICADA - SEM FILTROS AVANÇADOS
        search_query = st.text_input("Buscar em todas as planilhas:", 
                                   placeholder="Digite palavras-chave, autores, temas...",
                                   key="search_input_main")

        # Executar busca - SEM FILTROS AVANÇADOS
        if st.button("🔍 Executar Busca", use_container_width=True):
            if search_query:
                results = df_total.copy()
                
                # Busca em todas as colunas de texto
                mask = pd.Series(False, index=results.index)
                for col in results.columns:
                    if results[col].dtype == 'object':
                        mask = mask | results[col].str.contains(search_query, case=False, na=False)
                results = results[mask]
                
                st.session_state.search_results = results
                st.session_state.search_page = 1
                
                if results.empty:
                    st.info("Nenhum resultado encontrado.")
                else:
                    st.success(f"Encontrados {len(results)} resultados!")
            else:
                st.warning("Digite um termo de busca.")

        # Mostrar resultados
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

                for idx, (_, row) in enumerate(page_df.iterrows()):
                    user_src = row.get("_artemis_username", "N/A")
                    title = str(row.get('título') or row.get('titulo') or '(Sem título)')
                    author_snippet = (row.get('autor') or "")[:100]
                    year = row.get('ano') or ""
                    
                    # Destacar termos de busca
                    if search_query:
                        title = highlight_search_terms(title, search_query)
                        author_snippet = highlight_search_terms(author_snippet, search_query)
                    else:
                        title = escape_html(title)
                        author_snippet = escape_html(author_snippet)

                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">{title}</div>
                        <div class="small-muted">{author_snippet}</div>
                        <div class="small-muted">Ano: {escape_html(str(year))} • Fonte: {escape_html(user_src)}</div>
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
                            st.session_state.search_view_index = start + idx
                            safe_rerun()
                    st.markdown("---")
                
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
# Page: mensagens - MOSTRAR NOME EM VEZ DE CPF
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
            st.write(f"**{len(inbox_msgs)} mensagem(s) não lida(s)**" if UNREAD_COUNT > 0 else "Todas as mensagens lidas.")
            
            for msg in inbox_msgs:
                is_unread = not msg.get('read', False)
                unread_indicator = "🔵" if is_unread else "⚪"
                
                # CORREÇÃO: Mostrar nome em vez de CPF
                sender_name = load_users().get(msg['from'], {}).get('name', msg['from'])
                
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
            for msg in sent_msgs:
                # CORREÇÃO: Mostrar nome em vez de CPF
                recipient_name = load_users().get(msg['to'], {}).get('name', msg['to'])
                
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
        
        # Se é uma resposta, preencher alguns campos automaticamente
        reply_to_msg = None
        if st.session_state.get('reply_message_id'):
            reply_to_msg = next((m for m in all_msgs if m['id'] == st.session_state.reply_message_id), None)
        
        with st.form("compose_message", clear_on_submit=True):
            # Lista de usuários disponíveis - MOSTRAR NOMES
            users = load_users()
            user_options = {}
            for username, user_data in users.items():
                if username != USERNAME:
                    user_options[f"{user_data.get('name', username)} ({username})"] = username
            
            recipients = st.multiselect("Para:", options=list(user_options.keys()))
            subject = st.text_input("Assunto:", 
                                  value=f"Re: {reply_to_msg['subject']}" if reply_to_msg else "")
            body = st.text_area("Mensagem:", height=200,
                              value=f"\n\n---\nEm resposta à mensagem de {load_users().get(reply_to_msg['from'], {}).get('name', reply_to_msg['from'])}:\n{reply_to_msg['body'][:500]}..." if reply_to_msg else "")
            
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
                            sent_msg = send_message(USERNAME, recipient_username, subject, body, attachment)
                            st.success(f"Mensagem enviada para {recipient_display.split('(')[0].strip()}!")
                        
                        # Limpar estado de resposta se existir
                        if st.session_state.get('reply_message_id'):
                            st.session_state.reply_message_id = None
                        if st.session_state.get('compose_inline'):
                            st.session_state.compose_inline = False
                        
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
        # Escala de fonte
        font_scale = st.slider("Tamanho da fonte:", 
                              min_value=0.8, 
                              max_value=1.5, 
                              value=st.session_state.settings.get("font_scale", 1.0),
                              step=0.1,
                              help="Ajusta o tamanho geral do texto")
        
        # Tamanho da fonte dos nós (mapa mental)
        node_font_size = st.slider("Tamanho da fonte nos mapas:", 
                                  min_value=10, 
                                  max_value=24, 
                                  value=st.session_state.settings.get("node_font_size", 14),
                                  step=1,
                                  help="Tamanho do texto nos nós do mapa mental")
    
    with col2:
        # Altura dos gráficos
        plot_height = st.slider("Altura dos gráficos (px):", 
                               min_value=400, 
                               max_value=1200, 
                               value=st.session_state.settings.get("plot_height", 600),
                               step=100,
                               help="Altura padrão para visualizações de gráficos")
        
        # Opacidade dos nós
        node_opacity = st.slider("Opacidade dos nós:", 
                                min_value=0.3, 
                                max_value=1.0, 
                                value=st.session_state.settings.get("node_opacity", 0.8),
                                step=0.1,
                                help="Transparência dos elementos no mapa mental")

    # Aplicar configurações
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
                # Limpar estado da sessão
                for key in list(st.session_state.keys()):
                    if key not in ['authenticated', 'username', 'user_obj']:
                        del st.session_state[key]
                
                # Limpar arquivos de estado
                if USER_STATE.exists():
                    USER_STATE.unlink()
                
                st.success("Todos os dados locais foram removidos!")
                time.sleep(2)
                safe_rerun()
    
    with col4:
        if st.button("📥 Exportar Backup Completo", use_container_width=True):
            # Criar um arquivo ZIP com todos os dados do usuário
            import zipfile
            from io import BytesIO
            
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                # Adicionar estado atual
                state_data = {
                    "notes": st.session_state.get("notes", ""),
                    "favorites": st.session_state.get("favorites", []),
                    "settings": st.session_state.get("settings", {}),
                    "tutorial_completed": st.session_state.get("tutorial_completed", False)
                }
                zip_file.writestr("user_state.json", json.dumps(state_data, indent=2))
                
                # Adicionar backup da planilha se existir
                backup_path = st.session_state.get("last_backup_path")
                if backup_path and Path(backup_path).exists():
                    zip_file.write(backup_path, "planilha_backup.csv")
            
            st.download_button(
                "⬇️ Baixar Backup",
                data=zip_buffer.getvalue(),
                file_name=f"nugep_pqr_backup_{USERNAME}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip",
                use_container_width=True
            )

    # Informações do sistema
    st.subheader("ℹ️ Informações do Sistema")
    
    st.write(f"**Usuário:** {USERNAME}")
    st.write(f"**Nome:** {USER_OBJ.get('name', 'Não informado')}")
    st.write(f"**Bolsa:** {USER_OBJ.get('scholarship', 'Não informada')}")
    st.write(f"**Cadastrado em:** {USER_OBJ.get('created_at', 'Data não disponível')}")
    
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
