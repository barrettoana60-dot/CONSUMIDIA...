import streamlit as st
import datetime
import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from collections import Counter
from pathlib import Path
from PIL import Image
import uuid
import base64
import io

# ======================================================
# CONFIG BÁSICA
# ======================================================
st.set_page_config(
    page_title="PQR – Social Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)
STATE_FILE = "pqr_state.json"
AVATAR_DIR = Path("avatars")
AVATAR_DIR.mkdir(exist_ok=True)

# ======================================================
# CSS MODERNO AZUL ESCURO COM GLASSMORPHISM
# ======================================================
CSS = """
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f1f5f9;
    min-height: 100vh;
}
[data-testid="stSidebar"] { display: none; }
.main-container {
    display: flex;
    gap: 20px;
    padding: 20px;
    max-width: 1600px;
    margin: 0 auto;
}
.sidebar {
    width: 280px;
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px rgba(2, 8, 32, 0.3);
    height: fit-content;
    position: sticky;
    top: 20px;
}
.sidebar-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
}
.sidebar-avatar {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    border: 2px solid rgba(255, 255, 255, 0.1);
}
.sidebar-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.sidebar-user-info {
    display: flex;
    flex-direction: column;
}
.sidebar-user-name {
    font-size: 14px;
    font-weight: 600;
    color: #f8fafc;
}
.sidebar-user-role {
    font-size: 12px;
    color: #94a3b8;
}
.sidebar-menu {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.sidebar-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 12px;
    cursor: pointer;
    font-size: 14px;
    color: #cbd5e1;
    transition: all 0.3s;
    user-select: none;
    background: transparent;
    border: none;
    text-align: left;
    width: 100%;
}
.sidebar-item:hover {
    background: rgba(30, 41, 59, 0.5);
    color: #e2e8f0;
}
.sidebar-item-active {
    background: rgba(79, 70, 229, 0.2);
    color: #818cf8;
    font-weight: 600;
}
.sidebar-icon {
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.sidebar-label {
    flex: 1;
}
.content-wrapper {
    display: flex;
    gap: 20px;
    flex: 1;
    flex-direction: column;
}
.profile-card {
    width: 100%;
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px rgba(2, 8, 32, 0.3);
}
.profile-header {
    text-align: center;
    margin-bottom: 20px;
}
.profile-avatar-big {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    margin: 0 auto 16px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 3px solid rgba(255, 255, 255, 0.1);
}
.profile-avatar-big img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.profile-name {
    font-size: 20px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 4px;
}
.profile-role {
    font-size: 14px;
    color: #94a3b8;
    margin-bottom: 16px;
}
.profile-follow-btn {
    width: 100%;
    padding: 12px;
    border: none;
    border-radius: 12px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.profile-follow-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
}
.profile-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.3), transparent);
    margin: 16px 0;
}
.profile-stat {
    display: flex;
    justify-content: space-around;
    margin: 16px 0;
    text-align: center;
}
.profile-stat-item {
    flex: 1;
}
.profile-stat-number {
    font-size: 18px;
    font-weight: 700;
    color: #f8fafc;
}
.profile-stat-label {
    font-size: 12px;
    color: #94a3b8;
}
.profile-bio {
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.6;
    margin: 16px 0;
    text-align: center;
}
.profile-interests {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
    justify-content: center;
}
.profile-interest-tag {
    display: inline-block;
    background: rgba(30, 41, 59, 0.5);
    color: #e2e8f0;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.feed-card {
    flex: 1;
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px rgba(2, 8, 32, 0.3);
}
.feed-header {
    font-size: 18px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
}
.feed-item {
    display: flex;
    gap: 12px;
    padding: 16px 0;
    border-bottom: 1px solid rgba(148, 163, 184, 0.1);
}
.feed-item:last-child {
    border-bottom: none;
}
.feed-item-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    flex-shrink: 0;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px solid rgba(255, 255, 255, 0.1);
}
.feed-item-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.feed-item-content {
    flex: 1;
}
.feed-item-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
}
.feed-item-name {
    font-size: 14px;
    font-weight: 600;
    color: #f8fafc;
}
.feed-item-time {
    font-size: 12px;
    color: #94a3b8;
}
.feed-item-text {
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.5;
}
.feed-item-action {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 6px;
}
.auth-container {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}
.auth-card {
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(10px);
    border-radius: 24px;
    padding: 40px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 20px 50px rgba(2, 8, 32, 0.4);
    width: 100%;
    max-width: 400px;
}
.auth-title {
    font-size: 28px;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 8px;
}
.auth-subtitle {
    font-size: 14px;
    color: #94a3b8;
    text-align: center;
    margin-bottom: 24px;
}
.avatar-upload-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 20px;
}
.avatar-upload-btn {
    margin-top: 10px;
    background: rgba(79, 70, 229, 0.2);
    color: #818cf8;
    border: 1px solid rgba(129, 140, 248, 0.3);
    border-radius: 12px;
    padding: 8px 16px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.3s;
}
.avatar-upload-btn:hover {
    background: rgba(79, 70, 229, 0.3);
}
.glass-card {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 20px;
}
.glass-card h3 {
    font-size: 18px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 15px;
}
.glass-input {
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 12px 16px;
    color: #f1f5f9;
    width: 100%;
    margin-bottom: 15px;
}
.glass-input:focus {
    outline: none;
    border-color: #6366f1;
}
.glass-button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
    width: 100%;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.glass-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
}
.post-card {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.post-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 12px;
}
.post-author {
    font-size: 14px;
    font-weight: 600;
    color: #f8fafc;
}
.post-time {
    font-size: 12px;
    color: #94a3b8;
}
.post-content {
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.6;
    margin-bottom: 12px;
}
.post-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 12px;
}
.post-tag {
    display: inline-block;
    background: rgba(79, 70, 229, 0.2);
    color: #818cf8;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
}
.post-actions {
    display: flex;
    gap: 20px;
    margin-top: 12px;
    font-size: 14px;
}
.post-action {
    cursor: pointer;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: color 0.3s;
}
.post-action:hover {
    color: #818cf8;
}
.stories-container {
    display: flex;
    gap: 15px;
    overflow-x: auto;
    padding: 10px 0;
    margin-bottom: 20px;
}
.story {
    width: 80px;
    display: flex;
    flex-direction: column;
    align-items: center;
    flex-shrink: 0;
}
.story-avatar {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    border: 2px solid transparent;
    padding: 2px;
    background-clip: padding-box;
}
.story-avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 50%;
}
.story-name {
    font-size: 12px;
    color: #cbd5e1;
    margin-top: 6px;
    text-align: center;
    max-width: 80px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ======================================================
# DATACLASSES
# ======================================================
@dataclass
class User:
    id: str
    name: str
    email: str
    password: str
    role: str = "Pesquisador"
    bio: str = ""
    avatar_path: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    followers: List[str] = field(default_factory=list)
    following: List[str] = field(default_factory=list)

@dataclass
class Post:
    id: str
    author_id: str
    content: str
    tags: List[str] = field(default_factory=list)
    likes: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    shares: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@dataclass
class Comment:
    id: str
    post_id: str
    author_id: str
    text: str
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@dataclass
class TimelineStep:
    id: str
    owner_id: str
    title: str
    description: str
    status: str = "Não iniciado"
    due_date: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))

@dataclass
class ResearchDoc:
    id: str
    owner_id: str
    title: str
    category: str
    content: str
    link: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))

@dataclass
class MindNode:
    id: str
    owner_id: str
    title: str
    description: str
    tags: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

@dataclass
class Slide:
    id: str
    owner_id: str
    title: str
    objective: str
    content: str
    theme: str = "default"
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))

@dataclass
class ChatMessage:
    id: str
    from_id: str
    text: str
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@dataclass
class Notification:
    id: str
    user_id: str
    type: str
    from_user_id: str
    text: str = ""
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ======================================================
# INICIALIZAÇÃO DE SESSION STATE
# ======================================================
def init_session_state():
    if "users" not in st.session_state:
        st.session_state.users = []
    if "posts" not in st.session_state:
        st.session_state.posts = []
    if "comments" not in st.session_state:
        st.session_state.comments = []
    if "timeline" not in st.session_state:
        st.session_state.timeline = []
    if "docs" not in st.session_state:
        st.session_state.docs = []
    if "mindnodes" not in st.session_state:
        st.session_state.mindnodes = []
    if "slides" not in st.session_state:
        st.session_state.slides = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "notifications" not in st.session_state:
        st.session_state.notifications =

