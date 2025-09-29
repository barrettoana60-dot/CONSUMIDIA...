import streamlit as st
import json
import os
from datetime import datetime

USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4, ensure_ascii=False)

def register_ui():
    st.markdown("<h2 style='text-align:center;'>Cadastro de Usuário</h2>", unsafe_allow_html=True)

    nome = st.text_input("Nome completo")
    bolsa = st.selectbox("Tipo de bolsa", [
        "IC (Iniciação Científica)",
        "BIA (Bolsa de Incentivo Acadêmico)",
        "Extensão",
        "Doutorado"
    ])
    username = st.text_input("Username")
    senha = st.text_input("Senha", type="password")

    if st.button("Cadastrar", use_container_width=True):
        users = load_users()
        if username in users:
            st.error("❌ Esse username já está cadastrado.")
        else:
            users[username] = {
                "nome": nome,
                "bolsa": bolsa,
                "senha": senha,
                "created_at": str(datetime.now())
            }
            save_users(users)
            st.success("✅ Cadastro realizado com sucesso! Agora faça login.")
            st.session_state["show_login"] = True  

def login_ui():
    st.markdown("<h2 style='text-align:center;'>Login</h2>", unsafe_allow_html=True)

    username = st.text_input("Username", key="login_user")
    senha = st.text_input("Senha", type="password", key="login_pass")

    if st.button("Entrar", use_container_width=True):
        users = load_users()
        if username in users and users[username]["senha"] == senha:
            st.session_state["user"] = users[username]
            st.session_state["username"] = username
            st.success(f"👋 Bem-vindo, {users[username]['nome']}!")
            return True
        else:
            st.error("❌ Usuário ou senha incorretos.")
            return False
    return False

def auth_flow():
    if "user" in st.session_state:  # já logado
        return True  

    if st.session_state.get("show_login", False):  # mostrar login
        return login_ui()
    
    register_ui()  # caso contrário mostra cadastro
    return False
