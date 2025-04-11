import streamlit as st

st.set_page_config(
    page_title="Selective RAG Chatbot",
    page_icon="🦜",
)

st.write("# Welcome to the selective RAG chatbot 🦜")

st.markdown(
    """
    This is a chatbot that lets you talk to any uploaded documents.
    Upload documents, then start chatting to the bot!"""
)