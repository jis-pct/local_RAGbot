import streamlit as st
import os

from dotenv import load_dotenv
from src.opensearch import get_opensearch_client, create_index, get_embedding_model
from src.chat import pull_ollama_model, get_ollama_response_stream

load_dotenv()

st.title("Selective RAG Chatbot")

# Connect to opensearch
client = get_opensearch_client()

# Create index
create_index(client)

# Load models
if "embedding_models_loaded" not in st.session_state:
    get_embedding_model()
    pull_ollama_model(os.environ['OLLAMA_MODEL_NAME'])
    st.session_state["embedding_models_loaded"] = True

# Init message history and params
if "messages" not in st.session_state:
    st.session_state.messages = []
if "parameters" not in st.session_state:
    st.session_state.parameters = {}

# Sidebar inputs for system message and other parameters
st.session_state.system_msg = st.sidebar.text_area("System Message", value=
"""You are an AI assistant that helps users find information.""")

# Search parameters
st.sidebar.header("Search Parameters")
st.session_state.parameters['use_hybrid'] = st.sidebar.checkbox("Use hybrid search", value = True)
st.session_state.parameters['strictness'] = st.sidebar.slider("Strictness", min_value=1, max_value=5, value=3)
st.session_state.parameters['top_n_documents'] = st.sidebar.slider("Retrieved documents", min_value=3, max_value=20, value=5)

# Model parameters
st.sidebar.header("Model Parameters")
st.session_state.parameters['past_messages_included'] = st.sidebar.slider("Past messages included", min_value=1, max_value=20, value=10)
st.session_state.parameters['num_predict'] = st.sidebar.slider("Max response tokens", min_value=1, max_value=16000, value=800)
st.session_state.parameters['num_ctx'] = st.sidebar.slider("Context window", min_value=0, max_value=10000, value=4096)
st.session_state.parameters['repeat_last_n'] = st.sidebar.slider("Repeat last n", min_value=-1, max_value=1000, value=64)
st.session_state.parameters['temperature'] = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0)
st.session_state.parameters['top_p'] = st.sidebar.slider("Top P", min_value=0.10, max_value=1.0, value=0.9)
st.session_state.parameters['top_k'] = st.sidebar.slider("Top K", min_value=0, max_value=10, value=40)
st.session_state.parameters['min_p'] = st.sidebar.slider("Min P", min_value=0.0, max_value=0.10, value=0.0)
stop_phrase = [st.sidebar.text_input("Stop phrase")]
st.session_state.parameters['phrases'] = stop_phrase if stop_phrase[0] != '' else None
st.session_state.parameters['repeat_penalty'] = st.sidebar.slider("Repeat penalty", min_value=0.0, max_value=2.0, value=1.1)

# Write all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Receive user input and generate a response
if prompt := st.chat_input("..."):
    
    # Display user input
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Send the prompt to the model and write the response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        response_text = ""
        response_stream = get_ollama_response_stream(prompt, st.session_state.messages, st.session_state.parameters, st.session_state.system_msg)

        if response_stream is not None:
            for chunk in response_stream:
                if (
                    chunk and 
                    "message" in chunk
                    and "content" in chunk["message"]
                ):
                    response_text += chunk["message"]["content"]
                    response_placeholder.markdown(response_text + "▌")
                else:
                    print("Unexpected chunk format:", chunk)
        st.session_state.messages += [{"role": "user", "content": prompt},
                                    {"role": "assistant", "content": response_text}]
        response_placeholder.markdown(response_text)
