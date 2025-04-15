A local RAG LLM (Ollama) using OpenSearch. Run using streamlit. Supports parameter modification and custom system messages.

Requires Docker and Ollama to be installed.

Run `python monitor_files.py` to start a background script that watches for changes in UPLOAD_DIR (can be modified in .env) and automatically indexes and removes files

`streamlit run ./Welcome.py` to run streamlit chatbot

Refer to https://jamwithai.substack.com/p/build-a-local-llm-based-rag-system-628 for setup instructions.
