import streamlit as st
import os
from PyPDF2 import PdfReader
from streamlit_autorefresh import st_autorefresh

from src.opensearch import get_opensearch_client, create_index, save_uploaded_file, get_embedding_model
from dotenv import load_dotenv

load_dotenv()

st.title("Document Selection")

os.makedirs(os.environ['UPLOAD_DIR'], exist_ok=True)
model_loading_placeholder = st.empty()

# Display the loading spinner at the top for loading the embedding model
if "embedding_models_loaded" not in st.session_state:
    with model_loading_placeholder:
        with st.spinner("Loading models for document processing..."):
            get_embedding_model()
            st.session_state["embedding_models_loaded"] = True
    model_loading_placeholder.empty()  # Clear the placeholder after loading

# Initialize OpenSearch client
with st.spinner("Connecting to OpenSearch..."):
    client = get_opensearch_client()
index_name = os.environ['OPENSEARCH_INDEX']
create_index(client)

st_autorefresh(interval=2000, key="auto_rerun")

# Initialize or clear the documents list in session state
st.session_state["documents"] = []

# Query OpenSearch to get the list of unique document names
query = {
    "size": 0,
    "aggs": {"unique_docs": {"terms": {"field": "document_name", "size": 10000}}},
}
response = client.search(index=index_name, body=query)
buckets = response["aggregations"]["unique_docs"]["buckets"]
document_names = [bucket["key"] for bucket in buckets]

# Load document information from the index
for document_name in document_names:
    file_path = os.path.join(os.environ['UPLOAD_DIR'], document_name)
    if os.path.exists(file_path):
        reader = PdfReader(file_path)
        text = "".join([page.extract_text() for page in reader.pages])
        st.session_state["documents"].append(
            {"filename": document_name, "content": text, "file_path": file_path}
        )
    else:
        st.session_state["documents"].append(
            {"filename": document_name, "content": "", "file_path": None}
        )

if "deleted_file" in st.session_state:
    st.success(
        f"The file '{st.session_state['deleted_file']}' was successfully deleted."
    )
    del st.session_state["deleted_file"]

# Allow users to upload PDF files
uploaded_files = st.file_uploader(
    "Upload PDF documents", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Uploading and processing documents. Please wait..."):
        for uploaded_file in uploaded_files:
            if uploaded_file.name in document_names:
                st.warning(
                    f"The file '{uploaded_file.name}' already exists in the index."
                )
                continue

            file_path = save_uploaded_file(uploaded_file)

if st.session_state["documents"]:
    st.markdown("### Uploaded Documents")
    with st.expander("Manage Uploaded Documents", expanded=True):
        for idx, doc in enumerate(st.session_state["documents"], 1):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(
                    f"{idx}. {doc['filename']} - {len(doc['content'])} characters extracted"
                )
            with col2:
                delete_button = st.button(
                    "Delete",
                    key=f"delete_{doc['filename']}_{idx}",
                    help=f"Delete {doc['filename']}",
                )
                if delete_button:
                    if doc["file_path"] and os.path.exists(doc["file_path"]):
                        try:
                            os.remove(doc["file_path"])
                        except FileNotFoundError:
                            st.error(
                                f"File '{doc['filename']}' not found in filesystem."
                            )