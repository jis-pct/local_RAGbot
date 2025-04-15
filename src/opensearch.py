import json
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
import os
import numpy as np
from dotenv import load_dotenv
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from PyPDF2 import PdfReader
import functools

load_dotenv()

@functools.cache
def get_opensearch_client() -> OpenSearch:
    client = OpenSearch(
        hosts=[{"host": os.environ['OPENSEARCH_HOST'], "port": os.environ['OPENSEARCH_PORT']}],
        http_compress=True,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )
    return client

@functools.cache
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(os.environ['EMBEDDING_MODEL_PATH'])

def generate_embeddings(chunks: list) -> list:
    model = get_embedding_model()
    embeddings = [np.array(model.encode(chunk)) for chunk in chunks]
    return embeddings

def hybrid_search(
    query_text: str, query_embedding: list[float], top_k: int
) -> list:
    client = get_opensearch_client()

    query_body = {
        "_source": {"exclude": ["embedding"]},  # Exclude embeddings from the results
        "query": {
            "hybrid": {
                "queries": [
                    {"match": {"text": {"query": query_text}}},  # Text-based search
                    {"knn": {"embedding": {
                                "vector": query_embedding,
                                "k": top_k,
                            }
                        }
                    },
                ]
            }
        },
        "size": top_k,
    }

    response = client.search(
        index=os.environ['OPENSEARCH_INDEX'], body=query_body, search_pipeline="nlp-search-pipeline"
    )

    # Type casting for compatibility with expected return type
    hits = response["hits"]["hits"]
    return hits

# Index functions
@functools.cache
def load_index_config() -> dict:
    with open("src/index_config.json", "r") as f:
        config = json.load(f)

    config["mappings"]["properties"]["embedding"]["dimension"] = os.environ['EMBEDDING_DIMENSION']
    return config if isinstance(config, dict) else {}

def create_index(client: OpenSearch) -> None:
    index_body = load_index_config()
    index = os.environ['OPENSEARCH_INDEX']
    if not client.indices.exists(index=index):
        client.indices.create(index=index, body=index_body)

def index_documents(documents: list) -> tuple:
    actions = []
    client = get_opensearch_client()

    for doc in documents:
        doc_id = doc["doc_id"]
        embedding_list = doc["embedding"].tolist()
        document_name = doc["document_name"]

        # Prefix each document's text with "passage: " for the asymmetric embedding model
        if os.environ['ASYMMETRIC_EMBEDDING']:
            prefixed_text = f"passage: {doc['text']}"
        else:
            prefixed_text = f"{doc['text']}"

        action = {
            "_index": os.environ['OPENSEARCH_INDEX'],
            "_id": doc_id,
            "_source": {
                "text": prefixed_text,
                "embedding": embedding_list,  # Precomputed embedding
                "document_name": document_name,
            },
        }
        actions.append(action)

    # Perform bulk indexing and capture response details explicitly
    success, errors = helpers.bulk(client, actions)
    return success, errors

def delete_documents(document_name: str) -> dict:
    client = get_opensearch_client()
    query = {"query": {"term": {"document_name": document_name}}}
    response: dict = client.delete_by_query(
        index=os.environ['OPENSEARCH_INDEX'], body=query
    )
    return response

def clean_text(text: str) -> str:
    """
    Cleans OCR-extracted text by removing unnecessary newlines, hyphens, and correcting common OCR errors.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    # Remove hyphens at line breaks (e.g., 'exam-\nple' -> 'example')
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Replace newlines within sentences with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Replace multiple newlines with a single newline
    text = re.sub(r"\n+", "\n", text)

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)

    cleaned_text = text.strip()
    return cleaned_text

def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """
    Splits text into chunks with a specified overlap.

    Args:
        text (str): The text to split.
        chunk_size (int): The number of tokens in each chunk.
        overlap (int): The number of tokens to overlap between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    # Clean the text before chunking
    text = clean_text(text)

    # Tokenize the text into words
    tokens = text.split(" ")

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + int(chunk_size)
        chunk_tokens = tokens[start:end]
        chunk_text = " ".join(chunk_tokens)
        chunks.append(chunk_text)
        start = end - int(overlap)  # Move back by 'overlap' tokens
    return chunks

def save_uploaded_file(uploaded_file) -> str:  # type: ignore
    """
    Saves an uploaded file to the local file system.

    Args:
        uploaded_file: The uploaded file to save.

    Returns:
        str: The file path where the uploaded file is saved.
    """
    file_path = os.path.join(os.environ['UPLOAD_DIR'], uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def chunk_and_index_files(file_path: str):
    filename = os.path.basename(file_path)
    reader = PdfReader(file_path)
    text = "".join([page.extract_text() for page in reader.pages])
    chunks = chunk_text(text, chunk_size=os.environ['TEXT_CHUNK_SIZE'], overlap=os.environ['OVERLAP'])
    embeddings = generate_embeddings(chunks)

    documents_to_index = [
        {
            "doc_id": f"{filename}_{i}",
            "text": chunk,
            "embedding": embedding,
            "document_name": filename,
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    index_documents(documents_to_index)
    return filename

class DocumentHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.pdf')):
            print(f"[+] New file detected: {event.src_path}")
            time.sleep(0.1)
            chunk_and_index_files(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        print(f"[-] File deleted: {event.src_path}")
        time.sleep(0.1)
        delete_documents(os.path.basename(event.src_path))

def start_watching(directory: str):
    observer = Observer()
    handler = DocumentHandler()
    observer.schedule(handler, directory, recursive=False)
    observer.start()
    print(f"👀 Watching: {directory}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()