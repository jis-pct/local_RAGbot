import streamlit as st
import ollama
import os

from src.opensearch import hybrid_search, get_embedding_model
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource()
def pull_ollama_model(model: str) -> bool:
    try:
        available_models = ollama.list()
        if model not in available_models:
            ollama.pull(model)
    except ollama.ResponseError as e:
        print(f"Error checking or pulling model: {e.error}")
        return False
    return True

def get_ollama_response_stream(
    query: str,
    history: list,
    parameters: dict,
    system: str,
):
    
    history = history[-parameters['past_messages_included']:]
    context = ""

    # Include hybrid search results if enabled
    if parameters['use_hybrid']:
        if os.environ['ASYMMETRIC_EMBEDDING']:
            prefixed_query = f"passage: {query}"
        else:
            prefixed_query = f"{query}"
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode(
            prefixed_query
        ).tolist()  # Convert tensor to list of floats
        search_results = hybrid_search(query, query_embedding, top_k=parameters['top_n_documents'])

        # Collect text from search results
        for i, result in enumerate(search_results):
            context += f"{result['_source']['document_name']}:\n{result['_source']['text']}\n\n"

    # Generate prompt using the prompt_template function
    messages = prompt_template(query, context, history, system)

    return run_llama(messages, parameters)

def prompt_template(query: str, context: str, history: list, system: str) -> list:
    messages = [{"role": "system", "content": system}]
    messages += history

    if context:
        messages.append({"role": "user", "content": f"You will be asked a question. Here is the context to answer the question. The document name is above the respective chunk. Do not use any other sources of information: \n{context}"})
        messages.append({"role": "user", "content": f"Now, use the context to answer this question. DO NOT use ANY other sources of information outside of the documents above. Do not many any inferences. Cite the documents that you use: \n{query}"})
    else:
        messages.append({"role": "user", "content": query})

    return messages

def run_llama(messages: list, parameters: dict):
    try:
        [print(x.get("content")) for x in messages]
        stream = ollama.chat(
            model=os.environ['OLLAMA_MODEL_NAME'],
            messages=messages,
            stream=True,
            options={"temperature": parameters['temperature'],
                        "top_p": parameters['top_p'],
                        "top_k": parameters['top_k'],
                        "min_p": parameters['min_p'],
                        "repeat_penalty": parameters['repeat_penalty'],
                        "repeat_last_n": parameters['repeat_last_n'],
                        "num_predict": parameters['num_predict'],
                        "num_ctx": parameters['num_ctx'],
                        "stop": parameters['phrases']},
        )
    except ollama.ResponseError as e:
        print("stream error")
        return None

    return stream