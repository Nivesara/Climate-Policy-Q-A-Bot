# app.py (Full Chat Application Version)
import streamlit as st
import os
import pickle
import time
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    import openai
except ImportError:
    openai = None

# --- Core RAG Logic (No changes needed here) ---

INDEX_FILE = os.path.join(os.path.dirname(__file__), "index.pkl")
PERSONAS = {
    "plain": "You are a helpful assistant who explains climate policy in plain English. Avoid jargon and be concise.",
    "policy": "You are a policy analyst. Provide concise, formal responses and cite sources.",
    "journalist": "You are a climate journalist. Lead with the key finding, then support it with evidence from the context. Keep it readable.",
}

@st.cache_resource
def load_index():
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)

def retrieve(question: str, data: dict, top_k: int) -> tuple[list[str], list[str]]:
    """Retrieve top-k snippets and their sources."""
    vectorizer = data["vectorizer"]
    matrix = data["matrix"]
    q_vec = vectorizer.transform([question])
    sim = cosine_similarity(q_vec, matrix).flatten()
    top_indices = np.argsort(-sim)[:top_k].tolist()
    
    snippets = [data['chunks'][idx] for idx in top_indices]
    sources = []
    seen_sources = set()
    for idx in top_indices:
        meta = data['metadata'][idx]
        source_str = f"📄 **{meta['source']}** (page {meta['page']})"
        if source_str not in seen_sources:
            sources.append(source_str)
            seen_sources.add(source_str)
            
    return snippets, sorted(list(sources))

def build_prompt(question: str, snippets: List[str], persona: str) -> str:
    persona_prompt = PERSONAS.get(persona, PERSONAS["plain"])
    context = "\n\n".join(snippets)
    return (
        f"{persona_prompt}\n\n"
        "Answer the question using only the information provided in the context."
        " If the context does not contain enough information, say 'I don't know based on the provided documents.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

def call_llm(prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("OPENAI_API_KEY is not set. LLM generation is disabled.")
        return ""
    if openai is None:
        st.error("The `openai` library is not installed. Please run `pip install openai`.")
        return ""

    model = os.environ.get("LLM_MODEL", "llama3:8b")
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    max_tokens = int(os.environ.get("MAX_TOKENS", 512))
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True # Enable streaming for a better user experience
        )
        # Stream the response
        return response
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        return ""

# --- Streamlit Chat Interface ---

st.set_page_config(page_title="Climate Policy Chatbot", layout="wide")
st.title("🌍 Climate Policy Chatbot")

# Load data and initialize chat history
data = load_index()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    persona = st.selectbox("Answer Style (Persona):", list(PERSONAS.keys()))
    top_k = st.slider("Number of Passages to Retrieve:", min_value=1, max_value=15, value=6)
    st.info("Remember to run `ollama serve` in a separate terminal for this app to work.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Show Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

# Accept user input
if prompt := st.chat_input("Ask a question about climate policy..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 1. Retrieve context
        with st.spinner("Searching documents..."):
            snippets, sources = retrieve(prompt, data, top_k)
        
        # 2. Generate response
        prompt_for_llm = build_prompt(prompt, snippets, persona)
        response_stream = call_llm(prompt_for_llm)
        
        if response_stream:
            for chunk in response_stream:
                full_response += chunk.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        else:
            # Fallback if LLM fails or is disabled
            full_response = "I couldn't generate a summary. Here are the most relevant passages I found:\n\n---\n\n" + "\n\n".join(snippets)
            message_placeholder.markdown(full_response)
        
        # Show sources in an expander
        with st.expander("Show Sources"):
            for source in sources:
                st.markdown(f"- {source}")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})
