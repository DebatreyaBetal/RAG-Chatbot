import os
import streamlit as st
import logging
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
# ─── Logging ───
logging.basicConfig(level=logging.INFO)

# Replace the placeholder strings below with your actual values:
AZURE_OPENAI_API_KEY="dc6ccd6606544f64a6321dfd11abda27"

AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="AI-GPT4o-MINI-Model"

AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION="2024-08-01-preview"

AZURE_OPENAI_CHAT_ENDPOINT="https://tml-az-dev-ca-aac-openai.openai.azure.com/"

AZURE_OPENAI_CHAT_MODEL_NAME="gpt-4o-mini"
#######################################################
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="text-embedding-ada-002"

AZURE_OPENAI_EMBEDDING_DEPLOYMENT_VERSION="2023-05-15"

AZURE_OPENAI_EMBEDDING_ENDPOINT="https://tml-az-dev-ca-aac-openai.openai.azure.com/"

AZURE_OPENAI_EMBEDDING_MODEL_NAME="text-embedding-ada-002"


# ─── Instantiate Azure OpenAI clients ───────────────────────────────────
llm = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
    deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    model=AZURE_OPENAI_CHAT_MODEL_NAME,
    api_version=AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION,
    temperature=0.8,
    max_tokens=4096,
)
embed_model = AzureOpenAIEmbedding(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
    deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    model=AZURE_OPENAI_EMBEDDING_MODEL_NAME,
    api_version=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_VERSION,
)

Settings.llm = llm
Settings.embed_model = embed_model

# ─── Configuration ──────────────────────────────────────────────────────
# Point this at the folder where your index sub-stores (docstore/, vector_store/, etc.) live
PERSIST_DIR = r"D:\d.betal\OneDrive - TATA MOTORS LTD\Vectorembed"


# 1) full-width layout & zero padding


st.set_page_config(layout="wide")
st.markdown(
    "<style>.block-container{padding-left:0;padding-right:0;}</style>",
    unsafe_allow_html=True,
)

logo_left   = r"C:\Users\d.betal\Downloads\TML_Logo.png"
logo_center = r"C:\Users\d.betal\Downloads\engaige-logo.png"
logo_right  = r"C:\Users\d.betal\Downloads\TopSopt _Logo.png"

# Top row: left and right logos
col1, col2, col3 = st.columns([1, 0.97, 0.3])
with col1:
    try:
        st.image(Image.open(logo_left), width=200)
    except:
        st.error("Left logo not found")
with col2:
    #st.empty()
    try:
        st.image(Image.open(logo_center), width=200)
    except:
        st.error("Center logo not found")
with col3:
    try:
        st.image(Image.open(logo_right), width=120)
    except:
        st.error("Right logo not found")

# Center the main logo below, full width
# col_center = st.columns([1, 0.35, 1])[1]  # grab the middle column
# with col_center:
#     try:
#         st.image(Image.open(logo_center), width=200)
#     except:
#         st.error("Center logo not found")

# st.markdown("---")


# ─── Question Input ──────────────────────────────────────────────────────
# question = st.text_input("Ask a question:")
col_q1, col_q2, col_q3 = st.columns([2, 4, 2])  # Center the middle column
with col_q2:
    question = st.text_input("Ask a question:", placeholder="Type your question here...")

    if question:
        # Load index (once per query)
        try:
            ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            idx = load_index_from_storage(ctx, index_id="vector_index")
            engine = idx.as_query_engine()
        except Exception as e:
            st.error(f"❌ Could not load vector store:\n{e}")
            st.stop()
    
        with st.spinner("🔍 Retrieving answer…"):
            resp = engine.query(question)
    
        # Answer
        st.subheader("Answer")
        st.write(getattr(resp, "response", str(resp)))
    
        # Sources
        if hasattr(resp, "source_nodes"):
            st.subheader("Sources")
            for src in resp.source_nodes:
                node = src.node
                text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
                snippet = text.replace("\n", " ")[:200].strip() + "…"
                st.write(f"- {snippet} _(score: {src.score:.4f})_")