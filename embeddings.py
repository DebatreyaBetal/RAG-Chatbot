import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
from docx import Document as DocxDocument
import json   # <-- NEW import

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

# Logging setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure OpenAI credentials & endpoints (hard-coded)
AZURE_OPENAI_API_KEY="your api key"
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME="AI-GPT4o-MINI-Model"
AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION="2024-08-01-preview"
AZURE_OPENAI_CHAT_ENDPOINT="https://tml-az-dev-ca-aac-openai.openai.azure.com/"
AZURE_OPENAI_CHAT_MODEL_NAME="gpt-4o-mini"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="text-embedding-ada-002"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_VERSION="2023-05-15"
AZURE_OPENAI_EMBEDDING_ENDPOINT="https://tml-az-dev-ca-aac-openai.openai.azure.com/"
AZURE_OPENAI_EMBEDDING_MODEL_NAME="text-embedding-ada-002"

# Instantiate Azure OpenAI clients
llm = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_CHAT_ENDPOINT,
    deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
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

# Embed all documents in directory
def embed_all_documents(doc_dir: str, output_index_dir: str):
    doc_dir = Path(doc_dir)
    out_dir = Path(output_index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading docs from {doc_dir}...")
    reader = SimpleDirectoryReader(input_dir=str(doc_dir), recursive=True)
    docs = reader.load_data(show_progress=True)
    if not docs:
        logger.warning("No documents found to embed.")
        return
    index = VectorStoreIndex.from_documents(docs)
    index.set_index_id("vector_index")
    index.storage_context.persist(str(out_dir))
    logger.info(f" Embedded and saved index to {out_dir}")

# Extract and embed DOCX tables
def extract_and_embed_tables_from_docx(docx_path: str, output_index_dir: str):
    path = Path(docx_path)
    if not path.exists():
        logger.error(f"DOCX not found: {path}")
        return
    docx = DocxDocument(str(path))
    table_texts = []
    for tbl_idx, table in enumerate(docx.tables, start=1):
        rows = []
        for row in table.rows:
            cells = [cell.text.replace('\n', ' ').strip() for cell in row.cells]
            rows.append("|".join(cells))
        table_text = "\n".join(rows)
        logger.info(f"Extracted table #{tbl_idx} with {len(rows)} rows")
        table_texts.append((tbl_idx, table_text))
    if not table_texts:
        logger.info("No tables found.")
        return
    storage_context = StorageContext.from_defaults(persist_dir=output_index_dir)
    index = load_index_from_storage(storage_context, index_id="vector_index")
    temp_dir = Path(tempfile.mkdtemp(prefix="tbl_"))
    try:
        for tbl_idx, txt in table_texts:
            tmp_file = temp_dir / f"table_{tbl_idx}.txt"
            tmp_file.write_text(txt, encoding="utf-8")
            docs = SimpleDirectoryReader(input_files=[str(tmp_file)]).load_data()
            for doc in docs:
                index.insert(doc)
            logger.info(f"Inserted table #{tbl_idx}")
        index.storage_context.persist(str(output_index_dir))
        logger.info(" Embedded DOCX tables into index")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# --- NEW: Embed a JSON file as text -----------------------------------
def embed_json_file(json_path: str, output_index_dir: str):
    """
    Loads a JSON file, pretty-prints as string, embeds into vector index.
    """
    json_path = Path(json_path)
    output_index_dir = Path(output_index_dir)
    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            logger.error(f"Could not parse JSON: {e}")
            return

    # Convert JSON object to a pretty-printed string
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    # Write to temp .txt so SimpleDirectoryReader can pick it up
    tmp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt", encoding="utf-8")
    tmp_file.write(json_str)
    tmp_file.close()

    # Load and insert to index
    storage_context = StorageContext.from_defaults(persist_dir=str(output_index_dir))
    index = load_index_from_storage(storage_context, index_id="vector_index")
    docs = SimpleDirectoryReader(input_files=[tmp_file.name]).load_data()
    for doc in docs:
        index.insert(doc)
    index.storage_context.persist(str(output_index_dir))
    logger.info(f" Embedded JSON from {json_path.name} into index.")
    os.unlink(tmp_file.name)

# Main
if __name__ == "__main__":
    ALL_DOCS_DIR = r"D:\d.betal\OneDrive - TATA MOTORS LTD\QuickRAG"
    INDEX_DIR = r"D:\d.betal\OneDrive - TATA MOTORS LTD\Vectorembed"
    # 1. Embed PDFs + DOCX text
    embed_all_documents(ALL_DOCS_DIR, INDEX_DIR)
    # 2. Only embed real tables from one known DOCX
    #extract_and_embed_tables_from_docx(r"data\Ph_D_Policy_Dtaft_V1_updated_as_on_25_Dec_2024.docx", INDEX_DIR)
    # 3. Embed a JSON file (example path):
    # embed_json_file(r"D:\d.betal\OneDrive - TATA MOTORS LTD\QuickRAG\yourdata.json", INDEX_DIR)
