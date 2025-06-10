"""
chat.py
=======

A command-line chatbot for querying the fre-cli source code and documentation using a local LLM (Llama.cpp) and FAISS vector store.

This script loads `.md`, `.py`, and `.rst` files from the `fre-cli` directory, builds or loads a FAISS index using HuggingFace sentence-transformer embeddings, and enables interactive question-answering over the indexed content. The chatbot uses a local LLaMA model for generating answers and can optionally display the source documents used in each answer.

Arguments
---------
--sources : bool, optional
    If set, the chatbot will display the source documents used to generate each answer.
--chunk-size : int, optional
    The size of each text chunk for embedding (default: 500).
--chunk-overlap : int, optional
    The overlap between text chunks for embedding (default: 150).
--model-path : str, optional
    Path to the LLaMA model file (default: models/mistral-7b-instruct-v0.2.Q4_K_M.gguf).

Usage
-----
Run the chatbot from the command line:

    python chat.py [--sources] [--chunk-size N] [--chunk-overlap N] [--model-path PATH]

- Type your question at the prompt.
- Type 'exit' or 'quit' to end the session.

Requirements
------------
- Python 3.8+
- All dependencies listed in environment.yml
- Downloaded LLaMA model in the models/ directory

"""

import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader, UnstructuredRSTLoader
import logging
from typing import List, Dict, Any
import textwrap

DATA_PATH = "fre-cli"
INDEX_PATH = "faiss_index"
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
#MODEL_PATH = "models/capybarahermes-2.5-mistral-7b.Q8_0.gguf"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

EXTENSION_LOADERS: Dict[str, Any] = {
    ".md": UnstructuredMarkdownLoader,  # Markdown files
    ".py": TextLoader,                  # Python source files
    ".rst": TextLoader,                 # reStructuredText files
}
"""
EXTENSION_LOADERS maps file extensions to their respective loader classes. To add support for a new file type,
add an entry here with the extension as the key and the loader class as the value.
"""

def load_documents() -> List[Any]:
    """
    Load all `.md`, `.py`, and `.rst` files from the fre-cli directory using multithreading.

    Returns
    -------
    list
        A list of langchain Document objects, each with metadata including the source file path.
    """
    logging.info("📂 Loading .md, .py, and .rst documents from fre-cli...")
    files_to_process = []
    for root, _, files in os.walk(DATA_PATH):
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext in EXTENSION_LOADERS:
                files_to_process.append(os.path.join(root, filename))

    def load_file(full_path: str) -> List[Any]:
        ext = os.path.splitext(full_path)[1]
        try:
            loader = EXTENSION_LOADERS[ext](full_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = full_path
            return docs
        except Exception as e:
            logging.warning(f"⚠️ Skipping {full_path}: {e}")
            return []

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(load_file, files_to_process), total=len(files_to_process), desc="🔍 Loading files"))
        documents = [doc for sublist in results for doc in sublist]

    return documents


def create_or_load_index(documents: List[Any], chunk_size: int = 500, chunk_overlap: int = 150) -> Any:
    """
    Load an existing FAISS index or create a new one from the provided documents using HuggingFace embeddings.

    Parameters
    ----------
    documents : list
        List of langchain Document objects to index.
    chunk_size : int, optional
        The size of each text chunk for embedding (default is 500).
    chunk_overlap : int, optional
        The overlap between chunks (default is 150).

    Returns
    -------
    FAISS
        A FAISS vectorstore object for document retrieval.
    """
    logging.info("📦 Loading or creating FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    index_file = os.path.join(INDEX_PATH, "index.faiss")

    if os.path.exists(index_file):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(INDEX_PATH)
        return vectorstore


def smart_print(text, width=100):
    """
    Print text to the console, wrapping non-list lines to the specified width while preserving list formatting.

    Parameters
    ----------
    text : str
        The text to print, possibly containing bulleted or numbered lists.
    width : int, optional
        The maximum line width for non-list lines (default is 100).
    """
    for line in text.splitlines():
        if line.strip().startswith(("-", "*", "•")):
            print(line)
        else:
            print(textwrap.fill(line, width=width))


def main() -> None:
    """
    Command-line entry point for the local chatbot over the fre-cli source code.
    Parses arguments, loads documents, builds or loads the FAISS index, loads the LLaMA model, and starts the chat loop.
    """
    parser = argparse.ArgumentParser(description="Local chatbot over fre-cli source")
    parser.add_argument("--sources", action="store_true", help="Show source documents used in answers")
    parser.add_argument("--chunk-size", type=int, default=500, help="Text chunk size for embedding (default: 500)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap for embedding (default: 150)")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Path to the LLaMA model file")
    args = parser.parse_args()

    try:
        documents = load_documents()
        if not documents:
            logging.error("❌ No documents found in fre-cli/. Exiting.")
            return

        vectorstore = create_or_load_index(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

        logging.info("🧠 Loading local LLaMA model...")
        try:
            n_oscores = os.cpu_count()
            n_oscores = n_oscores if n_oscores is not None else 0
            llm = LlamaCpp(
                model_path=args.model_path,
                temperature=0.4,
                max_tokens=768,
                top_p=1,
                n_ctx=4096,
                verbose=False,
                n_threads=max(1, n_oscores // 4),  # Use half of CPU cores to avoid oversubscription
            )
        except Exception as e:
            logging.error(f"❌ Failed to load LLaMA model: {e}")
            return

        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=args.sources
        )

        print("🤖 Ready to chat! Type 'exit' to quit.\n")
        while True:
            try:
                query = input("🗨️  You: ")
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Exiting chat.")
                break
            if query.lower() in ("exit", "quit"):
                break

            try:
                result = qa.invoke({"query": query})
                print("\n🤖 Bot:")
                smart_print(result['result'])
                print()
            except Exception as e:
                logging.error(f"❌ Error during QA invocation: {e}")
                continue

            if args.sources and result.get("source_documents"):
                print("📚 Sources used:")
                for i, doc in enumerate(result["source_documents"], 1):
                    source = doc.metadata.get("source", "unknown")
                    snippet = doc.page_content.strip().replace("\n", " ")
                    print(f"\n  {i}. 📄 {source}\n     📝 {textwrap.shorten(snippet, width=300, placeholder='...')}")
                print()
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
