"""
chat.py

A local chatbot that uses LangChain, FAISS vectorstore, and a local LLaMA model to answer questions
based on documents in a specified directory. Embeds documents using HuggingFace sentence transformers
and supports source-aware QA.

Usage:
    Place your `.py` and `.md` files in the `fre-cli/` directory.
    Place the LLaMA model in the `models/` directory.
    Run the script and ask questions in the terminal.

Requirements:
    - langchain
    - langchain-community
    - langchain-unstructured
    - langchain-huggingface
    - sentence-transformers
    - llama-cpp-python (built with GPU support)
    - faiss-cpu
    - unstructured
    - torch

Author:
    Thomas Robinson
"""

import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import TextLoader

DATA_PATH = "fre-cli"
INDEX_PATH = "faiss_index"
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

def load_documents():
    print("📂 Loading .md and .py documents...")
    documents = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            full_path = os.path.join(root, file)
            try:
                if file.endswith(".md"):
                    loader = UnstructuredLoader(full_path, mode="elements")
                elif file.endswith(".py"):
                    loader = TextLoader(full_path)
                else:
                    continue

                loaded_docs = loader.load()
                for doc in loaded_docs:
                    if "LICENSE" in file.upper():
                        doc.metadata["is_license"] = True
                    doc.metadata["source"] = full_path
                documents.extend(loaded_docs)

            except Exception as e:
                print(f"⚠️ Skipping {file}: {e}")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []

    for doc in documents:
        if doc.metadata.get("is_license"):
            chunks.append(doc)
        else:
            chunks.extend(splitter.split_documents([doc]))

    return chunks

def load_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

def create_or_load_index(chunks, embeddings):
    print("📦 Loading or creating FAISS index...")
    index_file = os.path.join(INDEX_PATH, "index.faiss")

    if os.path.exists(index_file):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_PATH)
        return vectorstore

def load_llm():
    print("🧠 Loading local model...")
    n_threads = os.cpu_count() or 4

    force_cpu = os.getenv("FORCE_CPU", "false").lower() in ("1", "true", "yes")
    use_gpu = torch.cuda.is_available() and not force_cpu

    if force_cpu:
        print("🚫 FORCE_CPU enabled. Using CPU only.")
        n_gpu_layers = 0
    elif use_gpu:
        try:
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if total_mem_gb >= 24:
                n_gpu_layers = 0
            elif total_mem_gb >= 16:
                n_gpu_layers = 40
            elif total_mem_gb >= 8:
                n_gpu_layers = 20
            else:
                n_gpu_layers = 10
            print(f"⚡️ Detected CUDA GPU with {total_mem_gb:.1f} GB memory — using {n_gpu_layers} GPU layers.")
        except Exception as e:
            print(f"⚠️ GPU detection failed: {e}. Falling back to CPU.")
            n_gpu_layers = 0
    else:
        print("🖥️ No GPU available. Using CPU only.")
        n_gpu_layers = 0

    return LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.5,
        max_tokens=1024,
        top_p=0.9,
        top_k=50,
        n_ctx=4096,
        verbose=True,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        use_mlock=True,
    )

def build_qa_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

def start_chat(qa):
    print("🤖 Ready to chat! Type 'exit' to quit.\n")
    while True:
        query = input("🪨  You: ")
        if query.lower() in ("exit", "quit"):
            break
        result = qa.invoke({"query": query})
        print(f"\n🤖 Bot: {result['result']}\n")

        if result.get("source_documents"):
            print("📄 Sources:")
            for doc in result["source_documents"]:
                print(f"• {doc.metadata.get('source', 'Unknown')}")
            print()

def main():
    documents = load_documents()
    if not documents:
        print("❌ No documents loaded. Exiting.")
        return

    chunks = split_documents(documents)
    embeddings = load_embeddings()
    vectorstore = create_or_load_index(chunks, embeddings)
    llm = load_llm()
    qa = build_qa_chain(llm, vectorstore)
    start_chat(qa)

if __name__ == "__main__":
    main()
