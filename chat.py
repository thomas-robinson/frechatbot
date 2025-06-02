import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

DATA_PATH = "fre-cli"
INDEX_PATH = "faiss_index"
MODEL_PATH = "models/ggml-model-q4_0.gguf"


def load_documents():
    """
    Load all `.md` and `.py` documents from the specified data path.

    Uses `UnstructuredMarkdownLoader` for markdown files and `TextLoader` for Python files.
    Skips any files that raise an exception during loading.

    :return: A list of loaded documents.
    :rtype: list
    """
    print("📂 Loading .md and .py documents...")
    documents = []
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            full_path = os.path.join(root, file)
            try:
                if file.endswith(".md"):
                    loader = UnstructuredMarkdownLoader(full_path)
                elif file.endswith(".py"):
                    loader = TextLoader(full_path)
                else:
                    continue
                documents.extend(loader.load())
            except Exception as e:
                print(f"⚠️ Skipping {file}: {e}")
    return documents


def create_or_load_index(documents):
    """
    Create a new FAISS vector index or load an existing one from disk.

    Splits the documents into chunks and generates embeddings using HuggingFace's
    'all-MiniLM-L6-v2' model.

    :param documents: List of documents to index.
    :type documents: list
    :return: FAISS vector store.
    :rtype: FAISS
    """
    print("📦 Loading or creating FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_file = os.path.join(INDEX_PATH, "index.faiss")

    if os.path.exists(index_file):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(INDEX_PATH)
        return vectorstore


def main():
    """
    Entry point for the chatbot script.

    Loads the documents, creates or loads a FAISS index, initializes the LLM,
    and starts an interactive chat loop with the user.
    """
    documents = load_documents()
    if not documents:
        print("❌ No documents loaded. Exiting.")
        return

    vectorstore = create_or_load_index(documents)

    print("🧠 Loading local model...")
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        n_ctx=2048,
        verbose=True,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    print("🤖 Ready to chat! Type 'exit' to quit.\n")
    while True:
        query = input("🗨️  You: ")
        if query.lower() in ("exit", "quit"):
            break
        result = qa.run(query)
        print(f"🤖 Bot: {result}\n")


if __name__ == "__main__":
    main()
