from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from datetime import datetime, timezone
import numpy as np
import os

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = os.environ.get("QDRANT_PORT", 6333)
QDRANT_USERNAME = os.environ.get("QDRANT_USERNAME", "admin")
QDRANT_PASSWORD = os.environ.get("QDRANT_PASSWORD", "admin")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "document_embeddings")

def main():
    model_name = os.environ.get("EMBBADING_MODEL", "HuggingFaceH4/mistral-7b")
    documents = []
    count = 0
    for root, _, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                count += 1
                print(f"{datetime.now(tz=timezone.utc).isoformat()} [ Loading file ] loading {file} to memory")
                loader = PyPDFLoader(os.path.join(root, file))
                documents.extend(loader.load())

    print(f"{datetime.now(tz=timezone.utc).isoformat()} [ Preparing text splitter ] using chunk size of 500 and chunk overlap of 100")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    print(f"{datetime.now(tz=timezone.utc).isoformat()} [ Preparing text splitter ] splitting {count} files")
    texts = text_splitter.split_documents(documents)

    print(f"{datetime.now(tz=timezone.utc).isoformat()} [ Preparing model ] preparing model <{model_name}>")
    embeddings = HuggingFaceEmbeddings(model_name="HuggingFaceH4/mistral-7b")

    print(f"{datetime.now(tz=timezone.utc).isoformat()} [ Generating embeddings ] generating embeddings for {len(texts)} chunks")
    embedding_vectors = []
    for text in texts:
        embedding = embeddings.embed_query(text.page_content)
        embedding_vectors.append(embedding)

    print(f"{datetime.now(tz=timezone.utc).isoformat()} [ Writing to database ] inserting embeddings into Qdrant")

    vector_store = Qdrant(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        client_kwargs={
            "host": QDRANT_HOST,
            "port": QDRANT_PORT,
        }
    )

    ids = [str(i) for i in range(len(embedding_vectors))]
    payload = [{"text": texts[i].page_content} for i in range(len(texts))]

    vector_store.add_texts(
        texts=[text.page_content for text in texts],
        metadatas=payload,
        ids=ids,
        embeddings=embedding_vectors
    )

    print(f"{datetime.now(tz=timezone.utc).isoformat()} [ Done ] embeddings inserted successfully")

if __name__ == "__main__":
    main()
