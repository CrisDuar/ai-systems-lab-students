"""Pipeline de indexación (offline): documentos → chunks → embeddings → vector store en disco.

Se ejecuta una vez, y de nuevo cada vez que cambian los documentos o la configuración
de chunking o de embeddings. No usa el LLM ni necesita API key.

Uso:
    uv run python labs/04-agent-orchestration/ingest.py [--show]
"""

import sys
import time

from config import load_rag_settings
from documents import chunk_documents, load_documents
from embeddings import Embedder
from vector_store import VectorStore

DOCUMENTS_PATHS = [
    'publico',
    'restringido/docente',
    'restringido/estudiante'
]

def create_embeddings(path, index_dir, chunk_size, chunk_overlap, embedding_model):

    documents = load_documents(path)

    if not documents:
        print(f"No hay documentos .md en {path}")
        return

    chunks = chunk_documents(documents, chunk_size, chunk_overlap)

    start = time.perf_counter()
    embedder = Embedder(embedding_model)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    elapsed = time.perf_counter() - start

    print(f"{vectors.shape[0]} vectores de dimensión {vectors.shape[1]} en {elapsed:.1f} s")
    
    store = VectorStore(embedding_model)
    store.add(chunks, vectors)
    store.save(index_dir)

def main() -> None:

    try:
        settings = load_rag_settings()
    except ValueError as exc:
        print(f"[configuración] {exc}")
        return

    for doc in settings.document_paths:

        create_embeddings(settings.data_dir / doc, settings.index_dir / doc, settings.chunk_size, settings.chunk_overlap, settings.embedding_model)
    """ 
    show = "--show" in sys.argv
    try:
        settings = load_rag_settings()
    except ValueError as exc:
        print(f"[configuración] {exc}")
        return

    documents = load_documents(settings.data_dir / 'publico')

    if not documents:
        print(f"No hay documentos .md en {settings.data_dir}")
        return

    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)

    print(
        f"{len(documents)} documentos → {len(chunks)} chunks "
        f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap} palabras)"
    )

    if show:
        for chunk in chunks:
            print(f"\n[{chunk.id}] ({len(chunk.text.split())} palabras)\n{chunk.text}")

    print(f"\nCalculando embeddings con {settings.embedding_model} ...")

    start = time.perf_counter()
    embedder = Embedder(settings.embedding_model)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    elapsed = time.perf_counter() - start

    print(f"{vectors.shape[0]} vectores de dimensión {vectors.shape[1]} en {elapsed:.1f} s")

    store = VectorStore(settings.embedding_model)
    store.add(chunks, vectors)
    store.save(settings.index_dir)

    print(f"Índice guardado en {settings.index_dir}")
    """


if __name__ == "__main__":
    main()
