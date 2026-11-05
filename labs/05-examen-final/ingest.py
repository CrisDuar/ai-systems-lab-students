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

from evaluate_face_threshold import evaluate_far_frr

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

    #create_embeddings(settings.data_dir, settings.index_dir, settings.chunk_size, settings.chunk_overlap, settings.embedding_model)

    resultado = evaluate_far_frr(
    r"rostros\9062983711.npy",
        target_far=0.05,  # opcional: limitar FAR al 5 %
    )
    print(resultado)

if __name__ == "__main__":
    main()
