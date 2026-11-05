"""Vector store en memoria: guarda fragmentos con sus vectores y busca los más parecidos.

Es deliberadamente simple (una matriz de NumPy y una lista de fragmentos) para ver
qué hacen por dentro Chroma, FAISS o pgvector: guardar vectores y encontrar los más
cercanos a una consulta. Busca por fuerza bruta (compara la consulta con todos los
vectores), suficiente para miles de fragmentos.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np

from documents import Chunk

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"

WARNING_CHUNK = Chunk(
    id="system#auth_required",
    source="sistema_de_seguridad",
    text=(
        "[AVISO DEL SISTEMA]: La información solicitada corresponde a un documento "
        "restringido al que este usuario no tiene acceso. Debes responderle de forma "
        "amable que necesita iniciar sesión o autenticarse con el rol adecuado para "
        "consultar este contenido, sin inventar respuestas ni revelar información protegida."
    )
)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # similitud coseno entre la consulta y el fragmento (de -1 a 1)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Divide cada vector por su norma. Después, producto punto = similitud coseno."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class VectorStore:
    def __init__(self, embedding_model: str):
        # Los vectores solo son comparables con otros producidos por el mismo modelo.
        self.embedding_model = embedding_model
        self.chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None  # una fila normalizada por fragmento

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Debe haber exactamente un vector por fragmento.")
        vectors = normalize(np.asarray(vectors, dtype=np.float32))
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int, role: str | None = None) -> list[SearchResult]:
        """Devuelve los k fragmentos más parecidos a la consulta, del más al menos parecido.""" 
        if self._vectors is None:
            return []

        print(f"ROL: {role}")

        #Primero se busca en general para tener toda la información a la mano
        query_vector = normalize(np.asarray(query_vector, dtype=np.float32))
        scores = self._vectors @ query_vector#normalize(query_vector)
        top = np.argsort(scores)[::-1][:k]

        if len(top) == 0:
            return []

        # Se revisa el mejor resultado para ver su origen y determinar si la información requiere permisos
        best_chunk = self.chunks[top[0]]
        best_score = float(scores[top[0]])

        if (not best_chunk.source.lower().startswith("publico/")):
            if (not best_chunk.source.lower().startswith(f"{role}/")):
                return [SearchResult(chunk=WARNING_CHUNK, score=best_score)]
            
        valid_data = []
        for i in top:
            # Revisar si los chunks que se van a devolver deben ser accesibles por el rol o no
            chunk = self.chunks[i]
            #valid_data.append(i)
            if (chunk.source.lower().startswith("publico/") or chunk.source.lower().startswith(f"{role}/")):
                    valid_data.append(i)

        if (valid_data):
            return [SearchResult(chunk=self.chunks[i], score=float(scores[i])) for i in valid_data[:k]]
        
        best_score = float(scores[top[0]])
        return [SearchResult(chunk=WARNING_CHUNK, score=best_score)]
        return results

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / VECTORS_FILE, self._vectors)
        data = {"embedding_model": self.embedding_model, "chunks": [asdict(c) for c in self.chunks]}
        (index_dir / CHUNKS_FILE).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        chunks_path = index_dir / CHUNKS_FILE
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"No existe el índice en {index_dir}. Ejecuta primero ingest.py."
            )
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        store = cls(data["embedding_model"])
        store.chunks = [Chunk(**c) for c in data["chunks"]]
        store._vectors = np.load(index_dir / VECTORS_FILE)
        return store

    @classmethod
    def load_multiple(cls, index_dir: Path, levels: List[str]) -> "VectorStore":

        all_chunks = []
        all_vectors = []

        for lvl in levels:
            
            chunks_path = index_dir / lvl / CHUNKS_FILE

            if not chunks_path.exists():
                raise FileNotFoundError(
                    f"No existe el índice en {index_dir / lvl}. Ejecuta primero ingest.py."
                )
        
            data = json.loads(chunks_path.read_text(encoding="utf-8"))
            

            chunks = [Chunk(**c) for c in data["chunks"]]
            vectors = np.load(index_dir / lvl / VECTORS_FILE)

            all_chunks.extend(chunks)
            all_vectors.append(vectors)

        if not all_chunks:
            raise FileNotFoundError("No se encontraron chunks en las rutas especificadas.")

        store = cls(data["embedding_model"])
        store.chunks = all_chunks
        store._vectors = np.vstack(all_vectors)
        return store
