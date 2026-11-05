"""Capacidades del asistente expresadas como tools.

El RAG del Lab 02 se convierte aquí en una tool más (buscar_documentos). Solo hace la
RECUPERACIÓN: devuelve los fragmentos numerados con su fuente, y el LLM que la pidió
redacta la respuesta. Para el agente no hay diferencia entre buscar en documentos,
consultar un servidor MCP o ejecutar una función local: todas son `Tool`.

Uso (sin LLM ni API key; requiere el índice de ingest.py):
    uv run python labs/04-agent-orchestration/capabilities.py "¿Cuándo es el primer parcial?"
"""

import asyncio
import sys
from typing import Any

from rag import CourseRAG, format_context
from tools import Tool


def rag_tool(rag: CourseRAG) -> Tool:

    async def rag_func(consulta: str):

        resultados = rag.retrieve(consulta)
        return format_context(resultados)

    return Tool(
        name="buscar_documentos",
        origin="rag",
        read_only=True,
        description="Recupera información relacionada de los siguientes documentos: programa, docente, horarios, evaluación, fechas de parciales, políticas, anuncios y preguntas frecuentes. Devuelve fragmentos numerados",
        parameters= {
            "type": "object",
            "properties": {
                "consulta": {
                    "type": "string",
                    "description": "Pide una pregunta autocontenida, p. ej. 'fecha del primer parcial'."
                }
            },
            "required": ["consulta"]
        },
        run=rag_func,
    )


async def _demo(question: str) -> None:
    from config import load_rag_settings
    from embeddings import Embedder
    from vector_store import VectorStore

    settings = load_rag_settings()
    store = VectorStore.load(settings.index_dir)
    # llm=None: la tool solo usa rag.retrieve(), que no llama al LLM (rag.answer() sí lo haría).
    rag = CourseRAG(None, Embedder(store.embedding_model), store, settings.top_k, settings.min_score)
    tool = rag_tool(rag)
    print(f"{tool.name}({{'consulta': {question!r}}}) →\n")
    print(await tool.run({"consulta": question}))


if __name__ == "__main__":
    asyncio.run(_demo(" ".join(sys.argv[1:]) or "¿Cuándo es el primer parcial?"))
