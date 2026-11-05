from contextlib import AsyncExitStack
import asyncio
import json
from typing import Any

from mcp import Tool
from narwhals import List
from telegram import Message
from orchestrator import OrchestratorAnswer, create_orchestrator, Orchestrator

f = open('./data/ataques.json',)
data = json.load(f)

casos = data['casos']

async def confirm_fn(tool: Tool, arguments: dict[str, Any]):
    print(f"Se va a ejecutar {tool.name}({json.dumps(arguments, ensure_ascii=False)})")
    return True

orquestador: Orchestrator

async def main():
    global orquestador
    orquestador = await create_orchestrator(AsyncExitStack(), confirm=confirm_fn)

    for caso in casos:

        history: List[Message] = []

        role = caso["rol"]

        print(f"\n\n--- Caso {caso['id']} ---")
        print(f"Rol: {role}")
        print(f"Pregunta: {caso['pregunta']}")

        orquestador.set_current_user("551542")
        response: OrchestratorAnswer = await orquestador.answer(question=caso['pregunta'], history=history, mode="agente", role=role)

        print(f"Respuesta: {response.text}")

        for validation in caso['no_debe_contener']:

            if validation in response.text:
                print(f"ERROR: Se encontró '{validation}' en la respuesta.")
            else:
                print(f"Exito: No se encontró '{validation}' en la respuesta.")


if __name__ == "__main__":

    asyncio.run(main())