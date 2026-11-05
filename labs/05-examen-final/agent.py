"""Agente: un CICLO en el que el LLM decide el siguiente paso.

En el Lab 03 había una sola ronda: pedir → ejecutar → responder. Aquí se repite
mientras el modelo pida tools, así puede usar el resultado de una tool para decidir
la siguiente (consultar notas → calcular la nota final):

    repetir hasta max_steps:
        LLM(mensajes + definiciones) → ¿tool_calls?
            no → esa es la respuesta final
            sí → ejecutar cada tool, agregar resultados a los mensajes y volver a empezar
    si se agotan los pasos → una última llamada sin tools para responder con lo que hay

El límite existe porque el ciclo lo controla el modelo: sin él, un error o una mala
decisión podría repetirse indefinidamente (y cada vuelta cuesta tokens).
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from llm_client import LLMClient, LLMResponse, Message, ToolCall
from prompts import build_agent_messages
from tools import Tool

MAX_STEPS = 5

# Quien usa el agente (la interfaz) decide cómo pedir autorización al usuario.
ConfirmFn = Callable[[Tool, dict[str, Any]], Awaitable[bool]]

LIMIT_MESSAGE = (
    "[Aviso del sistema] Se alcanzó el límite de pasos: ya no puedes usar herramientas. "
    "Responde ahora al estudiante con la información que ya tienes y di qué quedó pendiente."
)

REJECTED = "El usuario NO autorizó esta acción, así que no se ejecutó."

ROLE_PERMISSIONS = {
    "estudiante": ["fecha_actual", "dias_hasta", "consultar_notas", "consultar_solicitudes", "solicitar_supletorio", "buscar_documentos"],
    "docente": ["fecha_actual", "dias_hasta", "consultar_notas", "buscar_documentos"],
    "user": ["fecha_actual", "buscar_documentos"]
}

@dataclass
class ToolRun:
    call: ToolCall  # lo que pidió el modelo
    status: Literal["ok", "error", "rechazada"]
    result: str  # lo que recibe el modelo como resultado


@dataclass
class AgentStep:
    response: LLMResponse  # la decisión del modelo en este paso
    tool_runs: list[ToolRun] = field(default_factory=list)  # vacío = respuesta final


@dataclass
class AgentAnswer:
    text: str
    steps: list[AgentStep]  # la traza: qué decidió el modelo en cada vuelta
    messages: list[Message]  # lo que recibió el LLM en la última llamada
    stopped_by_limit: bool  # True si se agotaron los pasos


class Agent:
    def __init__(self, llm: LLMClient, tools: list[Tool], confirm: ConfirmFn, max_steps: int = MAX_STEPS):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("Hay dos tools con el mismo nombre.")
        self.confirm = confirm
        self.max_steps = max_steps

        # Se establece el rol de la ejecución
        self.current_role = "user"

    def get_tools_for_role(self, role: str = "user") -> dict[str, Tool]:

        allowed_names = ROLE_PERMISSIONS.get(role, [])
        return {name: tool for name, tool in self.tools.items() if name in allowed_names}

    async def run(self, question: str, history: list[Message] | None = None, role="user") -> AgentAnswer:

        self.current_role = role

        messages = build_agent_messages(history or [], question)
        current_tools = self.get_tools_for_role(role)
        #definitions = [tool.definition() for tool in self.tools.values()]
        definitions = [tool.definition() for tool in current_tools.values()]
        steps: list[AgentStep] = []

        for _ in range(self.max_steps):

            response = self.llm.chat(messages=messages, tools=definitions)

            if (not response.tool_calls):
                steps.append(AgentStep(response))

                return AgentAnswer(
                    text=response.text, 
                    steps=steps, 
                    messages=messages, 
                    stopped_by_limit=False
                )

            messages.append(response.as_message())
            step = AgentStep(response=response)

            for call in response.tool_calls:

                # Validación
                tool = current_tools.get(call.name)
                if tool is None:
                    tool_run = ToolRun(call, "error", f"Error: no tienes permisos para usar la tool {call.name!r} o no existe.")
                else:
                    tool_run = await self.execute(call)
                
                #tool_run = await self.execute(call)
                step.tool_runs.append(tool_run)

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_run.result
                })

            # Agrega el paso completado a la lista de pasos
            steps.append(step)

        messages.append(Message({"role": "user", "content": LIMIT_MESSAGE}))
        response = self.llm.chat(messages=messages, tools=definitions, allow_tools=False)
        steps.append(AgentStep(response=response))

        return AgentAnswer(
            text=response.text, 
            steps=steps, 
            messages=messages, 
            stopped_by_limit=True
        ) 


    async def execute(self, call: ToolCall) -> ToolRun:
        """Ejecuta una tool pedida por el modelo (igual que en el Lab 03)."""
        tool = self.tools.get(call.name)
        if tool is None:
            return ToolRun(call, "error", f"Error: no existe la tool {call.name!r}.")
        try:
            arguments = call.arguments()
        except ValueError as exc:
            return ToolRun(call, "error", f"Error: {exc}")

        # Generar no es ejecutar: una acción con efectos solo se ejecuta si el usuario la autoriza.
        if not tool.read_only and not await self.confirm(tool, arguments):
            return ToolRun(call, "rechazada", REJECTED)

        try:
            result = await tool.run(arguments)
        except Exception as exc:  # argumentos inválidos, regla de negocio, servidor caído...
            return ToolRun(call, "error", f"Error: {exc}")
        return ToolRun(call, "ok", result)
