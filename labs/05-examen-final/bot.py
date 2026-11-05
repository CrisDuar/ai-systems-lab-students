from asyncio.log import logger
from contextlib import AsyncExitStack
import io
import logging
import os
import random
import tempfile
from typing import List
import liveness
from llm_client import Message
import logs
import sessions
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes, CommandHandler,  MessageHandler, filters
from dotenv import load_dotenv

from orchestrator import OrchestratorAnswer, create_orchestrator, Orchestrator

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

MAX_CHALLENGES = 2
CHALLENGE_MSG = [
    "Envía un video mirando hacia la izquierda por al menos dos segundos.",
    "Envía un video mirando hacia la derecha por al menos dos segundos.",
    "Envía un video acercándote a la cámara por al menos un segundo"
]

orquestador: Orchestrator

# login_status determina si el usuario se está logeando, con tal de procesar los videos únicamente si este intenta acceder
login_status = False
login_challenge = 0

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def ask_agent(question: str, user):

    history: List[Message] = sessions.get_user_history(user)

    user_data = sessions.get_user_data(user)

    role = user_data["rol"]
    user_id = user_data["codigo_estudiante"]

    print(f"User ID: {user_id}")

    orquestador.set_current_user(str(user_id))
    response: OrchestratorAnswer = await orquestador.answer(question=question, history=history, mode="agente", role=role)

    return response

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    ## Crear el objeto del orquestador
    global orquestador
    orquestador = await create_orchestrator(AsyncExitStack(), confirm=telegram_confirm)

    # Inicializa el historial del usuario concreto
    user_id = update.effective_user.id
    sessions.init_history(user_id)
    sessions.register(user_id)

    await context.bot.send_message(chat_id=update.effective_chat.id, text="Bienvenido al asistente del curso de inteligencia artificial desarrollado por Christian Duarte!\nUtilice el comando /ayuda para ver los comandos disponibles.")

import asyncio
import contextvars
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

# Contexto para rastrear a qué chat y bot pertenece la ejecución actual
telegram_context_var = contextvars.ContextVar("telegram_context", default=None)
# Almacena los futuros pendientes por usuario para resolverlos al hacer clic en el botón
# Un diccionario simple para guardar respuestas: { user_id: True/False/None }
pending_responses = {}

async def telegram_confirm(tool, arguments) -> bool:
    ctx = telegram_context_var.get()
    if not ctx:
        return True  
        
    bot, chat_id, user_id = ctx["bot"], ctx["chat_id"], int(ctx["user_id"])
    
    # Marcamos inicialmente como pendiente (None)
    pending_responses[user_id] = None
    
    keyboard = [
        [
            InlineKeyboardButton("Autorizar", callback_data=f"conf_yes_{user_id}"),
            InlineKeyboardButton("Rechazar", callback_data=f"conf_no_{user_id}"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await bot.send_message(
        chat_id=chat_id,
        text=f"⚠️ La herramienta *{tool.name}* requiere tu autorización.\nParámetros: `{arguments}`",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )
    
    # Esperamos activamente hasta que el usuario presione el botón (máx 5 minutos)
    timeout = 300
    elapsed = 0
    while pending_responses.get(user_id) is None:
        await asyncio.sleep(1)
        elapsed += 1
        if elapsed >= timeout:
            pending_responses.pop(user_id, None)
            return False
            
    # Obtenemos el resultado y limpiamos
    result = pending_responses.pop(user_id)
    return result

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    # Extraemos el usuario y la acción del callback_data (ej: conf_yes_123456)
    try:
        action, user_id_str = data.split("_")[1], data.split("_")[2]
        user_id = int(user_id_str)
    except IndexError:
        return

    if user_id in pending_responses:
        if action == "yes":
            await query.edit_message_text(text="Acción autorizada. Ejecutando...")
            pending_responses[user_id] = True
        else:
            await query.edit_message_text(text="Acción rechazada.")
            pending_responses[user_id] = False
    else:
        await query.edit_message_text(text="Esta solicitud ya expiró.")


async def responses(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.effective_user.id
    question: str = update.message.text

    # Fijamos el contexto de Telegram actual para que la confirmación sepa a dónde responder
    token = telegram_context_var.set({
        "bot": context.bot,
        "chat_id": update.effective_chat.id,
        "user_id": user_id
    })

    try:
        response = await ask_agent(question=question, user=user_id)
        response_txt = response.text
    finally:
        telegram_context_var.reset(token)

    #response: OrchestratorAnswer = await ask_agent(question=question, user=user_id)
    #response_txt = response.text

    sessions.add_history_message(user_id, response_txt)

    await context.bot.send_message(chat_id=update.effective_chat.id, text=response_txt)

async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_data = sessions.get_user_data(update.effective_user.id)

    if (user_data == None): 
        response = "Tu usuario no está registrado"
    else:
        response = f"Eres el {user_data["rol"]} {user_data["nombre"]} con el ID {update.effective_user.id} y el código {user_data["codigo_estudiante"]}."

    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)

async def login(update: Update, context: ContextTypes.DEFAULT_TYPE):

    """status = sessions.login(update.effective_user.id)

    if (status == None):
        response = f"No se pudo iniciar sesión"
    else:
        response = f"Se inició sesión: {status["nombre"]}, {status["login_time"].strftime("%c")}"""

    if (sessions.blocked_status(update.effective_user.id)):
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Vuelve a intentarlo más tarde")
        return

    # Iniciar el tiempo para validarse
    sessions.check_verification(update.effective_user.id)

    global login_challenge
    login_challenge = random.randint(0, MAX_CHALLENGES)
    response = f"Vamos a validar tu sesión. {CHALLENGE_MSG[login_challenge]}"

    global login_status
    login_status = True

    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


async def logout(update: Update, context: ContextTypes.DEFAULT_TYPE):

    status = sessions.logout(update.effective_user.id)

    if (status):

        logs.add_record(
            update.effective_user.id,
            2,
            "Se cerró la sesión correctamente"
        )

        response = "Se cerró la sesión correctamente."
    else:

        response = "No ha iniciado sesión."
        
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):

    help_txt = "Comandos disponibles\n/quiensoy - Da la información del usuario\n/login - Inicia sesión\n/logout - Cierra sesión\n/ayuda - Muestra esta ayuda"

    await context.bot.send_message(chat_id=update.effective_chat.id, text=help_txt)

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="El comando no existe.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Error:", exc_info=context.error)

    logs.add_record(
        "SYSTEM",
        1,
        f"{context.error}"
    )

    # Error por actualización de un chat
    if isinstance(update, Update) and update.effective_chat:
        text = "Ocurrió un error inesperado al procesar tu solicitud. Por favor, intenta de nuevo más tarde"
        try:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=text)
        except Exception as e:
            logger.error(f"No se pudo enviar el mensaje de error al usuario: {e}")

async def verification(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:

    global login_status
    global login_challenge
    if (login_status == False):
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Utiliza el comando /login para iniciar sesión.")
        return

    if (sessions.check_verification(update.effective_user.id) == False):

        logs.add_record(
            update.effective_user.id,
            4,
            f"Se acabó el tiempo para enviar la prueba de video"
        )

        login_status = False
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Se acabó el tiempo para validar la sesión. Utiliza el comando /login nuevamente.")
        return

    file_id = update.message.video_note.file_id
    new_file = await context.bot.get_file(file_id)

    # Crear una ruta temporal para poder cargar el video en liveness
    file_buffer = io.BytesIO()
    await new_file.download_to_memory(file_buffer)
    file_buffer.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(file_buffer.read())
        temp_path = temp_video.name

    try:
        result, reason = liveness.process_video(temp_path, update.effective_user.id, challenge=login_challenge)

        if (result):

            logs.add_record(
                update.effective_user.id,
                2,
                "Inicio de sesión exitoso"
            )

            user_data = sessions.login(update.effective_user.id)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Se creó la sesión con éxito. Bienvenido {user_data['nombre']}.")
        else:
            attempts_status = sessions.add_failed_attempt(update.effective_user.id)

            if (attempts_status):
                login_status = False

                logs.add_record(
                    update.effective_user.id,
                    4,
                    f"Se superó el límite de intentos de sesión: {reason["reason"]}: Se esperaba un valor de al menos {reason["expected"]} y se obtuvo {reason["result"]}"
                )

                await context.bot.send_message(chat_id=update.effective_chat.id, text="Superaste el límite de intentos. Vuelve a intentarlo más tarde.")
            else:

                logs.add_record(
                    update.effective_user.id,
                    3,
                    f"Falló el login: {reason["reason"]}: Se esperaba un valor de al menos {reason["expected"]} y se obtuvo {reason["result"]}"
                )

                await context.bot.send_message(chat_id=update.effective_chat.id, text="Inicio fallido. Vuelve a intentarlo")

    finally:
    
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':

    ## Iniciar la aplicación

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(4)
        .build()
    )

    buttons_confirmation_handler = CallbackQueryHandler(button_callback)
    application.add_handler(buttons_confirmation_handler)
    
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    ## Comandos requisito
    whoami_handler = CommandHandler('quiensoy', whoami)
    login_handler = CommandHandler('login', login)
    logout_handler = CommandHandler('logout', logout)
    help_handler = CommandHandler('ayuda', help)

    application.add_handler(whoami_handler)
    application.add_handler(login_handler)
    application.add_handler(logout_handler)
    application.add_handler(help_handler)

    # Respuestas

    verification_handler = MessageHandler(filters.VIDEO_NOTE & (~ filters.FORWARDED), verification)
    application.add_handler(verification_handler)

    agent_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), responses)
    application.add_handler(agent_handler)

    #Comodín por si no entiende algún comando

    unknown_handler = MessageHandler(filters.COMMAND, unknown)
    application.add_handler(unknown_handler)

    # Registrar errores

    application.add_error_handler(error_handler)
    
    application.run_polling()