import json
import datetime
import os
import random
from typing import List
import uuid

from dotenv import load_dotenv
load_dotenv()
LOGIN_MAX_ATTEMPTS = int(os.getenv('LOGIN_MAX_ATTEMPTS'))
SESSION_MINUTES = int(os.getenv('SESSION_MINUTES'))
LIVENESS_TIMEOUT_SECONDS = int(os.getenv('LIVENESS_TIMEOUT_SECONDS'))
BLOCKED_MINUTES = 1

f = open('./data/usuarios.json',)
data = json.load(f)

users_data = data['usuarios']

f.close()

sessions: dict = {}
histories: dict = {}

def user_exists(telegram_id):
    for u in users_data:
    
        if (u["telegram_user_id"] == telegram_id):
            return True

    return False

def get_enrolled_user_data(telegram_id):

    for u in users_data:

        if (u["telegram_user_id"] == telegram_id):
            return u

    return None

def get_user_data(telegram_id):
    global sessions
    return sessions[telegram_id]


def add_failed_attempt(telegram_id):

    global sessions
    sessions[telegram_id]["attempts"] += 1

    if (sessions[telegram_id]["attempts"] >= LOGIN_MAX_ATTEMPTS):

        sessions[telegram_id]["blocked"] = datetime.datetime.now() + datetime.timedelta(minutes=BLOCKED_MINUTES)
        return True

    return False

def blocked_status(telegram_id):
    global sessions
    blocked = sessions[telegram_id]["blocked"]

    if (blocked == None):
        return False

    if (datetime.datetime.now() > blocked):
        sessions[telegram_id]["blocked"] = None
        sessions[telegram_id]["attempts"] = 0
        return False

    return True


def init_user():
    random_id  = str(random.randint(0, 99999))
        
    session = {
        "id": f'visitante_{random_id}',
        "nombre": 'Usuario de Telegram',
        "rol": 'user',
        "codigo_estudiante": -1,

        "blocked": None,
        "expiration": None,
        "login_time": None,
        "verification_time": None,
        "attempts": 0,
        "logged": False,
        "timeout": None
    }

    return session
# Inicializa una sesión no validada, no sirve para registrar usuarios nuevos
def register(telegram_id):

    global sessions

    if (telegram_id in sessions):
        return
    
    session = init_user()
    sessions[telegram_id] = session

def login(telegram_id):

    user_data = get_enrolled_user_data(telegram_id)

    if (user_data == None): return None
    
    global sessions

    sessions[telegram_id]["id"] = user_data["id"]
    sessions[telegram_id]["nombre"] = user_data["nombre"]
    sessions[telegram_id]["rol"] = user_data["rol"]
    sessions[telegram_id]["codigo_estudiante"] = user_data["codigo_estudiante"]

    sessions[telegram_id]["expiration"] = datetime.datetime.now() + datetime.timedelta(minutes=SESSION_MINUTES),
    sessions[telegram_id]["login_time"] = datetime.datetime.now()

    sessions[telegram_id]["logged"] = True
    sessions[telegram_id]["attempts"] = 0
    sessions[telegram_id]["blocked"] = None

    return sessions[telegram_id]
    

def check_verification(telegram_id):

    global sessions
    verification = sessions[telegram_id]["verification_time"]

    if (verification == None):
        sessions[telegram_id]["verification_time"] = datetime.datetime.now() + datetime.timedelta(seconds=LIVENESS_TIMEOUT_SECONDS)
    else:

        if (verification < datetime.datetime.now()):
            # Se acabó el tiempo
            sessions[telegram_id]["verification_time"] = None
            return False

    return True

def logout(telegram_id):
    global sessions
    
    if (telegram_id not in sessions):
        return
    
    expiration = sessions[telegram_id]["expiration"]
    login_time = sessions[telegram_id]["login_time"]

    sessions[telegram_id] = init_user()

    sessions[telegram_id]["expiration"] = expiration
    sessions[telegram_id]["login_time"] = login_time

    sessions[telegram_id]["logged"] = False

    return True

def init_history(telegram_id):

    global histories
    histories[telegram_id] = []

def get_user_history(telegram_id):

    global histories
    return histories[telegram_id]

def add_history_message(telegram_id, message):

    global histories
    global sessions

    role = sessions[telegram_id]["rol"]

    if (len(histories) > 15):
        histories[telegram_id].pop()

    histories[telegram_id].append({"role": role, "content": message})