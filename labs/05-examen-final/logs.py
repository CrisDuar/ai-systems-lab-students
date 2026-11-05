# Registros de auditoría

import datetime
import json
from pathlib import Path
import uuid

LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "data"
LOGS_FILE_NAME = 'auditoria.jsonl'

RECORD_TYPES = [
    'system_test',
    'system_error',
    'login_success',
    'login_failed',
    'login_denied',
    'tool_execution'
]


def add_record(user, record_type, description):

    record = {
        "id": str(uuid.uuid4()),
        "date": datetime.datetime.now().strftime("%c"),
        "user": user,
        "type": RECORD_TYPES[record_type],
        "description": description,
    }

    with open(DATA_DIR / LOGS_FILE_NAME, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f'Se creó el registro {record["id"]} en ./data/{LOGS_FILE_NAME}')


if __name__ == '__main__':

    add_record('TEST', 0, "Esto es una prueba del sistema")