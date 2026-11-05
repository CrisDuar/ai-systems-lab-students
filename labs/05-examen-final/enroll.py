import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image

from sface import SFace
from yunet import YuNet
import cv2

LAB_DIR = Path(__file__).resolve().parent
INDEX_DIR = LAB_DIR / "rostros"
IMAGES_DIR = LAB_DIR / "face_enroll"

MODELS_DIR = LAB_DIR / "modelos"
MODEL_YUNET = "face_detection_yunet_2023mar.onnx"
MODEL_SFACE = "face_recognition_sface_2021dec.onnx"

IDS = ['9062983711', '9145484361']

model = SentenceTransformer('clip-ViT-B-32')

def process_images(directorio_raiz: str, user_id: str):

    ruta_base = Path(directorio_raiz)
    extensiones_validas = {".jpg", ".jpeg", ".png"}
    
    for archivo in ruta_base.rglob("*"):

        if archivo.is_file() and archivo.suffix.lower() in extensiones_validas:
            if str(user_id) in archivo.parts:

                imagen = cv2.imread(str(archivo))
                #image_embedding = model.encode(imagen)

                # Ajustar tamaño del detector a la imagen actual
                h, w = imagen.shape[:2]
                detector.setInputSize((w, h))
        
                # Detectar rostros en el frame con YuNet
                faces = detector.infer(imagen)

                if faces is None or len(faces) == 0:
                    print("La imagen no contiene rostros.")
                    continue

                num_faces = faces.shape[0]
                if num_faces > 1: 

                    print(f"La imagen {str(archivo)} tiene más de un rostro")
                    continue
                
                image_embedding = recognizer.infer(imagen, faces[0][:-1])

                # Esto salva los vectores
                vector_path = INDEX_DIR / f"{user_id}.npy"
                np.save(vector_path, image_embedding)

                metadata = {
                    "id": user_id,
                    "source_image": str(archivo),
                    "embedding_path": str(vector_path)
                }

                # Guarda el archivo

                json_path = INDEX_DIR / f"{user_id}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=4, ensure_ascii=False)

                print(f"¡Embedding guardado con éxito para el usuario {user_id}!")


def load_photo_data(telegram_id):
    vector = np.load(INDEX_DIR / f"{telegram_id}.npy")

    return vector

if __name__ == '__main__':

    recognizer = SFace(
        modelPath=MODELS_DIR / MODEL_SFACE
    )
    detector = YuNet(
        modelPath=MODELS_DIR / MODEL_YUNET
    )

    for id in IDS:
        process_images(IMAGES_DIR, id)

    image_data = load_photo_data(9145484361)

    #imagen = Image.open(IMAGES_DIR / "test/2.png")
    imagen = cv2.imread(IMAGES_DIR / "test/ed1.jpg")
    h, w = imagen.shape[:2]

    detector.setInputSize((w, h))
    
    faces = detector.infer(imagen)
    num_faces = faces.shape[0]

    image_embedding = recognizer.infer(imagen, faces[0][:-1])

    # Puedes calcular la similitud entre ambos

    similitud = recognizer.match(image_embedding, image_data)
    #similitud = np.dot(image_embedding.flatten(), image_data.flatten())
    #similitud = model.similarity(image_embedding, image_data)
    print(f"Similitud: {similitud}")