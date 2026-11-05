import os
from pathlib import Path
import cv2 as cv
from dotenv import load_dotenv
import sessions
from sface import SFace
from yunet import YuNet
import numpy as np

load_dotenv()
FACE_MATCH_THRESHOLD = float(os.getenv('FACE_MATCH_THRESHOLD'))

LAB_DIR = Path(__file__).resolve().parent
FACES_DIR = LAB_DIR / "face_enroll/test"
MODELS_DIR = LAB_DIR / "modelos"
MODEL_YUNET = "face_detection_yunet_2023mar.onnx"
MODEL_SFACE = "face_recognition_sface_2021dec.onnx"

previous_bbox_area = None
frame_skip = 1
frame_count = 0

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None, similarity=1,dist=""):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    sim_color=(255,255,255)

    if (similarity < 0.363): sim_color=(0,0,255)

    cv.putText(output, 'SIMILARITY: {:.2f}'.format(similarity), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, sim_color)

    for det in results:
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)

        ojo_der = landmarks[0]   # [x, y]
        ojo_izq = landmarks[1]   # [x, y]
        nariz = landmarks[2]     # [x, y]

        dist_nariz_ojo_der = abs(nariz[0] - ojo_der[0])
        dist_nariz_ojo_izq = abs(nariz[0] - ojo_izq[0])

        razon = dist_nariz_ojo_der / dist_nariz_ojo_izq

        txt = ""
        # 4. Definir un umbral de tolerancia para el giro (por ejemplo, fuera del rango 0.6 a 1.6)
        if razon < 0.6:
            txt = "¡Atención! La cabeza está girada hacia la izquierda (o derecha según perspectiva)."
        elif razon > 1.6:
            txt = "¡Atención! La cabeza está girada hacia el otro lado."
        else:
            txt = "La cabeza está de frente."

        cv.putText(output, txt, (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv.putText(output, dist, (0, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

    return output

def detect_looking_sideways(results, direction=0):

    # 0: IZQ
    # 1: DER
    LEFT_TOLERANCE = 0.6
    RIGHT_TOLERANCE = 1.6

    # Se supone que sólo habrá una cara en el video
    face = results[0]

    landmarks = face[4:14].astype(np.int32).reshape((5,2))

    eye_r = landmarks[0]
    eye_l = landmarks[1]
    nose = landmarks[2]

    ratio = abs(nose[0] - eye_r[0]) / abs(nose[0] - eye_l[0])

    # La cabeza está de frente
    looking_direction = -1

    # Definir un umbral de tolerancia para el giro (por ejemplo, fuera del rango 0.6 a 1.6)
    if ratio < LEFT_TOLERANCE:
        # La cabeza está girada hacia la izquierda.
        looking_direction = 0

    elif ratio > RIGHT_TOLERANCE:
        # La cabeza está girada hacia la derecha.
        looking_direction = 1

    if (looking_direction == direction):
        return True

    return False
        
def detect_zoom_in(area, frame_count, skip_frames=30, change_percentage=0.2):

    global previous_bbox_area

    if (previous_bbox_area == None or frame_count % skip_frames == 0): 
        previous_bbox_area = area

    diff = (area - previous_bbox_area) / previous_bbox_area

    if (diff > change_percentage):
        # Se acercó
        return 1
    elif (diff < -change_percentage):
        # Se alejó
        return -1
    else:
        # Estable
        return 0

def process_video(video_path, telegram_id, challenge=0):

    if (sessions.user_exists(telegram_id) == False):
        return False

    recognizer = SFace(
            modelPath=MODELS_DIR / MODEL_SFACE
    )
    detector = YuNet(
        modelPath=MODELS_DIR / MODEL_YUNET
    )

    enrolled_embedding = np.load(f"rostros/{telegram_id}.npy")
    cap = cv.VideoCapture(video_path)

    # Validar si se realiza el reto
    framerate = int(cap.get(cv.CAP_PROP_FPS))
    challenge_addition = (frame_skip)/framerate
    challenge_progress = 0
    challege_max_seconds = 2
    match_sum = 0
    match_am = 0

    # Verificar que el video se abrió correctamente
    if not cap.isOpened():
        print(f"Error al abrir el archivo de video: {video_path}")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o error al leer el fotograma.")
            break

        # Saltar frames para no procesar el video completo
        global frame_count
        frame_count += 1
        
        if frame_count % frame_skip != 0:
            continue

        # Ajustar tamaño del detector al frame actual
        h, w, _ = frame.shape
        detector.setInputSize((w, h))

        # Detectar rostros en el frame con YuNet
        faces = detector.infer(frame)

        #TEST ver el video
        output_frame = visualize(frame, faces)

        if faces is not None and len(faces) > 0:
            for face in faces:

                if (challenge == 0): # Girar la cabeza a la izquierda
                    if (detect_looking_sideways(faces, 0)):
                        challenge_progress += challenge_addition
                elif (challenge == 1): # Girar la cabeza a la derecha
                    if (detect_looking_sideways(faces, 1)):
                        challenge_progress += challenge_addition
                elif (challenge == 2): # Acercarse
                    challege_max_seconds = 1
                    box_w = face[2]
                    box_h = face[3]

                    area = box_w * box_h

                    if (detect_zoom_in(area=area, frame_count=frame_count, skip_frames=30) == 1):
                        challenge_progress += challenge_addition

                query_embedding = recognizer.infer(frame, face[:-1])
                cosine_score = recognizer.match(enrolled_embedding, query_embedding)#np.dot(enrolled_embedding, query_embedding)


                match_sum += cosine_score[0]
                match_am += 1

                # FACE_MATCH_THRESHOLD

            # 3. Mostrar el resultado en una ventana
            output_frame = visualize(frame, faces, similarity=cosine_score[0], dist=str(challenge_progress))
    
            cv.imshow("Video Demo", output_frame)
    
    
            # Pausa de 1ms y presionar 'q' para salir anticipadamente del video
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv.destroyAllWindows()

    if (match_am > 0):
        match_avg = match_sum/match_am
    else:
        match_avg = 0

    print(f"RESULTADOS:\nMatch:{match_avg}\nreto:{challenge}\nProgreso del reto{challenge_progress}\nAdd del reto{challenge_addition}")

    if (match_avg < FACE_MATCH_THRESHOLD): return False, {"reason":"Las caras no coinciden", "result": match_avg, "expected": FACE_MATCH_THRESHOLD}
    if (challenge_progress < challege_max_seconds): return False, {"reason":"La acción no se realizó por suficiente tiempo", "result": challenge_progress, "expected": challege_max_seconds}
    
    return True, {"reason":"Todo salió bien", "result": True, "expected": True}


if __name__ == '__main__':
    video_path = FACES_DIR / "2.mp4"

    process_video(video_path, 9062983711, 0)

"""
if __name__ == '__main__':
    # Reemplaza '0' o la ruta de la imagen por la ruta de tu video
    video_path = FACES_DIR / "1.mp4"
    enrolled_embedding = np.load(f"rostros/9062983711.npy")
    cap = cv.VideoCapture(video_path)

    # Verificar que el video se abrió correctamente
    if not cap.isOpened():
        print(f"Error al abrir el archivo de video: {video_path}")
        exit()

    # Obtener dimensiones del video para configurar el modelo si es necesario
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    recognizer = SFace(
        modelPath=MODELS_DIR / MODEL_SFACE
    )
    
    # Instantiate YuNet for face detection
    detector = YuNet(
        modelPath=MODELS_DIR / MODEL_YUNET
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o error al leer el fotograma.")
            break

        frame_count += 1

        # Si el residuo no es 0, nos saltamos este frame y pasamos al siguiente
        if frame_count % frame_skip != 0:
            continue
        
        # Ajustar tamaño del detector al frame actual
        h, w, _ = frame.shape
        detector.setInputSize((w, h))

        # Detectar rostros en el frame con YuNet
        faces = detector.infer(frame)

        output_frame = visualize(frame, faces)

        if faces is not None:
            for face in faces:


                # Extraer ancho (w) y alto (h) de la caja delimitadora
                # Índices 0: x, 1: y, 2: w, 3: h
                box_w = face[2]
                box_h = face[3]

                area_actual = box_w * box_h

                acercar = ""

                # Actualizar el tamaño anterior para la siguiente iteración
                                
                if (tamano_anterior == None or skip >= skip_frames): 
                    tamano_anterior = area_actual
                    skip=0

                skip+=1
                
                if tamano_anterior is not None:
                    # Calcular la diferencia proporcional respecto al frame anterior
                    diferencia = (area_actual - tamano_anterior) / tamano_anterior

                    if diferencia > porcentaje_cambio_requerido:
                        acercar = "¡Acción detectada! El usuario SE ACERCÓ a la cámara."
                    elif diferencia < -porcentaje_cambio_requerido:
                        acercar="El usuario SE ALEJÓ de la cámara."
                    else:
                        acercar="Distancia estable."

                acercar += f"\n{area_actual}\n{tamano_anterior}\n{diferencia}"

                




                # face[:-1] pasa la caja y los 5 landmarks necesarios para la alineación
                # .infer() alinea, procesa y extrae el embedding del frame actual de forma automática
                query_embedding = recognizer.infer(frame, face[:-1])

                # 4. Comparar el embedding del .npy con el vector extraído del video
                # Al estar normalizados, la similitud de coseno es simplemente el producto punto:
                cosine_score = recognizer.match(enrolled_embedding, query_embedding)#np.dot(enrolled_embedding, query_embedding)

                # 5. Evaluar contra tu umbral (p. ej., el de referencia 0.363 o tu umbral calibrado)
                umbral = 0.363
                
                if cosine_score[0] >= umbral:
                    print(f"¡Coincidencia exitosa! Similitud: {cosine_score[0]:.4f}")
                else:
                    print(f"No es el usuario (impostor o baja similitud). Similitud: {cosine_score[0]:.4f}")

        # 3. Mostrar el resultado en una ventana
        output_frame = visualize(frame, faces, similarity=cosine_score[0], dist=acercar)

        cv.imshow("Video Demo", output_frame)


        # Pausa de 1ms y presionar 'q' para salir anticipadamente del video
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()"""