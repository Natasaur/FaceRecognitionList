# facial_utils.py

import os
import cv2
import face_recognition
from pymongo import MongoClient

def procesar_imagen_y_guardar(matricula, imagen_file, media_root):
    """
    Procesa imagen para recortar rostro y obtener encoding
    """
    ruta_img = os.path.join(media_root, f"{matricula}.jpg")

    # Guardar imagen local
    with open(ruta_img, 'wb+') as f:
        for chunk in imagen_file.chunks():
            f.write(chunk)

    img = cv2.imread(ruta_img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ubicaciones = face_recognition.face_locations(rgb)

    if len(ubicaciones) != 1:
        return None, "Debe haber exactamente un rostro."

    top, right, bottom, left = ubicaciones[0]
    cara_recortada = img[top:bottom, left:right]

    # Redimensionar rostro para reducir tamaño
    cara_recortada = cv2.resize(cara_recortada, (300, 300))

    # Sobrescribir imagen
    cv2.imwrite(ruta_img, cara_recortada, [cv2.IMWRITE_JPEG_QUALITY, 50])

    # Codificar rostro
    rgb_cara = cv2.cvtColor(cara_recortada, cv2.COLOR_BGR2RGB)
    codificaciones = face_recognition.face_encodings(rgb_cara)

    if len(codificaciones) == 0:
        return None, "No se pudo codificar el rostro."

    return codificaciones[0].tolist(), None

def guardar_alumno_en_mongo(alumno, mongo_uri):
    cliente = MongoClient(mongo_uri)
    db = cliente["UConfortAsist"]
    alumnos = db["alumnos"]

    if alumnos.find_one({"matricula": alumno["matricula"]}):
        return False, f"Matrícula {alumno['matricula']} ya existe."

    alumnos.insert_one(alumno)
    return True, None
