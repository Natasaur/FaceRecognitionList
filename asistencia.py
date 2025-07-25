import cv2
import numpy as np
import face_recognition
from pymongo import MongoClient
from datetime import datetime, time
import os
import time as tiempo

# Conexión con MongoDB
cliente = MongoClient("mongodb+srv://uconfortasist:Udl8Q0APE93vt3BB@cluster0.g6qne.mongodb.net/UConfortAsist?retryWrites=true&w=majority")  # Cambia si usas un servidor remoto
db = cliente["UConfortAsist"]
col_alumnos = db["alumnos"]
col_asistencias = db["asistencias"]

# Hora límite de entrada (7:10 AM)
#hora_limite = time(7, 10)
hora_limite = time(20, 10)

# Obtener alumnos con codificaciones
alumnos_raw = list(col_alumnos.find({"encoding": {"$exists": True, "$ne": []}}))
codificados = [np.array(alumno["encoding"]) for alumno in alumnos_raw]

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Para evitar registrar múltiples veces al mismo alumno en el mismo día
asistencias_registradas = set()

print("Sistema de asistencia activado. Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar imagen.")
        break

    frame_peq = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb = cv2.cvtColor(frame_peq, cv2.COLOR_BGR2RGB)

    ubicaciones = face_recognition.face_locations(rgb)
    codificaciones = face_recognition.face_encodings(rgb, ubicaciones)

    for cod, loc in zip(codificaciones, ubicaciones):
        coincidencias = face_recognition.compare_faces(codificados, cod)
        distancias = face_recognition.face_distance(codificados, cod)

        if any(coincidencias):
            idx = np.argmin(distancias)
            alumno = alumnos_raw[idx]
            matricula = alumno["matricula"]
            nombre_completo = f"{alumno['nombre']} {alumno['apellido_paterno']}"

            hoy = datetime.now().strftime("%d/%m/%Y")
            hora_actual = datetime.now().time()

            clave_unica = f"{matricula}_{hoy}"

            if clave_unica not in asistencias_registradas and hora_actual <= hora_limite:
                # Verificar que no se haya registrado ya en la BD
                ya_registrado = col_asistencias.find_one({
                    "matricula": matricula,
                    "fecha": hoy
                })

                if not ya_registrado:
                    col_asistencias.insert_one({
                        "matricula": matricula,
                        "grupo": alumno["grupo"],
                        "ciclo_escolar": alumno["ciclo_escolar"],
                        "fecha": hoy,
                        "tipo_asistencia": "normal"
                    })
                    asistencias_registradas.add(clave_unica)
                    print(f"Asistencia registrada: {nombre_completo}")
                else:
                    print(f"Asistencia ya registrada previamente: {nombre_completo}")
            elif hora_actual > hora_limite:
                print(f"{nombre_completo} llegó tarde. No se registra asistencia.")
            else:
                print(f"Asistencia ya registrada hoy para {nombre_completo}")

            # Dibujar el recuadro
            top, right, bottom, left = [v * 4 for v in loc]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, nombre_completo, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("Asistencia por reconocimiento facial", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
