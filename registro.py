# registro.py

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import os
import mediapipe as mp
import math
import face_recognition
from pymongo import MongoClient
from facial_utils import procesar_imagen_y_guardar, guardar_alumno_en_mongo

MONGO_URI = "mongodb+srv://uconfortasist:Udl8Q0APE93vt3BB@cluster0.g6qne.mongodb.net/UConfortAsist?retryWrites=true&w=majority"
MEDIA_DIR = "./media"

os.makedirs(MEDIA_DIR, exist_ok=True)

def registrar_alumno():
    matricula = entry_matricula.get().strip()[:10]
    nombre = entry_nombre.get().strip()[:50]
    ap_pat = entry_apellido_paterno.get().strip()[:50]
    ap_mat = entry_apellido_materno.get().strip()[:50]
    grupo = entry_grupo.get().strip()[:10]
    ciclo = entry_ciclo.get().strip()[:10]
    contacto = entry_contacto.get().strip()[:20]

    if not all([matricula, nombre, ap_pat, ap_mat, grupo, ciclo, contacto]):
        messagebox.showerror("Error", "Por favor completa todos los campos.")
        return

    # Capturar foto automática
    foto = capturar_foto_auto()
    if foto is None:
        messagebox.showerror("Error", "No se pudo capturar la foto.")
        return

    # Verificar grupo válido
    cliente = MongoClient(MONGO_URI)
    db = cliente["UConfortAsist"]
    col_grupos = db["grupos"]

    grupo_valido = col_grupos.find_one({
        "grupo": grupo,
        "disponible": True
    })

    if not grupo_valido:
        messagebox.showerror("Error", "Verifique su grupo.")
        return

    # Guardar foto temporal
    temp_path = os.path.join(MEDIA_DIR, f"{matricula}_temp.jpg")
    cv2.imwrite(temp_path, foto)

    # Leer foto como bytes
    with open(temp_path, "rb") as f:
        class DummyFile:
            def __init__(self, data):
                self._data = data
            def chunks(self):
                yield self._data

        imagen_file = DummyFile(f.read())

    # Procesar imagen y codificar
    encoding, error = procesar_imagen_y_guardar(
        matricula,
        imagen_file,
        MEDIA_DIR
    )

    if os.path.exists(temp_path):
        os.remove(temp_path)

    if error:
        messagebox.showerror("Error", f"Error al procesar la imagen: {error}")
        return

    alumno = {
        "matricula": matricula,
        "nombre": nombre,
        "apellido_paterno": ap_pat,
        "apellido_materno": ap_mat,
        "grupo": grupo,
        "ciclo_escolar": ciclo,
        "contacto": contacto,
        "asistencias": [],
        "imagen": f"{matricula}.jpg",
        "encoding": encoding
    }

    ok, error = guardar_alumno_en_mongo(alumno, MONGO_URI)
    if ok:
        mostrar_info_alumno(alumno)
        limpiar_campos()
    else:
        messagebox.showerror("Error", f"No se pudo registrar: {error}")

def capturar_foto_auto():
    mpFaceMesh = mp.solutions.face_mesh
    mpDraw = mp.solutions.drawing_utils

    face_mesh = mpFaceMesh.FaceMesh(max_num_faces=1)
    cap = cv2.VideoCapture(0)

    parpadeos = 0
    parpadeo_anterior = False
    intentos = 0

    while intentos < 200:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)

        if result.multi_face_landmarks:
            for faceLms in result.multi_face_landmarks:
                # obtener puntos clave
                landmarks = faceLms.landmark
                # ojo derecho
                x1 = landmarks[145].x
                y1 = landmarks[145].y
                x2 = landmarks[159].x
                y2 = landmarks[159].y
                distancia_derecho = math.hypot(x2 - x1, y2 - y1)

                # ojo izquierdo
                x3 = landmarks[374].x
                y3 = landmarks[374].y
                x4 = landmarks[386].x
                y4 = landmarks[386].y
                distancia_izquierdo = math.hypot(x4 - x3, y4 - y3)

                # calcular parpadeo
                if distancia_derecho < 0.01 and distancia_izquierdo < 0.01:
                    if not parpadeo_anterior:
                        parpadeos += 1
                        parpadeo_anterior = True
                else:
                    parpadeo_anterior = False

                # head pose check (simplificado)
                # Ejemplo usando landmarks para ver orientación
                nariz = landmarks[1]
                frente = landmarks[10]
                diferencia_y = abs(frente.y - nariz.y)

                # supongamos que diferencia muy alta = cabeza inclinada
                if diferencia_y > 0.08:
                    cv2.putText(frame, "Por favor mire al frente", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    continue

                # detección de cubrebocas/lentes (simplificado)
                boca_superior = landmarks[13].y
                boca_inferior = landmarks[14].y
                boca_abierta = abs(boca_superior - boca_inferior) > 0.02

                if not boca_abierta:
                    cv2.putText(frame, "Retire cubrebocas/lentes", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    continue

                # Si ha parpadeado 3 veces, capturamos la foto
                if parpadeos >= 3:
                    cap.release()
                    cv2.destroyAllWindows()
                    return frame

        cv2.imshow("Liveness Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        intentos += 1

    cap.release()
    cv2.destroyAllWindows()
    return None

def mostrar_info_alumno(alumno):
    ventana_info = tk.Toplevel(ventana)
    ventana_info.title("Registro Exitoso")
    ventana_info.geometry("400x600")
    ventana_info.configure(bg="#ffffff")

    nombre_completo = f"{alumno['nombre']} {alumno['apellido_paterno']} {alumno['apellido_materno']}"

    tk.Label(ventana_info, text="REGISTRO EXITOSO", font=("Arial", 16, "bold"), bg="#ffffff").pack(pady=10)
    tk.Label(ventana_info, text=f"Nombre: {nombre_completo}", bg="#ffffff").pack(pady=5)
    tk.Label(ventana_info, text=f"Matrícula: {alumno['matricula']}", bg="#ffffff").pack(pady=5)
    tk.Label(ventana_info, text=f"Grupo: {alumno['grupo']}", bg="#ffffff").pack(pady=5)

    ruta_img = os.path.join(MEDIA_DIR, alumno["imagen"])
    if os.path.exists(ruta_img):
        img_cv = cv2.imread(ruta_img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_cv = cv2.resize(img_cv, (300, 300))
        img_pil = Image.fromarray(img_cv)
        img_tk = ImageTk.PhotoImage(img_pil)

        label_img = tk.Label(ventana_info, image=img_tk, bg="#ffffff")
        label_img.image = img_tk
        label_img.pack(pady=10)
    else:
        tk.Label(ventana_info, text="No se encontró la foto.", bg="#ffffff").pack()

    btn_cerrar = tk.Button(ventana_info, text="Cerrar", command=ventana_info.destroy, bg="#e74c3c", fg="white", width=20)
    btn_cerrar.pack(pady=20)

def limpiar_campos():
    entry_matricula.delete(0, tk.END)
    entry_nombre.delete(0, tk.END)
    entry_apellido_paterno.delete(0, tk.END)
    entry_apellido_materno.delete(0, tk.END)
    entry_grupo.delete(0, tk.END)
    entry_ciclo.delete(0, tk.END)
    entry_contacto.delete(0, tk.END)

# --- Ventana principal Tkinter ---
ventana = tk.Tk()
ventana.title("Registro de Alumnos")
ventana.geometry("500x600")
ventana.configure(bg="#f0f0f0")

def solo_numeros(nuevo_valor):
    return nuevo_valor.isdigit() and len(nuevo_valor) <= 10 or nuevo_valor == ""

vcmd_matricula = (ventana.register(solo_numeros), "%P")


# Campos de entrada
tk.Label(ventana, text="Matrícula:", bg="#f0f0f0").pack(anchor="w", padx=20, pady=5)
entry_matricula = tk.Entry(ventana, width=50, validate="key", validatecommand=vcmd_matricula)
entry_matricula.pack(padx=20)

tk.Label(ventana, text="Nombre:", bg="#f0f0f0").pack(anchor="w", padx=20, pady=5)
entry_nombre = tk.Entry(ventana, width=50, validate="key")
entry_nombre.pack(padx=20)

tk.Label(ventana, text="Apellido paterno:", bg="#f0f0f0").pack(anchor="w", padx=20, pady=5)
entry_apellido_paterno = tk.Entry(ventana, width=50, validate="key")
entry_apellido_paterno.pack(padx=20)

tk.Label(ventana, text="Apellido materno:", bg="#f0f0f0").pack(anchor="w", padx=20, pady=5)
entry_apellido_materno = tk.Entry(ventana, width=50, validate="key")
entry_apellido_materno.pack(padx=20)

tk.Label(ventana, text="Grupo:", bg="#f0f0f0").pack(anchor="w", padx=20, pady=5)
entry_grupo = tk.Entry(ventana, width=50, validate="key")
entry_grupo.pack(padx=20)

tk.Label(ventana, text="Ciclo escolar:", bg="#f0f0f0").pack(anchor="w", padx=20, pady=5)
entry_ciclo = tk.Entry(ventana, width=50, validate="key")
entry_ciclo.pack(padx=20)

tk.Label(ventana, text="Contacto:", bg="#f0f0f0").pack(anchor="w", padx=20, pady=5)
entry_contacto = tk.Entry(ventana, width=50, validate="key")
entry_contacto.pack(padx=20)

btn_registrar = tk.Button(ventana, text="Registrar Alumno", command=registrar_alumno,
                          bg="#4caf50", fg="white", height=2, width=20)
btn_registrar.pack(pady=30)

ventana.mainloop()