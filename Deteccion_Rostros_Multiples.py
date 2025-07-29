# Detección de varios rostros atraves de una cámara.
# Librerías
import cv2
import numpy as np
import face_recognition as fr
import os
import random
from datetime import datetime

# Acceso a la carpeta
path = 'Personal'
images = []
clases = []
lista = os.listdir(path)
#print(lista)

# Variables
comp1 = 100

# Lectura de rostros en la Base de Datos
for lis in lista:
    imgdb = cv2.imread(f'{path}/{lis}') # Se leen los rostros de las imágenes
    images.append(imgdb) # Se van agregando los rostros identificados
    clases.append(os.path.splitext(lis)[0]) # Se van almacenando los nombres

print(clases)

# Codificación de rostros
def codrostros(images):
    listacod = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Se hace una correción de color
        cod = fr.face_encodings(img)[0] # Se codifica la imágen
        listacod.append(cod) # Se almacena
    return listacod

# Horarios de ingreso
def horario(nombre):
    # Abrir archivo de horarios en modo lectura y escritura
    with open('Horario.csv', 'r+') as h:
        data = h.readlines() # Se lee la información
        listanombres = [] # Se crea una lista para los nombres

        for line in data:
            entrada = line.split(',') # Se busca la entrada y se diferencia con ,
            listanombres.append(entrada[0]) # Se van almacenando los nombre

        # Se verifica si ya está almacenado el nombre
        if nombre not in listanombres:
            info = datetime.now() # Se extrae la información actual
            fecha = info.strftime("%d/%m/%Y") # Se extrae la fecha
            hora = info.strftime("%H:%M:%S") # Se extrae la hora
            h.writelines(f'\n{nombre}, {fecha}, {hora}') # Se guarda la información
            print(info)

# Se llama la funcion
codificacion_de_rostros = codrostros(images)

# Video Captura
cap = cv2.VideoCapture(0)

# LOOP
while True:
    rwt, frame = cap.read() # Lectura de fotogramas
    frame2 = cv2.resize(frame, (0, 0), None, 0.25, 0.25) # Se reduce el tamaño de la imagen para un mejor procesamiento
    rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB) # Conversión de color

    # Se buscan los rostros
    faces = fr.face_locations(rgb)
    facescod = fr.face_encodings(rgb, faces)

    for facescod, faceloc in zip(facescod, faces):
        comparacion = fr.compare_faces(codificacion_de_rostros, facescod) # Compara el rostro en tiempo real contra los de la BD
        simi = fr.face_distance(codificacion_de_rostros, facescod) # Se calcula la similitud
        minimo = np.argmin(simi) # Se busca el valor mínimo, osea el que tenga mayor coincidencia

        if comparacion[minimo]:
            nombre = clases[minimo].upper()
            print(nombre)
            yi, xf, yf, xi = faceloc # Se estraen coordenadas
            yi, xf, yf, xi = yi*4, xf*4, yf*4, xi*4 # Escala

            indice = comparacion.index(True)

            # Comparamos
            if comp1 != indice:
                # Se define un colo RGB random
                r = random.randrange(0, 255, 50)
                g = random.randrange(0, 255, 50)
                b = random.randrange(0, 255, 50)

                comp1 = indice

            if comp1 == indice:
                # Se define un colo RGB random
                cv2.rectangle(frame, (xi, yi), (xf, yf), (r, g, b), 3)
                cv2.rectangle(frame, (xi, yf-35), (xf, yf), (r, g, b), cv2.FILLED)
                cv2.putText(frame, nombre, (xi+6, yf-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                horario(nombre)

    #Mostramos frames
    cv2.imshow("Reconocimiento Facial", frame)

    # Leemos el teclado
    t = cv2.waitKey(5)
    if t == 27: break