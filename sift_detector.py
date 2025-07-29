import cv2
import os
import numpy as np

NEGATIVE_DIR = "dataset-lbp/negatives"

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

def imread_safe(ruta, modo=cv2.IMREAD_GRAYSCALE):
    try:
        with open(ruta, 'rb') as f:
            bytes_data = f.read()
        img_array = np.frombuffer(bytes_data, dtype=np.uint8)
        return cv2.imdecode(img_array, modo)
    except Exception as e:
        print(f"❌ Error leyendo {ruta}: {e}")
        return None

def cargar_referencias():
    referencias = []
    for nombre in os.listdir(NEGATIVE_DIR):
        ruta = os.path.join(NEGATIVE_DIR, nombre)
        img = imread_safe(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Imagen no válida: {ruta}")
            continue
        kps, des = sift.detectAndCompute(img, None)
        referencias.append((nombre, img, kps, des))
    return referencias

referencias = cargar_referencias()

def detectar_sift(imagen_color):
    img_gray = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img_gray, None)

    mejores_resultados = []

    for nombre, ref_gray, kp2, des2 in referencias:
        if des2 is None or des1 is None:
            continue

        matches = bf.knnMatch(des1, des2, k=2)
        buenas = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(buenas) > 10:
            mejores_resultados.append((nombre, buenas, kp1, kp2))

    if mejores_resultados:
        nombre, buenas, kp1, kp2 = max(mejores_resultados, key=lambda x: len(x[1]))
        ref_color = imread_safe(os.path.join(NEGATIVE_DIR, nombre), cv2.IMREAD_COLOR)

        # Imagen con puntos conectados
        imagen_match = cv2.drawMatches(imagen_color, kp1, ref_color, kp2, buenas, None, flags=2)

        # Imagen original con solo keypoints (sin líneas ni combinación)
        imagen_keypoints = cv2.drawKeypoints(imagen_color, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Nivel de confianza como % de buenos matches sobre total de keypoints
        confianza = round((len(buenas) / max(len(kp1), 1)) * 100, 2)

        return imagen_match, imagen_keypoints, nombre, len(buenas), confianza
    else:
        return None, None, None, 0, 0.0
