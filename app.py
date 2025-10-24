import os
import time
import cv2
import torch
import numpy as np
import psutil
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from sift_detector import detectar_sift

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

modelo_yolo = YOLO('runs/detect/yolo_faces_final3/weights/best.pt')

def permitido(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pixelar_region(imagen, x1, y1, x2, y2, factor=0.05):
    cara = imagen[y1:y2, x1:x2]
    if cara.size == 0:
        return imagen
    cara_peq = cv2.resize(cara, (max(1, int((x2 - x1) * factor)), max(1, int((y2 - y1) * factor))), interpolation=cv2.INTER_LINEAR)
    cara_pix = cv2.resize(cara_peq, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    imagen[y1:y2, x1:x2] = cara_pix
    return imagen

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    cantidad = 0
    tiempo = 0
    imagenes_previas = sorted(os.listdir(UPLOAD_FOLDER), reverse=True)

    if request.method == 'POST':
        archivo = request.files['imagen']
        if archivo and permitido(archivo.filename):
            nombre_archivo = secure_filename(archivo.filename)
            ruta_imagen = os.path.join(UPLOAD_FOLDER, nombre_archivo)
            archivo.save(ruta_imagen)

            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                return render_template('index.html', error="Error leyendo la imagen.")

            start = time.time()
            resultados = modelo_yolo(imagen)[0]
            cantidad = 0

            for box in resultados.boxes.xyxy.cpu().numpy().astype(int):
                x1, y1, x2, y2 = box[:4]
                imagen = pixelar_region(imagen, x1, y1, x2, y2)
                cantidad += 1

            tiempo = round(time.time() - start, 2)
            resultado = 'detectado_' + nombre_archivo
            ruta_salida = os.path.join(UPLOAD_FOLDER, resultado)
            cv2.imwrite(ruta_salida, imagen)
            imagenes_previas.insert(0, resultado)

    return render_template('index.html', resultado=resultado, cantidad=cantidad, tiempo=tiempo, imagenes=imagenes_previas)

@app.route('/deteccion-avanzada', methods=['POST'])
def deteccion_avanzada():
    resultado = None
    cantidad = 0
    coincidencias = 0
    avion_detectado = None
    confianza = 0
    tiempo = 0
    memoria = 0
    imagen_keypoints_filename = None
    imagenes_previas = sorted(os.listdir(UPLOAD_FOLDER), reverse=True)

    archivo = request.files['imagen']
    if archivo and permitido(archivo.filename):
        nombre_archivo = secure_filename(archivo.filename)
        ruta_imagen = os.path.join(UPLOAD_FOLDER, nombre_archivo)
        archivo.save(ruta_imagen)

        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            return render_template('index.html', error="Error leyendo la imagen.")

        start = time.time()

        resultados = modelo_yolo(imagen)[0]
        for box in resultados.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box[:4]
            imagen = pixelar_region(imagen, x1, y1, x2, y2)
            cantidad += 1

        # Nuevo: obtener match y keypoints de SIFT
        imagen_match, imagen_keypoints, avion_detectado, coincidencias, confianza = detectar_sift(imagen)

        if imagen_match is not None:
            resultado = 'sift_match_' + nombre_archivo
            ruta_match = os.path.join(UPLOAD_FOLDER, resultado)
            cv2.imwrite(ruta_match, imagen_match)
            imagenes_previas.insert(0, resultado)

        if imagen_keypoints is not None:
            imagen_keypoints_filename = 'sift_keypoints_' + nombre_archivo
            ruta_keypoints = os.path.join(UPLOAD_FOLDER, imagen_keypoints_filename)
            cv2.imwrite(ruta_keypoints, imagen_keypoints)
            imagenes_previas.insert(0, imagen_keypoints_filename)

        tiempo = round(time.time() - start, 2)
        memoria = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)  # en MB

        return render_template('index.html',
                               resultado=resultado,
                               keypoints_img=imagen_keypoints_filename,
                               cantidad=cantidad,
                               coincidencias=coincidencias,
                               avion=avion_detectado,
                               confianza=confianza,
                               tiempo=tiempo,
                               memoria=memoria,
                               imagenes=imagenes_previas)

    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def gen_webcam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("❌ No se pudo abrir la cámara.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Medir FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Detección de rostros
        resultados = modelo_yolo(frame)[0]
        cantidad = 0
        for box in resultados.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box[:4]
            frame = pixelar_region(frame, x1, y1, x2, y2)
            cantidad += 1

        # Uso de memoria
        memoria = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)

        # Confianza simulada de SIFT (solo puntos detectados)
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp = cv2.SIFT_create().detect(gray, None)
            total_kp = len(kp)
            confianza = min(100, round((total_kp / 250.0) * 100, 2))  # simulación
        except:
            confianza = 0

        # Dibujar información en pantalla
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
        cv2.putText(frame, f'Rostros: {cantidad}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 50), 2)
        cv2.putText(frame, f'Memoria: {memoria} MB', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
        cv2.putText(frame, f'Confianza (SIFT): {confianza:.1f}%', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/webcam')
def webcam():
    return Response(gen_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
