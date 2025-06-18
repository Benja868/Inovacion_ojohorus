import cv2
import numpy as np
import time
import datetime
from ultralytics import YOLO
import threading
import pygame

# Inicializar sonido
pygame.mixer.init()
pygame.mixer.music.load("alerta.mp3")  # Debe estar en la misma carpeta

model = YOLO('yolov8n.pt')

# Cámaras
cap_tel = cv2.VideoCapture('http://192.168.1.87:8080/video')
cap_webcam = cv2.VideoCapture(0)

prev_area = None
alert_active = False
alert_start_time = 0
alert_duration = 3.0
sound_played = False

def reproducir_alerta():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

while True:
    ret_tel, frame_tel = cap_tel.read()
    ret_webcam, frame_webcam = cap_webcam.read()

    if not ret_tel or not ret_webcam:
        print("Error con cámaras")
        break

    frame_webcam = cv2.resize(frame_webcam, (640, 480))
    frame_tel = cv2.resize(frame_tel, (160, 120))

    results = model(frame_tel, verbose=False)[0]
    current_time = time.time()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if conf > 0.5:
            area = (x2 - x1) * (y2 - y1)
            if prev_area is not None and area > prev_area * 1.2:
                alert_start_time = current_time
                alert_active = True
                sound_played = False
            prev_area = area

            color = (0, 0, 255) if alert_active else (0, 255, 0)
            cv2.rectangle(frame_tel, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_tel, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if alert_active and (current_time - alert_start_time > alert_duration):
        alert_active = False
        sound_played = False

    if alert_active and not sound_played:
        threading.Thread(target=reproducir_alerta, daemon=True).start()
        sound_played = True

    # PIP
    x_offset = frame_webcam.shape[1] - frame_tel.shape[1] - 10
    y_offset = 10
    frame_webcam[y_offset:y_offset+frame_tel.shape[0], x_offset:x_offset+frame_tel.shape[1]] = frame_tel

    # Alerta visual
    if alert_active:
        texto_alerta = "OBJETO ACERCANDOSE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size, _ = cv2.getTextSize(texto_alerta, font, font_scale, 2)
        text_x = x_offset + (frame_tel.shape[1] - text_size[0]) // 2
        text_y = y_offset + frame_tel.shape[0] + text_size[1] + 5
        cv2.putText(frame_webcam, texto_alerta, (text_x, text_y), font, font_scale, (0, 0, 0), 3)
        cv2.putText(frame_webcam, texto_alerta, (text_x, text_y), font, font_scale, (255, 255, 255), 1)

    # Hora actual
    hora = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame_webcam, hora, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame_webcam, hora, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow("Visor Frontal con Camara Trasera PiP", frame_webcam)

    if cv2.waitKey(1) == 27:
        break

cap_tel.release()
cap_webcam.release()
cv2.destroyAllWindows()
