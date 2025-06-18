import cv2
import numpy as np
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('http://192.168.1.87:8080/video')  # Ajusta con tu IP

# Simulación de carretera como fondo (640x480)
visor_background = cv2.imread("img/visor.png")  # Asegúrate de tener esta imagen en tu carpeta
visor_background = cv2.resize(visor_background, (640, 480))

# Crear y posicionar ventanas
cv2.namedWindow("Camara Trasera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Visor de Casco", cv2.WINDOW_NORMAL)
cv2.moveWindow("Camara Trasera", 0, 0)
cv2.moveWindow("Visor de Casco", 650, 0)

prev_area = None
alert_active = False
alert_start_time = 0
alert_duration = 1.0  # duración en segundos

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el stream")
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, verbose=False)[0]

    current_time = time.time()
    new_alert_triggered = False

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if conf > 0.5:
            area = (x2 - x1) * (y2 - y1)
            if prev_area is not None and area > prev_area * 1.2:
                new_alert_triggered = True
                alert_start_time = current_time
                alert_active = True
            prev_area = area

            color = (0, 0, 255) if alert_active else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Verificar si la alerta debe seguir activa
    if alert_active and (current_time - alert_start_time > alert_duration):
        alert_active = False

    # Pantalla 1: Cámara trasera
    cv2.imshow("Camara Trasera", frame)

    # Pantalla 2: Visor del casco con fondo
    visor = visor_background.copy()
    if alert_active:
        cv2.putText(visor, 'OBJETO ACERCANDOSE', (80, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    cv2.imshow("Visor de Casco", visor)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
