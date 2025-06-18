import cv2
import numpy as np
import urllib.request
import time
from ultralytics import YOLO

# Modelo YOLO
model = YOLO('yolov8n.pt')

# URL del stream MJPEG desde IP Webcam
url = 'http://192.168.1.87:8080/video'

# Imagen de visor de casco
visor_background = cv2.imread("img/visor.png")
visor_background = cv2.resize(visor_background, (640, 480))

# Abrir conexión al stream
stream = urllib.request.urlopen(url)
bytes_data = b''

# Crear ventanas
cv2.namedWindow("Camara Trasera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Visor de Casco", cv2.WINDOW_NORMAL)
cv2.moveWindow("Camara Trasera", 0, 0)
cv2.moveWindow("Visor de Casco", 650, 0)

prev_area = None
alert_active = False
alert_start_time = 0
alert_duration = 1.0

while True:
    try:
        bytes_data += stream.read(1024)
        a = bytes_data.find(b'\xff\xd8')  # Inicio de JPEG
        b = bytes_data.find(b'\xff\xd9')  # Fin de JPEG

        if a != -1 and b != -1 and a < b:
            try:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]

                img_array = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is None:
                    print("⚠️ Frame inválido (None), se omite.")
                    time.sleep(0.01)
                    continue

            except Exception as e:
                print(f"⚠️ Error al decodificar imagen: {e}")
                time.sleep(0.01)
                continue

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

            if alert_active and (current_time - alert_start_time > alert_duration):
                alert_active = False

            # Pantalla 1: Cámara trasera
            cv2.imshow("Camara Trasera", frame)

            # Pantalla 2: Visor del casco con fondo
            visor = visor_background.copy()
            if alert_active:
                cv2.putText(visor, 'OBJETO ACERCANDOSE', (80, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Visor de Casco", visor)

            if cv2.waitKey(1) == 27:
                break

    except Exception as e:
        print(f"⚠️ Error inesperado en bucle principal: {e}")
        time.sleep(0.5)

cv2.destroyAllWindows()
