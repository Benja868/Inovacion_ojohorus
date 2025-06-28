import cv2, time, datetime, threading, webbrowser, pygame
from ultralytics import YOLO

pygame.mixer.init()
pygame.mixer.music.load("../assets/alerta.mp3")

model = YOLO('../assets/yolov8n.pt')

cap_tel = cv2.VideoCapture('http://192.168.1.87:8080/video')
cap_webcam = cv2.VideoCapture(0)

gps_path = [(-33.4489 + i*0.0002, -70.6693 + i*0.0003) for i in range(100)]
gps_index = 0
prev_area = None
alert_active = False
alert_start, alert_duration = 0, 3.0
sound_played = False

def reproducir_alerta():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

def actualizar_mapa(lat, lon):
    with open("../gps_mapa/data.js", "w") as f:
        f.write(f"var currentLat = {lat};\nvar currentLon = {lon};")

webbrowser.open("../gps_mapa/index.html")

while True:
    ret_tel, ft = cap_tel.read()
    ret_webcam, fw = cap_webcam.read()
    if not ret_tel or not ret_webcam: break

    fw = cv2.resize(fw, (640, 480))
    ft = cv2.resize(ft, (160, 120))
    results = model(ft, verbose=False)[0]
    now = time.time()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = model.names[int(box.cls[0])]
        if conf > 0.5:
            area = (x2-x1)*(y2-y1)
            if prev_area and area > prev_area * 1.2:
                alert_active, sound_played, alert_start = True, False, now
            prev_area = area
            clr = (0,0,255) if alert_active else (0,255,0)
            cv2.rectangle(ft, (x1,y1),(x2,y2), clr,2)
            cv2.putText(ft, f'{label} {conf:.2f}', (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,clr,2)

    if alert_active and now - alert_start > alert_duration:
        alert_active, sound_played = False, False
    if alert_active and not sound_played:
        threading.Thread(target=reproducir_alerta, daemon=True).start()
        sound_played = True

    x_off = fw.shape[1]-ft.shape[1]-10
    fw[10:10+ft.shape[0], x_off:x_off+ft.shape[1]] = ft

    if alert_active:
        cv2.putText(fw, "OBJETO ACERCANDOSE", (10,200),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.putText(fw, datetime.datetime.now().strftime("%H:%M:%S"),
                (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    lat, lon = gps_path[gps_index]
    gps_index = (gps_index + 1) % len(gps_path)
    cv2.putText(fw, f"GPS: {lat:.5f}, {lon:.5f}",
                (10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
    actualizar_mapa(lat, lon)

    cv2.imshow("Visor", fw)
    if cv2.waitKey(1) == 27: break

cap_tel.release()
cap_webcam.release()
cv2.destroyAllWindows()
