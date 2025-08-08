# Importación de librerías necesarias
import os                          # Para manejar archivos y carpetas
import cv2                         # OpenCV para captura y procesamiento de imágenes
import time                        # Para retardos y temporización
import random                      # Para generar retrasos aleatorios
import threading                   # Para ejecutar hilos concurrentes
import numpy as np                 # Para operaciones numéricas eficientes
from datetime import datetime     # Para crear timestamps legibles
from ultralytics import YOLO       # Para cargar y usar el modelo YOLO

# ─────────────────────────────────────────────────────────────
# Parámetros de configuración y rutas

VIDEO_PATH   = "VID-20250314-WA0009.mp4"   # Video de entrada
MODEL_PATH   = "bestcanasta.pt"            # Modelo YOLO entrenado
RAW_FOLDER   = "raw_snapshots"             # Carpeta para imágenes crudas
ERROR_FOLDER = "error_images"              # Carpeta para imágenes anotadas de error
CYCLES       = 5                           # Ciclos consecutivos para fallo/pase
CONF_THRESH  = 0.25                        # Umbral de confianza YOLO
DELAY_RANGE  = (0.1, 0.5)                  # Retrasos aleatorios entre análisis

LOG_FILE     = "event_log.txt"             # Archivo de log para registrar eventos
# ─────────────────────────────────────────────────────────────

# Crear carpetas si no existen
os.makedirs(RAW_FOLDER,   exist_ok=True)
os.makedirs(ERROR_FOLDER, exist_ok=True)

# Carga el modelo YOLO una sola vez
model = YOLO(MODEL_PATH)

# Evento para detener de forma segura el hilo de análisis
stop_event = threading.Event()

# Contadores de fallos/pases consecutivos (por ruta de imagen)
fail_counts = {}
pass_counts = {}

# Mapa para relacionar cada raw_path con su ID de canasta
raw_id_map = {}

# Fuente para anotaciones de texto en pantalla
font = cv2.FONT_HERSHEY_SIMPLEX

def write_log_entry(timestamp, canasta_id, bag_count, avg_conf):
    """
    Abre LOG_FILE en modo 'append', escribe una línea con:
    Fecha:...;Canasta:...;Bolsas:...;Confianza:...
    y cierra el archivo inmediatamente.
    """
    line = (
        f"Fecha:{timestamp};"
        f"Canasta:{canasta_id};"
        f"Bolsas:{bag_count};"
        f"Confianza:{avg_conf:.2f}\n"
    )
    with open(LOG_FILE, "a") as f:
        f.write(line)

def analysis_loop():
    """
    Hilo en segundo plano que procesa las imágenes guardadas en RAW_FOLDER:
      - Realiza inferencia YOLO y mantiene conteos de fallos/pases consecutivos.
      - Si hay CYCLES fallos seguidos, guarda la imagen anotada en ERROR_FOLDER y lo registra en el log.
      - Si hay CYCLES pases seguidos, registra el log FINAL y elimina el RAW.
    """
    while not stop_event.is_set():
        # Obtiene y mezcla las rutas de imagen en RAW_FOLDER
        paths = sorted([
            os.path.join(RAW_FOLDER, f)
            for f in os.listdir(RAW_FOLDER)
            if f.lower().endswith((".jpg", ".png"))
        ])
        random.shuffle(paths)

        for img_path in paths:
            if stop_event.is_set():
                return

            img = cv2.imread(img_path)   # Carga la imagen
            if img is None:
                continue                 # Salta si no pudo cargar

            # Inferencia con YOLO
            results = model(img, conf=CONF_THRESH)[0]
            boxes   = results.boxes.xyxy.cpu().numpy()
            confs   = results.boxes.conf.cpu().numpy()
            count   = len(boxes)
            passed  = (count == 22 and confs.mean() >= CONF_THRESH)

            # Actualiza contadores
            fail_counts[img_path] = 0 if passed else fail_counts.get(img_path, 0) + 1
            pass_counts[img_path] = pass_counts.get(img_path, 0) + 1 if passed else 0

            # 1) Si alcanzó CYCLES fallos, guarda la imagen anotada
            if fail_counts[img_path] == CYCLES:
                for (x1, y1, x2, y2), c in zip(boxes, confs):
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                    lbl = f"{c:.2f}"
                    (tw, th), base = cv2.getTextSize(lbl, font, 0.5, 1)
                    cv2.rectangle(img,
                                  (x1, y1-th-base),
                                  (x1+tw, y1),
                                  (0,255,0), -1)
                    cv2.putText(img, lbl,
                                (x1, y1-4),
                                font, 0.5,
                                (0,0,0), 1)

                text = f"Bolsas: {count}"
                fs, thk = 1.5, 3
                (tw, th), base = cv2.getTextSize(text, font, fs, thk)
                cv2.rectangle(img,
                              (550, 40-th-base),
                              (550+tw, 40+base),
                              (0,255,0), -1)
                cv2.putText(img, text, (550, 40),
                            font, fs, (255,255,255), thk)

                base = os.path.splitext(os.path.basename(img_path))[0]
                outp = os.path.join(ERROR_FOLDER, f"{base}_{count}.jpg")
                cv2.imwrite(outp, img)
                print(f"[Error] Guardado anotada → {outp}")

                # Registrar en log tras CYCLES fallos
                ts_log     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                canasta_id = raw_id_map.get(img_path, "desconocido")
                avg_conf   = float(confs.mean()) if count else 0.0
                write_log_entry(ts_log, canasta_id, count, avg_conf)

            # 2) Si alcanzó CYCLES pases, registrar LOG FINAL y eliminar RAW
            if pass_counts.get(img_path, 0) == CYCLES:
                # Obtiene el ID de canasta asociado al raw
                canasta_id = raw_id_map.get(img_path, "desconocido")
                # Calcula promedio de confianza final
                avg_conf = float(confs.mean()) if count else 0.0
                # Timestamp de registro
                ts_log = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Escribe en el log la información final tras CYCLES pases
                write_log_entry(ts_log, canasta_id, count, avg_conf)

                # Elimina la imagen cruda
                try:
                    os.remove(img_path)
                    print(f"[Info] Eliminado raw comprobado → {img_path}")
                except OSError as e:
                    print(f"[Warning] No se pudo eliminar {img_path}: {e}")

                # Limpia contadores y mapeo
                fail_counts.pop(img_path, None)
                pass_counts.pop(img_path, None)
                raw_id_map.pop(img_path,    None)

            # Pausa aleatoria antes del siguiente análisis de imagen
            time.sleep(random.uniform(*DELAY_RANGE))
        # Espera un segundo antes de re-escaneo completo
        time.sleep(1)

# Arranca el hilo de análisis en segundo plano
thread = threading.Thread(target=analysis_loop, daemon=True)
thread.start()

# ─────────────────────────────────────────────────────────────
# Bucle principal: captura de video y detección de canastas en vivo

cap           = cv2.VideoCapture(VIDEO_PATH)              # Inicia captura de video
fps           = cap.get(cv2.CAP_PROP_FPS) or 30            # FPS detectados
frame_delay   = int(1000 / fps)                            # Retardo entre frames (ms)
canasta_count = 0                                          # Contador de eventos únicos
prev_detected = False                                      # Prevención de duplicados

print(f"→ FPS: {fps:.2f}, delay/frame: {frame_delay} ms")
print("▶️ Iniciando captura de canastas…")

while True:
    ret, frame = cap.read()
    if not ret:
        # Al final del video, regresa al inicio
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 1) Detección rápida de la región de la canasta (HSV + contornos)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([100,150,50]),
                       np.array([140,255,255]))
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern)
    cnts,_ = cv2.findContours(mask,
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)

    # 2) Busca primer contorno rectangular grande
    found = False
    x = y = w = h = 0
    for cnt in cnts:
        if cv2.contourArea(cnt) < 10000:
            continue
        approx = cv2.approxPolyDP(cnt,
                                  0.02 * cv2.arcLength(cnt, True),
                                  True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            found = True
            break

    # 3) Si es un nuevo evento de canasta, guarda raw y asigna ID
    if found and not prev_detected:
        canasta_count += 1
        # Nombre del archivo incluye CAN + timestamp
        ts_file  = datetime.now().strftime("%Y%m%d%H%M%S")
        raw_path = os.path.join(RAW_FOLDER, f"CAN{ts_file}.jpg")
        cv2.imwrite(raw_path, frame)
        print(f"[Capture] Guardado raw → {raw_path}")

        # Asocia este raw_path con su ID de canasta
        raw_id_map[raw_path] = canasta_count

    # Actualiza flag para evitar múltiples capturas del mismo evento
    prev_detected = found

    # 4) Dibuja la canasta en pantalla y muestra contador global
    if found:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,255), 2)
    cv2.putText(frame,
                f"Canastas: {canasta_count}",
                (20, 40),
                font, 1, (255,255,255), 2,
                cv2.LINE_AA)

    # 5) Muestra la ventana y permite salir con ESC
    cv2.imshow("Detección de canastas", frame)
    if cv2.waitKey(frame_delay) & 0xFF == 27:
        break

# ─────────────────────────────────────────────────────────────
# Limpieza final

stop_event.set()           # Señala al hilo que termine
thread.join()              # Espera a que el hilo acabe
cap.release()              # Libera la cámara/video
cv2.destroyAllWindows()    # Cierra todas las ventanas