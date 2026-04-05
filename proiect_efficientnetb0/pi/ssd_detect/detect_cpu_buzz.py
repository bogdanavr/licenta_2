import cv2
import numpy as np
import onnxruntime as ort
import time
import threading
from picamera2 import Picamera2
from emotions_utils import EmotionSystem
from test_buzzer import buzz_for_emotion

# 1. Configurari
MODEL_PATH = "/home/bogdanavr/Desktop/pi/models/emotion_model.onnx"
IMG_SIZE = 224

LABELS = {
    0: "HAPPY",
    1: "NEUTRAL",
    2: "SAD",
    3: "SURPRISE"
}

# SSD face detector files (le-ai descarcat deja in /models)
PROTO = "/home/bogdanavr/Desktop/pi/models/deploy.prototxt.txt"
CAFFE = "/home/bogdanavr/Desktop/pi/models/res10_300x300_ssd_iter_140000.caffemodel"

CONF_THR = 0.6   # 0.5 daca rateaza, 0.7 daca are false positives
SCALE = 0.75     # 0.5 mai rapid, 1.0 mai precis
PADDING = 0.15   # extinde box-ul putin


def clamp(v, lo, hi):
    return max(lo, min(int(v), hi))
def detect_faces_ssd(net, frame_bgr):
    """
    Returneaza lista de fete ca (x, y, w, h, confidence) in coordonate frame original,
    cu padding deja aplicat.
    """
    (h, w) = frame_bgr.shape[:2]

    if SCALE != 1.0:
        small = cv2.resize(
            frame_bgr,
            (int(w * SCALE), int(h * SCALE)),
            interpolation=cv2.INTER_AREA
        )
    else:
        small = frame_bgr

    (hs, ws) = small.shape[:2]

    blob = cv2.dnn.blobFromImage(
        small, 1.0, (300, 300),
        (104.0, 177.0, 123.0),  # mean BGR pt modelul SSD
        swapRB=False, crop=False
    )
    net.setInput(blob)
    detections = net.forward()

    faces = []

    if detections is None or detections.size == 0:
        return faces

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < CONF_THR:
            continue

        box = detections[0, 0, i, 3:7] * np.array([ws, hs, ws, hs])
        x1, y1, x2, y2 = box.astype(int)

        # scale back to original
        if SCALE != 1.0:
            x1 = int(x1 / SCALE)
            y1 = int(y1 / SCALE)
            x2 = int(x2 / SCALE)
            y2 = int(y2 / SCALE)

        # clamp
        x1 = clamp(x1, 0, w - 1)
        y1 = clamp(y1, 0, h - 1)
        x2 = clamp(x2, 0, w - 1)
        y2 = clamp(y2, 0, h - 1)

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        # padding
        px = int(bw * PADDING)
        py = int(bh * PADDING)
        x1p = clamp(x1 - px, 0, w - 1)
        y1p = clamp(y1 - py, 0, h - 1)
        x2p = clamp(x2 + px, 0, w - 1)
        y2p = clamp(y2 + py, 0, h - 1)

        faces.append((x1p, y1p, x2p - x1p, y2p - y1p, conf))

    faces.sort(key=lambda t: t[4], reverse=True)
    return faces


def preprocess(face_img_bgr):
    img = cv2.resize(face_img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img.astype(np.float32), axis=0)


def main():
    # 2. Initializare ONNX Runtime pe CPU
    print("Incarcare model ONNX...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(MODEL_PATH, opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # 2b. Initializare SSD face detector
    print("Incarcare SSD face detector...")
    net = cv2.dnn.readNetFromCaffe(PROTO, CAFFE)

    # 3. Initializare Camera
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    try:
        picam2.set_controls({"AfMode": 2})
    except Exception:
        pass
    picam2.start()

    # 5. Sistem stabilitate
    system = EmotionSystem(window_size=15)

    # stare pentru buzzer
    last_buzzed_emotion = None
    last_buzz_time = 0.0
    BUZZ_COOLDOWN = 1.5  # secunde

    print("Sistem pornit! Apasa 'q' pentru a opri.")

    try:
        while True:
            frame_raw = picam2.capture_array()
            frame = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)

            fps = system.update_fps()

            # SSD faces (in loc de Haar)
            faces = detect_faces_ssd(net, frame) or []

            current_latency = 0.0

            for (x, y, w, h, conf) in faces:
                face_roi = frame[y:y + h, x:x + w]
                if face_roi.size == 0:
                    continue

                start_inf = time.perf_counter()
                input_tensor = preprocess(face_roi)
                outputs = session.run(None, {input_name: input_tensor})
                current_latency = (time.perf_counter() - start_inf) * 1000.0

                probs = outputs[0][0]
                idx = int(np.argmax(probs))
                raw_label = LABELS.get(idx, "UNK")

                system.update_buffer(raw_label)
                stable_label = system.get_stable_emotion()

                # apel buzzer doar la schimbarea emotiei stabile
                now = time.time()
                if (
                    stable_label
                    and stable_label != last_buzzed_emotion
                    and (now - last_buzz_time) > BUZZ_COOLDOWN
                ):
                    threading.Thread(
                        target=buzz_for_emotion,
                        args=(stable_label,),
                        daemon=True
                    ).start()

                    last_buzzed_emotion = stable_label
                    last_buzz_time = now

                color = (0, 255, 0) if stable_label == 'HAPPY' else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"{stable_label} ({conf:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )

            if current_latency > 0:
                cv2.putText(
                    frame, f"Lat: {current_latency:.1f} ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                )

            cv2.imshow("Robot Emotion Recognition (CPU + SSD Face)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

