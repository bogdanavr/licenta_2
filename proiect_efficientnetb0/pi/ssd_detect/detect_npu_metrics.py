import os
import csv
import cv2
import time
import numpy as np
from picamera2 import Picamera2
from emotions_utils import EmotionSystem

from hailo_platform import (
    HEF,
    VDevice,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)

# -------------------------
# Config
# -------------------------
MODEL_PATH = "/home/bogdanavr/Desktop/pi/models/emotion_model.hef"
PROTO = "/home/bogdanavr/Desktop/pi/models/deploy.prototxt.txt"
CAFFE = "/home/bogdanavr/Desktop/pi/models/res10_300x300_ssd_iter_140000.caffemodel"
RESULTS_DIR = "/home/bogdanavr/Desktop/pi/results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "controlled_conditions_npu.csv")

IMG_SIZE = 224
EVAL_SECONDS = 10.0
WINDOW_SIZE = 15

# Modifica inainte de fiecare rulare
SCENARIO_NAME = "iluminare_buna"
RUN_ID = 1

LABELS = {
    0: "HAPPY",
    1: "NEUTRAL",
    2: "SAD",
    3: "SURPRISE",
}

CONF_THR = 0.6
SCALE = 0.75
PADDING = 0.15

SHOW_WINDOW = True


# -------------------------
# Helpers
# -------------------------
def clamp(v, lo, hi):
    return max(lo, min(int(v), hi))


def preprocess_npu(face_img_bgr):
    img = cv2.resize(face_img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.uint8), axis=0)  # (1,224,224,3)


def detect_faces_ssd(net, frame_bgr):
    h, w = frame_bgr.shape[:2]

    if SCALE != 1.0:
        small = cv2.resize(
            frame_bgr,
            (int(w * SCALE), int(h * SCALE)),
            interpolation=cv2.INTER_AREA
        )
    else:
        small = frame_bgr

    hs, ws = small.shape[:2]

    blob = cv2.dnn.blobFromImage(
        small,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < CONF_THR:
            continue

        box = detections[0, 0, i, 3:7] * np.array([ws, hs, ws, hs])
        x1, y1, x2, y2 = box.astype(int)

        if SCALE != 1.0:
            x1 = int(x1 / SCALE)
            y1 = int(y1 / SCALE)
            x2 = int(x2 / SCALE)
            y2 = int(y2 / SCALE)

        x1 = clamp(x1, 0, w - 1)
        y1 = clamp(y1, 0, h - 1)
        x2 = clamp(x2, 0, w - 1)
        y2 = clamp(y2, 0, h - 1)

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        px = int(bw * PADDING)
        py = int(bh * PADDING)

        x1p = clamp(x1 - px, 0, w - 1)
        y1p = clamp(y1 - py, 0, h - 1)
        x2p = clamp(x2 + px, 0, w - 1)
        y2p = clamp(y2 + py, 0, h - 1)

        faces.append((x1p, y1p, x2p - x1p, y2p - y1p, conf))

    faces.sort(key=lambda t: t[4], reverse=True)
    return faces


def ensure_csv_header(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "backend",
                "scenario",
                "run_id",
                "window_size",
                "eval_seconds",
                "total_frames",
                "frames_with_face",
                "face_detection_pct",
                "label_changes_10s",
                "mean_latency_ms",
                "mean_fps"
            ])


def append_result(csv_path, row):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# -------------------------
# Main
# -------------------------
def main():
    ensure_csv_header(RESULTS_CSV)

    print("Loading HEF model on Hailo NPU...")
    hef = HEF(MODEL_PATH)

    print("Loading SSD face detector...")
    net = cv2.dnn.readNetFromCaffe(PROTO, CAFFE)

    with VDevice() as vdevice:
        network_group = vdevice.configure(hef)[0]
        ng_params = network_group.create_params()

        in_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=True, format_type=FormatType.UINT8
        )
        out_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        input_name = list(in_params.keys())[0]
        output_name = list(out_params.keys())[0]

        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        try:
            picam2.set_controls({"AfMode": 2})
        except Exception:
            pass
        picam2.start()

        system = EmotionSystem(window_size=WINDOW_SIZE)
        eval_start_time = time.time()
        total_frames = 0
        frames_with_face = 0

        prev_stable_label = None
        label_changes = 0

        latency_sum_ms = 0.0
        latency_count = 0

        print(f"Started evaluation on NPU | scenariu={SCENARIO_NAME} | run={RUN_ID}")

        try:
            with InferVStreams(network_group, in_params, out_params) as infer_pipeline:
                with network_group.activate(ng_params):
                    while True:
                        frame_raw = picam2.capture_array()
                        frame = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
                        frame = cv2.flip(frame, 1)

                        total_frames += 1
                        fps = system.update_fps()

                        faces = detect_faces_ssd(net, frame)
                        current_latency = 0.0
                        stable_label = None

                        if len(faces) > 0:
                            frames_with_face += 1

                            # Proceseaza doar fata dominanta
                            x, y, w, h, conf = faces[0]
                            face_roi = frame[y:y + h, x:x + w]

                            if face_roi.size > 0:
                                start_inf = time.perf_counter()
                                input_tensor = preprocess_npu(face_roi)
                                outputs = infer_pipeline.infer({input_name: input_tensor})
                                current_latency = (time.perf_counter() - start_inf) * 1000.0

                                latency_sum_ms += current_latency
                                latency_count += 1

                                probs = outputs[output_name][0]
                                idx = int(np.argmax(probs))
                                raw_label = LABELS.get(idx, "UNK")

                                system.update_buffer(raw_label)
                                stable_label = system.get_stable_emotion()

                                if stable_label not in [None, "", "Collecting..."]:
                                    if prev_stable_label is None:
                                        prev_stable_label = stable_label
                                    elif stable_label != prev_stable_label:
                                        label_changes += 1
                                        prev_stable_label = stable_label

                                if SHOW_WINDOW:
                                    color = (0, 255, 0) if stable_label == "HAPPY" else (0, 0, 255)
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                    cv2.putText(
                                        frame,
                                        f"{stable_label} ({conf:.2f})",
                                        (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.75,
                                        color,
                                        2,
                                    )
                        else:
                            if SHOW_WINDOW:
                                cv2.putText(
                                    frame,
                                    "NO FACE",
                                    (10, 95),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 0, 255),
                                    2,
                                )

                        if SHOW_WINDOW:
                            cv2.putText(
                                frame,
                                f"FPS: {fps:.1f}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 0),
                                2,
                            )
                            if current_latency > 0:
                                cv2.putText(
                                    frame,
                                    f"Lat: {current_latency:.1f} ms",
                                    (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 255),
                                    2,
                                )

                            cv2.imshow("NPU Metrics Evaluation", frame)

                        elapsed = time.time() - eval_start_time
                        if elapsed >= EVAL_SECONDS:
                            break

                        if SHOW_WINDOW and (cv2.waitKey(1) & 0xFF == ord("q")):
                            break

        finally:
            picam2.stop()
            cv2.destroyAllWindows()

        elapsed = max(time.time() - eval_start_time, 1e-6)
        face_detection_pct = 100.0 * frames_with_face / max(total_frames, 1)
        mean_latency_ms = latency_sum_ms / max(latency_count, 1)
        mean_fps = total_frames / elapsed

        print("\n=== REZULTATE NPU ===")
        print(f"Scenariu: {SCENARIO_NAME}")
        print(f"Rulare: {RUN_ID}")
        print(f"Durata [s]: {elapsed:.2f}")
        print(f"Cadre totale: {total_frames}")
        print(f"Cadre cu fata detectata: {frames_with_face}")
        print(f"Detectie fata [%]: {face_detection_pct:.2f}")
        print(f"Schimbari eticheta / 10 s: {label_changes}")
        print(f"Latenta medie [ms]: {mean_latency_ms:.2f}")
        print(f"FPS mediu: {mean_fps:.2f}")
        print("=====================\n")

        append_result(RESULTS_CSV, [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            "NPU",
            SCENARIO_NAME,
            RUN_ID,
            WINDOW_SIZE,
            round(elapsed, 2),
            total_frames,
            frames_with_face,
            round(face_detection_pct, 2),
            label_changes,
            round(mean_latency_ms, 2),
            round(mean_fps, 2)
        ])


if __name__ == "__main__":
    main()
