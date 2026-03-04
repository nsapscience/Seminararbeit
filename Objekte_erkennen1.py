import cv2
from ultralytics import YOLO
import numpy as np

def main():

    model = YOLO('yolov8n.pt')  # YOLOv8 Modell laden

    # Zwei Kameras öffnen (Kamera 1 und 2)
    cap1 = cv2.VideoCapture(1)
    cap2 = cv2.VideoCapture(2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Fehler: Eine oder beide Kameras konnten nicht geöffnet werden.")
        return

    class_names = model.names

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Fehler beim Lesen einer Kamera.")
            break

        # YOLO-Erkennung für beide Kameras
        results1 = model(frame1, conf=0.4)
        results2 = model(frame2, conf=0.4)

        # Funktion zur Darstellung von Erkennungen
        def draw_results(frame, results):
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy()

                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{class_names[int(class_id)]}: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return frame

        frame1 = draw_results(frame1, results1)
        frame2 = draw_results(frame2, results2)

        # Beide Bilder gleich groß machen und nebeneinander anzeigen
        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))
        combined = np.hstack((frame1, frame2))

        cv2.imshow('YOLOv8 Multi-Camera', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()