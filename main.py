#Import aller benötigter Sachen
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO(r"E:\Seminararbeit\runs\detect\train20\weights\best.pt") #Pfad zur KI

model.model.names = {0: 'Kratzer', 1: 'fehlendes Material', 2: 'Fleck', 3: 'Anguss noch dran'} #Beschreiung der Klassen

cap1 = cv2.VideoCapture(0) #Initialisierung der Kameras
cap2 = cv2.VideoCapture(1)

while True:
    ret1, img1 = cap1.read()
    ret2, img2 = cap2.read()

    if not ret1 or not ret2:
        break

    TARGET_HEIGHT = 480

    img1 = cv2.resize(img1, (int(img1.shape[1] * TARGET_HEIGHT / img1.shape[0]), TARGET_HEIGHT))
    img2 = cv2.resize(img2, (int(img2.shape[1] * TARGET_HEIGHT / img2.shape[0]), TARGET_HEIGHT))

    # YOLO Inferenz
    results1 = model(img1, conf=0.25)
    results2 = model(img2, conf=0.25)

    # Bounding Boxes einzeichnen
    annotated1 = results1[0].plot()
    annotated2 = results2[0].plot()

    # Zusammenfügen
    combined = np.hstack((annotated1, annotated2))

    cv2.imshow("Beide Webcams", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()


#Gibt die Kameras nach Beenden des Programms wieder frei
cap1.release()
cap2.release()
cv2.destroyAllWindows()