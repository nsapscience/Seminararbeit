#Import aller benötigter Sachen
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO(r"-") #Pfad zur KI

model.model.names = {0: 'Kratzer', 1: 'fehlendes Material', 2: 'Fleck', 3: 'Anguss noch dran'} #Beschreiung der Klassen

cap1 = cv2.VideoCapture(0) #Initialisierung der Kameras
cap2 = cv2.VideoCapture(1)

while True: #Kameras in Echtzeit anzeigen
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    #Rückmeldung falls die Kameras nicht geöffnet werden konnten
    if not ret1 or not ret2:
        print("Fehler beim Zugriff auf Kameras")
        break

    # YOLO auf beide Frames anwenden
    results1 = model(frame1)
    results2 = model(frame2)
    
    # Ergebnisse als Bilder zurückgeben
    img1 = results1[0].plot()
    img2 = results2[0].plot()
    
    # Bilder nebeneinander zusammenfügen
    combined = np.hstack((img1, img2))
    
    #Zeigt Webcams in einem Fenster mit Namen: "Beide Webcams"
    cv2.imshow("Beide Webcams", combined)
    
    #Schließt das Fenster mit ESC
    ESC_KEY = 27
    if cv2.waitKey(1) & 0xFF == ESC_KEY:
        break


#Gibt die Kameras nach Beenden des Programms wieder frei
cap1.release()
cap2.release()
cv2.destroyAllWindows()