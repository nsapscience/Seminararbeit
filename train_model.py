from ultralytics import YOLO #import von YOLO

model = YOLO("yolov8n.pt") #Pfad zur zu trainierenden KI

model.train(
    data = r"E:\Datenbank\data.yaml", #Pfad zur data.yaml
    epochs = 50, #Anzahl der Durchläufe
    imgsz = 640, #Bildgröße (Standard für YOLO)
    augment = True #Trainingsgenauigkeit verbessern
)

print("Training abgeschlossen!") #Rückmeldung das Training abgeschlossen ist