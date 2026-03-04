from ultralytics import YOLO #import von YOLO

model = YOLO(r"E:\Seminararbeit\runs\detect\train17\weights\best.pt") #Pfad zur zu trainierenden KI

model.train(
    data = r"E:\Seminararbeit\dataset\data.yaml", #Pfad zur data.yaml
    epochs = 120, #Anzahl der Durchläufe
    imgsz = 640, #Bildgröße (Standard für YOLO)
    augment = True #Trainingsgenauigkeit verbessern
)

print("Training abgeschlossen!") #Rückmeldung das Training abgeschlossen ist