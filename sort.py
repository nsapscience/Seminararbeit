import os #Interaktion mit dem Betriebssystem
import shutil #Verschieben von Dateien
import random #Mischen der Daten

base_path = r"E:\Datenbank" #Pfad zu der Datenbank
images_path = os.path.join(base_path, "images") #Pfad zu den Bildern
labels_path = os.path.join(base_path, "labels") #Pfad zu den Label

for split in ["train", "val"]:
    os.makedirs(os.path.join(images_path, split), exist_ok=True) #Ordner Erstellen, exist_ok=True damit nicht Abturz
    os.makedirs(os.path.join(labels_path, split), exist_ok=True)

images = [
    f for f in os.listdir(images_path) 
    if f.lower().endswith((".JPG", ".jpeg", ".jpg"))
]

print(f"Gefundene Bilder: {len(images)}") # Rückmeldung fall keine Bilder gefunden wurden
if len(images) == 0:
    print("Keine Bilder gefunden! Prüfe den Pfad oder die Dateiendungen.")
    exit()

random.shuffle(images) #Mischt die Liste der Bildnamen

split_idx = int(0.7 * len(images)) #Gewünschte Größe der Trainingsbilder
train_images = images[:split_idx] #Ersten 70% erhalten die Trainingsbilder
val_images = images[split_idx:] #Erhalten die restlichen 30%

print(f"Train: {len(train_images)} Bilder")
print(f"Val:   {len(val_images)} Bilder")

for img in train_images: #Verschiebung der Trainingsbilder
    label = os.path.splitext(img)[0] + ".txt"
    src_img = os.path.join(images_path, img)
    src_label = os.path.join(labels_path, label)

    if os.path.exists(src_img) and os.path.exists(src_label): #Sortiert wie Bilder und Label nach "train"
        shutil.move(src_img, os.path.join(images_path, "train", img))
        shutil.move(src_label, os.path.join(labels_path, "train", label))

for img in val_images:
    label = os.path.splitext(img)[0] + ".txt"
    src_img = os.path.join(images_path, img)
    src_label = os.path.join(labels_path, label)

    if os.path.exists(src_img) and os.path.exists(src_label): #Sortiert nach "val"
        shutil.move(src_img, os.path.join(images_path, "val", img))
        shutil.move(src_label, os.path.join(labels_path, "val", label))

print("Aufteilung abgeschlossen!")
