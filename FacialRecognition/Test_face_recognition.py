import cv2
import numpy as np
import os 
import openpyxl
from datetime import date

# Chargement du modèle de reconnaissance de visage
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/salma/OneDrive/Documents/GitHub/OpenCV-Face-Recognition/trainer/trainer.yml')

# Chargement du classificateur Haar Cascade pour la détection de visages
cascadePath = "C:/Users/salma/OneDrive/Documents/GitHub/OpenCV-Face-Recognition/FacialRecognition/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Noms des personnes associés à leurs IDs
names = ['None', 'Salma Jaraf', 'Aya Akly', 'Amina elghazouani', 'Ejjiyar Youssef', 'Manal Wardi', 'Barhmi Yousra']

# Initialisation de la vidéo en temps réel
cam = cv2.VideoCapture(0)
cam.set(3, 640) # largeur de la vidéo
cam.set(4, 480) # hauteur de la vidéo

# Définir la taille minimale de la fenêtre pour être reconnue comme un visage
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

# Création d'un objet Workbook pour gérer le fichier Excel
wb = openpyxl.Workbook()
sheet = wb.active
sheet['A1'] = 'Nom'
sheet['B1'] = 'Date'

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Si la confiance est inférieure à 100, une correspondance est trouvée
        if (confidence < 100):
            id = names[id]
            confidence = round(100 - confidence)
            
            # Ajout du nom et de la date dans le fichier Excel
            today = date.today()
            date_str = today.strftime("%d/%m/%Y")
            sheet.append([id, date_str])
            
        else:
            id = "Inconnu"
            confidence = round(100 - confidence)
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence) + '%', (x+5,y+h-5), font, 1, (255,255,0), 1)  

    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Appuyez sur 'ESC' pour quitter la vidéo
    if k == 27:
        break

# Sauvegarde des modifications dans le fichier Excel
excel_file_path = 'output.xlsx'
wb.save(excel_file_path)
wb.close()

# Nettoyage
print("\n [INFO] Fin du programme et nettoyage")
cam.release()
cv2.destroyAllWindows()
