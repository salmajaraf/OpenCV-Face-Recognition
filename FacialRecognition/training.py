

import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'C:/Users/salma/OneDrive/Documents/GitHub/OpenCV-Face-Recognition/dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create() # Création d'un objet cv2.face.LBPHFaceRecognizer pour le modèle de reconnaissance faciale LBPH.
detector = cv2.CascadeClassifier('C:/Users/salma/OneDrive/Documents/GitHub/OpenCV-Face-Recognition/FacialRecognition/haarcascade_frontalface_default.xml');

# function to get the images and label data
def getImagesAndLabels(path):# Cette fonction parcourt chaque image dans le répertoire spécifié.

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])#Elle extrait l'ID du visage à partir du nom de fichier.
        faces = detector.detectMultiScale(img_numpy)#Elle détecte les visages dans l'image à l'aide du classificateur Haar Cascade.

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])#Elle extrait les régions de visage détectées et les stocke dans une liste (faceSamples) avec les étiquettes correspondantes (ids).
            ids.append(id)

    return faceSamples,ids#Enfin, elle retourne ces listes.

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))#Entraînement du modèle de reconnaissance faciale en utilisant les échantillons de visages et les étiquettes obtenus à l'étape précédente avec la méthode train() du reconnaiseur.

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
