import cv2, os
import numpy as np
from PIL import Image
recognizer = cv2.face.EigenFaceRecognizer_create()

detector = cv2.CascadeClassifier("./dataset/haarcascade_frontalface_default.xml");

def getImagesAndLabels():
    path = "./dataset/images"
    faceSamples = []
    ids = np.intc([])

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    for x in imagePaths:
        print(cv2.imread(x))
        faceSamples.append(cv2.imread(x))
        ids = np.append(ids,1)
    """ for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        print(PIL_img.size())
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id) """
    return faceSamples,ids

faces,ids = getImagesAndLabels()
#for x in np.array(ids):
#    print(x)
#print(np.array(ids))
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml')
