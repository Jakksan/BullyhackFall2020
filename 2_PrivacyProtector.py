import cv2
import numpy as np
import os
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "/Users/jacksondegruiter/Documents/Bullyhack/python/dataset/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
cam = cv2.VideoCapture(0)
frame_count = 0
people_count_array=[]

last_count = 0
while True:
    ret, im =cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2,5)
    num_people = 0
    for(x,y,w,h) in faces:
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        Id_name = ""
        print(Id, confidence)
        if(Id == 1 and confidence < 100):
            Id_name = "Person"
            num_people+=1
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (200,30,100), 4)
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (200,30,100), -1)
            # os.system("open -a FireFox")
        else:
            Id_name = "Unknown"
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)

        # cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id_name), (x,y-40), font, 2, (255,255,255), 3)
        cv2.putText(im, str(int(confidence)), (x,y+10), font, 1, (255,255,255), 1)        # str(int(100-100*(confidence/186))) + "%"


    people_count_array.append(num_people)
    cv2.rectangle(im, (0,0), (300,100), (200,50,0), -1)
    cv2.putText(im, str("People " + str(num_people)), (50,50), font, 1, (255,255,255), 1)

    # if(last_count>1 and num_people>=last_count):
        # os.system("open -a FireFox")

    last_count = num_people

    cv2.imshow('im',im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    frame_count+=1
    if frame_count>10000:
        break
cam.release()
cv2.destroyAllWindows()
