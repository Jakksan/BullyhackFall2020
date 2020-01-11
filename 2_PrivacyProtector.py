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
while True:
    ret, im =cam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2,5)
    num_people = 0
    for(x,y,w,h) in faces:
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        Id_name = ""
        print(Id, confidence)
        if(Id == 1 and confidence < 50):
            Id_name = "Person"
            num_people+=1
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (200,30,0), 4)
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (200,30,0), -1)
            # os.system("open -a FireFox")
        else:
            Id_name = "Unknown"
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)
            cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)

        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id_name), (x,y-40), font, 2, (255,255,255), 3)
        cv2.putText(im, str("Confidence: " + str(int(confidence))), (x,y-100), font, 2, (255,255,255), 3)


    people_count_array.append(num_people)
    cv2.rectangle(im, (0,0), (300,100), (200,50,0), -1)
    cv2.putText(im, str("People " + str(num_people)), (50,50), font, 1, (255,255,255), 1)

    if(frame_count%60 == 0):
        frame_count=0
        ones=0
        gt_ones=0
        zeros=0
        sum=0
        for count in people_count_array:
            sum+=count
            if count == 1:
                ones+=1
            elif count >1:
                gt_ones+=1
            elif count == 0:
                zeros+=1
        avg=sum/len(people_count_array)
        try:
            one_to_gt_one_ratio=ones/gt_ones
        except:
            one_to_gt_one_ratio=2
        people_count_array=[]

        print("Average People: " + str(avg))
        print("1:X>1 ratio: " + str(one_to_gt_one_ratio))

        if( one_to_gt_one_ratio > 1):
            os.system("open -a FireFox")



    cv2.imshow('im',im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    frame_count+=1
    if frame_count>100:
        break
cam.release()
cv2.destroyAllWindows()
