import cv2
import os 
from os import listdir
from os.path import isfile, join

#This program doesnt actually provide "live video through  webcam", it only displays once a face is detected
#0 is the default camera
vid_cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('./dataset/haarcascade_frontalface_default.xml')

face_id = 1
mypath = "./dataset/images"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
count = len(onlyfiles)
detected_count = 0
while(vid_cam.isOpened()):
    if detected_count>5:
        print("New face detected.")
#Ret is boolean that returns true if image_frame returns
    ret, image_frame = vid_cam.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    #detectMultiScale returns list of rectangles of faces?
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    print(faces)
    for (x,y,w,h) in faces:
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        cv2.imwrite("./dataset/images/User." + str(face_id) + '.' + str(count) + ".jpg", cv2.resize(gray[y:y+h,x:x+w], dsize=(100,100)))
        cv2.imshow('frame', image_frame)
        detected_count = detected_count + 1


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    elif count>100000000:
        break
vid_cam.release()
cv2.destroyAllWindows()
