import cv2
import os
from os import listdir
from os.path import isfile, join

#This program doesnt actually provide "live video through  webcam", it only displays once a face is detected
#0 is the default camera
vid_cam = cv2.VideoCapture(0)
# face_detector = cv2.CascadeClassifier('./dataset/haarcascade_frontalface_default.xml')
#
# face_id = 1
try:
    mypath = "./dataset/images"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(onlyfiles)
    count0 = len(onlyfiles)
    count1 = len(onlyfiles)
    count2 = len(onlyfiles)
    count3 = len(onlyfiles)
except:
    os.system("mkdir ./dataset/images")
    mypath = "./dataset/images"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(onlyfiles)
    count0 = len(onlyfiles)
    count1 = len(onlyfiles)
    count2 = len(onlyfiles)
    count3 = len(onlyfiles)

frames = 0

face0_detector = cv2.CascadeClassifier('./dataset/haarcascade_frontalface_default.xml')
face1_detector = cv2.CascadeClassifier('./dataset/haarcascade_frontalface_alt.xml')
face2_detector = cv2.CascadeClassifier('./dataset/haarcascade_frontalface_alt2.xml')
profile_detector = cv2.CascadeClassifier('./dataset/haarcascade_profileface.xml')


def generate_rectangle(focus, screen, gray_img, id, tuple_color, counter):
    count = counter
    for (x,y,w,h) in focus:
        cv2.rectangle(screen, (x,y), (x+w,y+h), tuple_color, 2)
        count += 1
        cv2.imwrite("./dataset/images/User." + str(id) + '.' + str(count) + ".jpg", cv2.resize(gray_img[y:y+h,x:x+w],dsize=(100,100)))
        cv2.imshow('Training...', screen)
    return(count)


while(vid_cam.isOpened()):

    #Ret is boolean that returns true if image_frame returns
    ret, image_frame = vid_cam.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    #detectMultiScale returns list of rectangles of faces?
    faces0 = face0_detector.detectMultiScale(gray, 1.3, 5)
    faces1 = face1_detector.detectMultiScale(gray, 1.3, 5)
    faces2 = face2_detector.detectMultiScale(gray, 1.3, 5)
    profiles = profile_detector.detectMultiScale(gray, 1.3, 5)

    count0=generate_rectangle(faces0, image_frame, gray, 1, (255,0,0), count0)
    count1=generate_rectangle(faces1, image_frame, gray, 2, (0,255,0), count1)
    count2=generate_rectangle(faces2, image_frame, gray, 3, (0,0,255), count2)
    count3=generate_rectangle(profiles, image_frame, gray,4, (30,200,0), count3)
    #     detected_count = detected_count + 1

    frames+=1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    elif frames>100000000:
        break
vid_cam.release()
cv2.destroyAllWindows()
