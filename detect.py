import cv2, os
import numpy as np
from PIL import Image
import pickle
import sqlite3

faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
cascadePath = "Classifiers/face.xml"
path = 'dataset'

def getProfile(Id):
    conn = sqlite3.connect("facedb.db")
    cmd = "SELECT * FROM faces WHERE id="+str(Id)
    cursor = conn.execute(cmd)
    pro = None
    for row in cursor:
        pro = row
    conn.close()
    return pro
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        id, conf = recognizer.predict(gray[y:y + h, x:x + w])
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        profile = getProfile(id)
        if profile is not None:
            cv2.putText(im, str(profile[1]), (x, y + h + 30), font, 1, (245, 255, 200), 2, cv2.LINE_AA)
            cv2.putText(im, str(profile[2]), (x, y + h + 60), font, 1, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(im, str(profile[3]), (x, y + h + 90), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(im, str(profile[4]), (x, y + h + 120), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
            #cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('im', im)
    cv2.waitKey(10)
