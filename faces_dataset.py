import cv2
import sqlite3

cam = cv2.VideoCapture(0)
detetcor = cv2.CascadeClassifier('Classifiers/face.xml')


def insertOrUpdate(Id, name, flight, age,gender):
    conn = sqlite3.connect("facedb.db")
    cmd = "SELECT * FROM faces WHERE id=" + str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if (isRecordExist == 1):
        cmd="UPDATE faces SET names='" + str(name) + "', age='" + age + "',gender='" + str(gender) + "',  flightno='" + flight + "' WHERE id='" + Id + "'"
    else:
        cmd="INSERT INTO faces(id,names,flightno,gender,age) Values('"+str(Id)+"','"+str(name)+"','"+flight+"','"+gender+"','"+age+"' )"
    conn.execute(cmd)
    conn.commit()
    conn.close()


id = input('enter your id')
names = input('enter your name')
age = input('enter your age')
gender = input('enter your gender')
flight = input('enter flight no')
insertOrUpdate(id, names, flight, age,gender)
sampleNum = 0
while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = detetcor.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sampleNum = sampleNum + 1
        cv2.imwrite("datase/user." + id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(im, (x - 25, y - 25), (x + w + 25, y + h + 25), (255, 0, 0), 2)
    cv2.imshow('im', im)
    cv2.waitKey(100)
    if sampleNum > 50:
        cam.release()
        cv2.destroyAllWindows()
        break
