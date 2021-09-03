# Face-detetction/authentication-mask-detection-social-distancing-at-airport using ML
A system that helps in ensuring social distancing guidelines and wearing masks, and the face recognition and authentication of the passengers with the airport database at the airport through a website that serves as the control panel for the airport staff members.

1. Unzip the folder from the drive link.
2. Create new folder called datase.
3. Open sqlite studio.
4. Database -> Add a database.
5. Name it facedb and put it's location in the same folder that you had unzipped.
6. Click the created facedb -> Tables and put the table name as faces.
7. Add new columns i.e. id of int datatype and this is the primary key, names of string type, age as int type, gender as string type and flightno as string type.
8. Save the table and run the faces_dataset.py file and enter your id,name,age,gender and flight number.
9. Then run train_face.py
10. Finally run the main.py which will give the website url, click this and you will be diretced to the website which has face detection,mask detection and social distancing features.
11.  if len(red_zone_list)>0:
           playsound('social_dist.mp3', True)
There will be a line on line 318 in main.py, remove the comments and again run main.py  .
Whenever the social distancing button is clicked and the video starts playing, if users in video do not follow social distancing, an audio starts palying saying maintain social distancing but it decreases the frame rate of the video.

# Research paper link
https://ijcrr.com/uploads/3806_pdf.pdf
