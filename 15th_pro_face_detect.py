import cv2 as cv

img = cv.imread('imgs/more_faces.png') # Add an image in imread for detecting how many faces are there in an image.

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


haar_cascade = cv.CascadeClassifier('haar_faces.xml')
faces_rectangle = haar_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=6)

print(f"Number of Faces IN the Image Found = {len(faces_rectangle)}")



for (x, y,w,h) in faces_rectangle:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0),thickness=2)


cv.imshow("FAce DEtcted", img)

cv.waitKey(0)

