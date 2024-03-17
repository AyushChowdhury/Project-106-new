import cv2
vid = cv2.VideoCapture('walking.avi')

bodyClassifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')


# Loop once video is successfully loaded
while True:
    ret, frame = vid.read()

    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies = bodyClassifier.detectMultiScale(gray)
    print(bodies)

    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Video",frame)

    if cv2.waitKey(1) == 32: 
        break

vid.release()
cv2.destroyAllWindows()
