import cv2

playerCascade = cv2.CascadeClassifier('resources/Haarcascade/fullBody.xml')

img = cv2.imread('resources/images/match.png')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

players = playerCascade.detectMultiScale(img, 1.1, 5)

for (x,y,w,h) in players:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Match Snap', img)
cv2.waitKey(0)
