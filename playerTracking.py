import cv2
import numpy as np

matchVideo = cv2.VideoCapture("resources/videos/1.avi")

while True:
    success, img = matchVideo.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
    invertedBinary = ~binary
    contours, hierarchy = cv2.findContours(invertedBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower = np.array([43, 31, 4])
    higher = np.array([250, 88, 50])

    mask = cv2.inRange(img, lower, higher)
    blueTeam = cv2.bitwise_and(img, img, mask=mask)

    gray = cv2.cvtColor(blueTeam, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
    invertedBinary = ~binary
    blueContours, hierarchy = cv2.findContours(invertedBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Make sure contour area is large enough
        if (cv2.contourArea(c)) > 10:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

    for c in blueContours:
        x, y, w, h = cv2.boundingRect(c)
        # Make sure contour area is large enough
        if (cv2.contourArea(c)) > 10:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 720, 500)

    cv2.imshow("mask", mask)
    cv2.imshow("output", img)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break


# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)