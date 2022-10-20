import cv2
import numpy as np
print("imported successfully")

# reading an image
# img = cv2.imread('resources/image1.jpeg')
# cv2.imshow("output", img)
# cv2.waitKey(0)

# capturing a video
# cap = cv2.VideoCapture("resources/videos/sample.mp4")

# while True:
#     success, img = cap.read()
#     cv2.imshow("video", img)
#     if cv2.waitKey(1) & 0xFF==ord('q'):
#         break

# using web camera
# camera = cv2.VideoCapture(0)
# camera.set(3, 640)
# camera.set(4, 480)
# camera.set(10, 10)

# while True:
#     success, img = camera.read()
#     cv2.imshow("stream", img)
#     if cv2.waitKey(1) & 0xFF==ord('q'):
#         break

# img = cv2.imread("resources/images/image2.jpeg")
#
# imgResized = cv2.resize(img, (1080, 640))
# imgCropped = img[0:200, 200:500]

# cv2.imshow("Image", imgResized)
# cv2.imshow("Cropped Image", imgCropped)
# cv2.waitKey(0)


# img = np.zeros((512,512,3))
# print(img.shape)
# cv2.line(img, (0, 0), (img.shape[0], img.shape[1]), 5)
# cv2.rectangle(img, (10, 10), (200, 200), (0, 255, 0), 3)
# cv2.circle(img, (200, 100), 50, (255,0,255), 2)

# cv2.imshow("numpy image", img)
# cv2.waitKey(0)

# img = cv2.imread("resources/images/image1.jpeg")
# imgCropped = img[329:463, 482:638]
#
# cv2.imshow("image", img)
# cv2.imshow("image2", imgCropped)
# cv2.waitKey(0)

# color detection
img = cv2.imread("resources/images/match.png")
cv2.imshow("Original", img)

lower = np.array([43, 31, 4])
higher = np.array([250, 88, 50])

mask = cv2.inRange(img, lower, higher)
blueTeam = cv2.bitwise_and(img, img, mask=mask)

gray = cv2.cvtColor(blueTeam, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
invertedBinary = ~binary
blueContours, hierarchy = cv2.findContours(invertedBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in blueContours:
    x, y, w, h = cv2.boundingRect(c)

    # Make sure contour area is large enough
    if (cv2.contourArea(c)) > 5:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow("blue", img)
cv2.waitKey(0)

