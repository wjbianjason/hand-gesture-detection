import sys
import cv2
import numpy as np

#camera 0 is the integrated webcam on the notebook
camera_port = 0

#no of frames to throw up 
ramp_frames = 30

#initalizing the camera object with function VideoCapture which needs camera port as the parameter
camera = cv2.VideoCapture(camera_port)

#function to save the image
def saveImage(img):
	file = "./images/image.png"
	cv2.imwrite(file,img)

#function to detect hand motion
def detectHand(img):
	grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(grayImg,(5,5),0)
	ret , thresh = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	contours , hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#basic function to extract the largest contour (continous area having same color intensity)
	maxArea = 0
	maxci = 0 
	for i in range(len(contours)):
		cnt = contours[i]
		cntArea = cv2.contourArea(cnt)
		if cntArea>maxArea:
			maxArea = cntArea
			maxci = i
	#now consider the largest contour found out
	maxcnt = contours[maxci]
	#now draw the convex hull of the contour
	hull = cv2.convexHull(maxcnt)
	#displaying the largest contour with particular color on the image
	cv2.drawContours(img,[maxcnt],0,(0,255,0),2)
	cv2.drawContours(img,[hull],0,(0,0,255),2)
	cv2.imshow("Gesture Detection",img)

#function to show live webcam
def webcamLive():
	retval , img = camera.read()
	while retval:
		#reading image from the camera 
		# cv2.imshow("Gesture Detection", img)
		keyPress = cv2.waitKey(10)
		detectHand(img)
		if keyPress == 27:     #if ESC is pressed
			saveImage(img)
			break
		retval , img = camera.read()


webcamLive()
del(camera)