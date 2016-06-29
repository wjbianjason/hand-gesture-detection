#-*- coding:utf-8 -*-
"""[hand_track]
[hand gesture track,tracking the center of hand and point of finger]
"""
import cv2
import numpy as np
import math
import pickle
def thresholding(image, min, max):
	val, mask = cv2.threshold(image, max, 255, cv2.THRESH_BINARY)
	val, mask_inv = cv2.threshold(image, min, 255, cv2.THRESH_BINARY_INV)
	return cv2.add(mask, mask_inv)

def YCrCbSplit(image):
	im_YCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
	channels = cv2.split(im_YCrCb)
	# y_img = self.thresholding(channels[0], 59, 112)
	cr_img = thresholding(channels[1], 133, 175)  #有建议140-175
	cb_img = thresholding(channels[2], 77, 127)   #有建议100-120  还可已考虑用inrange
	skin = cv2.add(cr_img,cb_img)
	return cv2.bitwise_not(skin)

cap = cv2.VideoCapture(0)
# cv2.namedWindow("orige",cv2.WINDOW_NORMAL)
# cv2.namedWindow("filterImg",cv2.WINDOW_NORMAL)
# cv2.namedWindow("blurImg",cv2.WINDOW_NORMAL)
cv2.namedWindow("handImg",cv2.WINDOW_NORMAL)
cv2.namedWindow("color",cv2.WINDOW_NORMAL)
file2 = open("img.pkl",'rb')
cn1 = pickle.load(file2)
while True:
	for i in range (10):
	    ret, img = cap.read()
	    if ret:
	    	# print "get"
	        break
	else:
		print "error"
	# img = cv2.medianBlur(img,5)
	crop_img = img.copy()
	# cv2.imshow('orige',img)
	img = cv2.GaussianBlur(img,(35,35),0)
	MIN = np.array([0,30,30],np.uint8)
	MAX = np.array([40,170,255],np.uint8) #HSV: V-79%
	MIN_2 = np.array([150,30,30],np.uint8)
	MAX_2 = np.array([180,170,255],np.uint8) #HSV: V-79%
	

	# HSVImg = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	# cv2.COLOR_BGR2YC
	# filterImg = cv2.inRange(HSVImg,MIN,MAX)
	# filterImg_2 = cv2.inRange(HSVImg,MIN_2,MAX_2)
	# filterImg = YCrCbSplit(img)
	# filterImg = cv2.add(filterImg,filterImg_2)
	# 
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	_, filterImg = cv2.threshold(gray, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	filterImg = cv2.bitwise_not(filterImg)
	# res = cv2.bitwise_and(img,img,mask = filterImg)
	# cv2.imshow("filterImg",filterImg)
	cv2.imshow("color",filterImg)
	# cv2.imshow("")

	#add gray to delete 
	# res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
	# _, thresh1 = cv2.threshold(res, 127, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


	blurImg=cv2.blur(filterImg,(5,5))
	# cv2.imshow("blurImg",blurImg)

	contours, hierarchy = cv2.findContours(blurImg.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	if len(contours) == 0:
		continue	
	max_area = -1
	ci = 0
	for i in range(len(contours)):
	    cnt=contours[i]
	    area = cv2.contourArea(cnt)
	    if(area>max_area):
	        max_area=area
	        ci=i
	cnt=contours[ci]
	cnt_out = cnt.copy()




	# exit(1)


	# print ci
	cnt_oringe =cnt.copy()
	cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
	hull = cv2.convexHull(cnt)
	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)

	drawing = np.zeros(crop_img.shape,np.uint8)
	cv2.drawContours(drawing,[cnt_oringe],0,(0,255,0),0)
	moments = cv2.moments(cnt)
	if moments['m00']!=0:
		cx = int(moments['m10']/moments['m00']) # cx = M10/M00
		cy = int(moments['m01']/moments['m00']) # cy = M01/M00
	centr=(cx,cy)
	cv2.circle(drawing,centr,5,[0,0,255],-1)  
	hull = cv2.convexHull(cnt,returnPoints = False)
	defects = cv2.convexityDefects(cnt,hull)
	if defects == None:
		continue
	print defects.shape[0]
	max_angle = 0
	end_max = 0
	for i in range(defects.shape[0]):
		s,e,f,d = defects[i,0]
		start = tuple(cnt[s][0])
		end = tuple(cnt[e][0])
		far = tuple(cnt[f][0])
		a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
		b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
		c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
		angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
		# print dist,d
		if angle <= 90:
			if angle > max_angle:
				max_angle = angle
				end_max = e
			cv2.circle(drawing,start,5,[0,0,255],2)
		# if i == defects.shape[0]-1:
			# cv2.circle(drawing,end,5,[255,0,0],2)
	cv2.circle(drawing,tuple(cnt[end_max][0]),5,[0,0,255],2)
	cv2.imshow("handImg",drawing)



	k = cv2.waitKey(10)
	if k == 27:
		file1 = open("img.pkl",'wb')
		pickle.dump(cnt_out,file1)
		file1.close()
		break




