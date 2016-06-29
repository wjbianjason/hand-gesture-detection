#!/usr/bin/env python
#-*- coding:utf-8-*-
"""
Created on 2016-3-11
@version:1
@author: wjbian
gesture detect and track
"""
import cv2
import numpy as np
import math
class HandDetect(object):
	def __init__(self,windowList=["oringe","result"]):
		self.cap = cv2.VideoCapture(0)
		self.fgbg = cv2.BackgroundSubtractorMOG(history = 10,  nmixtures = 5, backgroundRatio= 0.8)
		self.im_out = None
		self.handCenter = None
		self.hull = None
		self.fingerTip = []

		# self.th_CR_min = 142 #133
		# self.th_CR_max = 173 #173
		# self.th_CB_min = 93  #77
		# self.th_CB_max = 119 #127
		self.th_CR_min = 133
		self.th_CR_max = 173
		self.th_CB_min = 77
		self.th_CB_max = 127
		self.count_flag = 0
		cv2.namedWindow('Control Panel')
		self.adjustPannel()

		for windowName in windowList:
			cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
	
	def processing(self,img):
		# blurResult = cv2.blur(img,(5,5))
		blurResult = cv2.GaussianBlur(img,(35,35),0)
		# blurHighResult = self.adjustGamma(blurResult,gamma=1.5) # 补光
		# return cv2.add(blurResult,blurHighResult)
		return blurResult
	
	def adjustPannel(self):
		cv2.createTrackbar('CR_min', 'Control Panel', self.th_CR_min, 255, self.onChange_th_CR_min)
		cv2.createTrackbar('CR_max', 'Control Panel', self.th_CR_max, 255, self.onChange_th_CR_max)
		cv2.createTrackbar('CB_min', 'Control Panel', self.th_CB_min, 255, self.onChange_th_CB_min)
		cv2.createTrackbar('CB_max', 'Control Panel', self.th_CB_max, 255, self.onChange_th_CB_max)		# pass

	def onChange_th_CR_min(self, value):
		self.th_CR_min = value

	def onChange_th_CR_max(self, value):
		self.th_CR_max = value

	def onChange_th_CB_min(self, value):
		self.th_CB_min = value

	def onChange_th_CB_max(self, value):
		self.th_CB_max = value

	def adjustGamma(self,image,gamma=1.0):
		invGamma = 1.0 / gamma
		table = np.array([((i/255.0)**invGamma)*255 for i in np.arange(0,256)]).astype("uint8")
		return cv2.LUT(image,table)

	def HSVSplit(self,image):
		MIN = np.array([0,30,30],np.uint8)
		MAX = np.array([30,170,200],np.uint8)
		HSVImg = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
		filterImg = cv2.inRange(HSVImg,MIN,MAX)
		# filterImg_spe = cv2.bilateralFilter(filterImg, 11, 17, 17) #没有大的改变所以去掉，减少计算量
		return filterImg

	def ContoursFilter(self,image):
		contours, hierarchy = cv2.findContours(image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
		cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
		# if len(cnts) == 0:
		# 	return False
		return cnts

	def BackgroundSub(self,image):
		return self.fgbg.apply(image,learningRate=0.01)

	def thresholding(self, image, min, max):
		val, mask = cv2.threshold(image, max, 255, cv2.THRESH_BINARY)
		val, mask_inv = cv2.threshold(image, min, 255, cv2.THRESH_BINARY_INV)
		return cv2.add(mask, mask_inv)

	def YCrCbSplit(self,image):
		im_YCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
		channels = cv2.split(im_YCrCb)
		# y_img = self.thresholding(channels[0], 59, 112)
		cr_img = self.thresholding(channels[1], self.th_CR_min, self.th_CR_max)  #有建议140-175
		cb_img = self.thresholding(channels[2], self.th_CB_min, self.th_CB_max)   #有建议100-120  还可已考虑用inrange
		skin = cv2.add(cr_img,cb_img)
		return cv2.bitwise_not(skin)

	def Ycc_2(self,image):
		im_YCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
		MIN = np.array([0,self.th_CR_min,self.th_CB_min],np.uint8)
		MAX = np.array([255,self.th_CR_max,self.th_CB_max],np.uint8)
		filterImg = cv2.inRange(im_YCrCb,MIN,MAX)
		return filterImg



	def drawHandCenter(self,contours):
		self.hull = cv2.convexHull(contours)
		moments = cv2.moments(contours)
		centroid_x = int(moments['m10']/moments['m00'])
		centroid_y = int(moments['m01']/moments['m00'])
		realCenter = (centroid_x, centroid_y)
		# cnt_oringe =cnt.copy()
		cnt = cv2.approxPolyDP(contours.copy(),0.001*cv2.arcLength(contours,True),True)
		# cv2.circle(self.im_out, realCenter, 5, (0,0,255), -1)
		cv2.drawContours(self.im_out,[cnt],0,(255,0,0),2)
		# cv2.drawContours(self.im_out,[hull],0,(255,0,0),2)
		self.handCenter = realCenter
		return cnt

	def distance(self, p0, p1):
	    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


	def drawFingers(self,cnt):
		hullNew = cv2.convexHull(cnt,returnPoints = False)
		defects = cv2.convexityDefects(cnt,hullNew)
		if len(defects) == 0:
			return False
		max_angle = 0
		end_max = 0
		farList = []
		self.fingerTip= []
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
			# print "angle:",d
			if angle <= 90 and self.distance(far,start) > 100 and d >20000:
				# print "true",d
				farList.append(far)
				if angle > max_angle:
					max_angle = angle
					end_max = e
				# cv2.circle(self.im_out,start,10,[0,0,255],2)
				self.fingerTip.append(start)
				self.fingerTip.append(end)
		# self.fingerTip.append(tuple(cnt[end_max][0]))
		# cv2.circle(self.im_out,tuple(cnt[end_max][0]),10,[0,0,255],2)
		fingerTip = sorted(self.fingerTip, key = self.momentDist, reverse = True)[:8]

		for i in range(len(fingerTip)):
			if i != 0:
				flag = False
				for j in range(i):
					print self.distance(fingerTip[i],fingerTip[j])
					if self.distance(fingerTip[i],fingerTip[j]) < 70:
						flag = True
						break
				if flag:
					continue
			cv2.circle(self.im_out,fingerTip[i],10,[0,0,255],2)
		print "len:"+str(len(farList))
		if len(farList) >= 3:
			self.drawcenter(farList)

	def momentDist(self,point):
		return self.distance(self.handCenter,point)

	def drawcenter(self,farList):
		a = 2*(farList[1][0]-farList[0][0])
		b = 2*(farList[1][1]-farList[0][1])
		c = farList[1][0]**2 + farList[1][1]**2 -(farList[0][0]**2 + farList[0][1]**2)
		d = 2*(farList[2][0]-farList[1][0])
		e = 2*(farList[2][1]-farList[1][1])
		f = farList[2][0]**2 + farList[2][1]**2 -(farList[1][0]**2 + farList[1][1]**2)
		x = (b*f-e*c)/(b*d-e*a)
		y = (d*c-a*f)/(b*d-e*a)
		cv2.circle(self.im_out, (x,y), 10, (0,255,0), 2)


	# def calibrate


	def track_init(self,image):
		# termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
		# roiBox = None
		x,y = self.handCenter
		track_window = (y,x,10,10)
		# roi = image[x:x+10,y:y+10]
		hsv_roi =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
		roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
		cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
		term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
		self.track_init_flag = True
		return track_window,roi_hist,term_crit


	# def drawcenter(self,farList):
	# 	pass

	def findfinger(self):
		pass



	def detect(self):
		FrontImg = None
		while True:
			for i in range(10):
				ret,img = self.cap.read()
				if ret:
					break
				else:
					pass	
			self.im_out = img
			self.count_flag += 1
			# if not self.track_flag :
				# if count < 3:
				# 	if count == 0:
				# 		img_ori = img
				# 	else:
				# 		img_ori += img
				# 		count +=1
				# 		continue
				# img_ori = img_ori/2
				# count = 0
				# img = img_ori
					
					
			img_pro= self.processing(img)
			fgmask = self.BackgroundSub(img_pro)
			# test = cv2.bitwise_not(fgmask)
			# YCCimg = self.YCrCbSplit(img_pro)
			# YCCimg = self.processing(YCCimg)
			YCC2 = self.Ycc_2(img_pro)
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
			for i in xrange(2):
				YCC2 = cv2.erode(YCC2,kernel)
				# YCC2 = cv2.dilate(YCC2,kernel)
			for i in xrange(5):
				YCC2 = cv2.morphologyEx(YCC2, cv2.MORPH_CLOSE, kernel)
			# blurRe = cv2.medianBlur(YCC2,5)
			FrontImg = YCC2
			if self.count_flag != 1:
				YCC2 = cv2.add(FrontImg,YCC2)
			contours = self.ContoursFilter(YCC2)
			if len(contours) == 0:
				print "not found"
				continue
			# self.findfinger(coutours[0])

			newCnt = self.drawHandCenter(contours[0])
			self.drawFingers(newCnt)
			# self.drawHandCenter(contours)
			# self.drawFingers(contours)
			cv2.imshow('result2',fgmask)
			cv2.imshow('result',YCC2)
			# cv2.imshow('blur',blurRe)
			cv2.imshow('oringe',self.im_out)
			k = cv2.waitKey(10)
			if k == 27:
				break



if __name__ == '__main__':
	windowNames = ["oringe","result","result2","result3","img2"]
	handdetect = HandDetect(windowNames)
	handdetect.detect()




	"""
	cap = cv2.VideoCapture(0)
	cv2.namedWindow("oringe",cv2.WINDOW_NORMAL)
	while True:
		# for i in range(10):
		# 	ret,img = cap.read()
		# 	if ret:
		# 		break
		# 	else:
		# 		pass	
	# 	mask = np.zeros(img.shape[:2],np.uint8)
	# 	bgdModel = np.zeros((1,65),np.float64)
	# 	fgdModel = np.zeros((1,65),np.float64)
	# 	rect = (50,50,450,290)
		img = np.zeros((200,200))
		cout = np.array([[50,50],[50,150],[150,150],[150,50]])
		moment = cv2.moments(cout)
		print moment
		# cout  = cout.reshape((-1,1,2))
		# print cout
		# cv2.drawContours(img,[cout],0,(255,0,0),2)
		cv2.polylines(img,[cout],True,(255,255,255))
		fillpoly(img,[cout],(255,255,255))

	# 	cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
	# 	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	# 	img = img*mask2[:,:,np.newaxis]
		cv2.imshow("oringe",img)
	# 	# plt.imshow(img),plt.colorbar(),plt.show()
	# 	# break

		k = cv2.waitKey(10)
		if k == 27:
			break


	"""


