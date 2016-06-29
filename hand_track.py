#-*- coding:utf-8 -*-
"""[hand_track]
[hand gesture track,tracking the center of hand and point of finger]
"""
import cv2

cap = cv2.VideoCapture(0)
ret,img = cap.read()
median = cv2.medianBlur(img,5)
gaussian = cv2.GaussianBlur(median,(5,5),0)

gray = cv2.cvtColor(gaussian,cv2.COLOR_RGB2GRAY)
ret,thresh1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

