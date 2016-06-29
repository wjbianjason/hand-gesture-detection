
#           Incorrect Code

import numpy as np
import cv2

print "Welcome\n"


print "Done Calibration\n"
print "Starting Rectification\n"

width = 320
height = 240

capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(0)

capL.set(3,width)
capL.set(4,height)

capR.set(3,width)
capR.set(4,height)


M1 = np.array([[ 2.0340227345156890e+002 , 0, 1.5950000000000000e+002 ],[ 0, 2.0340227345156890e+002 , 1.1950000000000000e+002 ],[ 0,0, 1 ]])
M2 = np.array([[2.0340227345156890e+002 , 0, 1.5950000000000000e+002],[0, 2.0340227345156890e+002 , 1.1950000000000000e+002],[0, 0, 1 ]])
D1 = np.array([-4.1344532351734609e-001 , 1.4783234656467026e-001, 4.4055963467856136e-003 , 6.1661017124724649e-003, 0])
D2 = np.array([-4.5592829264409196e-001 , 2.2408910954289543e-001, 1.0425612281350783e-003 , -5.5920270905273290e-003, 0])
R = np.array([[9.9880267535072764e-001 ,1.0478131925395094e-003, -4.8909281325148393e-002],[-1.9549817734339412e-003, 9.9982687809750082e-001, -1.8503834199179697e-002],[4.8881425495729047e-002 ,  1.8577295855929390e-002 , 9.9863180918704308e-001]])
T = np.array([-6.0430960451649426e+001 , 9.5357972721080564e-001, -3.4515100808301149e+000])

R1 = np.zeros(shape=(3,3))
R2 = np.zeros(shape=(3,3))
P1 = np.zeros(shape=(3,4))
P2 = np.zeros(shape=(3,4))
Q = np.zeros(shape=(4,4))

aa,bb,cc,dd,ee,roi1,roi2 = cv2.stereoRectify(M1, D1, M2, D2,(width, height), R, T, R1, R2, P1, P2, Q, flags=cv2.cv.CV_CALIB_ZERO_DISPARITY, alpha=-1, newImageSize=(0,0))
print roi1

# print maskl.max()
maskl = np.zeros((height,width),dtype=np.uint8)

maskr = np.zeros((height,width),dtype=np.uint8)

cv2.rectangle(maskl,(roi1[0],roi1[1]),(roi1[2]+roi1[0],roi1[3]+roi1[1]),(255,255,255),-1)
cv2.rectangle(maskr,(roi2[0],roi2[1]),(roi2[2]+roi2[0],roi2[3]+roi2[1]),(255,255,255),-1)


map1x, map1y = cv2.initUndistortRectifyMap(M1, D1, R1, P1, (width, height), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(M2, D2, R2, P2, (width, height), cv2.CV_32FC1)

print "Undistort complete\n"

while(True):

    for i in range(10):
        flagL, img1 = capL.read()
        flagR, img2 = capR.read()
        if flagL and flagR:
            break
        else:
            pass


    imgU1 = np.zeros((height,width,3), np.uint8)
    imgU2 = np.zeros((height,width,3), np.uint8)
    imgU1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR) #, cv2.BORDER_CONSTANT, 0)
    imgU2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR) #, cv2.BORDER_CONSTANT, 0)
    # print imgU1.shape
    new1 = cv2.bitwise_and(imgU1,imgU1,mask=maskl)
    new2 = cv2.bitwise_and(imgU2,imgU2,mask=maskr)

    # cv2.imshow("imageL", img1)
    # cv2.imshow("imageR", img2)
    # cv2.imshow("image1L", new1)
    # cv2.imshow("image2R", new2)





    window_size = 3 #size of blocks for matching
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM(
        minDisparity = min_disp, 
        numDisparities = num_disp, 
        SADWindowSize = window_size,
        uniquenessRatio = 25,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 1,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        fullDP = False
    )



    new11 = cv2.cvtColor(new1,cv2.COLOR_BGR2GRAY)
    new21 = cv2.cvtColor(new2,cv2.COLOR_BGR2GRAY)

    # print 'computing disparity...'
    disp = stereo.compute(new1, new2).astype(np.float32) / 16.0

    disp2 =stereo.compute(new11, new21).astype(np.float32) / 16.0
    # print 'generating 3d point cloud...'

    points = cv2.reprojectImageTo3D(disp2, Q)

    # display disparity map
    dispUs = cv2.pyrUp( ((disp-min_disp)/num_disp) )
    
    dispUs2 = cv2.pyrUp( ((disp2-min_disp)/num_disp) )
    cv2.imshow('disparity', dispUs) # noise much more but more precise
    # 
    cv2.imshow('disparity2', dispUs2) # otherwise

    ch = cv2.waitKey(5)
    if ch == 27:
        break

