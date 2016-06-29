//
//  DoubleCamera.cpp
//  OpencvProject2
//
//  Created by 李东奎 on 16/3/7.
//  Copyright © 2016年 李东奎. All rights reserved.
//

#include "opencv2/core/core.hpp"//因为在属性中已经配置了opencv等目录，所以把其当成了本地目录一样
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <vector>

using namespace cv;
using namespace std;



int main()
{
    double R1[3][3],R2[3][3],P1[3][4],P2[3][4],Q[4][4];
    //图片大小
    double w = 1280, h = 720;
    CvSize imageSize={static_cast<int>(w),static_cast<int>(h)};
    //相机内参
    double M1[3][3]={1292.92931480942, 0, 641.381089276017, 0, 1292.89944149206, 379.495799005823, 0,0, 1};
    double M2[3][3]={1292.61557697612, 0, 636.357162687524, 0, 1292.46682545845, 353.683555602483, 0, 0, 1};
    double D1[5]={0.0875866259601781, 0.157308159037881, -0.00123274225866768, 0.000506260108660759, -1.70035167273529};
    double D2[5]={0.0706097692836469, 0.281566807665954, 0.00145619235872172, 0.00182086280439303, -1.68025668969097};
    //相机外参数
    double R[3][3]={0.999934801460936 ,-0.00152098373675532,    0.0113172185518997,  0.00144572046375141,   0.999976806538965,   0.00665554479164107, -0.0113270790418195,   -0.00663874932539074 ,  0.999913808429394};
    double T[3]={-62.3015780643189, 0.0212762329479081, -1.03254372557284};
    //参数变为矩阵形式
    CvMat _M1=cvMat(3,3,CV_64F,M1);
    CvMat _M2=cvMat(3,3,CV_64F,M2);
    CvMat _D1=cvMat(1,5,CV_64F,D1);
    CvMat _D2=cvMat(1,5,CV_64F,D2);
    CvMat _R=cvMat(3,3,CV_64F,R);
    CvMat _T=cvMat(3,1,CV_64F,T);
    CvMat _R1=cvMat(3,3,CV_64F,R1);
    CvMat _R2=cvMat(3,3,CV_64F,R2);
    CvMat _P1=cvMat(3,4,CV_64F,P1);
    CvMat _P2=cvMat(3,4,CV_64F,P2);
    CvMat _Q =cvMat(4,4,CV_64F,Q);
    //得到立体相机相对位置关系R1，R2及三维-二维投影矩阵P1，P2
    
    cvStereoRectify(&_M1,&_M2,&_D1,&_D2,imageSize,&_R,&_T,&_R1,&_R2,&_P1,&_P2,&_Q,0);
    
    CvMat *mx1=cvCreateMat(imageSize.height,imageSize.width,CV_32F);
    CvMat *my1=cvCreateMat(imageSize.height,imageSize.width,CV_32F);
    CvMat *mx2=cvCreateMat(imageSize.height,imageSize.width,CV_32F);
    CvMat *my2=cvCreateMat(imageSize.height,imageSize.width,CV_32F);
    //获取镜头畸变矫正的映射关系参数mx，my
    cvInitUndistortRectifyMap(&_M1,&_D1,&_R1,&_P1,mx1,my1);
    cvInitUndistortRectifyMap(&_M2,&_D2,&_R2,&_P2,mx2,my2);
     
    //获取摄像头端口
    CvCapture* capture1 = cvCreateCameraCapture( 0 );
    CvCapture* capture2 = cvCreateCameraCapture( 1 );
    
    
    cvSetCaptureProperty ( capture1, CV_CAP_PROP_FRAME_WIDTH,  w );
    cvSetCaptureProperty ( capture1, CV_CAP_PROP_FRAME_HEIGHT, h );
    cvSetCaptureProperty ( capture2, CV_CAP_PROP_FRAME_WIDTH,  w );
    cvSetCaptureProperty ( capture2, CV_CAP_PROP_FRAME_HEIGHT, h );
    
    IplImage *bgrFrameL, *bgrFrameR;//矫正前的图片
    IplImage *img1r=cvCreateImage(imageSize,8,3);//矫正后图片初始化
    IplImage *img2r=cvCreateImage(imageSize,8,3);
    
    
    while(1)
    {
        bgrFrameL = cvQueryFrame( capture1 );
        bgrFrameR = cvQueryFrame( capture2 );
        
        cvRemap(bgrFrameL,img1r,mx1,my1);//根据上面求得的映射关系mx，my矫正图片
        cvRemap(bgrFrameR,img2r,mx2,my2);
        
        //显示矫正后的图片
        cvNamedWindow("1",CV_WINDOW_AUTOSIZE);
        cvShowImage("1",img1r);
       
        cvNamedWindow("2",CV_WINDOW_AUTOSIZE); 
        cvShowImage("2",img2r);  
        
        cvWaitKey(30);
    }
    
    return 0;
} 



