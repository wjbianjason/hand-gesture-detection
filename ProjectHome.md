Vision-based interfaces (VBI) are gaining much interest recently, maybe best illustrated by the commercial success of Sony's Eye Toy, an accessory for the company's PlayStation 2: a set-top camera recognizes full-body motions and projects the player directly into the game. However, more fine-grained control such as with hand gesture recognition, has not yet reached the same level of robustness and reliability. Outdoor and mobile environments in particular present additional difficulties due to camera motion and their variability in backgrounds and lighting conditions. In prior work, we presented a mobile VBI that allows control of wearable computer entirely with hand gesture commands. A collection of recently proposed and novel methods enables hand detection, tracking, and posture recognition for truly interactive interfaces, realized with a head-worn camera and display. For these vision-based hand gesture interfaces it is of tremendous importance to make available a background-invariant, lighting-insensitive, and person- and camera-independent classifier to reliably detect a human's most important manipulative tool, the hand. We will subsequently call these classifiers “detectors”.


Hand detection is a computer technology that determines the locations and sizes of human hands in arbitrary (digital) images. It detects hand features and ignores anything else, such as buildings, trees and bodies.
The objective of multiple hand detection is to detect all hand images regardless of the number of objects in the image frames from a live video stream input. We will implement this concept by utilizing OpenCV.  We must train the cascade classifiers images with and without applying distortions, which are needed to detect whether an object is a hand in the captured image even with alterations, by means of the Haartraining utility. The application should have a method to enclose the hand images within rectangles.
> Threshold values are calculated for different features of image of hands from the samples created. These threshold values are used to compare and hence, detect the hand in each frame of a video input. The detected hands are highlighted in rectangles on the output window.

Here we  are training the system to detect a fist gesture of Hand

And
here we thank the site describing openCV - OPENCV WIKI wich provided the basic code which we used and modified to obtain the source code required for the project


done by
MOHAFIZ RAZ.M.A
FEBIN JOSE
C.SANJAY ARVIND