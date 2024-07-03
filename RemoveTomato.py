import cv2, sys, time
import numpy as np
from utils import MaskProcessing, Detector, Gcode
from User_Interface import Draw, Buttons, Sliders
# cap = cv2.VideoCapture(0)
video_path = 'whale_example.MOV'
cap = cv2.VideoCapture(video_path)

cv2.namedWindow('image')
sliders = Sliders.Sliders('image')
detector = Detector.InitializeBlobDetector()


while 1:
	ret, img = cap.read()
	cv2.waitKey(1)
	# time.sleep(0.05)
	img = cv2.flip(img, 0)
	colorMask = MaskProcessing.GetColorMask(img, sliders)
	processedImg = MaskProcessing.ProcessImageMask(colorMask, img, sliders)

	keyPoints = detector.detect(processedImg)
	img = cv2.drawKeypoints(img, keyPoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	cv2.imshow('FinalMask', processedImg)
	cv2.imshow('image', img)
	cv2.imshow('color', colorMask)


cap.release()
cv2.destroyAllWindows()