import cv2, sys, time
import numpy as np
from utils import MaskProcessing, Detector, Gcode
from User_Interface import Draw, Buttons, Sliders
# cap = cv2.VideoCapture(0)
video_path = 'whale_example.mov'
cap = cv2.VideoCapture(video_path)

cv2.namedWindow('image')
sliders = Sliders.Sliders('image')
detector = Detector.InitializeBlobDetector()


cv2.namedWindow('FinalMask', cv2.WINDOW_NORMAL)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('color', cv2.WINDOW_NORMAL)

ROI_X_START = 200
ROI_Y_START = 400
ROI_WIDTH = 400
ROI_HEIGHT = 400

# todo: make roi mask
while 1:
	ret, img = cap.read()
	cv2.waitKey(1)
	time.sleep(0.02)
	img = cv2.flip(img, 0)
	colorMask = MaskProcessing.GetColorMask(img, sliders)
	processedImg = MaskProcessing.ProcessImageMask(colorMask, img, sliders)

	greyscaleProcessedImg = cv2.cvtColor(processedImg, cv2.COLOR_RGB2GRAY)
	greyscaleProcessedImg = greyscaleProcessedImg[ROI_Y_START:ROI_Y_START + ROI_HEIGHT, ROI_X_START:ROI_X_START + ROI_WIDTH]
	# contours, _ = cv2.findContours(greyscaleProcessedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours, _ = cv2.findContours(greyscaleProcessedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if contours:
		contours[0][:, :] += np.array([ROI_X_START, ROI_Y_START])
		cv2.drawContours(processedImg, contours, -1, (0, 255, 0), 3)
		c = max(contours, key=cv2.contourArea)
		cv2.drawContours(processedImg, [c], -1, (0, 0, 255), 3)

		M = cv2.moments(c)
		if M["m00"] != 0:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			cv2.circle(processedImg, (cX, cY), 30, (255, 250, 0), -1)
		# keyPoints = detector.detect(processedImg)
		# processedImg = cv2.drawKeypoints(processedImg, keyPoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.rectangle(processedImg, (ROI_X_START, ROI_Y_START), (ROI_X_START + ROI_WIDTH, ROI_Y_START + ROI_HEIGHT), (255, 0, 0),
				  2)
	cv2.imshow('FinalMask', processedImg)
	cv2.imshow('image', img)
	cv2.imshow('color', colorMask)


cap.release()
cv2.destroyAllWindows()