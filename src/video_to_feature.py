import threading
import time
import video_capture_gazebo
import feature_match_ardu
import traceback
import signal
import sys
import cv2 as cv
import numpy as np
from pymavlink import mavutil

x_prev = 0
y_prev = 0

baseImg = "satellite_image-main.png"

the_connection = mavutil.mavlink_connection('tcp:127.0.0.1:5762')
the_connection.wait_heartbeat()
print("Heartbeat from system (system %u component %u)" %
      (the_connection.target_system, the_connection.target_component))

class main:
	def __init__(self):
		self.lat = 0
		self.lon = 0
		self.alt = 0
		self.alt_ab_ter = 0
		self.video = video_capture_gazebo.Cam_stream()       # Get video frame
		self.feature = feature_match_ardu.Image_Process()
		# t1 = threading.Thread(target=self.video.setup)
		# t1.start()
		self.img1 = []
		self.sift = None
		self.kp = None
		self.des = None
		self.sr_pt = []
		self.frame = []

	def setPos(self,x=0,y=0,z=0):
		the_connection.mav.send(mavutil.mavlink.MAVLink_vision_position_estimate_message(
			10,
			x, y, 0,
			0, 0, 0,
			[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,1], 0
		))

	def sift_base(self):
		self.img1 = cv.imread(baseImg,1)
		self.img1 = cv.cvtColor(self.img1, cv.COLOR_BGR2GRAY) 
		# self.img1 = cv.resize(self.img1, (600,600))
		self.sift = cv.SIFT_create(nfeatures = 20000)
		self.kp, self.des = self.sift.detectAndCompute(self.img1,None)
	def getComp(self):
		global x_prev, y_prev
		while True:
			# print(len(self.frame))
			# self.feature.compParam("satellite_image.png", self.frame)
			if not self.video.frame_available():
				continue
			self.frame = self.video.frame
			try:
				[self.frame, x, y] = self.feature.compParam("satellite_image-main.png", self.frame,(self.kp,self.des))
			except:
				self.setPos(0,0)
			if type(self.frame) != type(None):
				self.setPos(x,y)
				cv.imshow('frame', self.frame)

				if cv.waitKey(1) & 0xFF == ord('q'):
					break
			else:
				self.setPos(x,y)

	def vid_sleep(self):
		time.sleep(0.3)




m = main()
m.sift_base()
m.getComp()