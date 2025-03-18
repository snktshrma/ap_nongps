import threading
import time
import video_capture_gazebo
import feature_match_ardu as feature_match_ardu
import feature_match_standalone as feature_match_standalone
import traceback
import signal
import sys
import cv2 as cv
import numpy as np
from pymavlink import mavutil

x_prev = 0
y_prev = 0

to_flag = False

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
		self.checkStart = feature_match_standalone.Image_Process()
		# t1 = threading.Thread(target=self.video.setup)
		# t1.start()
		self.img1 = []
		self.sift = None
		self.kp = None
		self.des = None
		self.sr_pt = []
		self.frame = []

		self.start_pos = ()

		self.curr_alt = 0.0

	def setPos(self,x=0,y=0,z=0):
		the_connection.mav.send(mavutil.mavlink.MAVLink_vision_position_estimate_message(
			10,
			x, y, 0,
			0, 0, 0,
			[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,1], 0
		))



	def check_takeoff_complete(self, target_altitude, tolerance=0.5):
		msg = the_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
		if msg:
			current_altitude = msg.relative_alt / 1000.0  # Convert altitude from millimeters to meters
			print(f"Current Altitude: {current_altitude:.2f} m, Target Altitude: {target_altitude:.2f} m")
			if abs(current_altitude - target_altitude) <= tolerance:
				print("Takeoff complete and target altitude reached.")
				return True
		time.sleep(0.1)
		return False

	def get_alt(self):
		msg = the_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
		if msg:
			current_altitude = msg.relative_alt / 1000.0  # Convert altitude from millimeters to meters
			self.curr_alt = current_altitude
			return current_altitude
		return self.curr_alt




	def sift_base(self):
		self.img1 = cv.imread(baseImg,1)
		self.img1 = cv.cvtColor(self.img1, cv.COLOR_BGR2GRAY) 
		# self.img1 = cv.resize(self.img1, (600,600))
		self.sift = cv.SIFT_create(nfeatures = 20000)
		self.kp, self.des = self.sift.detectAndCompute(self.img1,None)
	def getComp(self):
		global x_prev, y_prev

		while not self.check_takeoff_complete(10):
			if not self.video.frame_available():
				continue
			self.frame = self.video.frame
			self.start_pos = self.checkStart.compParam("satellite_image-main.png", self.frame)

		while True:
			# print(len(self.frame))
			# self.feature.compParam("satellite_image.png", self.frame)
			if not self.video.frame_available():
				continue
			self.frame = self.video.frame
			if not to_flag:
				[bb,x,y] = self.feature.compParam("satellite_image-main.png", self.frame,(self.kp,self.des), None, self.get_alt())
			elif to_flag:
				[bb,x,y] = self.feature.compParam("satellite_image-main.png", self.frame,(self.kp,self.des), self.start_pos, self.get_alt())

			x_prev = x
			y_prev = y

			# print(x, y)
			
			if type(bb) != type(None):
			 	self.setPos(x,y)
			 	# cv.imshow('frame', bb)

			 	# if cv.waitKey(1) & 0xFF == ord('q'):
			 	# 	break
			else:
			 	self.setPos(x_prev,y_prev)

	def vid_sleep(self):
		time.sleep(0.3)




m = main()
the_connection.mav.command_long_send(
	the_connection.target_system, the_connection.target_component,
      mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
      32,
      1e6 / 10,
      0, 0, 0, 0,
      0,
)

the_connection.mav.command_long_send(
	the_connection.target_system, the_connection.target_component,
      mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
      33,
      1e6 / 10,
      0, 0, 0, 0,
      0,
)

m.sift_base()
m.getComp()
