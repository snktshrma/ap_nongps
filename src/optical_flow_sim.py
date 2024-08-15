import cv2
import numpy as np

class OpticalFlowTracker:
    def __init__(self, lk_params=None, feature_params=None, resize_factor=0.5, altitude=10.0, altitude_ref=1.0):
        # Default parameters for Lucas-Kanade optical flow
        self.lk_params = lk_params if lk_params is not None else dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Default parameters for feature detection
        self.feature_params = feature_params if feature_params is not None else dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        self.resize_factor = resize_factor
        self.altitude = None
        self.altitude_ref = altitude_ref

        self.color = np.random.randint(0, 255, (100, 3))

        self.old_gray = None
        self.p0 = None
        self.mask = None
        self.x_pos = 0
        self.y_pos = 0

    def initialize(self, first_frame):
        self.old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
        self.mask = np.zeros_like(first_frame)

    def update(self, frame, alt):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.p0 is None:
            raise ValueError("Tracker not initialized.")

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

        if p1 is None or st is None or len(p1) == 0 or len(st) == 0:
            self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
            self.mask = np.zeros_like(frame)
            self.old_gray = frame_gray.copy()
            return frame, (self.x_pos, self.y_pos)

        good_new = p1[st == 1]
        good_old = self.p0[st == 1]

        if len(good_new) > 0 and len(good_old) > 0:
            flow_vectors = good_new - good_old
            avg_flow = np.mean(flow_vectors, axis=0)
            self.x_pos += (avg_flow[0] * alt / self.altitude_ref)
            self.y_pos += (avg_flow[1] * alt / self.altitude_ref)

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)

        img = cv2.add(frame, self.mask)
        cv2.putText(img, f"Position: X={self.x_pos:.2f}, Y={self.y_pos:.2f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        self.old_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)

        img_resized = cv2.resize(img, None, fx=self.resize_factor, fy=self.resize_factor)
        return img_resized, (self.x_pos, self.y_pos)

# tracker = OpticalFlowTracker()
# first_frame = cv2.imread('first_frame.png')
# tracker.initialize(first_frame)

# while capturing from video or other source:
#     frame = ...  # capture new frame
#     result_frame, (x_pos, y_pos) = tracker.update(frame)
#     cv2.imshow('Optical Flow', result_frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cv2.destroyAllWindows()
