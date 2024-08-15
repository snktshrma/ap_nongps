import cv2
import numpy as np

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parameters for feature detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

color = np.random.randint(0, 255, (100, 3))

cap = cv2.VideoCapture("a8-vid.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)

# Initialize tracking
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)

x_pos, y_pos = 0, 0
resize_factor = 0.5
altitude = 10.0
altitude_ref = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is None or st is None or len(p1) == 0 or len(st) == 0:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        mask = np.zeros_like(frame)
        old_gray = frame_gray.copy()
        cv2.putText(frame, f"Position: X={x_pos:.2f}, Y={y_pos:.2f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        frame_resized = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
        cv2.imshow('frame', frame_resized)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    if len(good_new) > 0 and len(good_old) > 0:
        flow_vectors = good_new - good_old
        avg_flow = np.mean(flow_vectors, axis=0)
        x_pos += (avg_flow[0] * altitude / altitude_ref)
        y_pos += (avg_flow[1] * altitude / altitude_ref)

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)
    cv2.putText(img, f"Position: X={x_pos:.2f}, Y={y_pos:.2f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    img_resized = cv2.resize(img, None, fx=resize_factor, fy=resize_factor)
    cv2.imshow('frame', img_resized)

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
