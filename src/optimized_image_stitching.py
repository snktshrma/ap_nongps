import cv2
import numpy as np
import os
import glob

# Function to read an image from a file and resize if it's too large for processing.
def read_and_resize(image_path, resize_max_dim=None):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if resize_max_dim and (image.shape[0] > resize_max_dim or image.shape[1] > resize_max_dim):
        ratio = resize_max_dim / max(image.shape[:2])
        image = cv2.resize(image, (int(ratio * image.shape[1]), int(ratio * image.shape[0])))
    return image

# Function to detect and extract features from an image
def detect_and_describe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

# Function to match descriptors between two images
def match_keypoints(desc1, desc2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    rawMatches = bf.knnMatch(desc1, desc2, k=2)
    matches = [m for m, n in rawMatches if m.distance < ratio * n.distance]
    return matches

# Function to find the homography between matched points and warp one image onto the other
def stitch_images(img1, img2, kp1, kp2, matches):
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    H, status = cv2.findHomography(points1, points2, cv2.RANSAC)
    result = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img2.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    return result

# Read all the images
resize_max_dim = 1000  # Resize for faster processing
image_folder_path = '/home/sanket/Downloads/highaltitude-nongps-testpics-30Mar2024/tests'
image_files = sorted(glob.glob(os.path.join(image_folder_path, '*.jpg')))
images = [read_and_resize(image_path, resize_max_dim) for image_path in image_files]

# Detect keypoints and descriptors in all images
keypoints = []
descriptors = []
for image in images:
    kp, desc = detect_and_describe(image)
    keypoints.append(kp)
    descriptors.append(desc)

# Start by stitching the first two images
imgA = images[0]
kpA = keypoints[0]
descA = descriptors[0]

for imgB, kpB, descB in zip(images[1:], keypoints[1:], descriptors[1:]):
    matches = match_keypoints(descA, descB)
    stitched_img = stitch_images(imgA, imgB, kpA, kpB, matches)
    
    # Update the 'base' image and its features for the next iteration
    imgA = stitched_img
    kpA, descA = detect_and_describe(stitched_img)

# Save the final stitched image
output_path = '/home/sanket/Downloads/stitched_output.jpg'
cv2.imwrite(output_path, stitched_img)

