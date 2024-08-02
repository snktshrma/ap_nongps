import sys
import cv2
import numpy as np
from imutils import paths


class mapper:
    def __init__(self):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        self.temp_img = []
        self.final_img = []

    def load_images(self):
        input_path = "/home/snkt/Downloads/timtuxworth-highalt-test-26Jul2024"
        output_path = "/home/snkt/Downloads/tt.png"

        img_paths = sorted(list(paths.list_images(input_path)))
        self.images = [cv2.resize(cv2.imread(p), (int(cv2.imread(p).shape[1] * 0.5), int(cv2.imread(p).shape[0] * 0.5))) for p in img_paths]

        self.output_path = output_path

    def stitch_images(self):
        num_images = len(self.images)
        print(f"{num_images} Images loaded")

        for i in range(num_images):
            if i == 0:
                self.temp_img = self.stitch_pair(self.images[i], self.images[i + 1])
            elif i < num_images - 1:
                self.temp_img = self.stitch_pair(self.temp_img, self.images[i + 1])
            else:
                self.final_img = self.temp_img

        cv2.imshow("output", self.final_img)
        cv2.imwrite(self.output_path, self.final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def stitch_pair(self, img1, img2):
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

        if len(good_matches) > 0:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            result = self.warp_images(img2, img1, M)
            return result
        else:
            print("Error")

    def warp_images(self, img1, img2, H):
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]
        pts1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
        pts2 = cv2.perspectiveTransform(pts2, H)
        pts = np.concatenate((pts1, pts2), axis=0)
        [x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)

        trans_dist = [-x_min, -y_min]
        H_trans = np.array([[1, 0, trans_dist[0]], [0, 1, trans_dist[1]], [0, 0, 1]])
        result_img = cv2.warpPerspective(img2, H_trans.dot(H), (x_max - x_min, y_max - y_min))
        result_img[trans_dist[1]:rows1 + trans_dist[1], trans_dist[0]:cols1 + trans_dist[0]] = img1
        return result_img


if __name__ == "__main__":
    map_fin = mapper()
    map_fin.load_images()
    map_fin.stitch_images()
