import sys
import cv2
import numpy as np
from imutils import paths


class mapper:
    def __init__(self):
        try:
            # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            pass
        except Exception as e:
            print(f"Error initializing the mapper: {e}")
            sys.exit(1)

    def load_images(self):
        try:
            input_path = "/home/snkt/Downloads/timtuxworth-highalt-test-26Jul2024"
            self.output_path_template = "/home/snkt/Downloads/batch_output_{}.png"

            img_paths = sorted(list(paths.list_images(input_path)))
            self.images = [
                cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_RGB2RGBA), (int(cv2.imread(p).shape[1] * 0.5), int(cv2.imread(p).shape[0] * 0.5)))
                for p in img_paths
            ]

            print(f"Loaded {len(self.images)} images successfully")
        except Exception as e:
            print(f"Error loading images: {e}")
            sys.exit(1)

    def stitch_images(self):
        try:
            num_images = len(self.images)
            print(f"{num_images} Images loaded")

            batch_size = 19
            batches = [self.images[i:i + batch_size] for i in range(0, num_images, batch_size)]
            print(f"Processing {len(batches)} batches of images")

            for batch_index, batch in enumerate(batches):
                print(f"Stitching batch {batch_index + 1} of {len(batches)}")

                if len(batch) > 1:
                    temp_img = self.stitch_pair(batch[0], batch[1])
                    for i in range(2, len(batch)):
                        if temp_img is None:
                            print(f"Skipping image {i + 1} in batch {batch_index + 1} due to errors.")
                            continue

                        temp_img = self.stitch_pair(temp_img, batch[i])
                    
                    r,g,b,a = cv2.split(temp_img)

                    _, alpha_mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY)

                    cont, _ = cv2.findContours(alpha_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if cont:
                        x,y,w,h = cv2.boundingRect(cont[0])

                        temp_img = temp_img[y:y+h, x:x+w]

                    if temp_img is not None:
                        # Display the batch
                        # resized_img = self.resize_image_to_fit_screen(temp_img)
                        # cv2.imshow(f"Batch {batch_index + 1}", resized_img)
                        # cv2.waitKey(0)  # Wait for a key press

                        # Save the batch to a file
                        batch_output_path = self.output_path_template.format(batch_index + 1)
                        cv2.imwrite(batch_output_path, temp_img)
                        print(f"Batch {batch_index + 1} saved to {batch_output_path}")

                        # Clear temp_img to free memory
                        temp_img = None
                        # cv2.destroyWindow(f"Batch {batch_index + 1}")
                    else:
                        print(f"Batch {batch_index + 1} stitching failed, skipping this batch.")
                else:
                    print(f"Batch {batch_index + 1} has only one image, skipping stitching.")

            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error during image stitching: {e}")

    def resize_image_to_fit_screen(self, img, max_width=800, max_height=600):
        h, w = img.shape[:2]
        if w > max_width or h > max_height:
            scaling_factor = min(max_width / float(w), max_height / float(h))
            img = cv2.resize(img, (int(w * scaling_factor), int(h * scaling_factor)))
        return img

    def stitch_pair(self, img1, img2):
        try:
            orb = cv2.ORB_create(nfeatures=1000)
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            if des1 is None or des2 is None:
                print("Descriptors not found, skipping this pair.")
                return None

            bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

            if len(good_matches) > 5:
                print(f"Found {len(good_matches)} good matches")
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    result = self.warp_images(img2, img1, M)
                    return result
                else:
                    print("Homography failed, skipping this pair.")
                    return None
            else:
                print("Not enough good matches to find homography.")
                return None
        except Exception as e:
            print(f"Error in stitch_pair: {e}")
            return None

    def warp_images(self, img1, img2, H):
        try:
            if H is None:
                raise ValueError("Homography matrix is None, cannot warp images.")

            rows1, cols1 = img1.shape[:2]
            rows2, cols2 = img2.shape[:2]
            pts1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
            pts2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

            pts2 = cv2.perspectiveTransform(pts2, H)
            pts = np.concatenate((pts1, pts2), axis=0)
            [x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)

            trans_dist = [-x_min, -y_min]

            # print((x_max - x_min, y_max - y_min))
            H_trans = np.array([[1, 0, trans_dist[0]], [0, 1, trans_dist[1]], [0, 0, 1]])
            result_img = cv2.warpPerspective(img2, H_trans.dot(H), (x_max - x_min, y_max - y_min), borderMode = cv2.BORDER_CONSTANT, borderValue = 0)
            result_img[trans_dist[1]:rows1 + trans_dist[1], trans_dist[0]:cols1 + trans_dist[0]] = img1
            return result_img
        except Exception as e:
            print(f"Error in warp_images: {e}")
            return None


if __name__ == "__main__":
    try:
        map_fin = mapper()
        map_fin.load_images()
        map_fin.stitch_images()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        sys.exit(1)
