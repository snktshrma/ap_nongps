import sys
import cv2
import numpy as np
from imutils import paths
import tempfile
import os
from exif import Image

class mapper:
    def __init__(self):
        try:
            pass
        except Exception as e:
            print(f"Error initializing the mapper: {e}")
            sys.exit(1)

    def decimal_coords(self, coords, ref):
        decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
        if ref == "S" or ref == 'W':
            decimal_degrees = -decimal_degrees
        return decimal_degrees

    def extract_gps(self, image_path):
        with open(image_path, 'rb') as src:
            img = Image(src)
        if img.has_exif:
            try:
                coords = (self.decimal_coords(img.gps_latitude, img.gps_latitude_ref),
                          self.decimal_coords(img.gps_longitude, img.gps_longitude_ref))
            except AttributeError:
                print('No Coordinates found in image:', image_path)
                return None
        else:
            print('The image has no EXIF information:', image_path)
            return None
        return coords

    def load_dataset(self, dataset):
        try:
            img_paths = sorted(list(paths.list_images(dataset)))
            self.dataset = []
            for p in img_paths:
                coords = self.extract_gps(p)
                if coords:
                    self.dataset.append({"path": p, "coords": coords})
            print(f"Loaded {len(self.dataset)} images with GPS data from dataset.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)

    def find_matching_images(self, new_img):
        try:
            orb = cv2.ORB_create(nfeatures=1000)
            kp_new, des_new = orb.detectAndCompute(new_img, None)
            
            best_matches = []
            for entry in self.dataset:
                img = cv2.imread(entry["path"])
                kp_img, des_img = orb.detectAndCompute(img, None)

                if des_img is None or des_new is None:
                    continue

                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des_new, des_img)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > 5:
                    best_matches.append({"path": entry["path"], "coords": entry["coords"], "matches": len(matches)})

            best_matches = sorted(best_matches, key=lambda x: x["matches"], reverse=True)
            if len(best_matches) >= 2:
                print("Found two best matching images.")
                return best_matches[:2]
            else:
                print("Could not find enough matching images in the dataset.")
                return None
        except Exception as e:
            print(f"Error finding matching images: {e}")
            return None

    def stitch_images(self, img1_path, img2_path, new_img):
        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            orb = cv2.ORB_create(nfeatures=1000)
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            kp_new, des_new = orb.detectAndCompute(new_img, None)

            if des1 is None or des2 is None or des_new is None:
                print("Descriptors not found, skipping this pair.")
                return None

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            matches1 = bf.match(des_new, des1)
            matches1 = sorted(matches1, key=lambda x: x.distance)

            # Match new image with the second dataset image
            matches2 = bf.match(des_new, des2)
            matches2 = sorted(matches2, key=lambda x: x.distance)

            if len(matches1) > 5 and len(matches2) > 5:
                print("Found good matches for both images.")
                src_pts1 = np.float32([kp_new[m.queryIdx].pt for m in matches1]).reshape(-1, 1, 2)
                dst_pts1 = np.float32([kp1[m.trainIdx].pt for m in matches1]).reshape(-1, 1, 2)

                src_pts2 = np.float32([kp_new[m.queryIdx].pt for m in matches2]).reshape(-1, 1, 2)
                dst_pts2 = np.float32([kp2[m.trainIdx].pt for m in matches2]).reshape(-1, 1, 2)

                M1, _ = cv2.findHomography(src_pts1, dst_pts1, cv2.RANSAC, 5.0)
                M2, _ = cv2.findHomography(src_pts2, dst_pts2, cv2.RANSAC, 5.0)

                if M1 is not None and M2 is not None:
                    stitched_img1 = self.warp_images(new_img, img1, M1)
                    stitched_img2 = self.warp_images(new_img, img2, M2)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img1:
                        cv2.imwrite(temp_img1.name, stitched_img1)
                        temp_img1_path = temp_img1.name

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img2:
                        cv2.imwrite(temp_img2.name, stitched_img2)
                        temp_img2_path = temp_img2.name

                    return temp_img1_path, temp_img2_path
                else:
                    print("Homography failed.")
                    return None
            else:
                print("Not enough good matches for stitching.")
                return None
        except Exception as e:
            print(f"Error in stitch_images: {e}")
            return None

    def warp_images(self, img1, img2, H):
        try:
            if H is None:
                raise ValueError("Homography matrix is None, cannot warp images.")

            rows1, cols1 = img1.shape[:2]
            rows2, cols2 = img2.shape[:2]

            pts1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
            pts2 = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
            pts2_transformed = cv2.perspectiveTransform(pts2, H)

            pts = np.concatenate((pts1, pts2_transformed), axis=0)
            [x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)

            translation_dist = [-x_min, -y_min]
            H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

            result_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            result_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

            return result_img
        except Exception as e:
            print(f"Error in warp_images: {e}")
            return None

    def interpolate_gps(self, gps_coords, stitched_img1, stitched_img2):
        try:
            h1, w1 = stitched_img1.shape[:2]
            h2, w2 = stitched_img2.shape[:2]

            center_pixel1 = np.float32([w1 // 2, h1 // 2])
            center_pixel2 = np.float32([w2 // 2, h2 // 2])

            gps_coords1, gps_coords2 = gps_coords

            lat_new = (gps_coords1[0] + gps_coords2[0]) / 2
            lon_new = (gps_coords1[1] + gps_coords2[1]) / 2

            print(f"Estimated Latitude: {lat_new}")
            print(f"Estimated Longitude: {lon_new}")

            return lat_new, lon_new

        except Exception as e:
            print(f"Error in interpolate_gps: {e}")
            return None

    def delete_temp_files(self, *file_paths):
        try:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted temporary file: {file_path}")
        except Exception as e:
            print(f"Error deleting temporary files: {e}")

if __name__ == "__main__":
    try:
        map_fin = mapper()
        dataset = "/home/sanket/Downloads/test_triang_sift/images"
        new_image = "/home/sanket/Downloads/test_triang_sift/new/XACT0017.JPG"

        map_fin.load_dataset(dataset)
        new_img = cv2.imread(new_image)

        matching_images = map_fin.find_matching_images(new_img)

        if matching_images:
            temp_paths = map_fin.stitch_images(matching_images[0]["path"], matching_images[1]["path"], new_img)
            if temp_paths:
                temp_img1_path, temp_img2_path = temp_paths
                stitched_img1 = cv2.imread(temp_img1_path)
                stitched_img2 = cv2.imread(temp_img2_path)

                lat, lon = map_fin.interpolate_gps([matching_images[0]["coords"], matching_images[1]["coords"]], stitched_img1, stitched_img2)

                map_fin.delete_temp_files(temp_img1_path, temp_img2_path)
            else:
                print("Stitching failed.")
        else:
            print("Could not find matching images.")
                
    except Exception as e:
        print(f"exception: {e}")
        sys.exit(1)
