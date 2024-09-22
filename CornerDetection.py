import numpy as np
import matplotlib.pyplot as plt
from OurCV2 import OurCV2
import cv2
import scipy.ndimage as ndimage


class CornerDetection:
    def __init__(self):
        self.ourcv2 = OurCV2()

    def apply_nms(self, R, window_size=3):
        # Perform max pooling to identify local maxima
        local_max = ndimage.maximum_filter(R, size=window_size)

        # Keep only those pixels that are equal to their local maximum
        corners = np.zeros_like(R)
        corners[(R == local_max) & (R > 0)] = (
            255  # Ensure thresholded R values are considered
        )

        return corners

    def harris_corner_detection_opencv(
        self, image, k=0.04, threshold=0.01, nms_window=3
    ):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        # Perform Harris corner detection using OpenCV
        R = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=k)

        # Apply 1% threshold to R
        R_max = np.max(R)
        R[R < 0.01 * R_max] = 0

        # Apply Non-Maximum Suppression
        corners = self.apply_nms(R, window_size=nms_window)

        # Mark detected corners on the original image
        corner_image = np.copy(image)
        for y in range(corners.shape[0]):
            for x in range(corners.shape[1]):
                if corners[y, x] == 255:
                    cv2.circle(corner_image, (x, y), 1, (0, 0, 255), 2)

        return corner_image, R

    def harris_corner_detection_custom(
        self, image, k=0.04, threshold=0.01, nms_window=3
    ):
        # Convert to grayscale
        gray = self.ourcv2.manual_gray(image)
        gray = np.float32(gray)

        # Compute gradients using Sobel
        Ix = self.ourcv2.apply_filter(gray, self.ourcv2.sobel_x)
        Iy = self.ourcv2.apply_filter(gray, self.ourcv2.sobel_y)

        # Compute products of derivatives at each pixel
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # Apply Gaussian blur
        Ixx = self.ourcv2.manual_gaussian_blur(Ixx, kernel_size=3, sigma=1)
        Iyy = self.ourcv2.manual_gaussian_blur(Iyy, kernel_size=3, sigma=1)
        Ixy = self.ourcv2.manual_gaussian_blur(Ixy, kernel_size=3, sigma=1)

        # Compute the Harris corner response matrix R for each pixel
        detM = (Ixx * Iyy) - (Ixy * Ixy)
        traceM = Ixx + Iyy
        R = detM - k * (traceM**2)

        # Apply 1% threshold to R
        R_max = np.max(R)
        R[R < 0.01 * R_max] = 0  # 1% of the max response

        # Apply Non-Maximum Suppression
        corners = self.apply_nms(R, window_size=nms_window)

        # Mark detected corners on the original image
        corner_image = np.copy(image)
        for y in range(corners.shape[0]):
            for x in range(corners.shape[1]):
                if corners[y, x] == 255:
                    cv2.circle(corner_image, (x, y), 1, (0, 0, 255), 2)

        return corner_image, R

    def main(self):
        image_path = "content/input/chess.png"
        image = cv2.imread(image_path)

        if image is None:
            print("Error: Image not found.")
            return

        # Perform Harris Corner Detection using OpenCV functions
        opencv_corners, _ = self.harris_corner_detection_opencv(image)

        # Perform Harris Corner Detection using custom functions
        custom_corners, _ = self.harris_corner_detection_custom(image)

        # Display the results side-by-side
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(self.ourcv2.manual_BGR2RGB(image))
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("OpenCV Harris Corners")
        plt.imshow(self.ourcv2.manual_BGR2RGB(opencv_corners))
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Custom Harris Corners")
        plt.imshow(self.ourcv2.manual_BGR2RGB(custom_corners))
        plt.axis("off")

        plt.show()


if __name__ == "__main__":
    detector = CornerDetection()
    detector.main()
