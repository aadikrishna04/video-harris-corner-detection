import numpy as np


class OurCV2:
    def __init__(self):
        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    def apply_filter(self, image, kernel):
        """
        Applies custom sobel filter to image
        Args:
        - image: Input image array.
        - kernel: Gaussian kernel.
        """
        width, height = image.shape
        filtered_image = np.zeros_like(image)
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                region = image[i - 1 : i + 2, j - 1 : j + 2]
                filtered_value = np.sum(region * kernel)
                filtered_image[i, j] = filtered_value
        return filtered_image

    def compute_gradients(self, image):
        """
        Compute's image gradients.
        Args:
        - image: Input image array.
        """
        Ix = self.apply_filter(image, self.sobel_x)
        Iy = self.apply_filter(image, self.sobel_y)
        return Ix, Iy

    def manual_gaussian_blur(self, image, kernel_size=3, sigma=1):
        """
        Applies a manual Gaussian blur to the image.
        Args:
        - image: Input image array.
        - kernel_size: Size of the Gaussian kernel.
        - sigma: Standard deviation for the Gaussian function.
        """
        center = kernel_size // 2
        kernel = np.exp(-0.5 * (np.arange(kernel_size) - center) ** 2 / sigma**2)
        kernel /= kernel.sum()

        blurred_image = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=image
        )
        blurred_image = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=blurred_image
        )

        return blurred_image

    def manual_gray(self, image):
        """
        Converts a BGR image to grayscale manually.
        Args:
        - image: Input BGR image.

        Returns:
        - Grayscale image.
        """
        B = image[:, :, 0]
        G = image[:, :, 1]
        R = image[:, :, 2]
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        return gray

    def manual_BGR2RGB(self, image):
        """
        Converts a BGR image to RGB manually.
        Args:
        - image: Input BGR image.

        Returns:
        - RGB image.
        """
        rgb_image = np.empty_like(image)
        rgb_image[:, :, 0] = image[:, :, 2]  # Red channel
        rgb_image[:, :, 1] = image[:, :, 1]  # Green channel
        rgb_image[:, :, 2] = image[:, :, 0]  # Blue channel
        return rgb_image
