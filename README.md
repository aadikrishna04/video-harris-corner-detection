## Harris Corner Detection for Video Processing

Implemented the Harris Corner Detection algorithm for video processing.
Implemented Non-Maximum Suppression and multiprocessing for video frames, resulting in approximately a **15x improvement** in time-to-process per frame.
Wrote various OpenCV functions from scratch:

- **`convolve()`:**
  - Used SciPy’s Fast Fourier Transform (FFT) functions to convert the array to the FFT domain.
  - Utilized the Convolution Theorem: Multiplying in the frequency domain = convolution in the spatial domain.
  - Applied the Inverse FFT function to convert back to the spatial domain.
 
- **`apply_along_axis()`:**
  - Applies convolution along each row and axis.

- **`cvtColor()`:**
  - Converts the original image to grayscale.
  - Handles conversion from BGR (OpenCV) to RGB (Matplotlib).

- **`gaussianBlur()`:**
  - Averages pixels in a region using a Gaussian function.
  - Smooths out the image, reducing noise and false positives.
  - The first attempt used a k-by-k kernel, but it was too slow.
  - Switched to separable convolution for improved efficiency.

### [Presentation Link](https://docs.google.com/presentation/d/11f1OSlC2Pb5wMTSbOezbI816ZdHqQ6261RltjsTJmB0/edit?usp=sharing)
