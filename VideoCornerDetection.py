import cv2
from CornerDetection import CornerDetection
from OurCV2 import OurCV2
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class VideoCornerDetection:
    def __init__(self):
        self.corner_detector = CornerDetection()
        self.ourcv2 = OurCV2()

    def process_frame(self, frame):
        # Resize frame to reduce processing time
        frame = cv2.resize(frame, (640, 360))
        corner_frame, _ = self.corner_detector.harris_corner_detection_custom(frame)
        return corner_frame

    def process_video(self, input_path, output_path):
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {input_path}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logging.info(
                f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames"
            )

            # Use a more efficient codec
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (640, 360))
            if not out.isOpened():
                raise IOError(f"Cannot create output video file: {output_path}")

            pbar = tqdm(total=total_frames, unit="frames")

            # Create a pool of worker processes
            with Pool(processes=cpu_count()) as pool:
                frame_count = 0
                while True:
                    frames = []
                    for _ in range(10):  # Process 10 frames at a time
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                        frame_count += 1

                    if not frames:
                        break

                    # Process frames in parallel
                    processed_frames = pool.map(self.process_frame, frames)

                    for processed_frame in processed_frames:
                        out.write(processed_frame)
                        pbar.update(1)

                    if frame_count % 100 == 0:
                        logging.info(f"Processed {frame_count} frames")

            pbar.close()
            cap.release()
            out.release()

            logging.info(f"Video processing complete. Output saved to {output_path}")

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")

    def main(self):
        input_video = "content/input/highway.mov"
        output_video = "content/output/output_video.mp4"

        logging.info("Starting video processing...")
        self.process_video(input_video, output_video)


if __name__ == "__main__":
    detector = VideoCornerDetection()
    detector.main()
