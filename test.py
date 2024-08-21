import cv2
import numpy as np
from ultralytics import YOLO


class HelicoilDepthCheck:
    def __init__(
        self,
        fins_detector_model_path: str,
        hand_detector_model_path: str,
        driver_detector_model_path: str,
        interpolation_points: int = 2,
        pixel_thresh: int = 120,  # Threshold for fin point hit detection
        driver_hand_thresh: int = 700,  # Threshold for driver-hand proximity
        color_thresh: int = 50,  # Color threshold to filter out non-fin surfaces
        size_range: tuple = (1000, 10000),  # Expected size range for the fin's top surface
        hand_far_thresh: int = 500,  # Distance threshold to consider hand far from the fin
    ):
        self.fins_model = self._load_model(fins_detector_model_path)
        self.hand_model = self._load_model(hand_detector_model_path)
        self.driver_model = self._load_model(driver_detector_model_path)
        self.point_checks = np.array([False] * (interpolation_points * 4 + 4))
        self.fin_coordinates = None
        self.pixel_thresh = pixel_thresh
        self.driver_hand_thresh = driver_hand_thresh
        self.color_thresh = color_thresh
        self.size_range = size_range
        self.hand_far_thresh = hand_far_thresh
        self.fin_top_color = None  # Placeholder for the fin's top surface color
        self.distances = []
        self.driver_hand_distances = []
        self.fin_point_hits = []
        self.frames_with_driver_hand_within_thresh = 0
        self.total_frames_checked = 0
        self.previous_box = None  # Store the previous yellow box
        self.hand_far_from_fin = True  # Flag to check if hand is far from fin

    def _load_model(self, model_path: str) -> YOLO:
        """Load model"""
        return YOLO(model_path)

    def _find_fin(self, frame: np.ndarray, imgsz: int = 640, conf: float = 0.25):
        """Find the fins' borders"""
        detections = self.fins_model(frame, imgsz=imgsz, conf=conf, verbose=False)
        if detections and hasattr(detections[0], 'obb') and len(detections[0].obb.xyxyxyxy.cpu().numpy()) > 0:
            self.fin_coordinates = self._interpolate_polygon_points(
                detections[0].obb.xyxyxyxy.cpu().numpy()[0]
            )
            # Sample the color of the fin's top surface
            self.fin_top_color = self._sample_color(frame, self.fin_coordinates)
            # Draw the interpolated points as circles on the frame
            for point in self.fin_coordinates:
                cv2.circle(frame, (int(point[0]), int(point[1])), radius=3, color=(0, 255, 0), thickness=-1)
        else:
            self.fin_coordinates = None
            print("No fins detected.")

    def _sample_color(self, frame: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        """Sample the color from the center of the detected fin area"""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        points = coordinates.reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
        mean_color = cv2.mean(frame, mask=mask)[:3]  # BGR color
        return np.array(mean_color)

    def _find_driver(self, frame: np.ndarray, imgsz: int = 640, conf: float = 0.25) -> list[int]:
        """Find the driver using OBB"""
        detections = self.driver_model(frame, imgsz=imgsz, conf=conf, verbose=False)
        if detections and hasattr(detections[0], 'obb') and len(detections[0].obb.xyxyxyxy.cpu().numpy()) > 0:
            obb = detections[0].obb.xyxyxyxy.cpu().numpy()[0]
            points = self._extract_obb_points(obb)
            c_x = np.mean(points[:, 0])
            c_y = np.mean(points[:, 1])
            print(f"Driver detected at ({c_x}, {c_y}).")
            return [int(c_x), int(c_y)]
        print("No driver detected.")
        return []

    def _find_hands(
        self, frame: np.ndarray, imgsz: int = 640, conf: float = 0.25
    ) -> list[list[int]]:
        """Find the hand borders"""
        detections = self.hand_model(frame, imgsz=imgsz, conf=conf, verbose=False)
        hand_coords = []
        for result in detections:
            if result.boxes is not None:
                boxes = result.boxes.data.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    c_x = int((x1 + x2) / 2)
                    c_y = int((y1 + y2) / 2)
                    hand_coords.append([c_x, c_y])
        return hand_coords

    def _extract_obb_points(self, obb: np.ndarray) -> np.ndarray:
        """Extract the points from OBB and reshape them."""
        return obb.reshape(-1, 2)

    def _interpolate_polygon_points(self, points: np.ndarray, additional_points: int = 2) -> np.ndarray:
        """Extrapolate points in a polygon. Retaining the shape"""
        new_points = []
        for i in range(len(points)):
            start_point = points[i]
            end_point = points[(i + 1) % len(points)]

            new_points.append([start_point[0], start_point[1]])
            for j in range(1, additional_points + 1):
                fraction = j / (additional_points + 1)
                # Linear interpolation
                x = start_point[0] + (end_point[0] - start_point[0]) * fraction
                y = start_point[1] + (end_point[1] - start_point[1]) * fraction
                new_points.append([x, y])

        return np.array(new_points)

    def _detect_top_surface(self, frame: np.ndarray):
        """Detect the top surface of the fin and annotate it"""
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv_frame, lower_white, upper_white)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            new_box_area = cv2.contourArea(box)

            # Filter based on size and color consistency
            if self._is_valid_top_surface(box, new_box_area, frame):
                if self.previous_box is None or (self.hand_far_from_fin and self._should_replace_box(box, new_box_area)):
                    self.previous_box = box
                    print("Updated the previous box based on size, color consistency, and hand distance.")
                cv2.drawContours(frame, [self.previous_box], 0, (0, 255, 255), 2)
                print("Top surface detected and annotated.")
            else:
                # Draw the previous box if the new one isn't valid
                if self.previous_box is not None:
                    cv2.drawContours(frame, [self.previous_box], 0, (0, 255, 255), 2)
                    print("Persisting previous yellow box.")

    def _is_valid_top_surface(self, box: np.ndarray, area: float, frame: np.ndarray) -> bool:
        """Check if the detected top surface is valid based on size and color."""
        if not (self.size_range[0] <= area <= self.size_range[1]):
            print(f"Detected area {area} is out of size range.")
            return False

        # Check color consistency with the sampled fin top color
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [box], 0, 255, -1)
        mean_color = cv2.mean(frame, mask=mask)[:3]
        color_diff = np.linalg.norm(np.array(mean_color) - self.fin_top_color)
        if color_diff > self.color_thresh:
            print(f"Color difference {color_diff} exceeds threshold.")
            return False

        return True

    def _should_replace_box(self, box: np.ndarray, area: float) -> bool:
        """Determine if the previous box should be replaced with the new one."""
        if self.previous_box is None:
            return True

        prev_box_area = cv2.contourArea(self.previous_box)
        new_box_orientation = cv2.minAreaRect(box)[2]
        prev_box_orientation = cv2.minAreaRect(self.previous_box)[2]

        # Replace the box if the new one is smaller or has a different orientation
        return area < prev_box_area or not np.isclose(new_box_orientation, prev_box_orientation, atol=5)

    def inspectHelicoilDepth(self, frame: np.ndarray, timestamp: float):
        """Analyze each frame where the driver is detected."""
        self._find_fin(frame)
        hand_coords_list = self._find_hands(frame)

        # Check if the hand is far from the fin
        if hand_coords_list and self.fin_coordinates is not None:
            for hand_coords in hand_coords_list:
                distances_to_fin = self._compute_distance_to_fin(hand_coords)
                if len(distances_to_fin) > 0:
                    min_distance_to_fin = np.min(distances_to_fin)
                    if min_distance_to_fin > self.hand_far_thresh:
                        self.hand_far_from_fin = True
                    else:
                        self.hand_far_from_fin = False
                        break
        else:
            self.hand_far_from_fin = True  # Assume hand is far if not detected

        self._detect_top_surface(frame)

    def _compute_distance_to_fin(self, hand_coords: list[int]) -> np.ndarray:
        """Compute distances between the hand and each point on the fin outline"""
        if hand_coords and self.fin_coordinates is not None:
            distances = np.sqrt(
                np.sum(
                    (np.array(self.fin_coordinates) - np.array(hand_coords)) ** 2, axis=1
                )
            )
            return distances

        return np.array([])

    # The final decision and other logic remain unchanged


if __name__ == "__main__":
    # Initialize model
    helicoil_depth_check = HelicoilDepthCheck("models/fin_detector.pt", "models/hand_detector.pt", "models/driver.pt")

    # This is just simulating grabbing frames from live stream
    example_video_path = "data/large/correct/Mar-11_ 24_09_16_30-clip.mkv"
    cap = cv2.VideoCapture(example_video_path)

    # Set up video writer to save output in MKV format
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output_with_visualization.mkv', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    if not cap.isOpened():
        raise Exception("Error opening video file")

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_count += 1
        timestamp = frame_count / fps  # Calculate the time in seconds
        print(f"Processing frame {frame_count} at {timestamp:.2f} seconds")

        # Analyze each frame for helicoil depth check
        helicoil_depth_check.inspectHelicoilDepth(frame, timestamp)

        # Write the frame with visualization to the output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Finished processing video.")
