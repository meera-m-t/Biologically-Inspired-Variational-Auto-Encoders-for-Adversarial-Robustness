import cv2
import numpy as np
import pandas as pd
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
    ):
        self.fins_model = self._load_model(fins_detector_model_path)
        self.hand_model = self._load_model(hand_detector_model_path)
        self.driver_model = self._load_model(driver_detector_model_path)
        self.point_checks = np.array(
            [False] * (interpolation_points * 4 + 4)
        )
        self.fin_coordinates = None
        self.pixel_thresh = pixel_thresh
        self.driver_hand_thresh = driver_hand_thresh
        self.distances = []
        self.driver_hand_distances = []
        self.fin_point_hits = []
        self.frames_with_driver_hand_within_thresh = 0
        self.total_frames_checked = 0
        self.driver_coords = []
        self.yellow_box_coords = None  # To keep track of the yellow box coordinates

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
            # Draw the interpolated points as circles on the frame
            for point in self.fin_coordinates:
                cv2.circle(frame, (int(point[0]), int(point[1])), radius=3, color=(0, 255, 0), thickness=-1)

            # Draw yellow box if the driver is not detected
            if not self.driver_coords:
                self.yellow_box_coords = self._draw_yellow_box(frame)
        else:
            self.fin_coordinates = None
            print("No fins detected.")

    def _find_driver(self, frame: np.ndarray, imgsz: int = 640, conf: float = 0.25) -> list[int]:
        """Find the driver using OBB"""
        detections = self.driver_model(frame, imgsz=imgsz, conf=conf, verbose=False)
        if detections and hasattr(detections[0], 'obb') and len(detections[0].obb.xyxyxyxy.cpu().numpy()) > 0:
            obb = detections[0].obb.xyxyxyxy.cpu().numpy()[0]
            points = self._extract_obb_points(obb)
            c_x = np.mean(points[:, 0])
            c_y = np.mean(points[:, 1])
            print(f"Driver detected at ({c_x}, {c_y}).")
            # Draw the driver on the frame
            for point in points:
                cv2.circle(frame, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 255), thickness=-1)
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
        """Extrapolate points in a polygon, retaining the shape."""
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

    def _compute_distance(self, point1: list[int], point2: list[int]) -> float:
        """Compute distance between two points"""
        if point1 and point2:
            distance = np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
            return distance
        return float('inf')

    def _check_operator(self, frame: np.ndarray, timestamp: float):
        """Determine if the operator is moving hands near the driver. Checks driver position relative to fins and flags if close enough."""
        self._find_fin(frame)
        self.driver_coords = self._find_driver(frame)
        hand_coords_list = self._find_hands(frame)

        if self.driver_coords and hand_coords_list:
            for hand_coords in hand_coords_list:
                # Compute distance between driver and hand
                driver_hand_distance = self._compute_distance(self.driver_coords, hand_coords)
                self.driver_hand_distances.append({"Time (seconds)": timestamp, "Driver-Hand Distance (pixels)": driver_hand_distance})
                print(f"Distance between driver and hand: {driver_hand_distance} pixels")

                # Check if the driver is within the threshold distance of the hand
                if driver_hand_distance <= self.driver_hand_thresh:
                    self.frames_with_driver_hand_within_thresh += 1

            # Compute distances between driver and each fin point
            distances_to_fin = self._compute_distance_to_fin(self.driver_coords)
            if len(distances_to_fin) > 0:
                min_distance = np.min(distances_to_fin)  # Store only the minimum distance
                self.distances.append({"Time (seconds)": timestamp, "Distance (pixels)": min_distance})
                print(f"Minimum distance between driver and fin: {min_distance} pixels")

            # Determine how many points on the fin outline are within the threshold
            hits = np.sum([d <= self.pixel_thresh for d in distances_to_fin])
            self.fin_point_hits.append(hits)
            print(f"Number of fin points 'hit' by the driver: {hits}")

        self.total_frames_checked += 1

        # Draw the yellow box if it was set in previous frames
        if self.yellow_box_coords and not self.driver_coords:
            self._draw_existing_yellow_box(frame)

    def _compute_distance_to_fin(self, driver_coords: list[int]) -> np.ndarray:
        """Compute distances between the driver and each point on the fin outline."""
        if driver_coords and self.fin_coordinates is not None:
            distances = np.sqrt(
                np.sum(
                    (np.array(self.fin_coordinates) - np.array(driver_coords)) ** 2, axis=1
                )
            )
            return distances

        return np.array([])

    def _draw_yellow_box(self, frame: np.ndarray):
        """Draw a yellow box similar to the fin box but 10% smaller from the left."""
        if self.fin_coordinates is not None and len(self.fin_coordinates) >= 2:
            # Get the left-most and right-most points (xmin, ymin) and (xmax, ymax)
            xmin = np.min(self.fin_coordinates[:, 0])
            ymin = np.min(self.fin_coordinates[:, 1])
            xmax = np.max(self.fin_coordinates[:, 0])
            ymax = np.max(self.fin_coordinates[:, 1])

            # Calculate the width and height
            width = xmax - xmin
            height = ymax - ymin

            # Calculate new coordinates for the yellow box
            new_xmin = int(xmin + 0.1 * width)
            new_xmax = xmax
            new_ymin = ymin
            new_ymax = ymax

            # Draw the yellow box on the frame
            cv2.rectangle(
                frame, (new_xmin, new_ymin), (new_xmax, new_ymax), (0, 255, 255), 2
            )
            print("Yellow box drawn.")

            # Return the yellow box coordinates
            return (new_xmin, new_ymin, new_xmax, new_ymax)

        return None

    def _draw_existing_yellow_box(self, frame: np.ndarray):
        """Draw the yellow box that was set in previous frames."""
        if self.yellow_box_coords is not None:
            new_xmin, new_ymin, new_xmax, new_ymax = self.yellow_box_coords
            cv2.rectangle(
                frame, (new_xmin, new_ymin), (new_xmax, new_ymax), (0, 255, 255), 2
            )
            print("Persistent yellow box drawn.")

    def inspectHelicoilDepth(self, frame: np.ndarray, timestamp: float):
        """Analyze each frame for helicoil depth check."""
        self._check_operator(frame, timestamp)

    def final_decision(self) -> bool:
        """Make the final decision based on driver-hand proximity and fin points hit."""
        if len(self.fin_point_hits) > 0:
            majority_hits = np.mean(self.fin_point_hits)
            print(f"Average number of fin points 'hit': {majority_hits}")

            # Consider the driver's proximity to the hand in the decision
            driver_hand_ratio = self.frames_with_driver_hand_within_thresh / self.total_frames_checked
            print(f"Ratio of frames where driver is within threshold distance of hand: {driver_hand_ratio:.2f}")

            # Adjusting the thresholds for acceptance
            if majority_hits >= 0.9 and driver_hand_ratio >= 0.27:
                return True

        print("Final Decision: Helicoil depth check failed.")
        return False

    def save_distances_to_csv(self, output_csv_path: str):
        """Save the distances and driver-hand distances to a CSV file."""
        df = pd.DataFrame(self.distances)
        df_hand = pd.DataFrame(self.driver_hand_distances)

        # Merge the distance and driver-hand distance DataFrames
        df_combined = pd.concat([df, df_hand["Driver-Hand Distance (pixels)"]], axis=1)

        # Rename columns for clarity
        df_combined.columns = ["Time (seconds)", "Distance (pixels)", "Driver-Hand Distance (pixels)"]

        df_combined.to_csv(output_csv_path, index=False)
        print(f"Distances and driver-hand distances saved to {output_csv_path}")


if __name__ == "__main__":
    # Initialize model
    helicoil_depth_check = HelicoilDepthCheck("models/fin_detector.pt", "models/hand_detector.pt", "models/driver.pt")

    # This is just simulating grabbing frames from a live stream
    example_video_path = "data/large/correct/Mar-11_ 24_09_16_30-clip.mkv"
    cap = cv2.VideoCapture(example_video_path)

    # Set up video writer to save output in MKV format
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output_with_visualization.mkv', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    if not cap.isOpened():
        raise ("Error opening video file")

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

    # After processing all frames, make the final decision
    if helicoil_depth_check.final_decision():
        print("Final Decision: Helicoil depth check passed.")
    else:
        print("Final Decision: Helicoil depth check failed.")

    # Save the distances to a CSV file
    helicoil_depth_check.save_distances_to_csv("distances.csv")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Finished processing video.")
