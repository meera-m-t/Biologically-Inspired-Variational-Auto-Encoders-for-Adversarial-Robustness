import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


class HelicoilDepthCheck:
    def __init__(
        self,
        fins_detector_model_path: str,
        hand_detector_model_path: str,
        interpolation_points: int = 2,
        pixel_thresh: int = 220,  # Threshold for fin point hit detection
    ):
        self.fins_model = self._load_model(fins_detector_model_path)
        self.hand_model = self._load_model(hand_detector_model_path)
        self.point_checks = np.array(
            [False] * (interpolation_points * 4 + 4)
        )
        self.fin_coordinates = None
        self.pixel_thresh = pixel_thresh
        self.distances = []
        self.fin_point_hits = []
        self.total_frames_checked = 0

    def _load_model(self, model_path: str) -> YOLO:
        """Load model"""
        return YOLO(model_path)

    def _find_fin(self, frame: np.ndarray, detections):
        """Find the fin using the provided class index (always 0)"""
        fin_index = 0  # Always 0 as per the requirement
        if detections and hasattr(detections[0], 'obb') and len(detections[0].obb.xyxyxyxy.cpu().numpy()) > 0:
            fin_class = int(detections[0].obb.cls.cpu().numpy()[fin_index])  # Get fin class
            print("fin_class**************", fin_class)
            
            # Differentiate between the 2 possibilities for the fin
            self.fin_coordinates = self._interpolate_polygon_points(
                detections[0].obb.xyxyxyxy.cpu().numpy()[fin_index]  # Using fin_index
            )
            
            # Assign color based on the fin class
            if fin_class == 0:
                color = (255, 0, 0)  # Blue (Back Fin)
            elif fin_class == 1:
                color = (0, 255, 0)  # Green (Front Fin)

            # Draw the interpolated points as circles on the frame with the assigned color
            for point in self.fin_coordinates:
                cv2.circle(frame, (int(point[0]), int(point[1])), radius=3, color=color, thickness=3)
        else:
            print("No fin detected.")

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
                    # Draw the hand on the frame
                    cv2.circle(frame, (c_x, c_y), radius=3, color=(0, 0, 255), thickness=3)
        return hand_coords

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

    def _compute_distance(self, point1: list[int], point2: list[int]) -> float:
        """Compute distance between two points"""
        if point1 and point2:
            distance = np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
            return distance
        return float('inf')

    def _check_operator(self, frame: np.ndarray, timestamp: float):
        """Determine if operator is moving hands near the fin. Checks hand position relative to fins and flags if close enough."""
        detections = self.fins_model(frame, imgsz=1024, conf=0.3, verbose=False)
        self._find_fin(frame, detections)
        hand_coords_list = self._find_hands(frame)

        if hand_coords_list:
            for hand_coords in hand_coords_list:
                # Compute distances between hand and each fin point
                distances_to_fin = self._compute_distance_to_fin(hand_coords)
                if len(distances_to_fin) > 0:
                    min_distance = np.min(distances_to_fin)  # Store only the minimum distance
                    self.distances.append({"Time (seconds)": timestamp, "Distance (pixels)": min_distance})
                    print(f"Minimum distance between hand and fin: {min_distance} pixels")

                # Determine how many points on the fin outline are within the threshold
                hits = np.sum([d <= self.pixel_thresh for d in distances_to_fin])
                self.fin_point_hits.append(hits)
                print(f"Number of fin points 'hit' by the hand: {hits}")

        self.total_frames_checked += 1

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

    def inspectHelicoilDepth(self, frame: np.ndarray, timestamp: float):
        """Analyze each frame where the hand is detected."""
        self._check_operator(frame, timestamp)

    def final_decision(self) -> bool:
        """Make the final decision based on hand proximity and fin points hit."""
        if len(self.fin_point_hits) > 0:
            majority_hits = np.mean(self.fin_point_hits)
            print(f"Average number of fin points 'hit': {majority_hits}")
    
            # Adjusting the thresholds for acceptance
            if majority_hits >= 0.9:
                return True
            
        print("Final Decision: Helicoil depth check failed.")
        return False

    def save_distances_to_csv(self, output_csv_path: str):
        """Save the distances to a CSV file"""
        df = pd.DataFrame(self.distances)
        
        # Rename columns for clarity
        df.columns = ["Time (seconds)", "Distance (pixels)"]
        
        df.to_csv(output_csv_path, index=False)
        print(f"Distances saved to {output_csv_path}")


if __name__ == "__main__":
    # Initialize model
    i = 6
    helicoil_depth_check = HelicoilDepthCheck("models/fin_detector2.pt", "models/hand_detector.pt")

    # This is just simulating grabbing frames from live stream
    example_video_path = "data/large/small/incorrect_Small Gromo CAC - 5004-05269_clip copy.mkv"
    cap = cv2.VideoCapture(example_video_path)

    # Set up video writer to save output in MKV format
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(f"results/video_output_{i}.mkv", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

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
