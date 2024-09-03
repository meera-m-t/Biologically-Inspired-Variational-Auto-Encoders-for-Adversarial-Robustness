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
        pixel_thresh: int = 700,  # Threshold for fin point hit detection
        driver_hand_thresh: int = 1100,  # Threshold for driver-hand proximity
        coverage_perc: float = 0.70,  # Minimum coverage percentage required for acceptance
        max_missing_frames: int = 1  # Maximum number of frames to wait before resetting fin coordinates
    ):
        self.fins_model = self._load_model(fins_detector_model_path)
        self.hand_model = self._load_model(hand_detector_model_path)
        self.driver_model = self._load_model(driver_detector_model_path)
        self.point_checks = np.array([False] * (interpolation_points * 4 + 4))
        self.fin_coordinates = None
        self.previous_fin_coordinates = None
        self.pixel_thresh = pixel_thresh
        self.driver_hand_thresh = driver_hand_thresh
        self.coverage_perc = coverage_perc
        self.distances = []
        self.driver_hand_distances = []
        self.fin_point_hits = []
        self.frames_with_driver_hand_within_thresh = 0
        self.total_frames_checked = 0
        self.max_missing_frames = max_missing_frames  # Initialize the missing frames threshold
        self.missing_frames = 0  # Counter for consecutive frames without fin detection

    def _load_model(self, model_path: str) -> YOLO:
        """Load model"""
        return YOLO(model_path)

    def _interpolate_polygon_points(self, points: np.ndarray, additional_points: int = 2) -> np.ndarray:
        """Extrapolate points in a polygon, retaining the shape"""
        new_points = []
        for i in range(len(points)):
            start_point = points[i]
            end_point = points[(i + 1) % len(points)]

            new_points.append([start_point[0], start_point[1]])
            for j in range(1, additional_points + 1):
                fraction = j / (additional_points + 1)
                x = start_point[0] + (end_point[0] - start_point[0]) * fraction
                y = start_point[1] + (end_point[1] - start_point[1]) * fraction
                new_points.append([x, y])

        return np.array(new_points)

    def _compute_orientation_angle(self, points: np.ndarray) -> float:
        """Compute the orientation angle of the OBB using the longest edge."""
        vectors = [points[(i + 1) % len(points)] - points[i] for i in range(4)]
        lengths = [np.linalg.norm(vec) for vec in vectors]
        max_index = np.argmax(lengths)
        vec = vectors[max_index]
        angle = np.arctan2(vec[1], vec[0])  
        return np.degrees(angle)

    def _process_detections(self, detections):
        """Process detections to separate fin and surface"""
        fin_index = None
        surface_index = None
        
        classes = detections[0].obb.cls.cpu().numpy()
        for i, cls in enumerate(classes):
            if cls == 3:  # Surface
                surface_index = i
            elif cls in [0, 1, 2]:  # Fin (large, medium, small)
                fin_index = i
        
        return fin_index, surface_index

    def _find_fin(self, frame: np.ndarray, detections, fin_index: int):
        """Find the fin using the provided class index"""
        if fin_index is None:
            self.missing_frames += 1
            if self.missing_frames >= self.max_missing_frames:
                self.fin_coordinates = None
                self.previous_fin_coordinates = None
                print("Fin not detected for several frames, resetting coordinates.")
            return  
        
        self.previous_fin_coordinates = self.fin_coordinates
        self.fin_coordinates = self._interpolate_polygon_points(
            detections[0].obb.xyxyxyxy.cpu().numpy()[fin_index]
        )

        if self.previous_fin_coordinates is not None:
            orientation_change = self._check_orientation_change(self.fin_coordinates, self.previous_fin_coordinates)
            if orientation_change:
                self.fin_coordinates = None
                self.previous_fin_coordinates = None
                self.missing_frames = 0  
                print("Fin orientation change detected, resetting coordinates.")
                return  

        fin_class = int(detections[0].obb.cls.cpu().numpy()[fin_index])
        color = (255, 0, 0) if fin_class == 0 else (0, 255, 0) if fin_class == 1 else (0, 255, 255)
        for point in self.fin_coordinates:
            cv2.circle(frame, (int(point[0]), int(point[1])), radius=3, color=color, thickness=3)

    def _check_orientation_change(self, current_coords: np.ndarray, previous_coords: np.ndarray) -> bool:
        """Check if there has been a significant orientation change between frames, including a 180-degree flip."""
        if current_coords is None or previous_coords is None:
            return False

        if current_coords.shape != previous_coords.shape:
            print("Shape mismatch detected. Possible flip or orientation change.")
            return True

        dist = np.sqrt(np.sum((current_coords - previous_coords) ** 2, axis=1))

        if np.mean(dist) > self.pixel_thresh:
            print("Significant distance change detected. Possible orientation change.")
            return True

        if self._check_flip(current_coords, previous_coords):
            print("Detected 180-degree flip.")
            return True

        return False

    def _check_flip(self, current_coords: np.ndarray, previous_coords: np.ndarray) -> bool:
        """Check if the fin has been flipped 180 degrees."""
        current_center = np.mean(current_coords, axis=0)
        previous_center = np.mean(previous_coords, axis=0)

        current_angle = self._compute_orientation_angle(current_coords)
        previous_angle = self._compute_orientation_angle(previous_coords)

        angle_diff = abs(current_angle - previous_angle)

        if 170 <= angle_diff <= 190:
            mirrored = np.allclose(current_coords - current_center, -(previous_coords - previous_center), atol=1e-2)
            if mirrored:
                print("Detected 180-degree flip based on angle and mirroring.")
                return True

        return False

    def _draw_surface(self, frame: np.ndarray, detections, surface_index: int):
        """Visualize the surface if detected and fin is also detected"""
        if surface_index is not None and self.fin_coordinates is not None:
            surface_coordinates = self._interpolate_polygon_points(
                detections[0].obb.xyxyxyxy.cpu().numpy()[surface_index]
            )
            cv2.polylines(frame, [surface_coordinates.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    def _find_driver(self, frame: np.ndarray, imgsz: int = 1024, conf: float = 0.7) -> list[int]:
        """Find the driver using OBB"""
        detections = self.driver_model(frame, imgsz=imgsz, conf=conf, verbose=False)
        if detections and hasattr(detections[0], 'obb') and len(detections[0].obb.xyxyxyxy.cpu().numpy()) > 0:
            obb = detections[0].obb.xyxyxyxy.cpu().numpy()[0]
            points = self._extract_obb_points(obb)
            c_x = np.mean(points[:, 0])
            c_y = np.mean(points[:, 1])
            for point in points:
                cv2.circle(frame, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 255), thickness=3)
            return [int(c_x), int(c_y)]
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

    def _compute_distance(self, point1: list[int], point2: list[int]) -> float:
        """Compute distance between two points"""
        if point1 and point2:
            return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
        return float('inf')

    def _check_operator(self, frame: np.ndarray, timestamp: float):
        """Determine if operator is moving hands near the driver. Checks driver position relative to fins and flags if close enough."""
        detections = self.fins_model(frame, imgsz=1024, conf=0.3, verbose=False)
        fin_index, surface_index = self._process_detections(detections)
        self._find_fin(frame, detections, fin_index)
        
        # Draw the surface only if the fin is detected
        if fin_index is not None:
            self._draw_surface(frame, detections, surface_index)
        
        driver_coords = self._find_driver(frame)
        hand_coords_list = self._find_hands(frame)

        if driver_coords and hand_coords_list:
            for hand_coords in hand_coords_list:
                driver_hand_distance = self._compute_distance(driver_coords, hand_coords)
                self.driver_hand_distances.append({"Time (seconds)": timestamp, "Driver-Hand Distance (pixels)": driver_hand_distance})

                if driver_hand_distance <= self.driver_hand_thresh:
                    self.frames_with_driver_hand_within_thresh += 1

            distances_to_fin = self._compute_distance_to_fin(driver_coords)
            if len(distances_to_fin) > 0:
                min_distance = np.min(distances_to_fin)
                self.distances.append({"Time (seconds)": timestamp, "Distance (pixels)": min_distance})

            hits = np.sum([d <= self.pixel_thresh for d in distances_to_fin])
            total_fin_points = len(self.fin_coordinates) if self.fin_coordinates is not None else 1
            hit_ratio = hits / total_fin_points
            self.fin_point_hits.append(hit_ratio)

        self.total_frames_checked += 1

    def _compute_distance_to_fin(self, driver_coords: list[int]) -> np.ndarray:
        """Compute distances between the driver and each point on the fin outline"""
        if driver_coords and self.fin_coordinates is not None:
            return np.sqrt(
                np.sum(
                    (np.array(self.fin_coordinates) - np.array(driver_coords)) ** 2, axis=1
                )
            )
        return np.array([])

    def inspectHelicoilDepth(self, frame: np.ndarray, timestamp: float):
        """Analyze each frame where the driver is detected."""
        self._check_operator(frame, timestamp)

    def final_decision(self) -> bool:
        """Make the final decision based on driver-hand proximity and fin points hit."""
        if len(self.fin_point_hits) > 0:
            majority_hits = np.mean(self.fin_point_hits)
            driver_hand_ratio = self.frames_with_driver_hand_within_thresh / self.total_frames_checked
    
            if majority_hits >= self.coverage_perc and driver_hand_ratio >= 0.20:
                return True
            
        return False

    def save_distances_to_csv(self, output_csv_path: str):
        """Save the distances and driver-hand distances to a CSV file"""
        if not self.distances:
            self.distances.append({"Time (seconds)": 0, "Distance (pixels)": np.nan})

        if not self.driver_hand_distances:
            self.driver_hand_distances.append({"Time (seconds)": 0, "Driver-Hand Distance (pixels)": np.nan})

        df = pd.DataFrame(self.distances)
        df_hand = pd.DataFrame(self.driver_hand_distances)
        
        if len(df) > len(df_hand):
            df_hand = pd.concat([df_hand, pd.DataFrame([{"Time (seconds)": np.nan, "Driver-Hand Distance (pixels)": np.nan}] * (len(df) - len(df_hand)))])
        elif len(df_hand) > len(df):
            df = pd.concat([df, pd.DataFrame([{"Time (seconds)": np.nan, "Distance (pixels)": np.nan}] * (len(df_hand) - len(df)))])

        df_combined = pd.concat([df.reset_index(drop=True), df_hand["Driver-Hand Distance (pixels)"].reset_index(drop=True)], axis=1)
        
        df_combined.columns = ["Time (seconds)", "Distance (pixels)", "Driver-Hand Distance (pixels)"]
        
        df_combined.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    i = 0
    helicoil_depth_check = HelicoilDepthCheck("models/fin_detector2.pt", "models/hand_detector.pt", "models/driver2.pt")

    example_video_path = "data/large/correct/066a8c18-7c7b-7951-8000-215fda47e19e-clip.mkv"
    cap = cv2.VideoCapture(example_video_path)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(f"results/video_output_{i}.mkv", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    if not cap.isOpened():
        raise ("Error opening video file")

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)  
    frames_to_skip = int(fps * 0.5)  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frames_to_skip != 0:
            continue

        timestamp = frame_count / fps  
        helicoil_depth_check.inspectHelicoilDepth(frame, timestamp)
        out.write(frame)

    if helicoil_depth_check.final_decision():
        print("Final Decision: Helicoil depth check passed.")
    else:
        print("Final Decision: Helicoil depth check failed.")

    helicoil_depth_check.save_distances_to_csv("distances.csv")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Finished processing video.")
