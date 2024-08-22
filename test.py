import cv2
import numpy as np
from ultralytics import YOLO

class HelicoilDepthCheck:
    def __init__(self, fins_detector_model_path, hand_detector_model_path, driver_detector_model_path,
                 interpolation_points=2, pixel_thresh=120, driver_hand_thresh=800, hand_far_thresh=5000):
        self.fins_model = self._load_model(fins_detector_model_path)
        self.hand_model = self._load_model(hand_detector_model_path)
        self.driver_model = self._load_model(driver_detector_model_path)
        self.interpolation_points = interpolation_points
        self.pixel_thresh = pixel_thresh
        self.driver_hand_thresh = driver_hand_thresh
        self.hand_far_thresh = hand_far_thresh
        self.fin_coordinates = None
        self.previous_box = None
        self.hand_close_to_fin = False

    def _load_model(self, model_path):
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
        else:
            self.fin_coordinates = None
            print("No fins detected.")

    def _find_driver(self, frame, imgsz=1024, conf=0.25):
        detections = self.driver_model(frame, imgsz=imgsz, conf=conf, verbose=False)
        if detections and hasattr(detections[0], 'obb') and len(detections[0].obb.xyxyxyxy.cpu().numpy()) > 0:
            points = detections[0].obb.xyxyxyxy.cpu().numpy()[0].reshape(-1, 2)
            c_x = np.mean(points[:, 0])
            c_y = np.mean(points[:, 1])
            return [int(c_x), int(c_y)]
        return []

    def _find_hands(self, frame, imgsz=640, conf=0.25):
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

    def _compute_distance(self, point1, point2):
        if point1 and point2:
            return np.linalg.norm(np.array(point1) - np.array(point2))
        return float('inf')

    def _detect_top_surface(self, frame):
        """Detect the top surface of the fin and annotate it"""
        if self.fin_coordinates and self.hand_close_to_fin:
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

                # Ensure the yellow box is within the fin's area and aligned with its orientation
                if self.previous_box is None or (self._is_box_within_fin(box) and self._should_replace_box(box)):
                    self.previous_box = box

                if self.previous_box is not None:
                    cv2.drawContours(frame, [self.previous_box], 0, (0, 255, 255), 2)
        else:
            self.previous_box = None

    def _should_replace_box(self, box):
        if self.previous_box is None:
            return True

        new_box_orientation = cv2.minAreaRect(box)[2]
        prev_box_orientation = cv2.minAreaRect(self.previous_box)[2]

        return not np.isclose(new_box_orientation, prev_box_orientation, atol=5)

    def _is_box_within_fin(self, box):
        """Check if the yellow box is within the fin area and has the same orientation"""
        fin_rect = cv2.minAreaRect(np.array(self.fin_coordinates))
        fin_box = cv2.boxPoints(fin_rect)
        fin_box = np.int0(fin_box)

        for point in box:
            if not cv2.pointPolygonTest(fin_box, (point[0], point[1]), False) >= 0:
                return False
        return True

    def inspectHelicoilDepth(self, frame, timestamp):
        self._find_fin(frame)
        driver_coords = self._find_driver(frame)
        hand_coords_list = self._find_hands(frame)

        if hand_coords_list and self.fin_coordinates is not None:
            for hand_coords in hand_coords_list:
                distances_to_fin = self._compute_distance_to_fin(hand_coords)
                if len(distances_to_fin) > 0:
                    min_distance_to_fin = np.min(distances_to_fin)
                    self.hand_close_to_fin = min_distance_to_fin > self.hand_far_thresh
                    if self.hand_close_to_fin:
                        break
        else:
            self.hand_close_to_fin = False

        if self.fin_coordinates and self.hand_close_to_fin:
            self._detect_top_surface(frame)
        elif self.fin_coordinates is None or not self.hand_close_to_fin:
            self.previous_box = None

    def _compute_distance_to_fin(self, coords):
        if coords and self.fin_coordinates is not None:
            return np.sqrt(np.sum((np.array(self.fin_coordinates) - np.array(coords)) ** 2, axis=1))
        return np.array([])

    def save_distances_to_csv(self, output_csv_path):
        pass  # Implement this method as needed

if __name__ == "__main__":
    helicoil_depth_check = HelicoilDepthCheck("models/fin_detector.pt", "models/hand_detector.pt", "models/driver.pt")

    example_video_path = "data/large/correct/Mar-11_ 24_09_16_30-clip.mkv"
    cap = cv2.VideoCapture(example_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output_with_visualization.mkv', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    if not cap.isOpened():
        raise Exception("Error opening video file")

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_count += 1
        timestamp = frame_count / fps

        helicoil_depth_check.inspectHelicoilDepth(frame, timestamp)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
