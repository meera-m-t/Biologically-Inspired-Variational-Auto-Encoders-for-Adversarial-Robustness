import cv2
import numpy as np
from ultralytics import YOLO
from bytetrack import BYTETracker

class HelicoilDepthCheck:
    def __init__(self, fins_detector_model_path: str, driver_detector_model_path: str, pixel_thresh: int = 700, num_holes: int = 14):
        self.fins_model = YOLO(fins_detector_model_path)  # YOLO for detecting fins/screw holes
        self.driver_model = YOLO(driver_detector_model_path)  # YOLO for detecting driver (hands/tools)
        self.tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)  # ByteTrack for tracking
        self.screw_hole_hits = set()  # Set to track unique screw holes hit
        self.pixel_thresh = pixel_thresh  # Pixel distance threshold for hit detection
        self.driver_coords = None
        self.screw_hole_coords = []
        self.num_holes = num_holes  # Total number of screw holes
        self.driver_id = None  # To track driver using ByteTrack

    def _euclidean_distance(self, point1, point2):
        """Compute the Euclidean distance between two points."""
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

    def _process_fins_detections(self, detections):
        """Process fin detections to extract fin holes (class 4)."""
        screw_hole_indices = []

        classes = detections[0].boxes.cls.cpu().numpy()  # YOLOv8 outputs classes

        for i, cls in enumerate(classes):
            if cls == 4:  # Class 4: Screw holes (fin holes)
                screw_hole_indices.append(i)

        return screw_hole_indices

    def _process_driver_detections(self, detections):
        """Process driver detections to find the driver (hands/tools)."""
        driver_index = None

        classes = detections[0].boxes.cls.cpu().numpy()  # YOLOv8 outputs classes

        for i, cls in enumerate(classes):
            if cls == 1:  # Class 0: Driver
                print("driver")
                driver_index = i

        return driver_index

    def _track_driver(self, frame, detections):
        """Track driver using ByteTrack and return driver coordinates with tracking ID."""
        tracks = self.tracker.update(detections[0], frame)

        for track in tracks:
            if track.cls == 1:  # Class 0: Driver
                self.driver_id = track.track_id  # Track driver ID
                driver_box = track.xyxy.cpu().numpy()  # Driver bounding box
                driver_coords = [(driver_box[0] + driver_box[2]) / 2, (driver_box[1] + driver_box[3]) / 2]  # Center point
                return driver_coords

        return None  # No driver found

    def _find_driver(self, frame):
        """Detect the driver in the frame using YOLOv8."""
        detections = self.driver_model(frame, imgsz=1024, conf=0.3)
        driver_index = self._process_driver_detections(detections)
        return driver_index, detections

    def _find_fins(self, frame):
        """Detect fins and screw holes in the frame using YOLOv8."""
        detections = self.fins_model(frame, imgsz=1024, conf=0.3)
        screw_hole_indices = self._process_fins_detections(detections)
        return screw_hole_indices, detections

    def _draw_objects(self, frame, driver_detections, driver_index, screw_hole_indices, fin_detections):
        """Draw detected driver and screw holes on the frame."""
        # Draw driver
        if driver_index is not None:
            driver_box = driver_detections[0].boxes.xyxy.cpu().numpy()[driver_index]
            self.driver_coords = [(driver_box[0] + driver_box[2]) / 2, (driver_box[1] + driver_box[3]) / 2]  # Center point
            cv2.rectangle(frame, (int(driver_box[0]), int(driver_box[1])), (int(driver_box[2]), int(driver_box[3])),
                          color=(255, 0, 0), thickness=2)
            cv2.putText(frame, f'Driver ID: {self.driver_id}', (int(driver_box[0]), int(driver_box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw screw holes (class 4)
        self.screw_hole_coords = []
        for idx in screw_hole_indices:
            screw_hole_box = fin_detections[0].boxes.xyxy.cpu().numpy()[idx]
            self.screw_hole_coords.append([(screw_hole_box[0] + screw_hole_box[2]) / 2,
                                           (screw_hole_box[1] + screw_hole_box[3]) / 2])  # Center point
            cv2.rectangle(frame, (int(screw_hole_box[0]), int(screw_hole_box[1])),
                          (int(screw_hole_box[2]), int(screw_hole_box[3])),
                          color=(0, 255, 0), thickness=2)
            cv2.putText(frame, 'Fin Hole', (int(screw_hole_box[0]), int(screw_hole_box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _check_driver_screw_hole_hit(self):
        """Check the distance between the driver and screw holes, and track unique holes hit."""
        if self.driver_coords is not None:
            for idx, screw_hole_point in enumerate(self.screw_hole_coords):
                distance = self._euclidean_distance(self.driver_coords, screw_hole_point)
                if distance <= self.pixel_thresh:  # Threshold for a "hit"
                    self.screw_hole_hits.add(idx)  # Add screw hole index to the set of hits

    def inspectHelicoilDepth(self, frame):
        """Analyze each frame to detect driver, screw holes, and compute hits."""
        # Detect driver (hands/tools)
        driver_index, driver_detections = self._find_driver(frame)

        # Track the driver using ByteTrack and retrieve the driver center coordinates
        self.driver_coords = self._track_driver(frame, driver_detections)

        # Detect fin holes (class 4)
        screw_hole_indices, fin_detections = self._find_fins(frame)

        # Draw objects (driver and screw holes) on the frame
        self._draw_objects(frame, driver_detections, driver_index, screw_hole_indices, fin_detections)

        # Check for hits between driver and screw holes
        self._check_driver_screw_hole_hit()

    def print_hit_count(self):
        """Print the number of unique screw holes hit."""
        print(f"Unique screw holes hit: {len(self.screw_hole_hits)} out of {self.num_holes}")
        if len(self.screw_hole_hits) == self.num_holes:
            print("Driver successfully hit all screw holes!")
        else:
            print(f"Driver missed {self.num_holes - len(self.screw_hole_hits)} screw holes.")


if __name__ == "__main__":
    helicoil_depth_check = HelicoilDepthCheck("models/holes.pt", "models/driver-caliper.pt", pixel_thresh=700)

    example_video_path = "data/large/correct/066a8c18-7c7b-7951-8000-215fda47e19e-clip.mkv"
    cap = cv2.VideoCapture(example_video_path)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(f"results/video_output.mkv", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    if not cap.isOpened():
        raise Exception("Error opening video file")

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * 0.5)  # Process every half second

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frames_to_skip != 0:
            continue

        helicoil_depth_check.inspectHelicoilDepth(frame)
        out.write(frame)

    helicoil_depth_check.print_hit_count()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Finished processing video.")
