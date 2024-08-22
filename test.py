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
        
        # Calculate width and height
        width = rect[1][0]
        height = rect[1][1]

        # Print the size, width, and height of the box created
        print(f"Detected box area: {new_box_area} pixels")
        print(f"Box width: {width} pixels, Box height: {height} pixels")

        # Filter based on size condition
        if self.min_surface_size <= new_box_area <= self.max_surface_size:
            if self.previous_box is None or (self.hand_far_from_fin and self._should_replace_box(box)):
                self.previous_box = box
                print("Updated the previous box based on hand distance and surface size.")
            
            # Draw the box with metallic silver (more white) color
            metallic_silver_color = (192, 192, 192)  # RGB for metallic silver with a white tint
            cv2.drawContours(frame, [self.previous_box], 0, metallic_silver_color, 2)
            print("Top surface detected and annotated.")
        else:
            print(f"Detected box area {new_box_area} is out of the accepted range ({self.min_surface_size}-{self.max_surface_size}).")

            # Draw the previous box if the new one isn't valid
            if self.previous_box is not None:
                metallic_silver_color = (192, 192, 192)
                cv2.drawContours(frame, [self.previous_box], 0, metallic_silver_color, 2)
                print("Persisting previous yellow box due to size constraint.")

