from threading import Thread
import time
from collections import deque

# Import OpenCV and MediaPipe independently so fallback works when MediaPipe fails
try:
    import cv2  # type: ignore
except Exception as e:
    print(f"Warning: OpenCV not available: {e}")
    cv2 = None

try:
    import mediapipe as mp  # type: ignore
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"Warning: MediaPipe not available: {e}")
    mp = None
    MEDIAPIPE_AVAILABLE = False


class GestureController:
    def __init__(self, action_handler):
        self.action_handler = action_handler
        self.running = False
        # Available if we at least have OpenCV/camera access
        self.available = cv2 is not None
        self.use_mediapipe = False
        self.last_gesture_time = 0
        self.last_gesture = None
        self.debounce_delay = 0.2  # Increased to prevent false triggers
        self.tap_buffer = deque(maxlen=3)  # Reduced buffer size
        self.tap_window = 0.5  # Increased window - need clearer double tap
        self.frame_skip = 0  # Frame skipping counter for faster processing
        self.min_movement_threshold = 0.05  # Minimum movement to register gesture
        self.accuracy = 0.0
        self.detection_count = 0
        self.success_count = 0
        # Fallback motion tracking state
        self._prev_gray = None
        self._prev_center = None
        self._motion_centers = deque(maxlen=8)
        self.bgsub = None

        if MEDIAPIPE_AVAILABLE and mp is not None:
            try:
                self.mp_hands = mp.solutions.hands  # type: ignore
                self.hands = self.mp_hands.Hands(  # type: ignore
                    max_num_hands=2,
                    model_complexity=0,  # Reduced for lower latency
                    min_detection_confidence=0.4,  # Lower threshold for faster detection
                    min_tracking_confidence=0.4,  # Lower threshold for faster tracking
                )
                self.use_mediapipe = True
                self.available = True
            except Exception as e:
                print(f"Warning: Could not initialize MediaPipe: {e}")
                # Keep available True if OpenCV is present; we’ll use fallback
                self.use_mediapipe = False

    def start(self):
        if self.running:
            return
        if not self.available:
            print("Gesture controller not available - No camera access")
            return
        self.running = True
        Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _loop(self):
        if cv2 is None:
            print("ERROR: OpenCV not available. Camera cannot be accessed.")
            return

        if self.use_mediapipe:
            self._loop_mediapipe()
        else:
            print("MediaPipe not available. Using camera-only fallback for basic gestures.")
            self._loop_fallback()

    def _loop_mediapipe(self):
        cap = None
        try:
            # Try multiple camera indices - prefer index 0 for gesture detection
            for camera_idx in [0, 1, 2]:
                try:
                    cap = cv2.VideoCapture(camera_idx)  # type: ignore
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret:
                            print(f"Camera {camera_idx} opened successfully for gesture detection!")
                            break
                        else:
                            cap.release()
                            cap = None
                    else:
                        if cap:
                            cap.release()
                        cap = None
                except Exception as e:
                    print(f"Error opening camera {camera_idx}: {e}")
                    if cap:
                        cap.release()
                    cap = None
                    continue

            if cap is None or not cap.isOpened():
                print("ERROR: Could not open any camera. Check camera permissions and connections.")
                return

            print("Gesture detection started (MediaPipe). Show your hand to the camera.")
            while self.running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    print("Warning: Could not read frame from camera")
                    break
                
                # Skip frames for faster processing (process every 2nd frame)
                self.frame_skip = (self.frame_skip + 1) % 2
                if self.frame_skip != 0:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
                results = self.hands.process(frame_rgb)  # type: ignore
                if results.multi_hand_landmarks:
                    for lms in results.multi_hand_landmarks:
                        gesture = self._detect_gesture(lms.landmark)
                        if gesture:
                            self._handle_gesture(gesture)
                # Reduced sleep for lower latency
                time.sleep(0.005)
        except Exception as e:
            print(f"Error in gesture loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cap is not None:
                cap.release()
                print("Camera released")

    def _loop_fallback(self):
        """Fallback motion-based gesture detection: left/right/up/down from movement."""
        cap = None
        try:
            for camera_idx in [0, 1, 2]:
                cap = cv2.VideoCapture(camera_idx)  # type: ignore
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        print(f"Camera {camera_idx} opened (fallback mode)")
                        break
                    cap.release()
                    cap = None

            if cap is None or not cap.isOpened():
                print("ERROR: Could not open camera for fallback mode.")
                return

            print("Gesture detection started (fallback). Move your hand left/right/up/down.")
            self._prev_gray = None
            self._prev_center = None
            self._motion_centers.clear()
            # Initialize background subtractor if available
            try:
                self.bgsub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)  # type: ignore
            except Exception:
                self.bgsub = None

            while self.running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    print("Warning: Could not read frame from camera")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore
                gray = cv2.GaussianBlur(gray, (7, 7), 0)  # type: ignore

                # Motion mask
                if self.bgsub is not None:
                    fgmask = self.bgsub.apply(gray)  # type: ignore
                    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)  # type: ignore
                else:
                    if self._prev_gray is None:
                        self._prev_gray = gray
                        continue
                    diff = cv2.absdiff(self._prev_gray, gray)  # type: ignore
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)  # type: ignore
                thresh = cv2.dilate(thresh, None, iterations=2)  # type: ignore
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # type: ignore

                center = None
                if contours:
                    # Pick largest moving area
                    cnt = max(contours, key=cv2.contourArea)  # type: ignore
                    if cv2.contourArea(cnt) > 800:  # minimum area to ignore noise
                        x, y, w, h = cv2.boundingRect(cnt)  # type: ignore
                        center = (x + w // 2, y + h // 2)
                        self._motion_centers.append(center)

                # Direction inference using recent centers
                if len(self._motion_centers) >= 4:
                    start = self._motion_centers[0]
                    end = self._motion_centers[-1]
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    mag = (dx * dx + dy * dy) ** 0.5
                    if mag > 40:  # significant movement
                        if abs(dx) > abs(dy):
                            gesture = "right" if dx > 0 else "left"
                        else:
                            gesture = "down" if dy > 0 else "up"
                        self._handle_gesture(gesture)
                        print(f"Fallback gesture detected: {gesture} (dx={dx}, dy={dy}, mag={mag:.1f})")
                        self._motion_centers.clear()

                self._prev_gray = gray
                # Reduced sleep for lower latency
                time.sleep(0.005)
        except Exception as e:
            print(f"Error in fallback loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cap is not None:
                cap.release()
                print("Camera released (fallback)")
    
    def _loop_camera_only(self):
        """Fallback: Show camera feed even without MediaPipe."""
        if cv2 is None:
            return
        cap = None
        try:
            for camera_idx in [0, 1, 2]:
                cap = cv2.VideoCapture(camera_idx)  # type: ignore
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        print(f"Camera {camera_idx} is accessible (MediaPipe disabled)")
                        break
                    cap.release()
                    cap = None
            
            if cap and cap.isOpened():
                print("Camera feed available but MediaPipe processing disabled")
                cap.release()
        except Exception as e:
            print(f"Camera check error: {e}")
        finally:
            if cap is not None:
                cap.release()

    def get_accuracy(self):
        """Return accuracy percentage for this mode."""
        if self.detection_count > 0:
            self.accuracy = (self.success_count / self.detection_count) * 100
        return self.accuracy

    def _handle_gesture(self, gesture):
        """Handle gesture with debouncing to prevent rapid firing."""
        current_time = time.time()
        
        # Only process if enough time has passed since last gesture
        time_since_last = current_time - self.last_gesture_time
        
        # Check for double tap - must be same gesture within tap window
        if gesture == self.last_gesture and time_since_last < self.tap_window and time_since_last > 0.1:
            # This is a potential double tap
            self.tap_buffer.append(current_time)
            if len(self.tap_buffer) >= 2:
                recent_taps = list(self.tap_buffer)[-2:]
                time_between = recent_taps[1] - recent_taps[0]
                # Double tap must be within 0.25-0.5 seconds (stricter)
                if 0.25 < time_between < self.tap_window:
                    self.success_count += 1
                    self.action_handler.handle_action("select", "gesture")
                    print(f"✓✓ Double tap CONFIRMED: {gesture} (time: {time_between:.2f}s)")
                    self.tap_buffer.clear()  # Clear buffer after double tap
                    self.last_gesture_time = current_time
                    self.last_gesture = None  # Reset to prevent triple tap
                    return
                else:
                    # Not a valid double tap - clear buffer
                    if time_between > self.tap_window:
                        self.tap_buffer.clear()
        
        # Regular gesture navigation - only if enough time passed
        if gesture != self.last_gesture or time_since_last >= self.debounce_delay:
            self.detection_count += 1
            self.action_handler.handle_action(gesture, "gesture")
            self.success_count += 1
            print(f"Gesture: {gesture}")
            # Clear tap buffer on new gesture type
            if gesture != self.last_gesture:
                self.tap_buffer.clear()
            self.last_gesture = gesture
            self.last_gesture_time = current_time

    def _detect_gesture(self, lm):
        if not self.available or not lm:
            return None
        # replicate your JS logic:
        # index finger only → direction from wrist
        thumb = lm[4]
        index = lm[8]
        middle = lm[12]
        ring = lm[16]
        pinky = lm[20]
        wrist = lm[0]

        thumb_up = thumb.y < lm[3].y
        index_up = index.y < lm[6].y
        middle_up = middle.y < lm[10].y
        ring_up = ring.y < lm[14].y
        pinky_up = pinky.y < lm[18].y

        if index_up and not (middle_up or ring_up or pinky_up):
            dx = index.x - wrist.x
            dy = index.y - wrist.y
            # Add minimum movement threshold to prevent false triggers
            if abs(dx) > self.min_movement_threshold or abs(dy) > self.min_movement_threshold:
                if abs(dx) > abs(dy):
                    return "right" if dx > 0 else "left"
                else:
                    return "down" if dy > 0 else "up"

        if thumb_up and index_up and middle_up and ring_up and pinky_up:
            return "home"

        if not (thumb_up or index_up or middle_up or ring_up or pinky_up):
            return "back"

        return None
