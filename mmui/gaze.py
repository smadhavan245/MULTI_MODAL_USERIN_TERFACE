from threading import Thread
import time
from collections import deque

# Import OpenCV and MediaPipe independently so fallback works when MediaPipe fails
try:
    import cv2
except Exception as e:
    print(f"Warning: OpenCV not available: {e}")
    cv2 = None

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"Warning: MediaPipe not available: {e}")
    MEDIAPIPE_AVAILABLE = False
    mp = None


class GazeController:
    def __init__(self, action_handler):
        self.action_handler = action_handler
        self.running = False
        # Available if camera via OpenCV is present
        self.available = cv2 is not None
        self.last_blink_time = 0
        self.blink_buffer = deque(maxlen=10)  # Track last 10 blinks
        self.blink_threshold = 0.25  # Threshold for eye closure
        self.blink_window = 0.8  # Reduced time window for faster double blink
        self.frame_skip = 0  # Frame skipping counter
        self.accuracy = 0.0
        self.detection_count = 0
        self.success_count = 0
        self.use_mediapipe = False
        # Fallback tracking state
        self._pupil_centers = deque(maxlen=8)
        self._prev_gray = None
        self._debounce_delay = 0.15  # Reduced for lower latency
        self._last_dir_time = 0.0
        self._eye_cascade = None
        self._face_cascade = None

        if MEDIAPIPE_AVAILABLE and mp is not None:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh  # type: ignore
                self.face_mesh = self.mp_face_mesh.FaceMesh(  # type: ignore
                    max_num_faces=1,
                    refine_landmarks=False,  # Disabled for lower latency
                    min_detection_confidence=0.4,  # Lower threshold for faster detection
                    min_tracking_confidence=0.4  # Lower threshold for faster tracking
                )
                # Eye landmarks (left and right)
                self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
                self.available = True
                self.use_mediapipe = True
            except Exception as e:
                print(f"Warning: Could not initialize MediaPipe Face Mesh: {e}")
                # Keep available True if OpenCV is present; fallback will be used
                self.use_mediapipe = False

    def start(self):
        if self.running:
            return
        if not self.available:
            print("Gaze controller not available - No camera access")
            return
        self.running = True
        Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def get_accuracy(self):
        """Return accuracy percentage for this mode."""
        if self.detection_count > 0:
            self.accuracy = (self.success_count / self.detection_count) * 100
        return self.accuracy

    def _calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR) to detect blinks."""
        if not landmarks:
            return None
        
        # Get eye landmark points
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks):
                point = landmarks[idx]
                eye_points.append([point.x, point.y])
        
        if len(eye_points) < 6:
            return None
        
        # Calculate distances
        # Vertical distances
        vertical1 = ((eye_points[1][1] - eye_points[5][1]) ** 2 + (eye_points[1][0] - eye_points[5][0]) ** 2) ** 0.5
        vertical2 = ((eye_points[2][1] - eye_points[4][1]) ** 2 + (eye_points[2][0] - eye_points[4][0]) ** 2) ** 0.5
        # Horizontal distance
        horizontal = ((eye_points[0][0] - eye_points[3][0]) ** 2 + (eye_points[0][1] - eye_points[3][1]) ** 2) ** 0.5
        
        if horizontal == 0:
            return None
        
        # EAR formula
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear

    def _detect_blink(self, landmarks):
        """Detect if eyes are closed (blink)."""
        if not landmarks or len(landmarks) < 468:
            return False
        
        left_ear = self._calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE_INDICES)
        right_ear = self._calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE_INDICES)
        
        if left_ear is None or right_ear is None:
            return False
        
        # Average EAR for both eyes
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Eye is closed if EAR is below threshold
        return avg_ear < self.blink_threshold

    def _detect_gaze_direction(self, landmarks):
        """Detect gaze direction based on eye landmarks."""
        if not landmarks or len(landmarks) < 468:
            return None
        
        # Use nose tip and eye centers for gaze estimation
        try:
            nose_tip = landmarks[4]
            left_eye_center = landmarks[33]
            right_eye_center = landmarks[263]
            
            # Calculate eye center
            eye_center_x = (left_eye_center.x + right_eye_center.x) / 2
            eye_center_y = (left_eye_center.y + right_eye_center.y) / 2
            
            # Calculate offset from nose
            dx = eye_center_x - nose_tip.x
            dy = eye_center_y - nose_tip.y
            
            # More sensitive gaze detection with lower threshold
            threshold = 0.015  # Reduced from 0.02 for better sensitivity
            if abs(dx) > threshold or abs(dy) > threshold:
                if abs(dx) > abs(dy):
                    return "right" if dx > 0 else "left"
                else:
                    return "down" if dy > 0 else "up"
        except (IndexError, AttributeError):
            return None

    def _handle_double_blink(self):
        """Handle double blink detection - open selected app."""
        current_time = time.time()
        self.blink_buffer.append(current_time)
        
        if len(self.blink_buffer) >= 2:
            # Check if last two blinks are within the window
            recent_blinks = list(self.blink_buffer)[-2:]
            if recent_blinks[1] - recent_blinks[0] < self.blink_window:
                self.success_count += 1
                self.action_handler.handle_action("select", "gaze")
                self.last_blink_time = current_time
                return True
        return False

    def _loop(self):
        if cv2 is None:
            print("ERROR: OpenCV not available. Camera cannot be accessed.")
            return
        
        if not MEDIAPIPE_AVAILABLE or not hasattr(self, 'face_mesh'):
            print("WARNING: MediaPipe not available. Using fallback eyeball movement tracking.")
            return self._loop_fallback()
        
        cap = None
        eyes_open = True
        last_blink_detection = time.time()
        
        try:
            # Try multiple camera indices - prefer index 0 for gaze detection
            for camera_idx in [0, 1, 2]:
                try:
                    cap = cv2.VideoCapture(camera_idx)  # type: ignore
                    if cap.isOpened():
                        # Test if we can read a frame
                        ret, test_frame = cap.read()
                        if ret:
                            print(f"Camera {camera_idx} opened successfully for gaze tracking!")
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
                print("ERROR: Could not open any camera for gaze tracking. Check camera permissions.")
                return
            
            print("Gaze tracking started. Look at the camera and blink twice to select.")
            
            while self.running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                
                # Skip frames for faster processing (process every 2nd frame)
                self.frame_skip = (self.frame_skip + 1) % 2
                if self.frame_skip != 0:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
                results = self.face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark
                        
                        # Detect blink
                        is_closed = self._detect_blink(landmarks)
                        current_time = time.time()
                        
                        # Detect blink state change (opening after closing)
                        if not is_closed and not eyes_open:
                            # Eye just opened - potential blink
                            if current_time - last_blink_detection > 0.15:  # Reduced debounce
                                if self._handle_double_blink():
                                    print("Double blink detected!")
                                last_blink_detection = current_time
                        
                        eyes_open = is_closed
                        
                        # Detect gaze direction - only if eyes are open
                        if eyes_open:
                            gaze = self._detect_gaze_direction(landmarks)
                            if gaze:
                                self.detection_count += 1
                                self.action_handler.handle_action(gaze, "gaze")
                                print(f"Gaze: {gaze}")
                
                # Reduced sleep for lower latency
                time.sleep(0.005)
        except Exception as e:
            print(f"Error in gaze loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cap is not None:
                cap.release()
                print("Gaze tracking camera released")

    def _loop_fallback(self):
        """OpenCV-only fallback: detect eyeball movement by tracking pupil center drift."""
        if cv2 is None:
            print("ERROR: OpenCV not available. Cannot run gaze fallback.")
            return
        cap = None
        try:
            # Try multiple camera indices
            for camera_idx in [0, 1, 2]:
                cap = cv2.VideoCapture(camera_idx)  # type: ignore
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        print(f"Camera {camera_idx} opened successfully for fallback gaze tracking!")
                        break
                    cap.release()
                    cap = None

            if cap is None or not cap.isOpened():
                print("ERROR: Could not open any camera for fallback gaze tracking.")
                return

            # Load cascades if available
            try:
                cascade_dir = getattr(cv2, 'data', None)
                base = cascade_dir.haarcascades if cascade_dir and hasattr(cascade_dir, 'haarcascades') else ''
                face_xml = base + 'haarcascade_frontalface_default.xml'
                eye_xml = base + 'haarcascade_eye.xml'
                if face_xml and eye_xml:
                    self._face_cascade = cv2.CascadeClassifier(face_xml)  # type: ignore
                    self._eye_cascade = cv2.CascadeClassifier(eye_xml)  # type: ignore
            except Exception:
                self._eye_cascade = None
                self._face_cascade = None

            print("Fallback gaze tracking started. Move your eyes left/right/up/down.")
            self._pupil_centers.clear()
            self._prev_gray = None
            self._last_dir_time = 0.0

            while self.running and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore
                gray = cv2.equalizeHist(gray)  # type: ignore

                # Detect regions likely containing eyes
                roi_list = []
                if self._face_cascade is not None:
                    faces = self._face_cascade.detectMultiScale(gray, 1.3, 5)  # type: ignore
                    for (x, y, w, h) in faces[:1]:
                        face_roi = gray[y:y+h, x:x+w]
                        if self._eye_cascade is not None:
                            eyes = self._eye_cascade.detectMultiScale(face_roi, 1.2, 10)  # type: ignore
                            for (ex, ey, ew, eh) in eyes[:2]:
                                roi_list.append((x+ex, y+ey, ew, eh))
                        else:
                            roi_list.append((x, y, w, h))
                else:
                    h, w = gray.shape
                    roi_list.append((int(w*0.25), int(h*0.25), int(w*0.5), int(h*0.5)))

                pupil_center = None
                for (rx, ry, rw, rh) in roi_list:
                    eye_roi = gray[ry:ry+rh, rx:rx+rw]
                    eye_blur = cv2.GaussianBlur(eye_roi, (7, 7), 0)  # type: ignore
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(eye_blur)  # type: ignore
                    cx, cy = rx + min_loc[0], ry + min_loc[1]
                    pupil_center = (cx, cy)
                    break

                if pupil_center is not None:
                    self._pupil_centers.append(pupil_center)
                    if len(self._pupil_centers) >= 5:
                        start = self._pupil_centers[0]
                        end = self._pupil_centers[-1]
                        dx = end[0] - start[0]
                        dy = end[1] - start[1]
                        mag = (dx*dx + dy*dy) ** 0.5
                        now = time.time()
                        # Reduced movement threshold for better sensitivity
                        if mag > 8 and (now - self._last_dir_time) >= self._debounce_delay:
                            if abs(dx) > abs(dy):
                                direction = 'right' if dx > 0 else 'left'
                            else:
                                direction = 'down' if dy > 0 else 'up'
                            self.detection_count += 1
                            self.success_count += 1
                            self.action_handler.handle_action(direction, 'gaze')
                            self._last_dir_time = now
                            self._pupil_centers.clear()

                # Reduced sleep for lower latency
                time.sleep(0.005)

        except Exception as e:
            print(f"Error in fallback gaze loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cap is not None:
                cap.release()
                print("Gaze tracking camera released (fallback)")
    
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

