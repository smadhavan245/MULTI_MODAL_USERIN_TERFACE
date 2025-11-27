from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle, Color
from kivy.clock import Clock
import cv2
import numpy as np


class CameraPreview(Widget):
    def __init__(self, camera_index=0, **kwargs):
        super().__init__(**kwargs)
        self.camera_index = camera_index
        self.cap = None
        self.texture = None
        self.running = False
        self._update_event = None
        self._rect = None
        
    def start(self):
        """Start the camera preview."""
        if self.running:
            print("Camera preview already running")
            return
            
        try:
            # Try different camera indices
            for idx in [0, 1, 2]:
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    ret, _ = self.cap.read()
                    if ret:
                        self.camera_index = idx
                        print(f"Camera preview opened on camera {idx}")
                        break
                    else:
                        self.cap.release()
                        self.cap = None
                else:
                    if self.cap:
                        self.cap.release()
                    self.cap = None
            
            if not self.cap or not self.cap.isOpened():
                print(f"Could not open any camera for preview")
                return
                
            self.running = True
            # Update at 20 FPS to reduce load
            self._update_event = Clock.schedule_interval(self._update, 1.0 / 20.0)
            print(f"Camera preview started successfully")
        except Exception as e:
            print(f"Error starting camera preview: {e}")
            import traceback
            traceback.print_exc()
            
    def stop(self):
        """Stop the camera preview."""
        self.running = False
        if self._update_event:
            self._update_event.cancel()
            self._update_event = None
        if self.cap:
            self.cap.release()
            self.cap = None
        print("Camera preview stopped")
        
    def _update(self, dt):
        """Update the camera frame."""
        if not self.running or not self.cap:
            return
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                return
        except Exception as e:
            print(f"Error reading camera frame: {e}")
            return
            
        # Resize frame to fit widget size
        if self.size[0] > 0 and self.size[1] > 0:
            h, w = frame.shape[:2]
            widget_w, widget_h = self.size
            
            # Calculate aspect ratio
            frame_aspect = w / h
            widget_aspect = widget_w / widget_h
            
            if frame_aspect > widget_aspect:
                # Frame is wider, fit to width
                new_w = int(widget_w)
                new_h = int(widget_w / frame_aspect)
            else:
                # Frame is taller, fit to height
                new_h = int(widget_h)
                new_w = int(widget_h * frame_aspect)
                
            if new_w > 0 and new_h > 0:
                frame = cv2.resize(frame, (new_w, new_h))
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Flip horizontally for mirror effect
                frame_rgb = cv2.flip(frame_rgb, 1)
                
                # Create texture
                if self.texture is None or self.texture.size[0] != new_w or self.texture.size[1] != new_h:
                    self.texture = Texture.create(size=(new_w, new_h), colorfmt='rgb')
                    self.texture.flip_vertical()
                    # Create rectangle for drawing
                    with self.canvas:
                        Color(1, 1, 1, 1)
                        self._rect = Rectangle(texture=self.texture, pos=self.pos, size=(new_w, new_h))
                
                # Update texture
                if self.texture:
                    self.texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
                    # Update rectangle position and size
                    if self._rect:
                        # Center the preview in the widget
                        x = self.x + (self.width - new_w) / 2
                        y = self.y + (self.height - new_h) / 2
                        self._rect.pos = (x, y)
                        self._rect.size = (new_w, new_h)
                    self.canvas.ask_update()
                
    def on_size(self, *args):
        """Handle size changes."""
        if self.texture:
            self.texture = None  # Force recreation on size change

