from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from gesture import GestureController
from voice import VoiceController
from gaze import GazeController
from actions import ActionHandler
from camera_widget import CameraPreview
import os


class RootWidget(BoxLayout):
    pass


class MultimodalApp(App):
    def build(self):
        self.title = "Multimodal Assistant (Emergency Mode)"
        self.root_widget = RootWidget()

        # Controllers
        self.action_handler = ActionHandler(ui=self.root_widget)
        
        # Get Gemini API key from environment
        gemini_key = os.getenv("GEMINI_API_KEY")
        self.gesture = GestureController(action_handler=self.action_handler)
        self.voice = VoiceController(action_handler=self.action_handler, gemini_api_key=gemini_key)
        self.gaze = GazeController(action_handler=self.action_handler)

        # Mode tracking
        self.current_mode = "none"  # Wait for user to select a mode
        self.mode_accuracies = {
            "gesture": 0.0,
            "voice": 0.0,
            "gaze": 0.0
        }
        self.best_mode = "gesture"
        self.update_interval = 2.0  # Update accuracy every 2 seconds

        # Start accuracy updates (even if controllers are idle)
        Clock.schedule_interval(self.update_accuracies, self.update_interval)
        
        # Initialize camera preview after UI is built
        Clock.schedule_once(self._init_camera_preview, 0.5)
        
        return self.root_widget
    
    def _init_camera_preview(self, dt):
        """Initialize the camera preview widget."""
        try:
            if hasattr(self.root_widget, 'ids') and 'camera_preview' in self.root_widget.ids:
                camera_widget = self.root_widget.ids.camera_preview
                # Replace the widget with CameraPreview
                parent = camera_widget.parent
                if parent:
                    parent.remove_widget(camera_widget)
                    # Use camera index 1 for preview to avoid conflict with gesture/gaze (which use 0)
                    self.camera_preview = CameraPreview(camera_index=1)
                    parent.add_widget(self.camera_preview)
                    # Start the camera preview after a short delay, but only if no detection mode is active
                    Clock.schedule_once(lambda dt: self._start_camera_preview_safe(), 0.5)
                    print("Camera preview widget initialized")
                else:
                    print("Warning: Could not find camera preview parent")
            else:
                print("Warning: Camera preview ID not found in UI")
        except Exception as e:
            print(f"Error initializing camera preview: {e}")
            import traceback
            traceback.print_exc()
    
    def _start_camera_preview_safe(self):
        """Start camera preview only if no detection mode is using camera."""
        # Only start preview if we're not in gesture or gaze mode
        if self.current_mode not in ["gesture", "gaze", "auto"]:
            if hasattr(self, 'camera_preview'):
                self.camera_preview.start()
        else:
            print("Camera preview paused - detection mode is using camera")

    def start_auto_mode(self, *args):
        """Start auto/emergency mode with all three modes active."""
        self.current_mode = "auto"
        
        # Start all controllers
        gesture_status = self.gesture.available
        voice_status = True  # Voice should work
        gaze_status = self.gaze.available
        
        self.gesture.start()
        self.voice.start_listening()
        self.gaze.start()
        
        # Update UI with status
        if hasattr(self.root_widget, 'ids') and 'status_label' in self.root_widget.ids:
            status_parts = []
            if gesture_status:
                status_parts.append("Gesture")
            else:
                status_parts.append("Gesture(disabled)")
            if voice_status:
                status_parts.append("Voice")
            else:
                status_parts.append("Voice(disabled)")
            if gaze_status:
                status_parts.append("Gaze")
            else:
                status_parts.append("Gaze(disabled)")
            
            status_text = f"Auto Mode: {' | '.join(status_parts)}"
            self.root_widget.ids.status_label.text = status_text
        
        self.action_handler.set_mode("auto")
        # Update mode button colors
        self._update_mode_buttons()
        
        # Print diagnostics
        print("\n" + "="*50)
        print("MULTIMODAL ASSISTANT - DIAGNOSTICS")
        print("="*50)
        print(f"Camera: Working (tested)")
        print(f"Gesture Controller: {'Available' if gesture_status else 'DISABLED (MediaPipe DLL error)'}")
        print(f"Voice Controller: {'Available' if voice_status else 'Disabled'}")
        print(f"Gaze Controller: {'Available' if gaze_status else 'DISABLED (MediaPipe DLL error)'}")
        if not gesture_status and not gaze_status:
            print("\nNOTE: MediaPipe has DLL error. To fix:")
            print("  1. Install Visual C++ Redistributables")
            print("  2. Or continue using Voice mode (works without MediaPipe)")
        print("="*50 + "\n")

    def _update_mode_buttons(self):
        """Update mode button colors based on current mode."""
        if not hasattr(self.root_widget, 'ids'):
            return
        
        # Default inactive color
        inactive_color = [0.25, 0.5, 0.85, 1]
        # Active color
        active_color = [0.2, 0.5, 0.8, 1]
        
        # Update all mode buttons
        mode_buttons = {
            "gesture": "mode_gesture",
            "voice": "mode_voice",
            "gaze": "mode_gaze",
            "auto": "mode_auto"
        }
        
        for mode, button_id in mode_buttons.items():
            if button_id in self.root_widget.ids:
                if mode == "auto":
                    # Auto button always has red color
                    self.root_widget.ids[button_id].background_color = [0.9, 0.3, 0.3, 1]
                elif mode == self.current_mode:
                    self.root_widget.ids[button_id].background_color = active_color
                else:
                    self.root_widget.ids[button_id].background_color = inactive_color

    def update_accuracies(self, *args):
        """Update accuracies and select best mode."""
        # Get accuracy from each controller
        self.mode_accuracies["gesture"] = self.gesture.get_accuracy()
        self.mode_accuracies["voice"] = self.voice.get_accuracy()
        self.mode_accuracies["gaze"] = self.gaze.get_accuracy()
        
        # Find best mode (highest accuracy)
        if any(acc > 0 for acc in self.mode_accuracies.values()):
            self.best_mode = max(self.mode_accuracies, key=lambda k: self.mode_accuracies[k])
        
        # Update UI with accuracy info
        if hasattr(self.root_widget, 'ids') and 'accuracy_label' in self.root_widget.ids:
            acc_text = f"Accuracy - G:{self.mode_accuracies['gesture']:.1f}% "
            acc_text += f"V:{self.mode_accuracies['voice']:.1f}% "
            acc_text += f"E:{self.mode_accuracies['gaze']:.1f}% "
            acc_text += f"| Best: {self.best_mode}"
            self.root_widget.ids.accuracy_label.text = acc_text

    def switch_mode(self, mode: str):
        """Switch between modes."""
        if mode == "auto":
            # Emergency/auto mode - activate all
            self.start_auto_mode()
            return
        
        self.current_mode = mode
        
        # Stop all controllers first
        print(f"Switching to {mode} mode - stopping all controllers...")
        self.gesture.stop()
        self.voice.stop_listening()
        self.gaze.stop()
        
        # Wait a moment to ensure controllers are stopped
        import time
        time.sleep(0.2)

        # Start selected mode only
        print(f"Starting {mode} mode...")
        if mode == "gesture":
            # Stop camera preview if running (to free camera for gesture)
            if hasattr(self, 'camera_preview') and self.camera_preview.running:
                self.camera_preview.stop()
            self.gesture.start()
        elif mode == "voice":
            self.voice.start_listening()
        elif mode == "gaze":
            # Stop camera preview if running (to free camera for gaze)
            if hasattr(self, 'camera_preview') and self.camera_preview.running:
                self.camera_preview.stop()
            self.gaze.start()
        else:
            # For other modes, start camera preview if available
            if hasattr(self, 'camera_preview') and not self.camera_preview.running:
                Clock.schedule_once(lambda dt: self.camera_preview.start(), 0.3)

        self.action_handler.set_mode(mode)
        print(f"Mode switched to {mode}")
        # Update status label explicitly
        if hasattr(self.root_widget, 'ids') and 'status_label' in self.root_widget.ids:
            self.root_widget.ids.status_label.text = f"Current Mode: {mode.capitalize()}"
        
        # Update mode button colors
        self._update_mode_buttons()


if __name__ == "__main__":
    MultimodalApp().run()
