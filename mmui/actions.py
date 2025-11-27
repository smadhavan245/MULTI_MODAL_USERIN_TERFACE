from collections import deque
from datetime import datetime


class ActionHandler:
    def __init__(self, ui):
        self.ui = ui
        self.current_mode = "none"  # Start with no mode to prevent auto-actions
        self.history = deque(maxlen=50)
        self.selected_app_index = 0
        self.apps = [
            {"name": "Calculator", "icon": "🔢", "path": "calc.exe"},
            {"name": "Notepad", "icon": "📝", "path": "notepad.exe"},
            {"name": "Browser", "icon": "🌐", "path": "chrome.exe"},
            {"name": "Camera", "icon": "📷", "path": "microsoft.windows.camera:"},
            {"name": "Music", "icon": "🎵", "path": "ms-windows-store://"},
        ]
        self.mode_initialized = False  # Track if mode has been explicitly set
        # Initialize app icons after a short delay to ensure UI is ready
        from kivy.clock import Clock
        Clock.schedule_once(lambda dt: self._update_app_icons(), 0.1)
        
    def set_mode(self, mode: str):
        self.current_mode = mode
        self.mode_initialized = True  # Mark that mode has been set
        if hasattr(self.ui, 'ids') and 'status_label' in self.ui.ids:
            self.ui.ids.status_label.text = f"Current Mode: {mode.capitalize()}"
        self._update_app_icons()
        print(f"Mode set to: {mode}")

    def _update_app_icons(self):
        """Initialize and update the app icons display in UI."""
        if not hasattr(self.ui, 'ids') or 'app_grid' not in self.ui.ids:
            return
        
        # Initialize app icons with their icons and names
        for i in range(len(self.apps)):
            app_id = f'app_{i}'
            if app_id in self.ui.ids:
                app = self.apps[i]
                app_button = self.ui.ids[app_id]
                
                # The BoxLayout is a child of the button
                if app_button.children:
                    box_layout = app_button.children[0]
                    # Find labels by checking their id or by position
                    # In vertical BoxLayout, first in KV = last in children list
                    if len(box_layout.children) >= 2:
                        # Try to find by id first
                        icon_label = None
                        name_label = None
                        for label in box_layout.children:
                            if hasattr(label, 'id'):
                                if label.id == 'icon_label':
                                    icon_label = label
                                elif label.id == 'name_label':
                                    name_label = label
                        
                        # If not found by id, use position (icon is larger font, name is smaller)
                        if not icon_label or not name_label:
                            # Larger font = icon, smaller font = name
                            for label in box_layout.children:
                                if hasattr(label, 'font_size'):
                                    if label.font_size >= 40:  # Icon has larger font
                                        icon_label = label
                                    elif label.font_size <= 15:  # Name has smaller font
                                        name_label = label
                        
                        # Set the text
                        if icon_label:
                            icon_label.text = app['icon']
                        if name_label:
                            name_label.text = app['name']

    def handle_action(self, action: str, source: str):
        """Handle actions from any source."""
        # Don't process actions if mode hasn't been initialized
        if not self.mode_initialized or self.current_mode == "none":
            print(f"Ignoring action '{action}' - mode not initialized yet")
            return
        
        # Only process actions from the current active mode (unless auto mode)
        if self.current_mode != "auto" and source != self.current_mode:
            print(f"Ignoring action '{action}' from {source} - current mode is {self.current_mode}")
            return
        
        entry = {
            "action": action,
            "source": source,
            "mode": self.current_mode,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        self.history.appendleft(entry)
        print(f"Action: {action} from {source} in {self.current_mode} mode")
        
        # Handle navigation actions
        if action == "up":
            self._navigate(-3)  # Move up (3 columns)
        elif action == "down":
            self._navigate(3)  # Move down
        elif action == "left":
            self._navigate(-1)  # Move left
        elif action == "right":
            self._navigate(1)  # Move right
        elif action == "select":
            self._open_selected_app()
        elif action == "open":
            self._open_selected_app()
        elif action == "home":
            self.selected_app_index = 0
        elif action == "back":
            pass
        
        # Update UI
        self._update_action_log()
        self._update_selection()

    def _navigate(self, direction: int):
        """Navigate through apps."""
        new_index = self.selected_app_index + direction
        if 0 <= new_index < len(self.apps):
            self.selected_app_index = new_index

    def _open_selected_app(self):
        """Open the currently selected app."""
        # Add confirmation check - require explicit select action
        if not self.mode_initialized:
            print("Cannot open app - mode not initialized")
            return
            
        if 0 <= self.selected_app_index < len(self.apps):
            app = self.apps[self.selected_app_index]
            # Double-check that this is an intentional action
            print(f"✓ SELECT action confirmed - Opening {app['name']}...")
            try:
                import subprocess
                import os
                if os.path.exists(app["path"]) or "://" in app["path"]:
                    subprocess.Popen(app["path"], shell=True)
                    print(f"✓ Successfully opened {app['name']}")
                else:
                    # Try to find app in system
                    subprocess.Popen(f'start {app["path"]}', shell=True)
                    print(f"✓ Opened {app['name']} via start command")
            except Exception as e:
                print(f"✗ Error opening app {app['name']}: {e}")
        else:
            print(f"✗ Invalid app index: {self.selected_app_index}")

    def _update_action_log(self):
        """Update action log in UI with enhanced formatting."""
        if hasattr(self.ui, 'ids') and 'action_log' in self.ui.ids:
            log_text = "Action Log:\n"
            for item in list(self.history)[:15]:  # Show more entries
                source_icon = {
                    "gesture": "👋",
                    "voice": "🎤",
                    "gaze": "👁️",
                    "auto": "🚨"
                }.get(item['source'], "•")
                log_text += f"{source_icon} [{item['timestamp']}] {item['action'].upper()} from {item['source']}\n"
            self.ui.ids.action_log.text = log_text
            self.ui.ids.action_log.height = self.ui.ids.action_log.texture_size[1]

    def _update_selection(self):
        """Update selected app indicator in UI."""
        if hasattr(self.ui, 'ids') and 'selected_label' in self.ui.ids:
            if 0 <= self.selected_app_index < len(self.apps):
                app = self.apps[self.selected_app_index]
                self.ui.ids.selected_label.text = f"Selected: {app['icon']} {app['name']}"
        
        # Update app icon colors with modern design and selection highlight
        if hasattr(self.ui, 'ids') and 'app_grid' in self.ui.ids:
            for i in range(len(self.apps)):
                app_id = f'app_{i}'
                if app_id in self.ui.ids:
                    app_button = self.ui.ids[app_id]
                    if i == self.selected_app_index:
                        # Selected: bright blue with modern gradient effect
                        app_button.background_color = [0.25, 0.55, 0.95, 1]
                        # Add visual feedback - slightly brighter
                        if hasattr(app_button, 'canvas'):
                            # The border will be highlighted via the canvas.after
                            pass
                    else:
                        # Unselected: subtle modern gray
                        app_button.background_color = [0.25, 0.3, 0.4, 0.9]
