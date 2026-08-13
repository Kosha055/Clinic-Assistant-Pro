"""
LCD3.5" Display Driver for Clinic Assistant Pro
- 3.5" LCD Display Support (320x480 or 480x320 resolution)
- Real-time vital signs visualization
- ECG waveform rendering
- Touch input support
- Status indicators and alerts

Compatible with: Adafruit PiTFT, Waveshare 3.5" LCD, etc.

Author: Clinic Assistant Pro
License: MIT
"""

import time
import threading
from PIL import Image, ImageDraw, ImageFont
import pygame
import os
from datetime import datetime
from collections import deque


class LCD35Display:
    """
    3.5" LCD Display for Raspberry Pi
    Supports 320x480 or 480x320 resolution
    """
    
    def __init__(self, width=480, height=320, rotate=0):
        """
        Initialize LCD display
        
        Args:
            width: Display width in pixels
            height: Display height in pixels
            rotate: Rotation (0, 90, 180, 270)
        """
        self.width = width
        self.height = height
        self.rotate = rotate
        
        # Try to initialize with pygame first (works with framebuffer)
        os.environ['SDL_VIDEODRIVER'] = 'fbcon'
        os.environ['SDL_FBDEV'] = '/dev/fb0'
        os.environ['SDL_MOUSEDRV'] = 'TSLIB'
        os.environ['SDL_MOUSEDEV'] = '/dev/input/touchscreen'
        
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Clinic Assistant Pro")
            self.use_pygame = True
            print(f"Display initialized with pygame (SDL) - {width}x{height}")
        except:
            print("Warning: pygame/SDL initialization failed, using PIL fallback")
            self.use_pygame = False
        
        # Create drawing surface
        self.image = Image.new('RGB', (width, height), color=(0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)
        
        # Try to load fonts
        self.try_load_fonts()
        
        # Display state
        self.display_mode = "vitals"  # vitals, ecg, settings
        self.last_update = 0
        self.update_interval = 0.5  # 500ms refresh
        
        # ECG waveform buffer for display (last 200 samples)
        self.ecg_display_buffer = deque(maxlen=200)
        
        # Color scheme
        self.colors = {
            'bg': (20, 20, 30),           # Dark blue-black
            'fg': (255, 255, 255),        # White
            'normal': (100, 200, 100),    # Green
            'warning': (255, 200, 0),     # Orange
            'critical': (255, 50, 50),    # Red
            'header': (50, 100, 150),     # Dark blue
            'border': (100, 100, 120),    # Gray-blue
            'ecg_line': (0, 255, 100),    # Cyan-green
            'grid': (40, 40, 50),         # Dark gray
        }
        
    def try_load_fonts(self):
        """Try to load system fonts"""
        font_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        ]
        
        self.fonts = {}
        
        try:
            # Try large font for main values
            for path in font_paths:
                if os.path.exists(path):
                    self.fonts['large'] = ImageFont.truetype(path, 48)
                    break
            if 'large' not in self.fonts:
                raise Exception("Large font not found")
                
            # Medium font for labels
            for path in font_paths:
                if os.path.exists(path):
                    self.fonts['medium'] = ImageFont.truetype(path, 24)
                    break
            if 'medium' not in self.fonts:
                raise Exception("Medium font not found")
                
            # Small font for status
            for path in font_paths:
                if os.path.exists(path):
                    self.fonts['small'] = ImageFont.truetype(path, 16)
                    break
            if 'small' not in self.fonts:
                raise Exception("Small font not found")
                
            print("Fonts loaded successfully")
            
        except:
            print("Warning: TrueType fonts not found, using default")
            self.fonts['large'] = ImageFont.load_default()
            self.fonts['medium'] = ImageFont.load_default()
            self.fonts['small'] = ImageFont.load_default()
    
    def clear_screen(self):
        """Clear display"""
        self.draw.rectangle(
            [(0, 0), (self.width, self.height)],
            fill=self.colors['bg']
        )
    
    def draw_header(self, title, timestamp=None):
        """Draw header with title and timestamp"""
        # Header background
        self.draw.rectangle(
            [(0, 0), (self.width, 40)],
            fill=self.colors['header']
        )
        
        # Title
        self.draw.text(
            (10, 10),
            title,
            font=self.fonts['medium'],
            fill=self.colors['fg']
        )
        
        # Timestamp
        if timestamp:
            time_str = datetime.now().strftime("%H:%M:%S")
            text_bbox = self.draw.textbbox((0, 0), time_str, font=self.fonts['small'])
            text_width = text_bbox[2] - text_bbox[0]
            self.draw.text(
                (self.width - text_width - 10, 12),
                time_str,
                font=self.fonts['small'],
                fill=self.colors['fg']
            )
    
    def get_status_color(self, value, normal_range, warning_range, critical_range):
        """
        Get color based on value ranges
        
        Args:
            value: Current value
            normal_range: (min, max) for normal
            warning_range: (min, max) for warning
            critical_range: (min, max) for critical
        
        Returns:
            Color tuple
        """
        if value is None:
            return self.colors['fg']
        
        if critical_range[0] <= value <= critical_range[1]:
            return self.colors['critical']
        elif warning_range[0] <= value <= warning_range[1]:
            return self.colors['warning']
        elif normal_range[0] <= value <= normal_range[1]:
            return self.colors['normal']
        else:
            return self.colors['warning']
    
    def draw_vital_box(self, x, y, label, value, unit, status_color):
        """
        Draw a vital sign box (temperature, heart rate, etc.)
        
        Args:
            x, y: Position
            label: Text label
            value: Numeric value
            unit: Unit text (°C, bpm, etc.)
            status_color: Color based on status
        """
        box_width = 140
        box_height = 100
        
        # Box border
        self.draw.rectangle(
            [(x, y), (x + box_width, y + box_height)],
            outline=status_color,
            width=3
        )
        
        # Background
        self.draw.rectangle(
            [(x + 2, y + 2), (x + box_width - 2, y + box_height - 2)],
            fill=self.colors['bg']
        )
        
        # Label
        self.draw.text(
            (x + 10, y + 8),
            label,
            font=self.fonts['small'],
            fill=self.colors['fg']
        )
        
        # Value
        if value is not None:
            value_str = f"{value:.1f}" if isinstance(value, float) else str(value)
            self.draw.text(
                (x + 10, y + 32),
                value_str,
                font=self.fonts['large'],
                fill=status_color
            )
            
            # Unit
            self.draw.text(
                (x + 10, y + 75),
                unit,
                font=self.fonts['small'],
                fill=self.colors['fg']
            )
        else:
            self.draw.text(
                (x + 20, y + 35),
                "N/A",
                font=self.fonts['medium'],
                fill=self.colors['fg']
            )
    
    def draw_vitals_screen(self, sensor_data):
        """
        Draw main vitals monitoring screen
        
        Args:
            sensor_data: Dict with temperature, heart_rate, spo2, bpm, etc.
        """
        self.clear_screen()
        self.draw_header("VITAL SIGNS", timestamp=True)
        
        # Extract data
        temp = sensor_data.get('object_temp')
        hr_ecg = sensor_data.get('ecg_bpm')
        leads = sensor_data.get('leads_connected', False)
        
        # Row 1: Temperature and ECG BPM
        # Temperature
        temp_color = self.get_status_color(
            temp,
            normal_range=(36.5, 37.5),
            warning_range=(36.0, 38.0),
            critical_range=(35.5, 38.5)
        )
        self.draw_vital_box(10, 50, "BODY TEMP", temp, "°C", temp_color)
        
        # ECG BPM
        bpm_color = self.get_status_color(
            hr_ecg,
            normal_range=(60, 100),
            warning_range=(50, 120),
            critical_range=(40, 150)
        )
        self.draw_vital_box(
            10 + 160, 50, "HEART RATE", hr_ecg, "bpm", bpm_color
        )
        
        # Row 2: Lead status and additional info
        status_y = 160
        
        # Lead status indicator
        lead_text = "LEADS: " + ("✓ CONNECTED" if leads else "✗ DISCONNECTED")
        lead_color = self.colors['normal'] if leads else self.colors['critical']
        self.draw.text(
            (10, status_y),
            lead_text,
            font=self.fonts['small'],
            fill=lead_color
        )
        
        # Ambient temperature
        amb_temp = sensor_data.get('ambient_temp')
        amb_str = f"Amb: {amb_temp:.1f}°C" if amb_temp else "Amb: N/A"
        self.draw.text(
            (10, status_y + 30),
            amb_str,
            font=self.fonts['small'],
            fill=self.colors['fg']
        )
        
        # Last reading status
        if all([temp is not None, hr_ecg is not None, leads]):
            status_str = "✓ ALL READINGS VALID"
            status_color = self.colors['normal']
        else:
            status_str = "⚠ CHECK SENSORS"
            status_color = self.colors['warning']
        
        self.draw.text(
            (10, status_y + 60),
            status_str,
            font=self.fonts['small'],
            fill=status_color
        )
        
        # Instructions at bottom
        self.draw.text(
            (10, self.height - 25),
            "Swipe for ECG | Settings | Hold sensors in place",
            font=self.fonts['small'],
            fill=self.colors['border']
        )
        
        self.update_display()
    
    def draw_ecg_screen(self, ecg_buffer):
        """
        Draw ECG waveform screen
        
        Args:
            ecg_buffer: Deque of ECG voltage samples
        """
        self.clear_screen()
        self.draw_header("ECG WAVEFORM", timestamp=True)
        
        # ECG display area
        ecg_x = 10
        ecg_y = 50
        ecg_width = self.width - 20
        ecg_height = self.height - 100
        
        # Draw grid
        self.draw_ecg_grid(ecg_x, ecg_y, ecg_width, ecg_height)
        
        # Draw waveform
        if len(ecg_buffer) > 1:
            # Normalize voltage to screen coordinates
            min_v = min(ecg_buffer)
            max_v = max(ecg_buffer)
            range_v = max_v - min_v if max_v > min_v else 1
            
            # Draw line
            points = []
            for i, v in enumerate(ecg_buffer):
                x = ecg_x + (i / len(ecg_buffer)) * ecg_width
                # Invert Y (higher voltage = higher on screen)
                y_norm = (v - min_v) / range_v
                y = ecg_y + ecg_height - (y_norm * ecg_height)
                points.append((int(x), int(y)))
            
            # Draw polyline
            if len(points) > 1:
                for i in range(len(points) - 1):
                    self.draw.line(
                        [points[i], points[i + 1]],
                        fill=self.colors['ecg_line'],
                        width=2
                    )
        
        # Scale information
        info_y = ecg_y + ecg_height + 10
        self.draw.text(
            (ecg_x, info_y),
            f"Samples: {len(ecg_buffer)} | Speed: 25mm/s (approx)",
            font=self.fonts['small'],
            fill=self.colors['fg']
        )
        
        self.draw.text(
            (ecg_x, info_y + 20),
            "Ensure proper electrode placement for best results",
            font=self.fonts['small'],
            fill=self.colors['border']
        )
        
        self.update_display()
    
    def draw_ecg_grid(self, x, y, width, height):
        """Draw ECG grid (like hospital monitor)"""
        # Major grid lines (every 50 pixels)
        grid_spacing = 50
        
        # Vertical lines
        for i in range(0, width, grid_spacing):
            self.draw.line(
                [(x + i, y), (x + i, y + height)],
                fill=self.colors['grid'],
                width=1
            )
        
        # Horizontal lines
        for i in range(0, height, grid_spacing):
            self.draw.line(
                [(x, y + i), (x + width, y + i)],
                fill=self.colors['grid'],
                width=1
            )
        
        # Border
        self.draw.rectangle(
            [(x, y), (x + width, y + height)],
            outline=self.colors['border'],
            width=2
        )
    
    def draw_alerts_screen(self, alerts):
        """
        Draw alerts/warnings screen
        
        Args:
            alerts: List of alert dicts
        """
        self.clear_screen()
        self.draw_header("ALERTS & STATUS", timestamp=True)
        
        if not alerts:
            self.draw.text(
                (self.width // 2 - 50, self.height // 2 - 20),
                "✓ NO ALERTS",
                font=self.fonts['large'],
                fill=self.colors['normal']
            )
        else:
            alert_y = 50
            for alert in alerts[:4]:  # Show max 4 alerts
                # Alert box
                alert_color = self.colors['critical'] if alert.get('level') == 'critical' else self.colors['warning']
                
                self.draw.rectangle(
                    [(10, alert_y), (self.width - 10, alert_y + 50)],
                    outline=alert_color,
                    width=2
                )
                
                # Alert text
                self.draw.text(
                    (20, alert_y + 5),
                    alert.get('message', 'Unknown alert'),
                    font=self.fonts['small'],
                    fill=alert_color
                )
                
                # Timestamp
                time_str = alert.get('time', 'N/A')
                self.draw.text(
                    (20, alert_y + 25),
                    f"Time: {time_str}",
                    font=self.fonts['small'],
                    fill=self.colors['fg']
                )
                
                alert_y += 60
        
        self.update_display()
    
    def update_ecg_buffer(self, voltage):
        """Add ECG sample to display buffer"""
        self.ecg_display_buffer.append(voltage)
    
    def update_display(self):
        """Push image to actual display"""
        if self.use_pygame:
            try:
                # Convert PIL image to pygame surface
                image_str = self.image.tobytes()
                image_surf = pygame.image.fromstring(
                    image_str, self.image.size, self.image.mode
                )
                self.screen.blit(image_surf, (0, 0))
                pygame.display.flip()
            except Exception as e:
                print(f"Error updating pygame display: {e}")
        else:
            # Fallback: save to file (for testing)
            try:
                self.image.save('/tmp/display_output.png')
            except:
                pass
    
    def set_display_mode(self, mode):
        """Set display mode: 'vitals', 'ecg', 'alerts'"""
        if mode in ['vitals', 'ecg', 'alerts']:
            self.display_mode = mode
    
    def render(self, sensor_data, alerts=None):
        """
        Main render function - update display based on mode
        
        Args:
            sensor_data: Current sensor readings
            alerts: List of active alerts
        """
        current_time = time.time()
        
        # Rate limit updates
        if current_time - self.last_update < self.update_interval:
            return
        
        # Update ECG buffer if data available
        if sensor_data.get('ecg_voltage'):
            self.update_ecg_buffer(sensor_data['ecg_voltage'])
        
        # Render based on mode
        if self.display_mode == 'vitals':
            self.draw_vitals_screen(sensor_data)
        elif self.display_mode == 'ecg':
            self.draw_ecg_screen(self.ecg_display_buffer)
        elif self.display_mode == 'alerts':
            self.draw_alerts_screen(alerts or [])
        
        self.last_update = current_time


if __name__ == "__main__":
    print("LCD3.5\" Display Test")
    print("=" * 50)
    
    # Initialize display
    display = LCD35Display(width=480, height=320)
    
    # Test data
    test_data = {
        'object_temp': 36.8,
        'ambient_temp': 24.2,
        'ecg_bpm': 72,
        'ecg_voltage': 1.2,
        'leads_connected': True,
    }
    
    print("Rendering vitals screen...")
    display.render(test_data)
    
    print("Test complete - check /tmp/display_output.png")
