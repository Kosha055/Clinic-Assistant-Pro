"""
Clinic Assistant Pro - Main Application
Complete integration of sensors and LCD35 display
- Real-time vital signs monitoring
- ECG waveform acquisition and analysis
- Advanced BPM calculation with slope detection
- Stable temperature reading with smoothing
- LCD35 display with multiple screens
- Alert management and logging

Author: Clinic Assistant Pro
License: MIT
"""

import time
import threading
import csv
import json
from datetime import datetime
from collections import deque
import traceback

# Import custom modules
from sensor_drivers import SensorAggregator
from lcd_display import LCD35Display


class AlertManager:
    """Manage clinical alerts and thresholds"""
    
    def __init__(self):
        """Initialize alert manager with default thresholds"""
        self.thresholds = {
            'heart_rate': {
                'critical_low': 40,
                'warning_low': 50,
                'warning_high': 120,
                'critical_high': 150,
            },
            'temperature': {
                'critical_low': 35.0,
                'warning_low': 36.0,
                'warning_high': 37.5,
                'critical_high': 38.5,
            },
            'spo2': {
                'critical_low': 90,
                'warning_low': 94,
            },
            'leads': {
                'disconnected': True,  # Alert if leads disconnected
            }
        }
        
        self.active_alerts = []
        self.alert_history = deque(maxlen=100)
    
    def check_alerts(self, sensor_data):
        """
        Check sensor data against thresholds
        
        Args:
            sensor_data: Current sensor readings dict
        
        Returns:
            List of active alerts
        """
        self.active_alerts = []
        
        # Check heart rate (from ECG BPM)
        bpm = sensor_data.get('ecg_bpm')
        if bpm is not None:
            if bpm <= self.thresholds['heart_rate']['critical_low']:
                self.add_alert(
                    f"CRITICAL: Heart rate critically low ({bpm:.0f} bpm)",
                    'critical',
                    'heart_rate'
                )
            elif bpm < self.thresholds['heart_rate']['warning_low']:
                self.add_alert(
                    f"WARNING: Heart rate low ({bpm:.0f} bpm)",
                    'warning',
                    'heart_rate'
                )
            elif bpm >= self.thresholds['heart_rate']['critical_high']:
                self.add_alert(
                    f"CRITICAL: Heart rate critically high ({bpm:.0f} bpm)",
                    'critical',
                    'heart_rate'
                )
            elif bpm > self.thresholds['heart_rate']['warning_high']:
                self.add_alert(
                    f"WARNING: Heart rate high ({bpm:.0f} bpm)",
                    'warning',
                    'heart_rate'
                )
        
        # Check temperature
        temp = sensor_data.get('object_temp')
        if temp is not None:
            if temp <= self.thresholds['temperature']['critical_low']:
                self.add_alert(
                    f"CRITICAL: Body temperature critically low ({temp:.1f}°C)",
                    'critical',
                    'temperature'
                )
            elif temp < self.thresholds['temperature']['warning_low']:
                self.add_alert(
                    f"WARNING: Body temperature low ({temp:.1f}°C)",
                    'warning',
                    'temperature'
                )
            elif temp >= self.thresholds['temperature']['critical_high']:
                self.add_alert(
                    f"CRITICAL: Body temperature critically high ({temp:.1f}°C)",
                    'critical',
                    'temperature'
                )
            elif temp > self.thresholds['temperature']['warning_high']:
                self.add_alert(
                    f"WARNING: Body temperature high ({temp:.1f}°C)",
                    'warning',
                    'temperature'
                )
        
        # Check leads
        if not sensor_data.get('leads_connected', False):
            self.add_alert(
                "WARNING: ECG leads disconnected - check electrode placement",
                'warning',
                'leads'
            )
        
        return self.active_alerts
    
    def add_alert(self, message, level, source):
        """Add alert to active list"""
        alert = {
            'timestamp': datetime.now(),
            'time': datetime.now().strftime("%H:%M:%S"),
            'message': message,
            'level': level,
            'source': source,
        }
        
        # Don't add duplicate alerts within 30 seconds
        for existing in self.active_alerts:
            if (existing['source'] == source and 
                (time.time() - existing['timestamp'].timestamp()) < 30):
                return
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)


class DataLogger:
    """Log vital signs and ECG data to files"""
    
    def __init__(self, log_dir="./logs"):
        """
        Initialize data logger
        
        Args:
            log_dir: Directory for log files
        """
        import os
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create CSV files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.vitals_file = os.path.join(log_dir, f"vitals_{timestamp}.csv")
        self.ecg_file = os.path.join(log_dir, f"ecg_{timestamp}.csv")
        self.alerts_file = os.path.join(log_dir, f"alerts_{timestamp}.json")
        
        # Write CSV headers
        with open(self.vitals_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp',
                'Object_Temp_C',
                'Ambient_Temp_C',
                'ECG_BPM',
                'Heart_Rate',
                'SpO2',
                'ECG_Voltage',
                'Leads_Connected',
                'Readings_Valid'
            ])
        
        with open(self.ecg_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Sample_Index', 'Voltage_V'])
        
        self.ecg_sample_count = 0
    
    def log_vitals(self, sensor_data):
        """Log vital signs to CSV"""
        try:
            with open(self.vitals_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    sensor_data.get('object_temp'),
                    sensor_data.get('ambient_temp'),
                    sensor_data.get('ecg_bpm'),
                    sensor_data.get('heart_rate'),
                    sensor_data.get('spo2'),
                    sensor_data.get('ecg_voltage'),
                    sensor_data.get('leads_connected'),
                    sensor_data.get('readings_valid'),
                ])
        except Exception as e:
            print(f"Error logging vitals: {e}")
    
    def log_ecg_sample(self, voltage):
        """Log single ECG sample"""
        try:
            with open(self.ecg_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.ecg_sample_count,
                    voltage
                ])
            self.ecg_sample_count += 1
        except Exception as e:
            print(f"Error logging ECG sample: {e}")
    
    def log_alerts(self, alerts):
        """Log alerts to JSON"""
        try:
            alerts_data = []
            for alert in alerts:
                alerts_data.append({
                    'timestamp': alert['timestamp'].isoformat(),
                    'time': alert['time'],
                    'message': alert['message'],
                    'level': alert['level'],
                    'source': alert['source'],
                })
            
            with open(self.alerts_file, 'a') as f:
                for alert in alerts_data:
                    f.write(json.dumps(alert) + '\n')
        except Exception as e:
            print(f"Error logging alerts: {e}")


class ClinicAssistantPro:
    """Main application class"""
    
    def __init__(self, config=None):
        """
        Initialize Clinic Assistant Pro
        
        Args:
            config: Configuration dict (optional)
        """
        print("=" * 60)
        print("CLINIC ASSISTANT PRO - Healthcare Monitoring System")
        print("=" * 60)
        print()
        
        # Configuration
        self.config = config or {
            'sensor_update_interval': 0.1,  # 100ms
            'display_update_interval': 0.5,  # 500ms
            'vitals_log_interval': 5.0,  # 5 seconds
            'display_mode': 'vitals',  # vitals, ecg, alerts
        }
        
        # Initialize components
        print("[1/4] Initializing sensors...")
        try:
            self.sensors = SensorAggregator()
            print("    ✓ Sensors initialized")
        except Exception as e:
            print(f"    ✗ Sensor initialization failed: {e}")
            self.sensors = None
        
        print("[2/4] Initializing display...")
        try:
            self.display = LCD35Display(width=480, height=320)
            print("    ✓ Display initialized")
        except Exception as e:
            print(f"    ✗ Display initialization failed: {e}")
            self.display = None
        
        print("[3/4] Initializing alert manager...")
        try:
            self.alert_manager = AlertManager()
            print("    ✓ Alert manager initialized")
        except Exception as e:
            print(f"    ✗ Alert manager initialization failed: {e}")
            self.alert_manager = None
        
        print("[4/4] Initializing data logger...")
        try:
            self.logger = DataLogger()
            print("    ✓ Data logger initialized")
            print(f"       Logs: {self.logger.log_dir}/")
        except Exception as e:
            print(f"    ✗ Data logger initialization failed: {e}")
            self.logger = None
        
        # Runtime state
        self.running = False
        self.threads = []
        self.last_vitals_log = 0
        
        print()
        print("Initialization complete!")
        print()
    
    def sensor_thread(self):
        """Background thread for sensor data acquisition"""
        print("[SENSOR THREAD] Started")
        
        while self.running:
            try:
                if self.sensors:
                    # Update sensor readings
                    self.sensors.update_readings()
                    
                    # Log ECG samples
                    if self.logger and self.sensors.current_data.get('ecg_voltage'):
                        self.logger.log_ecg_sample(
                            self.sensors.current_data['ecg_voltage']
                        )
                
                time.sleep(self.config['sensor_update_interval'])
            
            except Exception as e:
                print(f"[SENSOR THREAD] Error: {e}")
                traceback.print_exc()
    
    def display_thread(self):
        """Background thread for display updates"""
        print("[DISPLAY THREAD] Started")
        
        while self.running:
            try:
                if self.display and self.sensors:
                    # Get current data
                    sensor_data = self.sensors.get_current_readings()
                    
                    # Get alerts
                    alerts = []
                    if self.alert_manager:
                        alerts = self.alert_manager.check_alerts(sensor_data)
                    
                    # Render display
                    self.display.render(sensor_data, alerts)
                
                time.sleep(self.config['display_update_interval'])
            
            except Exception as e:
                print(f"[DISPLAY THREAD] Error: {e}")
                traceback.print_exc()
    
    def logging_thread(self):
        """Background thread for periodic logging"""
        print("[LOGGING THREAD] Started")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Log vitals periodically
                if (current_time - self.last_vitals_log > 
                    self.config['vitals_log_interval']):
                    
                    if self.logger and self.sensors:
                        sensor_data = self.sensors.get_current_readings()
                        self.logger.log_vitals(sensor_data)
                    
                    # Log any active alerts
                    if self.logger and self.alert_manager:
                        self.logger.log_alerts(self.alert_manager.active_alerts)
                    
                    self.last_vitals_log = current_time
                
                time.sleep(1.0)  # Check every second
            
            except Exception as e:
                print(f"[LOGGING THREAD] Error: {e}")
                traceback.print_exc()
    
    def start(self):
        """Start the application"""
        if self.running:
            print("Application already running!")
            return
        
        self.running = True
        
        print("Starting application...")
        print()
        
        # Start threads
        threads_to_start = [
            ('SENSOR', self.sensor_thread),
            ('DISPLAY', self.display_thread),
            ('LOGGING', self.logging_thread),
        ]
        
        for name, thread_func in threads_to_start:
            try:
                t = threading.Thread(target=thread_func, daemon=True)
                t.start()
                self.threads.append(t)
            except Exception as e:
                print(f"Failed to start {name} thread: {e}")
        
        print()
        print("All systems operational!")
        print()
        print("┌─ MONITORING ACTIVE ─────────────────────────────┐")
        print("│                                                  │")
        print("│  • Temperature: MLX90614 (GY906)                 │")
        print("│  • Heart Rate:  AD8232 ECG via ADS1115          │")
        print("│  • Display:     LCD35 (480x320)                 │")
        print("│  • Alerts:      Real-time monitoring            │")
        print("│  • Logging:     CSV + JSON files                │")
        print("│                                                  │")
        print("└──────────────────────────────────────────────────┘")
        print()
        print("Press Ctrl+C to stop monitoring...")
        print()
    
    def stop(self):
        """Stop the application"""
        print("\n\nShutting down...")
        self.running = False
        
        # Wait for threads to finish
        for t in self.threads:
            t.join(timeout=2)
        
        print("Application stopped cleanly")
    
    def print_status(self):
        """Print current status to console"""
        if not self.sensors:
            return
        
        data = self.sensors.get_current_readings()
        
        print("\n" + "=" * 60)
        print(f"Status @ {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
        # Temperature
        temp = data.get('object_temp')
        if temp:
            print(f"  Body Temperature:     {temp:.1f}°C")
        else:
            print(f"  Body Temperature:     N/A")
        
        # Ambient
        amb_temp = data.get('ambient_temp')
        if amb_temp:
            print(f"  Ambient Temperature:  {amb_temp:.1f}°C")
        
        # Heart Rate
        bpm = data.get('ecg_bpm')
        if bpm:
            print(f"  Heart Rate (ECG):     {bpm:.1f} bpm")
        else:
            print(f"  Heart Rate (ECG):     Calculating...")
        
        # ECG Voltage
        ecg_v = data.get('ecg_voltage')
        if ecg_v:
            print(f"  ECG Voltage:          {ecg_v:.3f}V")
        
        # Lead Status
        leads = "✓ CONNECTED" if data.get('leads_connected') else "✗ DISCONNECTED"
        print(f"  ECG Leads:            {leads}")
        
        # Validity
        valid = "✓ VALID" if data.get('readings_valid') else "⚠ INVALID"
        print(f"  Readings:             {valid}")
        
        # Alerts
        if self.alert_manager and self.alert_manager.active_alerts:
            print()
            print("  ACTIVE ALERTS:")
            for alert in self.alert_manager.active_alerts:
                prefix = "  ⚠" if alert['level'] == 'warning' else "  ✗"
                print(f"    {prefix} {alert['message']}")
        
        print()


def main():
    """Main entry point"""
    
    # Configuration
    config = {
        'sensor_update_interval': 0.05,   # 50ms for faster ECG sampling
        'display_update_interval': 0.5,   # 500ms for display refresh
        'vitals_log_interval': 5.0,       # Log vitals every 5 seconds
    }
    
    # Initialize application
    app = ClinicAssistantPro(config)
    
    try:
        # Start monitoring
        app.start()
        
        # Main loop - print status periodically
        status_interval = 10  # Print status every 10 seconds
        last_status = time.time()
        
        while True:
            current_time = time.time()
            
            # Print status
            if current_time - last_status >= status_interval:
                app.print_status()
                last_status = current_time
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupt received")
        app.stop()
    
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        app.stop()


if __name__ == "__main__":
    main()
