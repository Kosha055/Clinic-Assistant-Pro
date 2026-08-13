"""
Clinic Assistant Pro - Configuration Template
Copy this file to config/clinic_config.py and customize for your setup

Usage:
    from config.clinic_config import AppConfig
    app_config = AppConfig()
"""

class AppConfig:
    """Application configuration"""
    
    # ============================================================
    # APPLICATION SETTINGS
    # ============================================================
    
    APP_NAME = "Clinic Assistant Pro"
    APP_VERSION = "1.0.0"
    DEBUG_MODE = False
    LOG_DIR = "./logs"
    
    # ============================================================
    # SENSOR CONFIGURATION
    # ============================================================
    
    # I2C Bus Configuration
    I2C_BUS = 1  # /dev/i2c-1 on Raspberry Pi
    
    # Temperature Sensor (GY906/MLX90614)
    TEMP_SENSOR_I2C_ADDR = 0x5A
    TEMP_SMOOTHING_FACTOR = 0.7  # 0-1, higher = more responsive
    TEMP_UPDATE_INTERVAL = 1.0  # seconds between smoothed readings
    
    # Heart Rate Sensor (MAX30102)
    HR_SENSOR_I2C_ADDR = 0x57
    HR_SAMPLING_RATE = 100  # Hz
    
    # ECG Sensor (AD8232 via ADS1115)
    ADS_I2C_ADDR = 0x48  # ADDR pin to GND
    ECG_SAMPLING_RATE = 860  # SPS (8-860)
    ECG_BUFFER_SIZE = 500  # samples (~0.58 sec @ 860 SPS)
    ECG_PEAK_WINDOW_SIZE = 6  # peaks for stable BPM
    
    # ECG Lead-Off Detection
    ECG_LO_PLUS_GPIO = 22  # GPIO pin for LO+
    ECG_LO_MINUS_GPIO = 27  # GPIO pin for LO-
    
    # ============================================================
    # THREAD & TIMING CONFIGURATION
    # ============================================================
    
    SENSOR_UPDATE_INTERVAL = 0.05  # 50ms - sensor data acquisition
    DISPLAY_UPDATE_INTERVAL = 0.5  # 500ms - LCD refresh rate
    VITALS_LOG_INTERVAL = 5.0  # 5 sec - log vitals to CSV
    ALERT_CHECK_INTERVAL = 1.0  # 1 sec - check for alerts
    
    # ============================================================
    # DISPLAY CONFIGURATION
    # ============================================================
    
    # Display Type: 'lcd35', 'pygame', 'terminal'
    DISPLAY_TYPE = 'lcd35'
    
    # Resolution
    DISPLAY_WIDTH = 480
    DISPLAY_HEIGHT = 320
    DISPLAY_ROTATION = 0  # degrees (0, 90, 180, 270)
    
    # Color Scheme
    COLORS = {
        'bg': (20, 20, 30),           # Dark background
        'fg': (255, 255, 255),        # White text
        'normal': (100, 200, 100),    # Green - normal range
        'warning': (255, 200, 0),     # Orange - warning
        'critical': (255, 50, 50),    # Red - critical
        'header': (50, 100, 150),     # Blue header
        'border': (100, 100, 120),    # Gray border
        'ecg_line': (0, 255, 100),    # Cyan ECG trace
        'grid': (40, 40, 50),         # Dark grid
    }
    
    # Display Screens
    AVAILABLE_SCREENS = ['vitals', 'ecg', 'alerts']
    DEFAULT_SCREEN = 'vitals'
    
    # ============================================================
    # CLINICAL ALERT THRESHOLDS
    # ============================================================
    
    ALERT_THRESHOLDS = {
        'heart_rate': {
            'critical_low': 40,      # bpm
            'warning_low': 50,
            'warning_high': 120,
            'critical_high': 150,
        },
        'temperature': {
            'critical_low': 35.0,    # °C
            'warning_low': 36.0,
            'warning_high': 37.5,
            'critical_high': 38.5,
        },
        'spo2': {
            'critical_low': 90,      # %
            'warning_low': 94,
        },
        'leads': {
            'disconnected_alert': True,  # Alert if ECG leads off
        }
    }
    
    # ============================================================
    # DATA LOGGING CONFIGURATION
    # ============================================================
    
    # Logging Modes
    LOG_TO_CSV = True
    LOG_TO_JSON = True
    LOG_TO_DATABASE = False  # Set to True if using SQLite/PostgreSQL
    
    # CSV Logging
    CSV_VITALS_FILE_PREFIX = 'vitals_'
    CSV_ECG_FILE_PREFIX = 'ecg_'
    CSV_DELIMITER = ','
    
    # JSON Logging (Alerts)
    JSON_ALERTS_FILE_PREFIX = 'alerts_'
    
    # Retention Policy
    LOG_RETENTION_DAYS = 30  # Auto-delete logs older than 30 days
    
    # ============================================================
    # ADVANCED SIGNAL PROCESSING
    # ============================================================
    
    # BPM Calculation
    BPM_STABILITY_THRESHOLD = 3.0  # bpm - max change between readings
    BPM_REQUIRES_PEAKS = 6  # minimum peaks for stable BPM
    BPM_MINIMUM_DATA_SECONDS = 3.0  # need 3+ seconds before first reading
    
    # Peak Detection (ECG)
    PEAK_MIN_DISTANCE_SAMPLES = 200  # min samples between peaks
                                      # filters dicrotic notch
    PEAK_SLOPE_WINDOW = 5  # samples for slope calculation
    PEAK_AMPLITUDE_THRESHOLD = 0.1  # minimum peak height (volts)
    
    # Temperature Smoothing
    TEMP_SMOOTHING_ALPHA = 0.7  # exponential smoothing factor
                                # higher = more responsive
    TEMP_STABILITY_THRESHOLD = 0.1  # °C - max fluctuation allowed
    
    # ============================================================
    # AUDIO & NOTIFICATIONS (Optional)
    # ============================================================
    
    ENABLE_ALERTS_AUDIO = True
    ALERT_SOUND_CRITICAL = '/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga'
    ALERT_SOUND_WARNING = '/usr/share/sounds/freedesktop/stereo/bell.oga'
    
    # ============================================================
    # DATABASE CONFIGURATION (Optional)
    # ============================================================
    
    # Uncomment to enable database logging
    # DB_TYPE = 'sqlite'  # or 'postgresql', 'mysql'
    # DB_PATH = './clinic_data.db'  # for SQLite
    # DB_HOST = 'localhost'
    # DB_PORT = 5432
    # DB_USER = 'clinic_user'
    # DB_PASSWORD = 'secure_password'
    # DB_NAME = 'clinic_assistant'
    
    # ============================================================
    # NETWORK CONFIGURATION (Optional)
    # ============================================================
    
    # Cloud Upload
    ENABLE_CLOUD_SYNC = False
    # CLOUD_API_URL = 'https://api.clinic-assistant.com'
    # CLOUD_API_KEY = 'your-api-key'
    
    # Network Timeout
    NETWORK_TIMEOUT = 30  # seconds
    
    # ============================================================
    # DEVELOPMENT & DEBUG OPTIONS
    # ============================================================
    
    # Verbose logging
    VERBOSE_LOGGING = False
    
    # Simulate sensor data (for testing without hardware)
    SIMULATE_SENSORS = False
    
    # Print status to console every N seconds
    CONSOLE_STATUS_INTERVAL = 10  # seconds
    
    # Save debug frames from display
    SAVE_DEBUG_FRAMES = False
    DEBUG_FRAMES_DIR = './debug_frames/'


class PatientConfig:
    """Patient-specific configuration"""
    
    def __init__(self, patient_id, patient_name):
        self.patient_id = patient_id
        self.patient_name = patient_name
        self.date_admitted = None
        self.medications = []
        self.allergies = []
        self.medical_history = []
        self.normal_baseline = {
            'heart_rate': 70,  # baseline resting HR
            'temperature': 37.0,  # baseline temp
            'spo2': 98,  # baseline SpO2
        }


class SensorCalibraton:
    """Sensor calibration constants"""
    
    # Temperature sensor offset (in °C)
    # Set to 0 if sensor reads correctly, adjust if readings are off
    TEMP_OFFSET = 0.0
    
    # ADC reference voltage (typically 4.096V for ADS1115)
    ADC_REF_VOLTAGE = 4.096
    
    # ECG scaling factor (V/bit)
    ECG_SCALE_FACTOR = ADC_REF_VOLTAGE / 32768


def get_config():
    """Factory function to get app configuration"""
    return AppConfig()


if __name__ == "__main__":
    config = get_config()
    
    print("Clinic Assistant Pro - Configuration")
    print("=" * 60)
    print(f"App Name: {config.APP_NAME} v{config.APP_VERSION}")
    print(f"Display: {config.DISPLAY_WIDTH}x{config.DISPLAY_HEIGHT}")
    print(f"ECG Sampling Rate: {config.ECG_SAMPLING_RATE} SPS")
    print(f"Temp Smoothing: {config.TEMP_SMOOTHING_FACTOR}")
    print()
    print("Alert Thresholds:")
    for metric, thresholds in config.ALERT_THRESHOLDS.items():
        print(f"  {metric.upper()}: {thresholds}")
