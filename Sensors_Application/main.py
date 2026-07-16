#!/usr/bin/env python3
"""
Healthcare Assistant - Raspberry Pi 4B
Sensors: MAX30102 (SpO2/HR), MLX90614 (Temperature), AD8232 (ECG)
"""

import time
import threading
import logging
from datetime import datetime

from sensors.max30102_sensor import MAX30102Sensor
from sensors.mlx90614_sensor import MLX90614Sensor
from sensors.ad8232_sensor import AD8232Sensor
from data.data_logger import DataLogger
from alerts.alert_manager import AlertManager
from display.console_display import ConsoleDisplay

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('healthcare_assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HealthcareAssistant:
    def __init__(self):
        logger.info("Initializing Healthcare Assistant...")

        # Initialize sensors
        self.max30102 = MAX30102Sensor()
        self.mlx90614 = MLX90614Sensor()
        self.ad8232 = AD8232Sensor()

        # Initialize support modules
        self.data_logger = DataLogger()
        self.alert_manager = AlertManager()
        self.display = ConsoleDisplay()

        # Control flags
        self.running = False
        self.readings = {}

        # Sampling intervals (seconds)
        self.vitals_interval = 5     # SpO2, HR, Temp every 5s
        self.ecg_interval = 0.01     # ECG at ~100Hz

    def read_vitals(self):
        """Read SpO2, Heart Rate, and Temperature."""
        while self.running:
            try:
                # MAX30102: SpO2 + Heart Rate
                hr, spo2 = self.max30102.read()
                # MLX90614: Body Temperature
                temp_obj, temp_amb = self.mlx90614.read()

                timestamp = datetime.now().isoformat()

                self.readings.update({
                    'timestamp': timestamp,
                    'heart_rate': hr,
                    'spo2': spo2,
                    'body_temp': temp_obj,
                    'ambient_temp': temp_amb
                })

                # Log and display
                self.data_logger.log_vitals(self.readings)
                self.display.show_vitals(self.readings)

                # Check for alerts
                self.alert_manager.check_vitals(self.readings)

            except Exception as e:
                logger.error(f"Vitals read error: {e}")

            time.sleep(self.vitals_interval)

    def read_ecg(self):
        """Continuously read ECG signal from AD8232 via MCP3008/SPI."""
        while self.running:
            try:
                ecg_value, leads_off = self.ad8232.read()

                if not leads_off:
                    self.data_logger.log_ecg(ecg_value)
                    self.display.update_ecg(ecg_value)
                    self.alert_manager.check_ecg(ecg_value, leads_off)
                else:
                    logger.warning("ECG: Leads off detected!")

            except Exception as e:
                logger.error(f"ECG read error: {e}")

            time.sleep(self.ecg_interval)

    def start(self):
        """Start all sensor reading threads."""
        logger.info("Starting Healthcare Assistant...")
        self.running = True

        # Initialize sensors
        if not self.max30102.initialize():
            logger.error("MAX30102 initialization failed!")
        if not self.mlx90614.initialize():
            logger.error("MLX90614 initialization failed!")
        if not self.ad8232.initialize():
            logger.error("AD8232 initialization failed!")

        # Start threads
        self.vitals_thread = threading.Thread(
            target=self.read_vitals, daemon=True, name="VitalsThread"
        )
        self.ecg_thread = threading.Thread(
            target=self.read_ecg, daemon=True, name="ECGThread"
        )

        self.vitals_thread.start()
        self.ecg_thread.start()

        logger.info("Healthcare Assistant is running. Press Ctrl+C to stop.")
        self.display.show_header()

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Gracefully shut down."""
        logger.info("Shutting down Healthcare Assistant...")
        self.running = False
        self.max30102.cleanup()
        self.mlx90614.cleanup()
        self.ad8232.cleanup()
        self.data_logger.close()
        logger.info("Shutdown complete.")


if __name__ == '__main__':
    assistant = HealthcareAssistant()
    assistant.start()
