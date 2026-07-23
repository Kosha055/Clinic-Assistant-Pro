#!/usr/bin/env python3
"""
Healthcare Assistant - Raspberry Pi 4B
Sensors: MAX30102 (SpO2/HR), MLX90614 (Temperature), AD8232 (ECG)
Display: LCD35 3.5" TFT via framebuffer
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
from display.lcd35_display import LCD35Display

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

        # Sensors
        self.max30102  = MAX30102Sensor()
        self.mlx90614  = MLX90614Sensor()
        self.ad8232    = AD8232Sensor()

        # Support modules
        self.data_logger   = DataLogger()
        self.alert_manager = AlertManager()

        # LCD35 display  — change /dev/fb1 to /dev/fb0 if needed
        self.lcd = LCD35Display(fb_device="/dev/fb1")

        # Control
        self.running         = False
        self.readings        = {}
        self.vitals_interval = 5      # seconds
        self.ecg_interval    = 0.01   # ~100 Hz

    # ── Sensor Threads ───────────────────────────────────────────────────────

    def read_vitals(self):
        """Read SpO2, Heart Rate, Temperature every 5 seconds."""
        while self.running:
            try:
                hr, spo2           = self.max30102.read()
                temp_obj, temp_amb = self.mlx90614.read()

                self.readings = {
                    'timestamp':    datetime.now().isoformat(),
                    'heart_rate':   hr,
                    'spo2':         spo2,
                    'body_temp':    temp_obj,
                    'ambient_temp': temp_amb,
                }

                self.data_logger.log_vitals(self.readings)
                self.alert_manager.check_vitals(self.readings)
                self.lcd.update_vitals(self.readings)

                logger.info(
                    f"HR={hr} bpm | SpO2={spo2}% | "
                    f"Temp={temp_obj}°C | Ambient={temp_amb}°C"
                )

            except Exception as e:
                logger.error(f"Vitals read error: {e}")

            time.sleep(self.vitals_interval)

    def read_ecg(self):
        """Read ECG at ~100 Hz continuously."""
        while self.running:
            try:
                ecg_value, leads_off = self.ad8232.read()

                self.lcd.set_leads_off(leads_off)

                if not leads_off and ecg_value is not None:
                    self.data_logger.log_ecg(ecg_value)
                    self.lcd.update_ecg(ecg_value)
                    self.alert_manager.check_ecg(ecg_value, leads_off)
                else:
                    self.lcd.update_ecg(512)  # flat line when leads off

            except Exception as e:
                logger.error(f"ECG read error: {e}")

            time.sleep(self.ecg_interval)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        logger.info("Starting Healthcare Assistant...")
        self.running = True

        # Initialize sensors
        if not self.max30102.initialize():
            logger.error("MAX30102 initialization failed!")
        if not self.mlx90614.initialize():
            logger.error("MLX90614 initialization failed!")
        if not self.ad8232.initialize():
            logger.error("AD8232 initialization failed!")

        # Start LCD display
        self.lcd.start()

        # Start sensor threads
        self.vitals_thread = threading.Thread(
            target=self.read_vitals, daemon=True, name="VitalsThread"
        )
        self.ecg_thread = threading.Thread(
            target=self.read_ecg, daemon=True, name="ECGThread"
        )
        self.vitals_thread.start()
        self.ecg_thread.start()

        logger.info("Healthcare Assistant running. Press Ctrl+C to stop.")

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        logger.info("Shutting down...")
        self.running = False
        self.lcd.stop()
        self.max30102.cleanup()
        self.mlx90614.cleanup()
        self.ad8232.cleanup()
        self.data_logger.close()
        logger.info("Shutdown complete.")


if __name__ == '__main__':
    assistant = HealthcareAssistant()
    assistant.start()
