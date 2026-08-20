#!/usr/bin/env python3
"""
AD8232 ECG Sensor Driver
Uses ADS1115 16-bit ADC over I2C (replaces MCP3008 SPI).

Wiring:
  AD8232 OUTPUT -> ADS1115 A0
  AD8232 LO+    -> GPIO22 (Pin 15)
  AD8232 LO-    -> GPIO27 (Pin 13)
  AD8232 3.3V   -> 3.3V
  AD8232 GND    -> GND

  ADS1115 SDA   -> GPIO2 (Pin 3)   shared I2C bus
  ADS1115 SCL   -> GPIO3 (Pin 5)   shared I2C bus
  ADS1115 ADDR  -> GND             address = 0x48
  ADS1115 VDD   -> 3.3V
  ADS1115 GND   -> GND
"""

import logging
from sensors.ads1115_sensor import ADS1115Sensor

logger = logging.getLogger(__name__)


class AD8232Sensor:
    def __init__(self):
        self.adc         = ADS1115Sensor(i2c_bus=1, channel=0)
        self.initialized = False

        # Basic signal stats
        self._sample_count = 0
        self._ecg_min      = 32767
        self._ecg_max      = -32768

    def initialize(self):
        """Initialize ADS1115 ADC and GPIO."""
        result = self.adc.initialize()
        if result:
            self.initialized = True
            logger.info("AD8232 ECG sensor initialized via ADS1115.")
        else:
            logger.error("AD8232 initialization failed — check ADS1115 wiring.")
        return result

    def read(self):
        """
        Read ECG sample.
        Returns (ecg_normalized 0-1023, leads_off bool).
        Compatible with existing main.py and display code.
        """
        if not self.initialized:
            return None, True

        value, leads_off = self.adc.read_normalized()

        if value is not None:
            self._sample_count += 1
            if value < self._ecg_min:
                self._ecg_min = value
            if value > self._ecg_max:
                self._ecg_max = value

        return value, leads_off

    def read_voltage(self):
        """Read ECG as voltage (0.0 – 3.3V)."""
        return self.adc.read_voltage()

    def read_raw(self):
        """Read raw 16-bit ADS1115 value (-32768 to 32767)."""
        return self.adc.read_raw()

    def get_stats(self):
        return {
            'samples':   self._sample_count,
            'min':       self._ecg_min,
            'max':       self._ecg_max,
            'amplitude': self._ecg_max - self._ecg_min,
        }

    def cleanup(self):
        self.adc.cleanup()
        logger.info("AD8232 cleaned up.")
