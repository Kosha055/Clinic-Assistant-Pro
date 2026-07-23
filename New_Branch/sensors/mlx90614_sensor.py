#!/usr/bin/env python3
"""
MLX90614 Sensor Driver
Non-contact IR Temperature via I2C
Address: 0x5A
Pins: SDA (GPIO2/Pin3), SCL (GPIO3/Pin5)
"""

import time
import logging

try:
    import smbus2
    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False

logger = logging.getLogger(__name__)

MLX90614_ADDRESS = 0x5A
MLX_RAM_TA       = 0x06   # Ambient temp
MLX_RAM_TOBJ1    = 0x07   # Object temp
KELVIN_OFFSET    = 273.15


class MLX90614Sensor:
    def __init__(self, i2c_bus=1, address=MLX90614_ADDRESS):
        self.bus_num     = i2c_bus
        self.address     = address
        self.bus         = None
        self.initialized = False

    def initialize(self):
        if not I2C_AVAILABLE:
            logger.error("smbus2 not available.")
            return False
        try:
            self.bus = smbus2.SMBus(self.bus_num)
            time.sleep(0.1)
            self._read_raw(MLX_RAM_TA)  # test read
            self.initialized = True
            logger.info("MLX90614 initialized.")
            return True
        except Exception as e:
            logger.error(f"MLX90614 init error: {e}")
            return False

    def _read_raw(self, register):
        raw = self.bus.read_word_data(self.address, register)
        if raw & 0x8000:
            raise ValueError("MLX90614 error flag set")
        return raw & 0x7FFF

    def _raw_to_celsius(self, raw):
        return round(raw * 0.02 - KELVIN_OFFSET, 2)

    def read(self):
        """Returns (object_temp_c, ambient_temp_c)."""
        if not self.initialized:
            return None, None
        try:
            obj_temp = self._raw_to_celsius(self._read_raw(MLX_RAM_TOBJ1))
            amb_temp = self._raw_to_celsius(self._read_raw(MLX_RAM_TA))
            return obj_temp, amb_temp
        except Exception as e:
            logger.error(f"MLX90614 read error: {e}")
            return None, None

    def cleanup(self):
        if self.bus:
            try:
                self.bus.close()
            except Exception:
                pass
        logger.info("MLX90614 cleaned up.")
