#!/usr/bin/env python3
"""
MLX90614 Sensor Driver
Non-contact Infrared Body Temperature Sensor via I2C
I2C Address: 0x5A (default)
Pins: SDA (GPIO2 / Pin 3), SCL (GPIO3 / Pin 5)

Note: MLX90614 uses SMBus read_word_data with PEC (Packet Error Checking).
"""

import time
import logging

try:
    import smbus2
    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False
    logging.warning("smbus2 not found. Install with: pip install smbus2")

logger = logging.getLogger(__name__)

# MLX90614 I2C Address & Registers
MLX90614_ADDRESS    = 0x5A
MLX_RAM_TA          = 0x06   # Ambient temperature register
MLX_RAM_TOBJ1       = 0x07   # Object (body) temperature register
MLX_RAM_TOBJ2       = 0x08   # Object 2 (if dual zone)

KELVIN_OFFSET       = 273.15


class MLX90614Sensor:
    def __init__(self, i2c_bus=1, address=MLX90614_ADDRESS):
        self.bus_num = i2c_bus
        self.address = address
        self.bus = None
        self.initialized = False

    def initialize(self):
        """Initialize the MLX90614 sensor."""
        if not I2C_AVAILABLE:
            logger.error("smbus2 library not available.")
            return False
        try:
            self.bus = smbus2.SMBus(self.bus_num)
            time.sleep(0.1)

            # Quick test read to verify device is present
            self._read_raw(MLX_RAM_TA)
            self.initialized = True
            logger.info("MLX90614 initialized successfully.")
            return True

        except Exception as e:
            logger.error(f"MLX90614 initialization error: {e}")
            return False

    def _read_raw(self, register):
        """
        Read a 16-bit word from the MLX90614.
        Uses read_word_data which returns little-endian 16-bit value.
        The MSB (bit 15) is an error flag — should be 0.
        """
        raw = self.bus.read_word_data(self.address, register)
        # Check error bit
        if raw & 0x8000:
            raise ValueError(f"MLX90614 error flag set on register 0x{register:02X}")
        return raw & 0x7FFF

    def _raw_to_celsius(self, raw):
        """Convert raw sensor value to Celsius."""
        # Resolution: 0.02°C per LSB, output in Kelvin
        kelvin = raw * 0.02
        celsius = kelvin - KELVIN_OFFSET
        return round(celsius, 2)

    def read(self):
        """
        Read and return (object_temp_c, ambient_temp_c).
        Object temp = body/surface temperature (what you point at).
        Ambient temp = sensor's own temperature.
        """
        if not self.initialized:
            return None, None

        try:
            raw_obj = self._read_raw(MLX_RAM_TOBJ1)
            raw_amb = self._read_raw(MLX_RAM_TA)

            obj_temp = self._raw_to_celsius(raw_obj)
            amb_temp = self._raw_to_celsius(raw_amb)

            return obj_temp, amb_temp

        except Exception as e:
            logger.error(f"MLX90614 read error: {e}")
            return None, None

    def read_fahrenheit(self):
        """Read temperatures in Fahrenheit."""
        obj_c, amb_c = self.read()
        if obj_c is None:
            return None, None
        obj_f = round(obj_c * 9 / 5 + 32, 2)
        amb_f = round(amb_c * 9 / 5 + 32, 2)
        return obj_f, amb_f

    def cleanup(self):
        """Close I2C bus."""
        if self.bus:
            try:
                self.bus.close()
            except Exception:
                pass
        logger.info("MLX90614 cleaned up.")
