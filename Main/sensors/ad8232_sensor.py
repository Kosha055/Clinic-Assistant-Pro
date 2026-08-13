#!/usr/bin/env python3
"""
AD8232 ECG Sensor Driver
Reads ECG signal via MCP3008 ADC over SPI interface.

Wiring:
  AD8232 OUTPUT -> MCP3008 CH0
  AD8232 LO+    -> GPIO17 (Pin 11)   [Leads-Off detection +]
  AD8232 LO-    -> GPIO27 (Pin 13)   [Leads-Off detection -]
  AD8232 3.3V   -> 3.3V Pin
  AD8232 GND    -> GND

MCP3008 SPI Wiring:
  MCP3008 VDD   -> 3.3V
  MCP3008 VREF  -> 3.3V
  MCP3008 AGND  -> GND
  MCP3008 DGND  -> GND
  MCP3008 CLK   -> GPIO11 / SCLK (Pin 23)
  MCP3008 DOUT  -> GPIO9  / MISO (Pin 21)
  MCP3008 DIN   -> GPIO10 / MOSI (Pin 19)
  MCP3008 CS/SHDN -> GPIO8 / CE0 (Pin 24)
"""

import time
import logging

try:
    import spidev
    SPI_AVAILABLE = True
except ImportError:
    SPI_AVAILABLE = False
    logging.warning("spidev not found. Install with: pip install spidev")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("RPi.GPIO not found. Install with: pip install RPi.GPIO")

logger = logging.getLogger(__name__)

# GPIO Pins for Leads-Off detection (BCM numbering)
LO_PLUS_PIN  = 17
LO_MINUS_PIN = 27

# MCP3008 channel for ECG signal
ECG_CHANNEL  = 0


class AD8232Sensor:
    def __init__(self, spi_bus=0, spi_device=0, spi_speed=1350000):
        self.spi_bus = spi_bus
        self.spi_device = spi_device
        self.spi_speed = spi_speed
        self.spi = None
        self.initialized = False

        # ECG signal statistics
        self._sample_count = 0
        self._ecg_min = 4096
        self._ecg_max = 0

    def initialize(self):
        """Initialize SPI and GPIO for AD8232."""
        if not SPI_AVAILABLE:
            logger.error("spidev library not available.")
            return False

        if not GPIO_AVAILABLE:
            logger.error("RPi.GPIO library not available.")
            return False

        try:
            # Setup SPI
            self.spi = spidev.SpiDev()
            self.spi.open(self.spi_bus, self.spi_device)
            self.spi.max_speed_hz = self.spi_speed
            self.spi.mode = 0b00   # SPI Mode 0

            # Setup GPIO for leads-off detection
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(LO_PLUS_PIN,  GPIO.IN)
            GPIO.setup(LO_MINUS_PIN, GPIO.IN)

            self.initialized = True
            logger.info("AD8232 (via MCP3008 SPI) initialized successfully.")
            return True

        except Exception as e:
            logger.error(f"AD8232 initialization error: {e}")
            return False

    def _read_mcp3008(self, channel):
        """
        Read a 10-bit value from MCP3008 ADC.
        MCP3008 uses 3-byte SPI transaction:
          Byte 1: Start bit (0x01)
          Byte 2: Single-ended channel select (0x80 | channel << 4)
          Byte 3: Don't care (0x00)
        Returns 10-bit integer (0–1023).
        """
        if channel < 0 or channel > 7:
            raise ValueError(f"Invalid MCP3008 channel: {channel}")

        cmd = [0x01, (0x08 | channel) << 4, 0x00]
        response = self.spi.xfer2(cmd)

        # Combine last two bytes, mask to 10 bits
        value = ((response[1] & 0x03) << 8) | response[2]
        return value

    def is_leads_off(self):
        """
        Check if ECG electrodes are properly connected.
        Returns True if leads are off (disconnected), False if connected.
        """
        if not GPIO_AVAILABLE:
            return False
        lo_plus  = GPIO.input(LO_PLUS_PIN)
        lo_minus = GPIO.input(LO_MINUS_PIN)
        return bool(lo_plus or lo_minus)

    def read(self):
        """
        Read ECG sample and leads-off status.
        Returns (ecg_value: int 0-1023, leads_off: bool).
        ecg_value: raw ADC reading (10-bit), convert to voltage: v = val * 3.3 / 1023
        """
        if not self.initialized:
            return None, True

        try:
            leads_off = self.is_leads_off()
            ecg_value = self._read_mcp3008(ECG_CHANNEL)

            # Update statistics
            self._sample_count += 1
            if ecg_value < self._ecg_min:
                self._ecg_min = ecg_value
            if ecg_value > self._ecg_max:
                self._ecg_max = ecg_value

            return ecg_value, leads_off

        except Exception as e:
            logger.error(f"AD8232 read error: {e}")
            return None, True

    def read_voltage(self):
        """Read ECG sample and return voltage (0–3.3V)."""
        value, leads_off = self.read()
        if value is None:
            return None, leads_off
        voltage = round(value * 3.3 / 1023.0, 4)
        return voltage, leads_off

    def get_stats(self):
        """Return basic ECG signal statistics."""
        return {
            'samples': self._sample_count,
            'min_raw': self._ecg_min,
            'max_raw': self._ecg_max,
            'amplitude': self._ecg_max - self._ecg_min
        }

    def cleanup(self):
        """Close SPI and clean up GPIO."""
        if self.spi:
            try:
                self.spi.close()
            except Exception:
                pass
        if GPIO_AVAILABLE:
            try:
                GPIO.cleanup([LO_PLUS_PIN, LO_MINUS_PIN])
            except Exception:
                pass
        logger.info("AD8232 cleaned up.")
