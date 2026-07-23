#!/usr/bin/env python3
"""
AD8232 ECG Sensor via MCP3008 ADC over SPI
ECG output -> MCP3008 CH0
LO+  -> GPIO22 (Pin 15)   [moved from GPIO17 due to LCD35 conflict]
LO-  -> GPIO27 (Pin 13)

MCP3008 SPI Wiring:
  CLK  -> GPIO11/SCLK (Pin 23)
  DOUT -> GPIO9/MISO  (Pin 21)
  DIN  -> GPIO10/MOSI (Pin 19)
  CS   -> GPIO5       (Pin 29)  [software CS, moved from CE0 due to LCD35]
"""

import time
import logging

try:
    import spidev
    SPI_AVAILABLE = True
except ImportError:
    SPI_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

logger = logging.getLogger(__name__)

# GPIO pins (BCM)
LO_PLUS_PIN  = 22   # moved from 17 (LCD35 conflict)
LO_MINUS_PIN = 27
CS_PIN       = 5    # software CS for MCP3008

ECG_CHANNEL  = 0    # MCP3008 channel


class AD8232Sensor:
    def __init__(self, spi_bus=0, spi_device=0, spi_speed=1350000):
        self.spi_bus    = spi_bus
        self.spi_device = spi_device
        self.spi_speed  = spi_speed
        self.spi        = None
        self.initialized = False

    def initialize(self):
        if not SPI_AVAILABLE:
            logger.error("spidev not available. Run: pip install spidev --break-system-packages")
            return False
        if not GPIO_AVAILABLE:
            logger.error("RPi.GPIO not available. Run: pip install RPi.GPIO --break-system-packages")
            return False
        try:
            # SPI setup
            self.spi = spidev.SpiDev()
            self.spi.open(self.spi_bus, self.spi_device)
            self.spi.max_speed_hz = self.spi_speed
            self.spi.mode         = 0b00
            self.spi.no_cs        = True   # we drive CS manually

            # GPIO setup
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(CS_PIN,       GPIO.OUT, initial=GPIO.HIGH)
            GPIO.setup(LO_PLUS_PIN,  GPIO.IN)
            GPIO.setup(LO_MINUS_PIN, GPIO.IN)

            self.initialized = True
            logger.info("AD8232 via MCP3008 initialized.")
            return True
        except Exception as e:
            logger.error(f"AD8232 init error: {e}")
            return False

    def _read_mcp3008(self, channel):
        """Read 10-bit value from MCP3008 using software CS on GPIO5."""
        GPIO.output(CS_PIN, GPIO.LOW)
        cmd      = [0x01, (0x08 | channel) << 4, 0x00]
        response = self.spi.xfer2(cmd)
        GPIO.output(CS_PIN, GPIO.HIGH)
        return ((response[1] & 0x03) << 8) | response[2]

    def is_leads_off(self):
        if not GPIO_AVAILABLE:
            return False
        return bool(GPIO.input(LO_PLUS_PIN) or GPIO.input(LO_MINUS_PIN))

    def read(self):
        """Returns (ecg_raw 0-1023, leads_off bool)."""
        if not self.initialized:
            return None, True
        try:
            leads_off = self.is_leads_off()
            ecg_value = self._read_mcp3008(ECG_CHANNEL)
            return ecg_value, leads_off
        except Exception as e:
            logger.error(f"AD8232 read error: {e}")
            return None, True

    def cleanup(self):
        if self.spi:
            try:
                self.spi.close()
            except Exception:
                pass
        if GPIO_AVAILABLE:
            try:
                GPIO.cleanup([CS_PIN, LO_PLUS_PIN, LO_MINUS_PIN])
            except Exception:
                pass
        logger.info("AD8232 cleaned up.")
