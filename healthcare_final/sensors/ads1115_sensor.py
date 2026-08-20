#!/usr/bin/env python3
"""
ADS1115 16-bit ADC Driver
Used to read AD8232 ECG analog output via I2C.

Wiring:
  ADS1115 VDD  -> 3.3V
  ADS1115 GND  -> GND
  ADS1115 SDA  -> GPIO2 (Pin 3)   shared I2C bus
  ADS1115 SCL  -> GPIO3 (Pin 5)   shared I2C bus
  ADS1115 ADDR -> GND             I2C address = 0x48
  ADS1115 A0   -> AD8232 OUTPUT   ECG signal input
  ADS1115 A1   -> GND             (unused)
  ADS1115 A2   -> GND             (unused)
  ADS1115 A3   -> GND             (unused)

AD8232 Wiring:
  AD8232 OUTPUT -> ADS1115 A0
  AD8232 LO+    -> GPIO22 (Pin 15)
  AD8232 LO-    -> GPIO27 (Pin 13)
  AD8232 3.3V   -> 3.3V
  AD8232 GND    -> GND
"""

import time
import logging
import struct

try:
    import smbus2
    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False
    logging.warning("smbus2 not found. Run: pip install smbus2 --break-system-packages")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── ADS1115 I2C Address ───────────────────────────────────────────────────────
# ADDR pin -> GND  : 0x48 (default)
# ADDR pin -> VDD  : 0x49
# ADDR pin -> SDA  : 0x4A
# ADDR pin -> SCL  : 0x4B
ADS1115_ADDRESS = 0x48

# ── Registers ─────────────────────────────────────────────────────────────────
REG_CONVERSION = 0x00
REG_CONFIG     = 0x01
REG_LO_THRESH  = 0x02
REG_HI_THRESH  = 0x03

# ── Config Register Bits ──────────────────────────────────────────────────────
# OS: Operational status
OS_SINGLE      = 0x8000   # Start single conversion

# MUX: Input multiplexer (single-ended channels)
MUX_AIN0_GND  = 0x4000   # A0 vs GND  ← ECG channel
MUX_AIN1_GND  = 0x5000
MUX_AIN2_GND  = 0x6000
MUX_AIN3_GND  = 0x7000

# PGA: Programmable gain amplifier (full-scale range)
PGA_6_144V    = 0x0000   # ±6.144V
PGA_4_096V    = 0x0200   # ±4.096V
PGA_2_048V    = 0x0400   # ±2.048V  (default, good for 3.3V systems)
PGA_1_024V    = 0x0600
PGA_0_512V    = 0x0800
PGA_0_256V    = 0x0A00

# MODE
MODE_CONTINUOUS = 0x0000
MODE_SINGLE     = 0x0100

# DATA RATE (samples per second)
DR_8SPS        = 0x0000
DR_16SPS       = 0x0020
DR_32SPS       = 0x0040
DR_64SPS       = 0x0060
DR_128SPS      = 0x0080   # default
DR_250SPS      = 0x00A0
DR_475SPS      = 0x00C0
DR_860SPS      = 0x00E0   # fastest — best for ECG

# COMP: Comparator (disable)
COMP_QUE_DISABLE = 0x0003

# GPIO for AD8232 leads-off detection
LO_PLUS_PIN  = 22   # moved from GPIO17 (LCD35 conflict)
LO_MINUS_PIN = 27

# Voltage reference for conversion
PGA_VOLTAGE  = 2.048   # matches PGA_2_048V setting


class ADS1115Sensor:
    def __init__(self, i2c_bus=1, address=ADS1115_ADDRESS, channel=0):
        self.bus_num     = i2c_bus
        self.address     = address
        self.channel     = channel
        self.bus         = None
        self.initialized = False

        # Map channel number to MUX setting
        self._mux_map = {
            0: MUX_AIN0_GND,
            1: MUX_AIN1_GND,
            2: MUX_AIN2_GND,
            3: MUX_AIN3_GND,
        }

    def initialize(self):
        """Initialize ADS1115 and GPIO for leads-off detection."""
        if not I2C_AVAILABLE:
            logger.error("smbus2 not available.")
            return False
        try:
            self.bus = smbus2.SMBus(self.bus_num)
            time.sleep(0.1)

            # Quick test write to config register
            self._write_config(self.channel)

            # GPIO for leads-off
            if GPIO_AVAILABLE:
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                GPIO.setup(LO_PLUS_PIN,  GPIO.IN)
                GPIO.setup(LO_MINUS_PIN, GPIO.IN)

            self.initialized = True
            logger.info(f"ADS1115 initialized at 0x{self.address:02X}, channel A{self.channel}.")
            return True

        except Exception as e:
            logger.error(f"ADS1115 init error: {e}")
            return False

    def _write_config(self, channel):
        """Write config register for single-ended read on given channel."""
        mux = self._mux_map.get(channel, MUX_AIN0_GND)
        config = (
            OS_SINGLE       |
            mux             |
            PGA_2_048V      |   # ±2.048V range — suits AD8232 output
            MODE_SINGLE     |
            DR_860SPS       |   # 860 samples/sec — max rate for ECG
            COMP_QUE_DISABLE
        )
        # Write 16-bit config as two bytes, big-endian
        self.bus.write_i2c_block_data(
            self.address,
            REG_CONFIG,
            [(config >> 8) & 0xFF, config & 0xFF]
        )

    def _read_conversion(self):
        """Read 16-bit conversion result."""
        # Wait for conversion to complete (~1.2ms at 860SPS)
        time.sleep(0.002)
        data = self.bus.read_i2c_block_data(self.address, REG_CONVERSION, 2)
        # Big-endian signed 16-bit
        raw = struct.unpack('>h', bytes(data))[0]
        return raw

    def read_raw(self):
        """
        Read raw 16-bit signed ADC value (-32768 to 32767).
        Returns (raw_value, leads_off).
        """
        if not self.initialized:
            return None, True
        try:
            self._write_config(self.channel)
            raw = self._read_conversion()
            leads_off = self._is_leads_off()
            return raw, leads_off
        except Exception as e:
            logger.error(f"ADS1115 read error: {e}")
            return None, True

    def read_voltage(self):
        """
        Read voltage in volts.
        Returns (voltage_V, leads_off).
        AD8232 output: ~0V to ~3.3V, resting ~1.65V
        """
        raw, leads_off = self.read_raw()
        if raw is None:
            return None, leads_off
        # Convert: full scale = PGA_VOLTAGE at 32767 counts
        voltage = raw * PGA_VOLTAGE / 32767.0
        return round(voltage, 4), leads_off

    def read_normalized(self):
        """
        Read as 0-1023 value (same scale as MCP3008 10-bit ADC).
        Makes it a drop-in replacement for the MCP3008-based driver.
        Returns (normalized_0_1023, leads_off).
        """
        voltage, leads_off = self.read_voltage()
        if voltage is None:
            return None, leads_off
        # Map 0V–3.3V to 0–1023
        normalized = int((voltage / 3.3) * 1023)
        normalized = max(0, min(1023, normalized))
        return normalized, leads_off

    def _is_leads_off(self):
        """Check AD8232 leads-off detection pins."""
        if not GPIO_AVAILABLE:
            return False
        try:
            return bool(GPIO.input(LO_PLUS_PIN) or GPIO.input(LO_MINUS_PIN))
        except Exception:
            return False

    def get_sample_rate(self):
        """Return configured sample rate in Hz."""
        return 860  # DR_860SPS

    def cleanup(self):
        """Close I2C bus and clean up GPIO."""
        if self.bus:
            try:
                self.bus.close()
            except Exception:
                pass
        if GPIO_AVAILABLE:
            try:
                GPIO.cleanup([LO_PLUS_PIN, LO_MINUS_PIN])
            except Exception:
                pass
        logger.info("ADS1115 cleaned up.")
