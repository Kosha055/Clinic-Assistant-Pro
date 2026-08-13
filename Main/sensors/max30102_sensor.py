#!/usr/bin/env python3
"""
MAX30102 Sensor Driver
Measures SpO2 (blood oxygen) and Heart Rate via I2C
I2C Address: 0x57
Pins: SDA (GPIO2 / Pin 3), SCL (GPIO3 / Pin 5)
"""

import time
import logging
import struct

try:
    import smbus2
    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False
    logging.warning("smbus2 not found. Install with: pip install smbus2")

logger = logging.getLogger(__name__)

# MAX30102 Register Map
MAX30102_ADDRESS        = 0x57
REG_INTR_STATUS_1       = 0x00
REG_INTR_STATUS_2       = 0x01
REG_INTR_ENABLE_1       = 0x02
REG_INTR_ENABLE_2       = 0x03
REG_FIFO_WR_PTR         = 0x04
REG_OVF_COUNTER         = 0x05
REG_FIFO_RD_PTR         = 0x06
REG_FIFO_DATA           = 0x07
REG_FIFO_CONFIG         = 0x08
REG_MODE_CONFIG         = 0x09
REG_SPO2_CONFIG         = 0x0A
REG_LED1_PA             = 0x0C   # Red LED
REG_LED2_PA             = 0x0D   # IR LED
REG_PILOT_PA            = 0x10
REG_MULTI_LED_CTRL1     = 0x11
REG_MULTI_LED_CTRL2     = 0x12
REG_TEMP_INTR           = 0x1F
REG_TEMP_FRAC           = 0x20
REG_TEMP_CONFIG         = 0x21
REG_PART_ID             = 0xFF

EXPECTED_PART_ID        = 0x15

# IR and Red buffer size for SpO2/HR calculation
BUFFER_SIZE = 100


class MAX30102Sensor:
    def __init__(self, i2c_bus=1, address=MAX30102_ADDRESS):
        self.bus_num = i2c_bus
        self.address = address
        self.bus = None
        self.initialized = False

        # Circular buffers for signal processing
        self.ir_buffer = []
        self.red_buffer = []

        self._heart_rate = 0
        self._spo2 = 0

    def initialize(self):
        """Initialize the MAX30102 sensor."""
        if not I2C_AVAILABLE:
            logger.error("smbus2 library not available.")
            return False
        try:
            self.bus = smbus2.SMBus(self.bus_num)
            time.sleep(0.1)

            # Verify part ID
            part_id = self.bus.read_byte_data(self.address, REG_PART_ID)
            if part_id != EXPECTED_PART_ID:
                logger.error(f"Unexpected Part ID: 0x{part_id:02X} (expected 0x{EXPECTED_PART_ID:02X})")
                return False

            self._reset()
            self._setup()
            self.initialized = True
            logger.info("MAX30102 initialized successfully.")
            return True

        except Exception as e:
            logger.error(f"MAX30102 initialization error: {e}")
            return False

    def _reset(self):
        """Software reset the sensor."""
        self.bus.write_byte_data(self.address, REG_MODE_CONFIG, 0x40)
        time.sleep(0.1)

    def _setup(self):
        """Configure sensor for SpO2 + HR measurement."""
        # Interrupt: FIFO almost full
        self.bus.write_byte_data(self.address, REG_INTR_ENABLE_1, 0xC0)
        self.bus.write_byte_data(self.address, REG_INTR_ENABLE_2, 0x00)

        # FIFO: 4 samples avg, FIFO rollover on, almost full at 17
        self.bus.write_byte_data(self.address, REG_FIFO_WR_PTR, 0x00)
        self.bus.write_byte_data(self.address, REG_OVF_COUNTER, 0x00)
        self.bus.write_byte_data(self.address, REG_FIFO_RD_PTR, 0x00)
        self.bus.write_byte_data(self.address, REG_FIFO_CONFIG, 0x4F)

        # Mode: SpO2 (0x03)
        self.bus.write_byte_data(self.address, REG_MODE_CONFIG, 0x03)

        # SpO2: ADC range 4096nA, 100 samples/sec, 411μs pulse width
        self.bus.write_byte_data(self.address, REG_SPO2_CONFIG, 0x27)

        # LED pulse amplitude
        self.bus.write_byte_data(self.address, REG_LED1_PA, 0x24)   # Red ~7mA
        self.bus.write_byte_data(self.address, REG_LED2_PA, 0x24)   # IR  ~7mA
        self.bus.write_byte_data(self.address, REG_PILOT_PA, 0x7F)

    def _read_fifo(self):
        """Read one sample from FIFO. Returns (red, ir)."""
        # Read 6 bytes: 3 for Red, 3 for IR
        data = self.bus.read_i2c_block_data(self.address, REG_FIFO_DATA, 6)
        red = (data[0] << 16 | data[1] << 8 | data[2]) & 0x3FFFF
        ir  = (data[3] << 16 | data[4] << 8 | data[5]) & 0x3FFFF
        return red, ir

    def _calculate_hr_spo2(self):
        """
        Simple heart rate and SpO2 calculation.
        For production use, replace with a proper algorithm (e.g., Pan-Tompkins).
        """
        if len(self.ir_buffer) < BUFFER_SIZE:
            return None, None

        ir = self.ir_buffer[-BUFFER_SIZE:]
        red = self.red_buffer[-BUFFER_SIZE:]

        # Basic SpO2: ratio of AC/DC components
        ir_max = max(ir)
        ir_min = min(ir)
        red_max = max(red)
        red_min = min(red)

        if ir_max == ir_min or red_max == red_min:
            return None, None

        ir_ac = ir_max - ir_min
        ir_dc = sum(ir) / len(ir)
        red_ac = red_max - red_min
        red_dc = sum(red) / len(red)

        # R ratio for SpO2 (empirical formula)
        R = (red_ac / red_dc) / (ir_ac / ir_dc)
        spo2 = 110 - 25 * R
        spo2 = max(85.0, min(100.0, spo2))

        # Simple HR: detect peaks in IR signal
        threshold = ir_min + (ir_max - ir_min) * 0.6
        peaks = 0
        above = False
        for val in ir:
            if val > threshold and not above:
                peaks += 1
                above = True
            elif val < threshold:
                above = False

        # peaks / (BUFFER_SIZE / sample_rate) * 60
        sample_rate = 100  # samples per second
        duration = BUFFER_SIZE / sample_rate
        hr = (peaks / duration) * 60
        hr = max(40, min(200, hr))

        return round(hr, 1), round(spo2, 1)

    def read(self):
        """
        Read and return (heart_rate, spo2).
        Returns (None, None) if not enough data yet.
        """
        if not self.initialized:
            return None, None

        try:
            # Read several FIFO samples
            for _ in range(10):
                red, ir = self._read_fifo()
                self.ir_buffer.append(ir)
                self.red_buffer.append(red)

            # Keep buffer bounded
            if len(self.ir_buffer) > BUFFER_SIZE * 2:
                self.ir_buffer = self.ir_buffer[-BUFFER_SIZE:]
                self.red_buffer = self.red_buffer[-BUFFER_SIZE:]

            hr, spo2 = self._calculate_hr_spo2()
            if hr and spo2:
                self._heart_rate = hr
                self._spo2 = spo2

            return self._heart_rate, self._spo2

        except Exception as e:
            logger.error(f"MAX30102 read error: {e}")
            return None, None

    def cleanup(self):
        """Close I2C bus."""
        if self.bus:
            try:
                # Power down
                self.bus.write_byte_data(self.address, REG_MODE_CONFIG, 0x80)
                self.bus.close()
            except Exception:
                pass
        logger.info("MAX30102 cleaned up.")
