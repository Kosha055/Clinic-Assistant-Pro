#!/usr/bin/env python3
"""
MAX30102 Sensor Driver
SpO2 + Heart Rate via I2C
Address: 0x57
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

# Registers
MAX30102_ADDRESS  = 0x57
REG_INTR_ENABLE_1 = 0x02
REG_INTR_ENABLE_2 = 0x03
REG_FIFO_WR_PTR   = 0x04
REG_OVF_COUNTER   = 0x05
REG_FIFO_RD_PTR   = 0x06
REG_FIFO_DATA     = 0x07
REG_FIFO_CONFIG   = 0x08
REG_MODE_CONFIG   = 0x09
REG_SPO2_CONFIG   = 0x0A
REG_LED1_PA       = 0x0C
REG_LED2_PA       = 0x0D
REG_PILOT_PA      = 0x10
REG_PART_ID       = 0xFF
EXPECTED_PART_ID  = 0x15

BUFFER_SIZE = 100


class MAX30102Sensor:
    def __init__(self, i2c_bus=1, address=MAX30102_ADDRESS):
        self.bus_num     = i2c_bus
        self.address     = address
        self.bus         = None
        self.initialized = False
        self.ir_buffer   = []
        self.red_buffer  = []
        self._heart_rate = 0
        self._spo2       = 0

    def initialize(self):
        if not I2C_AVAILABLE:
            logger.error("smbus2 not available. Run: pip install smbus2 --break-system-packages")
            return False
        try:
            self.bus = smbus2.SMBus(self.bus_num)
            time.sleep(0.1)
            part_id = self.bus.read_byte_data(self.address, REG_PART_ID)
            if part_id != EXPECTED_PART_ID:
                logger.error(f"Wrong Part ID: 0x{part_id:02X}")
                return False
            self._reset()
            self._setup()
            self.initialized = True
            logger.info("MAX30102 initialized.")
            return True
        except Exception as e:
            logger.error(f"MAX30102 init error: {e}")
            return False

    def _reset(self):
        self.bus.write_byte_data(self.address, REG_MODE_CONFIG, 0x40)
        time.sleep(0.1)

    def _setup(self):
        self.bus.write_byte_data(self.address, REG_INTR_ENABLE_1, 0xC0)
        self.bus.write_byte_data(self.address, REG_INTR_ENABLE_2, 0x00)
        self.bus.write_byte_data(self.address, REG_FIFO_WR_PTR,   0x00)
        self.bus.write_byte_data(self.address, REG_OVF_COUNTER,   0x00)
        self.bus.write_byte_data(self.address, REG_FIFO_RD_PTR,   0x00)
        self.bus.write_byte_data(self.address, REG_FIFO_CONFIG,   0x4F)
        self.bus.write_byte_data(self.address, REG_MODE_CONFIG,   0x03)
        self.bus.write_byte_data(self.address, REG_SPO2_CONFIG,   0x27)
        self.bus.write_byte_data(self.address, REG_LED1_PA,       0x24)
        self.bus.write_byte_data(self.address, REG_LED2_PA,       0x24)
        self.bus.write_byte_data(self.address, REG_PILOT_PA,      0x7F)

    def _read_fifo(self):
        data = self.bus.read_i2c_block_data(self.address, REG_FIFO_DATA, 6)
        red  = (data[0] << 16 | data[1] << 8 | data[2]) & 0x3FFFF
        ir   = (data[3] << 16 | data[4] << 8 | data[5]) & 0x3FFFF
        return red, ir

    def _calculate_hr_spo2(self):
        if len(self.ir_buffer) < BUFFER_SIZE:
            return None, None

        ir  = self.ir_buffer[-BUFFER_SIZE:]
        red = self.red_buffer[-BUFFER_SIZE:]

        ir_max  = max(ir);  ir_min  = min(ir)
        red_max = max(red); red_min = min(red)

        if ir_max == ir_min or red_max == red_min:
            return None, None

        ir_ac  = ir_max - ir_min
        ir_dc  = sum(ir)  / len(ir)
        red_ac = red_max - red_min
        red_dc = sum(red) / len(red)

        R    = (red_ac / red_dc) / (ir_ac / ir_dc)
        spo2 = max(85.0, min(100.0, 110 - 25 * R))

        threshold = ir_min + (ir_max - ir_min) * 0.6
        peaks = 0
        above = False
        for val in ir:
            if val > threshold and not above:
                peaks += 1; above = True
            elif val < threshold:
                above = False

        hr = max(40, min(200, (peaks / (BUFFER_SIZE / 100)) * 60))
        return round(hr, 1), round(spo2, 1)

    def read(self):
        if not self.initialized:
            return None, None
        try:
            for _ in range(10):
                red, ir = self._read_fifo()
                self.ir_buffer.append(ir)
                self.red_buffer.append(red)

            if len(self.ir_buffer) > BUFFER_SIZE * 2:
                self.ir_buffer  = self.ir_buffer[-BUFFER_SIZE:]
                self.red_buffer = self.red_buffer[-BUFFER_SIZE:]

            hr, spo2 = self._calculate_hr_spo2()
            if hr and spo2:
                self._heart_rate = hr
                self._spo2       = spo2

            return self._heart_rate, self._spo2
        except Exception as e:
            logger.error(f"MAX30102 read error: {e}")
            return None, None

    def cleanup(self):
        if self.bus:
            try:
                self.bus.write_byte_data(self.address, REG_MODE_CONFIG, 0x80)
                self.bus.close()
            except Exception:
                pass
        logger.info("MAX30102 cleaned up.")
