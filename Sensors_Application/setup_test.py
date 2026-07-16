#!/usr/bin/env python3
"""
Setup & Hardware Test Script
Run this first to verify wiring and sensor connectivity.
Usage: python3 setup_test.py
"""

import sys
import time


def test_i2c_bus():
    """Scan I2C bus and check for expected devices."""
    print("\n── I2C Bus Scan ────────────────────────────────────────")
    try:
        import smbus2
        bus = smbus2.SMBus(1)
        found = []
        for addr in range(0x08, 0x78):
            try:
                bus.read_byte(addr)
                found.append(hex(addr))
            except Exception:
                pass
        bus.close()

        print(f"  Devices found: {found if found else 'None'}")
        if '0x57' in found:
            print("  ✅ MAX30102 found at 0x57")
        else:
            print("  ❌ MAX30102 NOT found at 0x57 — check wiring")
        if '0x5a' in found:
            print("  ✅ MLX90614 found at 0x5a")
        else:
            print("  ❌ MLX90614 NOT found at 0x5a — check wiring")

    except ImportError:
        print("  ❌ smbus2 not installed. Run: pip install smbus2")
    except Exception as e:
        print(f"  ❌ I2C error: {e}")
        print("     Enable I2C: sudo raspi-config → Interface Options → I2C")


def test_spi():
    """Verify SPI is available and MCP3008 responds."""
    print("\n── SPI / MCP3008 Test ──────────────────────────────────")
    try:
        import spidev
        spi = spidev.SpiDev()
        spi.open(0, 0)
        spi.max_speed_hz = 1350000
        spi.mode = 0

        # Read CH0
        cmd = [0x01, 0x80, 0x00]
        resp = spi.xfer2(cmd)
        value = ((resp[1] & 0x03) << 8) | resp[2]
        voltage = value * 3.3 / 1023.0

        spi.close()
        print(f"  ✅ MCP3008 CH0 read OK: raw={value}, voltage={voltage:.3f}V")
        print(f"     (AD8232 resting ECG should be ~1.65V / 511 raw)")

    except ImportError:
        print("  ❌ spidev not installed. Run: pip install spidev")
    except Exception as e:
        print(f"  ❌ SPI error: {e}")
        print("     Enable SPI: sudo raspi-config → Interface Options → SPI")


def test_gpio():
    """Test GPIO leads-off detection pins."""
    print("\n── GPIO / Leads-Off Pin Test ───────────────────────────")
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(17, GPIO.IN)
        GPIO.setup(27, GPIO.IN)

        lo_plus  = GPIO.input(17)
        lo_minus = GPIO.input(27)

        print(f"  GPIO17 (LO+):  {'HIGH - lead off' if lo_plus  else 'LOW  - lead connected'}")
        print(f"  GPIO27 (LO-):  {'HIGH - lead off' if lo_minus else 'LOW  - lead connected'}")

        if not lo_plus and not lo_minus:
            print("  ✅ Both leads appear connected")
        else:
            print("  ⚠️  Check electrode placement on AD8232")

        GPIO.cleanup()

    except ImportError:
        print("  ❌ RPi.GPIO not installed. Run: pip install RPi.GPIO")
    except Exception as e:
        print(f"  ❌ GPIO error: {e}")


def test_max30102():
    """Quick MAX30102 sensor read test."""
    print("\n── MAX30102 Sensor Test ────────────────────────────────")
    try:
        sys.path.insert(0, '.')
        from sensors.max30102_sensor import MAX30102Sensor
        sensor = MAX30102Sensor()
        if sensor.initialize():
            time.sleep(2)
            hr, spo2 = sensor.read()
            print(f"  Heart Rate: {hr} bpm")
            print(f"  SpO2:       {spo2} %")
            print("  ✅ MAX30102 reading OK (place finger on sensor for accurate values)")
            sensor.cleanup()
        else:
            print("  ❌ MAX30102 initialization failed")
    except Exception as e:
        print(f"  ❌ MAX30102 test error: {e}")


def test_mlx90614():
    """Quick MLX90614 sensor read test."""
    print("\n── MLX90614 Sensor Test ────────────────────────────────")
    try:
        from sensors.mlx90614_sensor import MLX90614Sensor
        sensor = MLX90614Sensor()
        if sensor.initialize():
            obj_temp, amb_temp = sensor.read()
            print(f"  Object (Body) Temp: {obj_temp}°C")
            print(f"  Ambient Temp:       {amb_temp}°C")
            if 20 <= amb_temp <= 35 and 30 <= obj_temp <= 42:
                print("  ✅ MLX90614 readings look reasonable")
            else:
                print("  ⚠️  Values outside expected range — check sensor distance (~5cm)")
            sensor.cleanup()
        else:
            print("  ❌ MLX90614 initialization failed")
    except Exception as e:
        print(f"  ❌ MLX90614 test error: {e}")


if __name__ == '__main__':
    print("=" * 58)
    print("  Healthcare Assistant — Hardware Setup Test")
    print("=" * 58)

    test_i2c_bus()
    test_spi()
    test_gpio()
    test_max30102()
    test_mlx90614()

    print("\n" + "=" * 58)
    print("  Test complete. Fix any ❌ errors before running main.py")
    print("=" * 58 + "\n")
