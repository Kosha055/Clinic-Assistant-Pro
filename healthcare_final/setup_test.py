#!/usr/bin/env python3
"""
Hardware Setup Test
Run this before main.py to verify all sensors and display are working.
Usage: python3 setup_test.py
"""

import sys, time, os
sys.path.insert(0, '.')


def test_i2c():
    print("\n── I2C Bus Scan ─────────────────────────────────")
    try:
        import smbus2
        bus   = smbus2.SMBus(1)
        found = []
        for addr in range(0x08, 0x78):
            try:
                bus.read_byte(addr)
                found.append(hex(addr))
            except Exception:
                pass
        bus.close()

        print(f"  Devices found: {found}")
        print("  MAX30102  (0x57):", "✅ Found" if '0x57' in found else "❌ NOT found")
        print("  MLX90614  (0x5a):", "✅ Found" if '0x5a' in found else "❌ NOT found")
        print("  ADS1115   (0x48):", "✅ Found" if '0x48' in found else "❌ NOT found")

        if '0x48' not in found:
            print("     → Check ADDR pin: GND=0x48, VDD=0x49, SDA=0x4A, SCL=0x4B")

    except ImportError:
        print("  ❌ smbus2 missing: pip install smbus2 --break-system-packages")
    except Exception as e:
        print(f"  ❌ I2C error: {e}")
        print("     Enable: sudo raspi-config → Interface Options → I2C")


def test_ads1115():
    print("\n── ADS1115 ADC Test ─────────────────────────────")
    try:
        from sensors.ads1115_sensor import ADS1115Sensor
        adc = ADS1115Sensor(channel=0)
        if adc.initialize():
            print("  Reading 5 ECG samples from ADS1115 A0...")
            for i in range(5):
                raw,  leads_off = adc.read_raw()
                norm, _         = adc.read_normalized()
                volt, _         = adc.read_voltage()
                print(f"  [{i+1}] raw={raw:6d} | normalized={norm:4d} | voltage={volt:.4f}V | leads_off={leads_off}")
                time.sleep(0.1)

            if volt is not None and 0.5 <= volt <= 2.8:
                print("  ✅ ADS1115 reading looks reasonable for AD8232")
            else:
                print("  ⚠️  Voltage outside expected range — check AD8232 wiring")
            adc.cleanup()
        else:
            print("  ❌ ADS1115 init failed")
    except Exception as e:
        print(f"  ❌ ADS1115 error: {e}")


def test_gpio():
    print("\n── GPIO Leads-Off Pins ──────────────────────────")
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(22, GPIO.IN)
        GPIO.setup(27, GPIO.IN)

        lo_plus  = GPIO.input(22)
        lo_minus = GPIO.input(27)
        print(f"  GPIO22 (LO+): {'HIGH — lead off'      if lo_plus  else 'LOW  — lead connected'}")
        print(f"  GPIO27 (LO-): {'HIGH — lead off'      if lo_minus else 'LOW  — lead connected'}")

        if not lo_plus and not lo_minus:
            print("  ✅ Both leads connected")
        else:
            print("  ⚠️  Check electrode placement on AD8232")
        GPIO.cleanup()

    except ImportError:
        print("  ❌ RPi.GPIO missing: pip install RPi.GPIO --break-system-packages")
    except Exception as e:
        print(f"  ❌ GPIO error: {e}")


def test_mlx90614():
    print("\n── MLX90614 Temperature ─────────────────────────")
    try:
        from sensors.mlx90614_sensor import MLX90614Sensor
        s = MLX90614Sensor()
        if s.initialize():
            obj, amb = s.read()
            print(f"  Object (body) temp: {obj}°C")
            print(f"  Ambient temp:       {amb}°C")
            if obj and 30 <= obj <= 42:
                print("  ✅ Reading looks normal")
            else:
                print("  ⚠️  Point sensor at skin from ~5cm")
            s.cleanup()
        else:
            print("  ❌ Init failed — check wiring")
    except Exception as e:
        print(f"  ❌ MLX90614 error: {e}")


def test_max30102():
    print("\n── MAX30102 SpO2 + HR ───────────────────────────")
    try:
        from sensors.max30102_sensor import MAX30102Sensor
        s = MAX30102Sensor()
        if s.initialize():
            print("  Place finger on sensor...")
            time.sleep(2)
            hr, spo2 = s.read()
            print(f"  Heart Rate: {hr} bpm")
            print(f"  SpO2:       {spo2} %")
            print("  ✅ MAX30102 reading OK")
            s.cleanup()
        else:
            print("  ❌ Init failed — check wiring")
    except Exception as e:
        print(f"  ❌ MAX30102 error: {e}")


def test_lcd():
    print("\n── LCD35 Display ────────────────────────────────")
    try:
        fbs = [f for f in os.listdir('/dev') if f.startswith('fb')]
        print(f"  Framebuffer devices: {fbs}")

        import numpy as np
        os.environ['SDL_VIDEODRIVER'] = 'offscreen'
        os.environ['SDL_NOMOUSE']     = '1'
        import pygame
        pygame.init()

        s = pygame.Surface((480, 320))
        s.fill((10, 10, 30))
        f = pygame.font.SysFont('monospace', 36, bold=True)
        t = f.render("LCD35 TEST OK", True, (0, 255, 100))
        s.blit(t, (60, 140))

        raw24  = pygame.image.tostring(s, 'RGB')
        arr    = np.frombuffer(raw24, dtype=np.uint8).reshape((320, 480, 3))
        r, g, b = arr[:,:,0].astype(np.uint16), arr[:,:,1].astype(np.uint16), arr[:,:,2].astype(np.uint16)
        rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)

        with open('/dev/fb1', 'wb') as fb:
            fb.write(rgb565.tobytes())

        print("  ✅ Written to /dev/fb1 — check LCD35 for dark blue screen with text")
        pygame.quit()

    except Exception as e:
        print(f"  ❌ LCD error: {e}")


if __name__ == '__main__':
    print("=" * 52)
    print("  Healthcare Assistant — Hardware Setup Test")
    print("=" * 52)

    test_i2c()
    test_ads1115()
    test_gpio()
    test_mlx90614()
    test_max30102()
    test_lcd()

    print("\n" + "=" * 52)
    print("  Done. Fix any ❌ before running main.py")
    print("=" * 52 + "\n")
