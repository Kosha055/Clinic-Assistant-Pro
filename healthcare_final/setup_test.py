#!/usr/bin/env python3
"""
Hardware Setup Test — run this before main.py to verify everything works.
Usage: python3 setup_test.py
"""

import sys, time, os

def test_i2c():
    print("\n── I2C Bus Scan ─────────────────────────────")
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
        print("  MAX30102 (0x57):", "✅ Found" if '0x57' in found else "❌ NOT found")
        print("  MLX90614 (0x5a):", "✅ Found" if '0x5a' in found else "❌ NOT found")
    except ImportError:
        print("  ❌ smbus2 missing: pip install smbus2 --break-system-packages")
    except Exception as e:
        print(f"  ❌ I2C error: {e}")
        print("     Enable: sudo raspi-config → Interface Options → I2C")

def test_spi():
    print("\n── SPI / MCP3008 ────────────────────────────")
    try:
        import spidev, RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(5, GPIO.OUT, initial=GPIO.HIGH)

        spi = spidev.SpiDev()
        spi.open(0, 0)
        spi.max_speed_hz = 1350000
        spi.mode         = 0
        spi.no_cs        = True

        GPIO.output(5, GPIO.LOW)
        r = spi.xfer2([0x01, 0x80, 0x00])
        GPIO.output(5, GPIO.HIGH)

        val     = ((r[1] & 0x03) << 8) | r[2]
        voltage = val * 3.3 / 1023.0
        spi.close()
        GPIO.cleanup([5])
        print(f"  ✅ MCP3008 CH0: raw={val}, voltage={voltage:.3f}V")
        print(f"     (AD8232 resting ~1.65V / 511 raw)")
    except ImportError:
        print("  ❌ spidev/RPi.GPIO missing")
        print("     pip install spidev RPi.GPIO --break-system-packages")
    except Exception as e:
        print(f"  ❌ SPI error: {e}")
        print("     Enable: sudo raspi-config → Interface Options → SPI")

def test_gpio():
    print("\n── GPIO Leads-Off Pins ──────────────────────")
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(22, GPIO.IN)
        GPIO.setup(27, GPIO.IN)
        lo_plus  = GPIO.input(22)
        lo_minus = GPIO.input(27)
        print(f"  GPIO22 (LO+): {'HIGH - lead off' if lo_plus  else 'LOW  - lead connected'}")
        print(f"  GPIO27 (LO-): {'HIGH - lead off' if lo_minus else 'LOW  - lead connected'}")
        if not lo_plus and not lo_minus:
            print("  ✅ Both leads connected")
        else:
            print("  ⚠️  Check electrode placement")
        GPIO.cleanup()
    except ImportError:
        print("  ❌ RPi.GPIO missing")
    except Exception as e:
        print(f"  ❌ GPIO error: {e}")

def test_lcd():
    print("\n── LCD35 Display ────────────────────────────")
    try:
        import os
        fbs = [f for f in os.listdir('/dev') if f.startswith('fb')]
        print(f"  Framebuffer devices: {fbs}")

        os.environ['SDL_VIDEODRIVER'] = 'fbcon'
        os.environ['SDL_FBDEV']       = '/dev/fb1'
        os.environ['SDL_NOMOUSE']     = '1'
        import pygame
        pygame.init()
        s = pygame.display.set_mode((480, 320))
        s.fill((10, 10, 30))
        f = pygame.font.SysFont('monospace', 36, bold=True)
        t = f.render("LCD35 TEST OK", True, (0, 255, 100))
        s.blit(t, (60, 140))
        pygame.display.flip()
        print("  ✅ pygame framebuffer OK — check LCD35 for blue screen with text")
        time.sleep(3)
        pygame.quit()
    except Exception as e:
        print(f"  ❌ LCD error: {e}")
        print("     Try changing SDL_FBDEV to /dev/fb0")

def test_mlx90614():
    print("\n── MLX90614 Temperature ─────────────────────")
    try:
        sys.path.insert(0, '.')
        from sensors.mlx90614_sensor import MLX90614Sensor
        s = MLX90614Sensor()
        if s.initialize():
            obj, amb = s.read()
            print(f"  Object temp:  {obj}°C")
            print(f"  Ambient temp: {amb}°C")
            if obj and 30 <= obj <= 42:
                print("  ✅ Readings look normal")
            else:
                print("  ⚠️  Point sensor at skin from ~5cm")
            s.cleanup()
        else:
            print("  ❌ Init failed — check wiring")
    except Exception as e:
        print(f"  ❌ MLX90614 error: {e}")

if __name__ == '__main__':
    print("=" * 52)
    print("  Healthcare Assistant — Hardware Setup Test")
    print("=" * 52)
    test_i2c()
    test_spi()
    test_gpio()
    test_mlx90614()
    test_lcd()
    print("\n" + "=" * 52)
    print("  Done. Fix any ❌ before running main.py")
    print("=" * 52 + "\n")
