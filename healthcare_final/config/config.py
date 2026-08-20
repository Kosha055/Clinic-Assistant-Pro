#!/usr/bin/env python3
"""
Configuration — all pins, addresses and settings in one place.
"""

# ── I2C ───────────────────────────────────────────────────────────────────────
I2C_BUS           = 1

# I2C Addresses
MAX30102_ADDRESS  = 0x57   # MAX30102  SpO2 + HR
MLX90614_ADDRESS  = 0x5A   # MLX90614  IR Temperature
ADS1115_ADDRESS   = 0x48   # ADS1115   ADC for AD8232 ECG
                           # (ADDR pin -> GND = 0x48)
                           # (ADDR pin -> VDD = 0x49)
                           # (ADDR pin -> SDA = 0x4A)
                           # (ADDR pin -> SCL = 0x4B)

# ── ADS1115 ───────────────────────────────────────────────────────────────────
ADS1115_CHANNEL   = 0      # A0 = ECG signal from AD8232
ADS1115_GAIN      = 2.048  # PGA ±2.048V — suits AD8232 output (0–3.3V)
ADS1115_RATE      = 860    # Samples per second (max = 860)

# ── GPIO (BCM) ────────────────────────────────────────────────────────────────
LO_PLUS_PIN       = 22     # AD8232 LO+  (Pin 15) — moved from GPIO17 (LCD35)
LO_MINUS_PIN      = 27     # AD8232 LO-  (Pin 13)

# ── LCD35 ─────────────────────────────────────────────────────────────────────
LCD_FB_DEVICE     = "/dev/fb1"   # change to /dev/fb0 if needed

# ── Sampling ──────────────────────────────────────────────────────────────────
VITALS_INTERVAL_S = 5.0    # seconds between SpO2/HR/Temp reads
ECG_SAMPLE_RATE   = 860    # Hz — ADS1115 max rate

# ── Alerts ────────────────────────────────────────────────────────────────────
ALERT_COOLDOWN_S  = 30     # minimum seconds between same alert type
