#!/usr/bin/env python3
"""
Configuration — all pins, addresses and settings in one place.
Edit this file if you need to change any hardware assignments.
"""

# ── I2C ──────────────────────────────────────────────────────────────────────
I2C_BUS           = 1
MAX30102_ADDRESS  = 0x57
MLX90614_ADDRESS  = 0x5A

# ── SPI / MCP3008 ─────────────────────────────────────────────────────────────
SPI_BUS           = 0
SPI_DEVICE        = 0
SPI_SPEED_HZ      = 1_350_000
ECG_CHANNEL       = 0        # MCP3008 channel for AD8232

# ── GPIO (BCM) ────────────────────────────────────────────────────────────────
CS_PIN            = 5        # MCP3008 software CS  (Pin 29)
LO_PLUS_PIN       = 22       # AD8232 LO+           (Pin 15)  moved from 17
LO_MINUS_PIN      = 27       # AD8232 LO-           (Pin 13)

# ── LCD35 ─────────────────────────────────────────────────────────────────────
LCD_FB_DEVICE     = "/dev/fb1"   # change to /dev/fb0 if needed

# ── Sampling ──────────────────────────────────────────────────────────────────
VITALS_INTERVAL_S = 5.0
ECG_SAMPLE_RATE   = 100     # Hz

# ── Alerts ────────────────────────────────────────────────────────────────────
ALERT_COOLDOWN_S  = 30
