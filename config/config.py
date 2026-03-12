#!/usr/bin/env python3
"""
Healthcare Assistant Configuration
Edit this file to customize sensor settings, thresholds, and behavior.
"""

# ── I2C Settings ────────────────────────────────────────────────────────────
I2C_BUS             = 1          # Raspberry Pi default I2C bus

MAX30102_ADDRESS    = 0x57       # MAX30102 I2C address
MLX90614_ADDRESS    = 0x5A       # MLX90614 I2C address (default)

# ── SPI Settings (MCP3008 for AD8232 ECG) ───────────────────────────────────
SPI_BUS             = 0          # SPI bus 0
SPI_DEVICE          = 0          # CE0 (GPIO8, Pin 24)
SPI_SPEED_HZ        = 1_350_000  # 1.35 MHz (MCP3008 max at 3.3V)
ECG_CHANNEL         = 0          # MCP3008 channel for AD8232 output

# ── GPIO Pins (BCM numbering) ────────────────────────────────────────────────
LO_PLUS_PIN         = 17         # AD8232 Leads-Off LO+ detection
LO_MINUS_PIN        = 27         # AD8232 Leads-Off LO- detection

# ── Sampling Rates ───────────────────────────────────────────────────────────
VITALS_INTERVAL_S   = 5.0        # Read SpO2/HR/Temp every N seconds
ECG_SAMPLE_RATE_HZ  = 100        # Target ECG samples per second

# ── Alert Cooldown ───────────────────────────────────────────────────────────
ALERT_COOLDOWN_S    = 30         # Minimum seconds between same alert type

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR             = "logs"
LOG_LEVEL           = "INFO"     # DEBUG | INFO | WARNING | ERROR
