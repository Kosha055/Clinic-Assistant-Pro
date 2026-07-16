# 🏥 Healthcare Assistant — Raspberry Pi 4B

A multi-sensor healthcare monitoring system using:
- **MAX30102** — SpO₂ & Heart Rate (I²C)
- **MLX90614** — Non-contact Body Temperature (I²C)
- **AD8232** — ECG Signal via **MCP3008 ADC** (SPI)

---

## 📦 Project Structure

```
healthcare_assistant/
├── main.py                  # Entry point — starts all sensor threads
├── setup_test.py            # Hardware verification script
├── requirements.txt
├── sensors/
│   ├── max30102_sensor.py   # MAX30102 driver (SpO2 + HR)
│   ├── mlx90614_sensor.py   # MLX90614 driver (IR temperature)
│   └── ad8232_sensor.py     # AD8232 driver via MCP3008 SPI
├── data/
│   └── data_logger.py       # CSV + JSON logging
├── alerts/
│   └── alert_manager.py     # Clinical threshold alerts
├── display/
│   └── console_display.py   # Live terminal display + ASCII ECG
└── config/
    └── config.py            # All settings in one place
```

---

## 🔌 Wiring Guide

### I²C — Shared Bus (MAX30102 + MLX90614)

| Signal | Raspberry Pi Pin |
|--------|-----------------|
| SDA    | Pin 3 (GPIO2)   |
| SCL    | Pin 5 (GPIO3)   |
| 3.3V   | Pin 1 or 17     |
| GND    | Pin 6, 9, etc.  |

> Both MAX30102 (0x57) and MLX90614 (0x5A) share the same I²C bus — they have different addresses so no conflict.

### SPI — MCP3008 ADC (for AD8232 ECG)

| MCP3008 Pin | Raspberry Pi    |
|-------------|-----------------|
| VDD / VREF  | 3.3V            |
| AGND / DGND | GND             |
| CLK         | Pin 23 (GPIO11/SCLK) |
| DOUT (MISO) | Pin 21 (GPIO9/MISO)  |
| DIN  (MOSI) | Pin 19 (GPIO10/MOSI) |
| CS/SHDN     | Pin 24 (GPIO8/CE0)   |

### AD8232 → MCP3008 → Raspberry Pi

| AD8232 Pin | Connects To       |
|------------|-------------------|
| OUTPUT     | MCP3008 CH0       |
| LO+        | GPIO17 (Pin 11)   |
| LO-        | GPIO27 (Pin 13)   |
| 3.3V       | 3.3V              |
| GND        | GND               |

---

## ⚙️ Raspberry Pi Setup

### 1. Enable I²C and SPI

```bash
sudo raspi-config
# Interface Options → I2C → Enable
# Interface Options → SPI → Enable
sudo reboot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

### 3. Verify I²C devices

```bash
i2cdetect -y 1
# Should show:  0x57 (MAX30102)  and  0x5a (MLX90614)
```

### 4. Run hardware test

```bash
python3 setup_test.py
```

### 5. Run the assistant

```bash
python3 main.py
```

---

## 📊 Output

**Live Console:**
```
  ⏱  2024-01-15T10:23:45
  Heart Rate                72.3 bpm             ✅ Normal
  SpO₂                      98.1 %               ✅ Normal
  Body Temp (Object)        36.8 °C              ✅ Normal
  Ambient Temp              24.2 °C
```

**Log files** (in `logs/` directory):
- `vitals_YYYYMMDD_HHMMSS.csv` — timestamped vitals readings
- `ecg_YYYYMMDD_HHMMSS.csv` — raw ECG samples at ~100Hz
- `alerts.json` — all triggered alerts

---

## 🚨 Alert Thresholds

| Metric         | Warning           | Critical         |
|----------------|-------------------|------------------|
| Heart Rate     | <50 or >120 bpm   | <40 or >150 bpm  |
| SpO₂           | <94%              | <90%             |
| Body Temp      | <36°C or >37.5°C  | <35°C or >38.5°C |
| ECG Leads      | Leads off warning | —                |

Edit thresholds in `alerts/alert_manager.py`.

---

## ⚠️ Notes

- This is a **prototype/educational project** — not a certified medical device.
- The SpO₂/HR algorithm in `max30102_sensor.py` is a simplified estimation. For clinical use, replace with a validated algorithm (e.g., Maxim's reference algorithm or pyMaxim).
- ECG interpretation requires trained personnel and proper lead placement.
- Always consult a medical professional for health decisions.
