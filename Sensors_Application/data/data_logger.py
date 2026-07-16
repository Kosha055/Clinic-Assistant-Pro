#!/usr/bin/env python3
"""
Data Logger
Logs vitals and ECG data to CSV files with timestamps.
"""

import csv
import json
import os
import logging
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

LOG_DIR = "logs"
VITALS_CSV  = os.path.join(LOG_DIR, "vitals.csv")
ECG_CSV     = os.path.join(LOG_DIR, "ecg.csv")
ALERTS_LOG  = os.path.join(LOG_DIR, "alerts.json")

VITALS_FIELDS = ['timestamp', 'heart_rate', 'spo2', 'body_temp', 'ambient_temp']
ECG_CHUNK_SIZE = 500   # Write ECG to CSV every N samples


class DataLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)

        self._ecg_buffer = deque()
        self._session_start = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use session-timestamped filenames
        self.vitals_file = os.path.join(LOG_DIR, f"vitals_{self._session_start}.csv")
        self.ecg_file    = os.path.join(LOG_DIR, f"ecg_{self._session_start}.csv")

        self._init_vitals_csv()
        self._init_ecg_csv()

        logger.info(f"DataLogger initialized. Logs directory: {LOG_DIR}/")

    def _init_vitals_csv(self):
        """Create vitals CSV with header."""
        with open(self.vitals_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=VITALS_FIELDS)
            writer.writeheader()

    def _init_ecg_csv(self):
        """Create ECG CSV with header."""
        with open(self.ecg_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'ecg_raw', 'ecg_voltage'])

    def log_vitals(self, data: dict):
        """Append a vitals reading to CSV."""
        try:
            row = {field: data.get(field, '') for field in VITALS_FIELDS}
            with open(self.vitals_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=VITALS_FIELDS)
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Vitals log error: {e}")

    def log_ecg(self, ecg_raw: int):
        """Buffer ECG sample; flush to CSV periodically."""
        ts = datetime.now().isoformat()
        voltage = round(ecg_raw * 3.3 / 1023.0, 4) if ecg_raw is not None else ''
        self._ecg_buffer.append((ts, ecg_raw, voltage))

        if len(self._ecg_buffer) >= ECG_CHUNK_SIZE:
            self._flush_ecg()

    def _flush_ecg(self):
        """Write buffered ECG samples to CSV."""
        try:
            with open(self.ecg_file, 'a', newline='') as f:
                writer = csv.writer(f)
                while self._ecg_buffer:
                    writer.writerow(self._ecg_buffer.popleft())
        except Exception as e:
            logger.error(f"ECG flush error: {e}")

    def log_alert(self, alert: dict):
        """Append alert to JSON log."""
        try:
            alerts = []
            if os.path.exists(ALERTS_LOG):
                with open(ALERTS_LOG, 'r') as f:
                    alerts = json.load(f)
            alerts.append(alert)
            with open(ALERTS_LOG, 'w') as f:
                json.dump(alerts, f, indent=2)
        except Exception as e:
            logger.error(f"Alert log error: {e}")

    def close(self):
        """Flush remaining data on shutdown."""
        self._flush_ecg()
        logger.info("DataLogger closed and flushed.")
