#!/usr/bin/env python3
"""
Data Logger — CSV logging for vitals and ECG.
"""

import csv
import os
import logging
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

LOG_DIR        = "logs"
VITALS_FIELDS  = ['timestamp', 'heart_rate', 'spo2', 'body_temp', 'ambient_temp']
ECG_CHUNK_SIZE = 500


class DataLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        session = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.vitals_file = os.path.join(LOG_DIR, f"vitals_{session}.csv")
        self.ecg_file    = os.path.join(LOG_DIR, f"ecg_{session}.csv")
        self._ecg_buffer = deque()

        with open(self.vitals_file, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=VITALS_FIELDS).writeheader()

        with open(self.ecg_file, 'w', newline='') as f:
            csv.writer(f).writerow(['timestamp', 'ecg_raw', 'ecg_voltage'])

        logger.info(f"Logging to {LOG_DIR}/")

    def log_vitals(self, data: dict):
        try:
            row = {field: data.get(field, '') for field in VITALS_FIELDS}
            with open(self.vitals_file, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=VITALS_FIELDS).writerow(row)
        except Exception as e:
            logger.error(f"Vitals log error: {e}")

    def log_ecg(self, ecg_raw: int):
        ts      = datetime.now().isoformat()
        voltage = round(ecg_raw * 3.3 / 1023.0, 4) if ecg_raw is not None else ''
        self._ecg_buffer.append((ts, ecg_raw, voltage))
        if len(self._ecg_buffer) >= ECG_CHUNK_SIZE:
            self._flush_ecg()

    def _flush_ecg(self):
        try:
            with open(self.ecg_file, 'a', newline='') as f:
                writer = csv.writer(f)
                while self._ecg_buffer:
                    writer.writerow(self._ecg_buffer.popleft())
        except Exception as e:
            logger.error(f"ECG flush error: {e}")

    def close(self):
        self._flush_ecg()
        logger.info("DataLogger closed.")
