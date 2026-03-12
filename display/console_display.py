#!/usr/bin/env python3
"""
Console Display
Real-time terminal display of sensor readings.
"""

import os
import sys
import logging
from collections import deque

logger = logging.getLogger(__name__)

ECG_CHART_WIDTH  = 60
ECG_CHART_HEIGHT = 10
ECG_BUFFER_SIZE  = ECG_CHART_WIDTH


class ConsoleDisplay:
    def __init__(self):
        self._ecg_buffer = deque([512] * ECG_BUFFER_SIZE, maxlen=ECG_BUFFER_SIZE)
        self._last_vitals = {}

    def show_header(self):
        """Print startup banner."""
        print("\n" + "=" * 70)
        print("  🏥  HEALTHCARE ASSISTANT — Raspberry Pi 4B")
        print("  Sensors: MAX30102 | MLX90614 | AD8232 via MCP3008")
        print("=" * 70)
        print(f"  {'METRIC':<25} {'VALUE':<20} {'STATUS'}")
        print("-" * 70)

    def _status_icon(self, metric, value):
        """Return a status emoji based on value and metric."""
        if value is None:
            return "❓ No Data"
        icons = {
            'heart_rate': (lambda v: "✅ Normal" if 60 <= v <= 100 else "⚠️  Abnormal"),
            'spo2':       (lambda v: "✅ Normal" if v >= 95 else ("⚠️  Low" if v >= 90 else "🚨 Critical")),
            'body_temp':  (lambda v: "✅ Normal" if 36.0 <= v <= 37.5 else "⚠️  Abnormal"),
        }
        fn = icons.get(metric)
        return fn(value) if fn else "—"

    def show_vitals(self, data: dict):
        """Print current vitals to console."""
        hr    = data.get('heart_rate')
        spo2  = data.get('spo2')
        temp  = data.get('body_temp')
        temp_a= data.get('ambient_temp')
        ts    = data.get('timestamp', '')[:19]

        print(f"\n  ⏱  {ts}")
        print(f"  {'Heart Rate':<25} {str(hr) + ' bpm' if hr else 'N/A':<20} {self._status_icon('heart_rate', hr)}")
        print(f"  {'SpO₂':<25} {str(spo2) + ' %' if spo2 else 'N/A':<20} {self._status_icon('spo2', spo2)}")
        print(f"  {'Body Temp (Object)':<25} {str(temp) + ' °C' if temp else 'N/A':<20} {self._status_icon('body_temp', temp)}")
        print(f"  {'Ambient Temp':<25} {str(temp_a) + ' °C' if temp_a else 'N/A':<20}")
        print("-" * 70)

        self._last_vitals = data

    def update_ecg(self, ecg_value: int):
        """Add ECG sample to buffer and periodically redraw chart."""
        if ecg_value is None:
            return
        self._ecg_buffer.append(ecg_value)

        # Redraw every 60 new samples (throttle output)
        if len(self._ecg_buffer) % 60 == 0:
            self._draw_ecg_chart()

    def _draw_ecg_chart(self):
        """Draw a simple ASCII ECG waveform in the terminal."""
        samples = list(self._ecg_buffer)
        lo = min(samples)
        hi = max(samples)
        span = max(hi - lo, 1)

        chart_lines = []
        for row in range(ECG_CHART_HEIGHT, -1, -1):
            threshold = lo + (row / ECG_CHART_HEIGHT) * span
            line = ""
            for val in samples:
                next_thresh = lo + ((row + 1) / ECG_CHART_HEIGHT) * span
                if threshold <= val < next_thresh:
                    line += "█"
                else:
                    line += " "
            chart_lines.append(f"  │{line}│")

        print("\n  ECG Signal (Live)")
        print("  " + "─" * (ECG_CHART_WIDTH + 2))
        print("\n".join(chart_lines))
        print("  " + "─" * (ECG_CHART_WIDTH + 2))
        print(f"  Raw: {samples[-1]:4d}  Voltage: {samples[-1]*3.3/1023:.3f}V\n")
