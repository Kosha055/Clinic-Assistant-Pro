#!/usr/bin/env python3
"""
Alert Manager — clinical threshold monitoring with rate limiting.
"""

import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

THRESHOLDS = {
    'heart_rate': {'critical_low': 40,  'low': 50,   'high': 120,  'critical_high': 150},
    'spo2':       {'critical_low': 90,  'low': 94},
    'body_temp':  {'critical_low': 35.0,'low': 36.0, 'high': 37.5, 'critical_high': 38.5},
}

COOLDOWN = 30  # seconds between same alert


class AlertManager:
    def __init__(self):
        self._last_alert = {}
        self._count      = 0

    def _can_alert(self, key):
        now  = time.time()
        last = self._last_alert.get(key, 0)
        if now - last >= COOLDOWN:
            self._last_alert[key] = now
            return True
        return False

    def _fire(self, level, metric, value, message):
        self._count += 1
        if level == 'CRITICAL':
            logger.critical(f"CRITICAL [{metric}]: {message} (value={value})")
        else:
            logger.warning(f"WARNING  [{metric}]: {message} (value={value})")

    def check_vitals(self, data):
        hr   = data.get('heart_rate')
        spo2 = data.get('spo2')
        temp = data.get('body_temp')

        if hr is not None:
            t = THRESHOLDS['heart_rate']
            if hr <= t['critical_low'] and self._can_alert('hr_crit_lo'):
                self._fire('CRITICAL', 'Heart Rate', hr, f"Severe bradycardia: {hr} bpm")
            elif hr >= t['critical_high'] and self._can_alert('hr_crit_hi'):
                self._fire('CRITICAL', 'Heart Rate', hr, f"Severe tachycardia: {hr} bpm")
            elif hr < t['low'] and self._can_alert('hr_lo'):
                self._fire('WARNING',  'Heart Rate', hr, f"Bradycardia: {hr} bpm")
            elif hr > t['high'] and self._can_alert('hr_hi'):
                self._fire('WARNING',  'Heart Rate', hr, f"Tachycardia: {hr} bpm")

        if spo2 is not None:
            t = THRESHOLDS['spo2']
            if spo2 < t['critical_low'] and self._can_alert('spo2_crit'):
                self._fire('CRITICAL', 'SpO2', spo2, f"Severe hypoxemia: {spo2}%")
            elif spo2 < t['low'] and self._can_alert('spo2_lo'):
                self._fire('WARNING',  'SpO2', spo2, f"Low SpO2: {spo2}%")

        if temp is not None:
            t = THRESHOLDS['body_temp']
            if temp <= t['critical_low'] and self._can_alert('temp_crit_lo'):
                self._fire('CRITICAL', 'Temperature', temp, f"Hypothermia risk: {temp}°C")
            elif temp >= t['critical_high'] and self._can_alert('temp_crit_hi'):
                self._fire('CRITICAL', 'Temperature', temp, f"High fever: {temp}°C")
            elif temp < t['low'] and self._can_alert('temp_lo'):
                self._fire('WARNING',  'Temperature', temp, f"Below normal: {temp}°C")
            elif temp > t['high'] and self._can_alert('temp_hi'):
                self._fire('WARNING',  'Temperature', temp, f"Elevated temp: {temp}°C")

    def check_ecg(self, ecg_value, leads_off):
        if leads_off and self._can_alert('ecg_leads_off'):
            self._fire('WARNING', 'ECG', 'N/A', "Leads off — check electrodes")
