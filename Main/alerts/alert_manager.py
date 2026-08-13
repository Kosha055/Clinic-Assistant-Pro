#!/usr/bin/env python3
"""
Alert Manager
Monitors sensor readings against clinical thresholds and triggers alerts.
"""

import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Clinical Thresholds ─────────────────────────────────────────────────────
THRESHOLDS = {
    'heart_rate': {
        'critical_low':  40,   # Severe bradycardia
        'low':           50,   # Bradycardia
        'high':         120,   # Tachycardia
        'critical_high': 150,  # Severe tachycardia
    },
    'spo2': {
        'critical_low':  90,   # Severe hypoxemia
        'low':           94,   # Mild hypoxemia
        # Normal: 95–100%
    },
    'body_temp': {
        'critical_low':  35.0,  # Hypothermia
        'low':           36.0,  # Below normal
        'high':          37.5,  # Low-grade fever
        'critical_high': 38.5,  # High fever
    },
    'ecg_amplitude': {
        'flatline_threshold': 10,  # Raw ADC units — possible lead contact loss
    }
}

COOLDOWN_SECONDS = 30   # Minimum seconds between same-type alerts


class AlertManager:
    def __init__(self):
        self._last_alert_time = {}   # { alert_key: timestamp }
        self._alert_count = 0

    def _can_alert(self, key: str) -> bool:
        """Rate-limit alerts by key."""
        now = time.time()
        last = self._last_alert_time.get(key, 0)
        if now - last >= COOLDOWN_SECONDS:
            self._last_alert_time[key] = now
            return True
        return False

    def _fire(self, level: str, metric: str, value, message: str):
        """Emit alert to logger and optionally notify external systems."""
        self._alert_count += 1
        alert = {
            'id': self._alert_count,
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'metric': metric,
            'value': value,
            'message': message
        }

        if level == 'CRITICAL':
            logger.critical(f"🚨 CRITICAL ALERT [{metric}]: {message} (value={value})")
        elif level == 'WARNING':
            logger.warning(f"⚠️  WARNING [{metric}]: {message} (value={value})")
        else:
            logger.info(f"ℹ️  INFO [{metric}]: {message} (value={value})")

        # Hook: log to file via DataLogger (import lazily to avoid circular deps)
        try:
            from data.data_logger import DataLogger
            # DataLogger is shared via main; here just log to module logger
        except Exception:
            pass

        return alert

    def check_vitals(self, data: dict):
        """Check SpO2, Heart Rate, and Temperature readings."""
        alerts = []

        # ── Heart Rate ──────────────────────────────────────────────────────
        hr = data.get('heart_rate')
        if hr is not None:
            t = THRESHOLDS['heart_rate']
            if hr <= t['critical_low'] and self._can_alert('hr_critical_low'):
                alerts.append(self._fire('CRITICAL', 'Heart Rate', hr,
                    f"Severe bradycardia: HR={hr} bpm (below {t['critical_low']})"))
            elif hr >= t['critical_high'] and self._can_alert('hr_critical_high'):
                alerts.append(self._fire('CRITICAL', 'Heart Rate', hr,
                    f"Severe tachycardia: HR={hr} bpm (above {t['critical_high']})"))
            elif hr < t['low'] and self._can_alert('hr_low'):
                alerts.append(self._fire('WARNING', 'Heart Rate', hr,
                    f"Bradycardia: HR={hr} bpm"))
            elif hr > t['high'] and self._can_alert('hr_high'):
                alerts.append(self._fire('WARNING', 'Heart Rate', hr,
                    f"Tachycardia: HR={hr} bpm"))

        # ── SpO2 ────────────────────────────────────────────────────────────
        spo2 = data.get('spo2')
        if spo2 is not None:
            t = THRESHOLDS['spo2']
            if spo2 < t['critical_low'] and self._can_alert('spo2_critical'):
                alerts.append(self._fire('CRITICAL', 'SpO2', spo2,
                    f"Severe hypoxemia: SpO2={spo2}% (below {t['critical_low']}%)"))
            elif spo2 < t['low'] and self._can_alert('spo2_low'):
                alerts.append(self._fire('WARNING', 'SpO2', spo2,
                    f"Low oxygen saturation: SpO2={spo2}%"))

        # ── Body Temperature ────────────────────────────────────────────────
        temp = data.get('body_temp')
        if temp is not None:
            t = THRESHOLDS['body_temp']
            if temp <= t['critical_low'] and self._can_alert('temp_critical_low'):
                alerts.append(self._fire('CRITICAL', 'Temperature', temp,
                    f"Hypothermia risk: {temp}°C"))
            elif temp >= t['critical_high'] and self._can_alert('temp_critical_high'):
                alerts.append(self._fire('CRITICAL', 'Temperature', temp,
                    f"High fever: {temp}°C — seek medical attention"))
            elif temp < t['low'] and self._can_alert('temp_low'):
                alerts.append(self._fire('WARNING', 'Temperature', temp,
                    f"Below normal body temperature: {temp}°C"))
            elif temp > t['high'] and self._can_alert('temp_high'):
                alerts.append(self._fire('WARNING', 'Temperature', temp,
                    f"Elevated temperature: {temp}°C"))

        return alerts

    def check_ecg(self, ecg_value, leads_off: bool):
        """Check ECG signal quality."""
        alerts = []

        if leads_off and self._can_alert('ecg_leads_off'):
            alerts.append(self._fire('WARNING', 'ECG', 'N/A',
                "ECG leads off — check electrode placement"))

        if ecg_value is not None:
            # Flatline detection (very low amplitude may indicate poor contact)
            pass   # Requires sliding window — implement in production

        return alerts

    def get_summary(self):
        """Return total alert count."""
        return {'total_alerts': self._alert_count}
