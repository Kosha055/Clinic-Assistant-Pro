#!/usr/bin/env python3
"""
LCD35 Display Module — 480x320 TFT via pygame framebuffer
No extra installs needed — pygame is pre-installed on Pi OS.

Change fb_device to /dev/fb0 if /dev/fb1 does not work.
"""

import os
import threading
import logging
from collections import deque

# Must be set BEFORE importing pygame
os.environ["SDL_VIDEODRIVER"] = "fbcon"
os.environ["SDL_NOMOUSE"]     = "1"

import pygame

logger = logging.getLogger(__name__)

# ── Screen ───────────────────────────────────────────────────────────────────
W, H = 480, 320
FPS  = 20

# ── Colours ──────────────────────────────────────────────────────────────────
BG         = (10,  12,  30)
PANEL      = (18,  25,  55)
PANEL2     = (22,  32,  65)
BORDER     = (40,  60, 110)
WHITE      = (255, 255, 255)
GREEN      = (  0, 220,  90)
CYAN       = (  0, 200, 230)
ORANGE     = (255, 165,   0)
RED        = (220,  50,  50)
GRAY       = (110, 115, 140)
DARKGRAY   = ( 40,  45,  65)
ECG_LINE   = (  0, 255, 120)
ECG_GRID   = ( 20,  45,  35)
WARN       = (255, 200,   0)
CRIT       = (220,  50,  50)
HEADER_BG  = ( 15,  20,  50)

ECG_POINTS = 300   # ECG samples shown across screen width


class LCD35Display:
    def __init__(self, fb_device="/dev/fb1"):
        os.environ["SDL_FBDEV"] = fb_device
        self.fb_device   = fb_device
        self.running     = False
        self.screen      = None
        self.clock       = None
        self._lock       = threading.Lock()
        self._thread     = None

        # Data updated by sensor threads
        self._data = {
            'timestamp':    '--:--:--',
            'heart_rate':   None,
            'spo2':         None,
            'body_temp':    None,
            'ambient_temp': None,
            'ecg_leads_off': False,
        }
        self._ecg_buf = deque([512] * ECG_POINTS, maxlen=ECG_POINTS)

        # Fonts — loaded after pygame.init()
        self.f_large  = None
        self.f_medium = None
        self.f_small  = None
        self.f_tiny   = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update_vitals(self, data: dict):
        with self._lock:
            self._data.update(data)

    def update_ecg(self, value: int):
        with self._lock:
            self._ecg_buf.append(value if value is not None else 512)

    def set_leads_off(self, state: bool):
        with self._lock:
            self._data['ecg_leads_off'] = state

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start display loop in background thread."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="LCD35Thread"
        )
        self._thread.start()
        logger.info(f"LCD35 display thread started ({self.fb_device}).")

    def stop(self):
        self.running = False

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def _run(self):
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("Healthcare Assistant")
            pygame.mouse.set_visible(False)
            self.clock  = pygame.time.Clock()

            self.f_large  = pygame.font.SysFont('monospace', 48, bold=True)
            self.f_medium = pygame.font.SysFont('monospace', 26, bold=True)
            self.f_small  = pygame.font.SysFont('monospace', 19)
            self.f_tiny   = pygame.font.SysFont('monospace', 14)

            self.running = True
            logger.info("LCD35 display running.")

            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        pass  # touch events — extend later

                self._draw()
                pygame.display.flip()
                self.clock.tick(FPS)

        except Exception as e:
            logger.error(f"LCD35 error: {e}")
        finally:
            pygame.quit()
            logger.info("LCD35 display stopped.")

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self):
        with self._lock:
            data = dict(self._data)
            ecg  = list(self._ecg_buf)

        self.screen.fill(BG)
        self._draw_header(data)
        self._draw_vitals(data)
        self._draw_ecg_panel(ecg, data['ecg_leads_off'])
        self._draw_footer(data)

    def _draw_header(self, data):
        pygame.draw.rect(self.screen, HEADER_BG, (0, 0, W, 34))
        pygame.draw.line(self.screen, BORDER, (0, 34), (W, 34), 1)

        title = self.f_small.render("  Healthcare Assistant", True, CYAN)
        self.screen.blit(title, (4, 7))

        ts = self.f_tiny.render(data['timestamp'][11:19], True, GRAY)
        self.screen.blit(ts, (W - ts.get_width() - 8, 10))

    def _value_color(self, metric, value):
        """Return colour based on clinical threshold."""
        if value is None:
            return GRAY
        limits = {
            'heart_rate': (40,  50,  120, 150),
            'spo2':       (90,  94,  999, 999),
            'body_temp':  (35.0, 36.0, 37.5, 38.5),
        }
        t = limits.get(metric)
        if not t:
            return WHITE
        clo, wlo, whi, chi = t
        if value <= clo or value >= chi:
            return CRIT
        if value < wlo or value > whi:
            return WARN
        return GREEN

    def _draw_vital_card(self, x, y, w, h, label, value, unit, metric):
        """Draw a single vital signs card."""
        # Card background + border
        pygame.draw.rect(self.screen, PANEL,  (x, y, w, h), border_radius=10)
        pygame.draw.rect(self.screen, BORDER, (x, y, w, h), 1, border_radius=10)

        # Label
        lbl = self.f_tiny.render(label, True, GRAY)
        self.screen.blit(lbl, (x + 8, y + 6))

        # Value
        color   = self._value_color(metric, value)
        val_str = str(value) if value is not None else "---"
        val     = self.f_large.render(val_str, True, color)

        # Centre value in card
        vx = x + (w - val.get_width()) // 2
        self.screen.blit(val, (vx, y + 22))

        # Unit
        unit_s = self.f_tiny.render(unit, True, GRAY)
        self.screen.blit(unit_s, (x + w - unit_s.get_width() - 8, y + h - 20))

        # Status dot
        dot_color = self._value_color(metric, value)
        pygame.draw.circle(self.screen, dot_color, (x + 10, y + h - 10), 5)

    def _draw_vitals(self, data):
        """Draw 3 vital sign cards side by side."""
        top = 40
        pad = 5
        pw  = (W - pad * 4) // 3
        ph  = 108

        cards = [
            (pad,              top, pw, ph, "HEART RATE", data['heart_rate'],  "bpm", "heart_rate"),
            (pad*2 + pw,       top, pw, ph, "SpO\u2082",  data['spo2'],        " %",  "spo2"),
            (pad*3 + pw*2,     top, pw, ph, "BODY TEMP",  data['body_temp'],   "\u00b0C", "body_temp"),
        ]
        for args in cards:
            self._draw_vital_card(*args)

    def _draw_ecg_panel(self, samples, leads_off):
        """Draw ECG waveform panel."""
        ex, ey = 0,   154
        ew, eh = W,   138

        pygame.draw.rect(self.screen, PANEL2, (ex, ey, ew, eh))
        pygame.draw.line(self.screen, BORDER, (ex, ey), (ex + ew, ey), 1)

        # ECG label
        lbl = self.f_tiny.render("ECG", True, GRAY)
        self.screen.blit(lbl, (ex + 6, ey + 4))

        if leads_off:
            # Show leads-off warning centred
            warn = self.f_medium.render("  LEADS OFF  ", True, WARN)
            wx   = W  // 2 - warn.get_width()  // 2
            wy   = ey + eh // 2 - warn.get_height() // 2
            pygame.draw.rect(self.screen, (60, 40, 0), (wx - 6, wy - 4, warn.get_width() + 12, warn.get_height() + 8), border_radius=6)
            self.screen.blit(warn, (wx, wy))
            return

        # Grid lines
        for i in range(1, 4):
            gx = ex + ew * i // 4
            pygame.draw.line(self.screen, ECG_GRID, (gx, ey + 18), (gx, ey + eh - 4), 1)
        for i in range(1, 3):
            gy = ey + 18 + (eh - 22) * i // 3
            pygame.draw.line(self.screen, ECG_GRID, (ex, gy), (ex + ew, gy), 1)

        # Normalise + draw waveform
        lo    = min(samples)
        hi    = max(samples)
        span  = max(hi - lo, 1)
        mgn   = 16
        avail = eh - mgn * 2

        points = []
        for i, v in enumerate(samples):
            px = ex + int(i * ew / len(samples))
            py = ey + mgn + avail - int((v - lo) / span * avail)
            points.append((px, py))

        if len(points) > 1:
            pygame.draw.lines(self.screen, ECG_LINE, False, points, 2)

        # Latest voltage
        volts = self.f_tiny.render(f"{samples[-1] * 3.3 / 1023:.3f}V", True, GRAY)
        self.screen.blit(volts, (W - volts.get_width() - 6, ey + 4))

    def _draw_footer(self, data):
        """Bottom status bar."""
        fy = H - 28
        pygame.draw.line(self.screen, BORDER, (0, fy), (W, fy), 1)
        pygame.draw.rect(self.screen, HEADER_BG, (0, fy, W, 28))

        # Ambient temp
        amb = data.get('ambient_temp')
        amb_str = f"Ambient: {amb}\u00b0C" if amb else "Ambient: ---"
        amb_s = self.f_tiny.render(amb_str, True, GRAY)
        self.screen.blit(amb_s, (10, fy + 7))

        # Live indicator
        hr     = data.get('heart_rate')
        status = "  LIVE" if hr else "  WAITING"
        color  = GREEN if hr else GRAY
        s_s    = self.f_tiny.render(status, True, color)
        pygame.draw.circle(
            self.screen, color,
            (W - s_s.get_width() - 22, fy + 14), 5
        )
        self.screen.blit(s_s, (W - s_s.get_width() - 10, fy + 7))
