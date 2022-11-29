import queue

import numpy as np
from scipy import signal
import voluptuous as vol

from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.hsv_effect import HSVEffect
from ledfx.effects import smooth
from ledfx.utils import empty_queue

class Water(AudioReactiveEffect, HSVEffect):
    """A rippling water effect.

    Bass, mid, and high powers create droplets evenly spaced along the
    span.

    References:
        * https://mikro.naprvyraz.sk/docs/Coding/1/WATER.TXT
        * https://github.com/Zygo/xscreensaver/blob/master/hacks/ripples.c
    """

    NAME = "Water"
    CATEGORY = "Atmospheric"

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "speed",
                description="Speed",
                default=1,
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=5)),
            vol.Optional(
                "vertical_shift",
                description="Vertical Shift",
                default=0.12,
            ): vol.All(vol.Coerce(float), vol.Range(min=-0.2, max=1)),
            vol.Optional(
                "bass_size",
                description="Size of bass ripples",
                default=8,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=15)),
            vol.Optional(
                "mids_size",
                description="Size of mids ripples",
                default=6,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=15)),
            vol.Optional(
                "high_size",
                description="Size of high ripples",
                default=3,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=15)),
            vol.Optional(
                "viscosity",
                description="Viscosity of ripples",
                default=6,
            ): vol.All(vol.Coerce(float), vol.Range(min=2, max=12)),
        }
    )

    def on_activate(self, pixel_count):
        # Double buffered rendering
        self._buffer = np.zeros((2, pixel_count))
        self._cur_buffer = 0
        # Queue will contain tuples of (pixel location, water height value)
        self._drops_queue = queue.Queue()

    def deactivate(self):
        empty_queue(self._drops_queue)
        self.onsets_queue = None
        return super().deactivate()

    def config_updated(self, config):
        self._lows_power = 0
        self._lows_filter = self.create_filter(alpha_decay=0.1, alpha_rise=0.1)
        self._mids_power = 0
        self._mids_filter = self.create_filter(alpha_decay=0.1, alpha_rise=0.1)
        self._high_power = 0
        self._high_filter = self.create_filter(alpha_decay=0.1, alpha_rise=0.1)

    def audio_data_updated(self, data):
        self._last_lows_power = self._lows_power
        self._lows_power = self._lows_filter.update(data.lows_power(filtered=True))
        self._last_mids_power = self._mids_power
        self._mids_power = self._mids_filter.update(data.mids_power(filtered=True))
        self._last_high_power = self._high_power
        self._high_power = self._mids_filter.update(data.high_power(filtered=True))

        # Evenly distribute drop locations throughout the span:
        # B    H    M    H    M    H    B    H     M    H    M     H     B
        # 0   1/12 1/6  3/12 2/6  5/12 1/2   7/12  4/6  9/12 5/6  11/12  12/12
        self._drops_queue.put((1, self._lows_power * self._config["bass_size"]))
        self._drops_queue.put((self.pixel_count // 2, self._lows_power * self._config["bass_size"]))
        self._drops_queue.put((self.pixel_count - 2, self._lows_power * self._config["bass_size"]))

        sixths = self.pixel_count / 6
        self._drops_queue.put((int(sixths), self._mids_power * self._config["mids_size"]))
        self._drops_queue.put((int(2*sixths), self._mids_power * self._config["mids_size"]))
        self._drops_queue.put((int(4*sixths), self._mids_power * self._config["mids_size"]))
        self._drops_queue.put((int(5*sixths), self._mids_power * self._config["mids_size"]))

        twefths = self.pixel_count / 12
        self._drops_queue.put((int(twefths), self._high_power * self._config["high_size"]))
        self._drops_queue.put((int(3*twefths), self._high_power * self._config["high_size"]))
        self._drops_queue.put((int(5*twefths), self._high_power * self._config["high_size"]))
        self._drops_queue.put((int(7*twefths), self._high_power * self._config["high_size"]))
        self._drops_queue.put((int(9*twefths), self._high_power * self._config["high_size"]))
        self._drops_queue.put((int(11*twefths), self._high_power * self._config["high_size"]))

    def render_hsv(self):
        # Run water calculations
        for _ in range(0,self._config["speed"]):
            # Flip buffers for each rendering pass.
            self._cur_buffer = 1 - self._cur_buffer
            self._do_ripple(self._buffer, self._cur_buffer, 2**self._config["viscosity"])

        # Create new drops if any
        if self._drops_queue is None:
            self._drops_queue = queue.Queue()
        while not self._drops_queue.empty():
            drop = self._drops_queue.get()
            self._create_drop(drop[0], drop[1])

        # Render

        # Hues are a triangle of the raw values which makes for some nice effects.
        self.hsv_array[:, 0] = self._triangle(self._buffer[self._cur_buffer])

        # Shift the values buffer up by the shift amount and then scale to fit.
        # Values can still be out of bounds, so we clamp.
        self._v = self._buffer[self._cur_buffer]
        shift_v = self._config["vertical_shift"]
        self._v = (self._v + shift_v) / (1 + shift_v)
        self.hsv_array[:, 2] = np.clip(self._v, 0.0, 1.0)

        # Saturation starts at 1.0, and then for over-bright values (above 1),
        # reduce saturation to make it look hot.
        self._s = np.clip(-1 * (self._v + shift_v) + 2.0, 0.0, 1.0)
        self.hsv_array[:, 1] = self._s

    def _create_drop(self, position, height):
        self._buffer[0][position] = self._buffer[0][position - 1] = self._buffer[0][position + 1] = height
        self._buffer[1][position] = self._buffer[1][position - 1] = self._buffer[1][position + 1] = height

    def _do_ripple(self, buf, buf_idx, damp_factor):
        """Apply ripple algorithm to the given buffer

        Arguments:
            buf: the double buffer to operate on
            buf_idx: the current destination buffer.
            damp_factor: the viscocity of the liquid.  Higher is less viscous.
        """

        src = 1 if buf_idx == 0 else 0
        dest = 0 if buf_idx == 0 else 1

        # The 2D version of this algorithm uses the north and south neighbors.
        # Since we don't have that here, I dropped in the value at the current
        # position.  This slows down the waves somewhat and still looks nice.
        for pixel in range(1, self.pixel_count - 1):
            buf[dest][pixel] = (((buf[src][pixel - 1]
                                + buf[src][pixel + 1]
                                + buf[src][pixel] * 2)
                                / 2)
                                - buf[dest][pixel])

        buf[dest] = smooth(buf[dest], 1.0)
        buf[dest] -= buf[dest] / damp_factor

    def _triangle(self, a):
        a = signal.sawtooth(a * np.pi * 2, 0.5)
        np.multiply(a, 0.5, out=a)
        return np.add(a, 0.5)