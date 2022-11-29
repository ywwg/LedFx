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

    Bass onsets create big, wide waves starting at the 0 position.
    Mid onsets create small waves at random points all around the range.

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
            ): vol.All(vol.Coerce(float), vol.Range(min=-1, max=1)),
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
                "viscosity",
                description="Viscosity of bass ripples",
                default=9,
            ): vol.All(vol.Coerce(float), vol.Range(min=2, max=12)),
        }
    )

    def on_activate(self, pixel_count):
        # Double buffered
        self._buffer = np.zeros((2, pixel_count))
        self._cur_buffer = 0
        self.drops_queue = queue.Queue()

    def deactivate(self):
        empty_queue(self.drops_queue)
        self.onsets_queue = None
        return super().deactivate()

    def config_updated(self, config):
        self._lows_power = 0
        self._lows_filter = self.create_filter(alpha_decay=0.1, alpha_rise=0.1)
        self._mids_power = 0
        self._mids_filter = self.create_filter(alpha_decay=0.1, alpha_rise=0.1)

    def audio_data_updated(self, data):
        self._last_lows_power = self._lows_power
        self._lows_power = self._lows_filter.update(data.lows_power(filtered=False))
        self._last_mids_power = self._mids_power
        self._mids_power = self._mids_filter.update(
            (data.mids_power(filtered=False) + data.high_power(filtered=False)))

        self.drops_queue.put((1, self._lows_power * self._config["bass_size"]))
        self.drops_queue.put((self.pixel_count // 2, self._lows_power * self._config["bass_size"]))
        self.drops_queue.put((self.pixel_count - 2, self._lows_power * self._config["bass_size"]))
        if data.onset():
            self.drops_queue.put((np.random.randint(1, self.pixel_count - 2),
                              self._mids_power * self._config["mids_size"]))

    def render_hsv(self):
        # Run water calculations
        for _ in range(0,self._config["speed"]):
            self._cur_buffer = 1 - self._cur_buffer
            self._do_ripple(self._buffer, self._cur_buffer, 2**self._config["viscosity"])

        # Create new drops if any
        if self.drops_queue is None:
            self.drops_queue = queue.Queue()
        while not self.drops_queue.empty():
            drop = self.drops_queue.get()
            self._create_drop(self._buffer, drop[0], drop[1])

        # Render
        shift_v = self._config["vertical_shift"]
        self._v = self._buffer[self._cur_buffer]

        # Hues are a triangle of the raw values which makes for some nice effects.
        self.hsv_array[:, 0] = self._triangle(self._v)

        # Shift the values buffer up by the shift amount and then scale to fit.
        # Values can still be out of bounds, so we clamp.
        self._v = (self._v + shift_v) / (1 + shift_v)
        self.hsv_array[:, 2] = np.clip(self._v, 0.0, 1.0)

        # Saturation starts at 1.0, and then for over-bright values (above 1),
        # reduce saturation to make it look hot.
        self._s = np.clip(-1 * self._v + 2.0, 0.0, 1.0)
        self.hsv_array[:, 1] = self._s

    def _create_drop(self, buf, position, height):
        buf[0][position] = buf[0][position - 1] = buf[0][position + 1] = height
        buf[1][position] = buf[1][position - 1] = buf[1][position + 1] = height

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