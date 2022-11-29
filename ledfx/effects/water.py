# import logging
# import queue
import time
import sys

import numpy as np
from scipy import signal
import voluptuous as vol

from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.hsv_effect import HSVEffect
# from ledfx.effects import smooth
# from ledfx.utils import empty_queue

# _LOGGER = logging.getLogger(__name__)


np.set_printoptions(threshold=sys.maxsize)


class Water(AudioReactiveEffect, HSVEffect):
    """A rippling water effect.

    Bass onsets create big, wide waves starting at the 0 position.
    High onsets create small waves at random points all around the range.

    References:
        * https://mikro.naprvyraz.sk/docs/Coding/1/WATER.TXT
        * https://github.com/Zygo/xscreensaver/blob/master/hacks/ripples.c
    """

    NAME = "Water"
    CATEGORY = "Atmospheric"

    CONFIG_SCHEMA = vol.Schema(
        {
        }
    )

    def on_activate(self, pixel_count):
        # Double buffered
        self._bass_buffer = np.zeros((2, pixel_count))
        self._cur_bass_buffer = 0
        self._highs_buffer = np.zeros((2, pixel_count))
        self._cur_highs_buffer = 0

        self._last_drop = 0

        # Saturation is always 100%
        self._s = np.ones(pixel_count)

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
            (data.mids_power(filtered=False) + data.high_power(filtered=False))
            )

        self._create_drop(self._bass_buffer, 1, self._lows_power * 4)
        self._create_drop(self._bass_buffer, self.pixel_count // 2, self._lows_power * 4)
        self._create_drop(self._bass_buffer, self.pixel_count - 2, self._lows_power * 4)
        # Init new droplets, if any
        if data.onset():
            # if time.time() - self._last_drop > 0.5:
            self._create_drop(self._highs_buffer, np.random.randint(1, self.pixel_count - 2),
                              self._mids_power * 10)
            # self._last_drop = time.time()
        #     self._create_drop(self._bass_buffer, np.random.randint(self.pixel_count), 10)

    def render_hsv(self):
        # map water height to hue and value, leave saturation at 1.
        # draw_buf = np.maximum(self._bass_buffer[self._cur_bass_buffer], 0.0)

        # Flip buffers
        # self._cur_bass_buffer = 1 - self._cur_bass_buffer

        # Run water calculations

        # XXXXX TODO: probably the reactivity param could adjust the damp factors.
        self._cur_bass_buffer = 1 - self._cur_bass_buffer
        self._do_ripple(self._bass_buffer, self._cur_bass_buffer, 2**9)

        self._cur_highs_buffer = 1 - self._cur_highs_buffer
        self._do_ripple(self._highs_buffer, self._cur_highs_buffer, 2**3)

        h = self._bass_buffer[self._cur_bass_buffer] + 1.0 - self._highs_buffer[self._cur_highs_buffer]
        self._triangle(h)
        self.hsv_array[:, 0] = h

        self.hsv_array[:, 1] = self._s

        v = np.abs(self._bass_buffer[self._cur_bass_buffer] + self._highs_buffer[self._cur_highs_buffer])
        self.hsv_array[:, 2] = np.minimum(v, 1.0)

    def _create_drop(self, buf, position, height):
        buf[0][position] = buf[0][position - 1] = buf[0][position + 1] = height
        buf[1][position] = buf[1][position - 1] = buf[1][position + 1] = height

    def _do_ripple(self, buf, buf_idx, damp_factor):
        """Apply ripple algorithm to the current buffer

        Arguments:
            buf: the double buffer to operate on
            buf_idx: the current destination buffer.
            damp_factor: the viscocity of the liquid.  Higher is less viscous.
        """

        src = 1 if buf_idx == 0 else 0
        dest = 0 if buf_idx == 0 else 1

        for pixel in range(1, self.pixel_count - 1):
            buf[dest][pixel] = (((buf[src][pixel - 1]
                                + buf[src][pixel + 1]
                                + buf[src][pixel] * 2)
                                / 2)
                                - buf[dest][pixel])

        # Smooth and damp it.
        for pixel in range(1, self.pixel_count - 1):
            damp = (
                buf[dest][pixel - 1]
                + buf[dest][pixel + 1]
                + buf[dest][pixel]) / 3
            buf[dest][pixel] = damp - (damp / damp_factor)

    def _triangle(self, a):
        a = signal.sawtooth(a * np.pi * 2, 0.5)
        np.multiply(a, 0.5, out=a)
        return np.add(a, 0.5)