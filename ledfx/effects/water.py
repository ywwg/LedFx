# import logging
import queue
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
        self._buffer = np.zeros((2, pixel_count))
        # Index into self.buffer of current data buffer.
        self._cur_buffer = 0
        self._draw_count = 0
        self._draw_buf = np.zeros(pixel_count)

        self._last_drop = 0

        # Saturation is always 100%
        self._s = np.ones(pixel_count)

        self._temp = np.zeros(pixel_count)

    def config_updated(self, config):
        self._lows_power = 0
        self._lows_filter = self.create_filter(alpha_decay=0.1, alpha_rise=0.1)

    def audio_data_updated(self, data):
        self._last_lows_power = self._lows_power
        self._lows_power = self._lows_filter.update(data.lows_power(filtered=False))

        # Init new droplets, if any
        # if data.onset():
        if time.time() - self._last_drop > 0.5:
            print("drop!")
            self._last_drop = time.time()
            self._create_drop(np.random.randint(self.pixel_count), 10)

    def render_hsv(self):
        # map water height to hue and value, leave saturation at 1.
        # draw_buf = np.maximum(self._buffer[self._cur_buffer], 0.0)

        # Flip buffers
        self._cur_buffer = 1 - self._cur_buffer

        # Run water calculation
        self._do_ripple()

        # for pixel in range(0, self.pixel_count - 1):
        #     offset = self._buffer[self._cur_buffer][pixel] - self._buffer[self._cur_buffer][pixel+1]
        #     self._draw_buf[pixel] = offset

        self.hsv_array[:, 0] = np.divide(self._buffer[self._cur_buffer], 2)
        self.hsv_array[:, 1] = self._s
        # self.hsv_array[:, 2] = np.abs(self._buffer[self._cur_buffer])
        self.hsv_array[:, 2] = self._buffer[self._cur_buffer]

        # print(self._buffer[self._cur_buffer][48:53])

    def _create_drop(self, position, height):
        self._buffer[0][position] = self._buffer[0][position - 1] = self._buffer[0][position + 1] = height
        self._buffer[1][position] = self._buffer[1][position - 1] = self._buffer[1][position + 1] = height

    def _do_ripple(self):
        """Apply ripple algorithm to the current buffer"""

        damp_factor = 2**4
        src = 1 if self._cur_buffer == 0 else 0
        dest = 0 if self._cur_buffer == 0 else 1

        for pixel in range(1, self.pixel_count - 1):
            self._buffer[dest][pixel] = (((self._buffer[src][pixel - 1]
                                           + self._buffer[src][pixel + 1]
                                           + self._buffer[src][pixel] * 2)
                                          / 2)
                                         - self._buffer[dest][pixel])

        # Smooth and damp it.
        for pixel in range(1, self.pixel_count - 1):
            damp = (
                self._buffer[dest][pixel - 1]
                + self._buffer[dest][pixel + 1]
                + self._buffer[dest][pixel]) / 3
            self._buffer[dest][pixel] = damp - (damp / damp_factor)
