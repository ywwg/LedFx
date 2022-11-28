import logging
import queue
import time

import numpy as np
import voluptuous as vol

from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.hsv_effect import HSVEffect
from ledfx.effects import smooth
from ledfx.utils import empty_queue

_LOGGER = logging.getLogger(__name__)


class HuxleyMelt(AudioReactiveEffect, HSVEffect):

    NAME = "Huxley Melt"
    CATEGORY = "Atmospheric"

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "speed",
                description="Effect Speed modifier",
                default=0.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.001, max=1)),
            vol.Optional(
                "reactivity",
                description="Audio Reactive modifier",
                default=0.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0001, max=1)),
            vol.Optional(
                "strobe_width",
                description="Percussive strobe width, in pixels",
                default=10,
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=1000)),
            vol.Optional(
                "strobe_decay_rate",
                description="Percussive strobe decay rate. Higher -> decays faster.",
                default=0.25,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
            vol.Optional(
                "strobe_blur",
                description="How much to blur the strobes",
                default=3.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=10)),
            vol.Optional(
                "bg_bright",
                description="How bright the melt bg should be",
                default=0.4,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
            vol.Optional(
                "strobe_threshold",
                description="Cutoff for quiet sounds. Higher -> only loud sounds are detected",
                default=0.75,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
            vol.Optional(
                "strobe_rate",
                description="higher numbers -> more strobes",
                default=0.75,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
        }
    )

    def on_activate(self, pixel_count):
        self.hl = pixel_count
        self.h = np.linspace(0, 1, pixel_count)

        self.timestep = 0
        self.last_time = time.time_ns()
        self.dt = 0

        self.onsets_queue = queue.Queue()
        self.strobe_overlay = np.zeros(self.pixel_count)

    def deactivate(self):
        empty_queue(self.onsets_queue)
        self.onsets_queue = None
        return super().deactivate()

    def config_updated(self, config):
        # lows power seems to be on a 0-1 scale
        self._lows_power = 0
        self._last_lows_power = 0
        self._lows_filter = self.create_filter(alpha_decay=0.1, alpha_rise=0.1)
        self._direction = 1.0

        # intensity comes from melbank so it's not capped at 1.
        self._mids_power = 0
        self._mids_filter = self.create_filter(alpha_decay=0.1, alpha_rise=0.1)

        self.bg_bright = self._config["bg_bright"]

        self.strobe_cutoff = self._config["strobe_threshold"] / 10.0
        self.strobe_width = self._config["strobe_width"]
        self.last_strobe_time = 0
        self.strobe_wait_time = 1.0 - self._config["strobe_rate"]
        self.strobe_decay_rate = 1.0 - self._config["strobe_decay_rate"]
        self.strobe_blur = self._config["strobe_blur"]

    def audio_data_updated(self, data):
        self._last_lows_power = self._lows_power
        self._lows_power = self._lows_filter.update(data.lows_power(filtered=False))
        # self._lows_power = 0
        # _LOGGER.debug(f"bass {self._lows_power}")

        if self._lows_power > self.strobe_cutoff and np.random.randint(0, 200) == 0:
            self._direction *= -1.0

        currentTime = time.time()

        intensities = np.fromiter(
             (i.max() ** 2 for i in self.melbank_thirds()), float
        )
        np.clip(intensities, 0, 1, out=intensities)
        self._mids_power = self._mids_filter.update(data.mids_power(filtered=True))
        if (
            data.onset()
            and currentTime - self.last_strobe_time > self.strobe_wait_time
            # and intensities[1] < self.strobe_cutoff
            and intensities[2] > self.strobe_cutoff):
            self.onsets_queue.put(True)
            self.last_strobe_time = currentTime

    def render_hsv(self):
        now_ns = time.time_ns()
        self.dt = now_ns - self.last_time
        self.last_time = now_ns

        self.timestep += self.dt
        self.timestep += (
            self._lows_power
            * self._config["reactivity"]
            * self._config["speed"]
            * 100000000.0
        )

        t1 = self.time(self._config["speed"] * 20, timestep=self.timestep)
        # _LOGGER.debug(f"t1 {t1}")
        # t1 = self._lows_power * self._config["reactivity"]
        # t2 = self.time(self._config["speed"] * 6.5, timestep=self.timestep)
        t2 = self._lows_power * self._config["reactivity"] * 0.5

        self.h[:] = np.linspace(0, 1, self.pixel_count)
        # big_sine = np.linspace(0, 4, self.pixel_count)
        np.subtract(1, self.h, out=self.h)
        self.array_sin(self.h)
        self.s = np.ones(self.pixel_count)
        self.v = np.copy(self.h)
        # self.v = np.ones(self.pixel_count)

        # np.add(self.h, (1.0 - t2) * self._config["reactivity"] * self._config["speed"] * 5, out=self.h)
        # self.array_sin(self.h)
        self.array_sin(self.h)

        # Value munging
        self.array_sin(self.v)
        np.add(self.v, t1 * self._direction, out=self.v)
        self.array_sin(self.v)
        # np.add(self.v, t1 * self._direction, out=self.v)
        # self.array_sin(self.v)
        # np.add(self.v, (1.0 - t1) * self._direction, out=self.v)
        # self.array_sin(self.v)
        np.add(self.v, t2 * self._direction, out=self.v)
        self.array_sin(self.v)
        # np.add(self.v, (1.0 - t1) * self._direction, out=self.v)
        # self.array_sin(self.v)

        # power = 30 - (self._mids_power * 20)
        power = 5 - (self._mids_power * 5)
        # _LOGGER.debug(f"{power}")
        np.power(self.v, power, out=self.v)
        # self.hsv_array[:, 2] = self.v


        # noise = np.random.normal(0.5,0.5, self.pixel_count)

        # np.subtract(1.0, self.v, out=self.v)

        # np.subtract(self.v, 1.0 - (self.bg_bright), out=self.v)
        # np.maximum(self.v, 0.0, out=self.v)
        np.multiply(self.v, self.bg_bright, out=self.v)

        # roll_amount = (self._lows_power - self._last_lows_power) * 1000.0 * self._config["reactivity"]
        # self.h = np.roll(self.h, int(roll_amount))
        # _LOGGER.debug(f"roll {roll_amount}")
        self.hsv_array[:, 0] = self.h

        if not self.onsets_queue.empty():
            self.onsets_queue.get()
            strobe_width = min(self.strobe_width, self.pixel_count)
            position = np.random.randint(self.pixel_count - strobe_width)
            self.strobe_overlay[position : position + strobe_width] = 1.0

        # adjust saturation by the strength of the overlay mask
        np.multiply(self.s, np.subtract(1, self.strobe_overlay), out=self.s)
        # add strobe_overlay strength to value, cap at 1.0
        np.add(self.v, self.strobe_overlay, out=self.v)
        np.minimum(self.v, 1.0, out=self.v)

        self.hsv_array[:, 1] = self.s
        # self.hsv_array[:, 1] = np.zeros(self.pixel_count)
        self.hsv_array[:, 2] = self.v

        # blur and decay the strobe
        self.strobe_overlay *= self.strobe_decay_rate
        self.strobe_overlay = smooth(self.strobe_overlay, self.strobe_blur)

