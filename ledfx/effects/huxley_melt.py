import queue
import time

import numpy as np
import voluptuous as vol

from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.hsv_effect import HSVEffect
from ledfx.effects import smooth
from ledfx.utils import empty_queue


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
                default=0.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
            vol.Optional(
                "strobe_blur",
                description="How much to blur the strobes",
                default=2.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=10)),
            vol.Optional(
                "bg_bright",
                description="How bright the melt bg should be",
                default=0.8,
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=1)),
        }
    )

    def on_activate(self, pixel_count):
        self.hl = pixel_count
        self.c1 = np.linspace(0, 1, pixel_count)

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
        self._lows_power = 0
        self._lows_filter = self.create_filter(alpha_decay=0.1, alpha_rise=0.1)

        self.bg_bright = self._config["bg_bright"]

        self.strobe_width = self._config["strobe_width"]
        self.last_strobe_time = 0
        self.strobe_wait_time = 0
        self.strobe_decay_rate = 1 - self._config["strobe_decay_rate"]
        self.strobe_blur = self._config["strobe_blur"]

    def audio_data_updated(self, data):
        self._lows_power = self._lows_filter.update(data.lows_power(filtered=False))

        currentTime = time.time()

        if (data.onset() and currentTime - self.last_strobe_time > self.strobe_wait_time):
            self.onsets_queue.put(True)
            self.last_strobe_time = currentTime

    def render_hsv(self):
        self.dt = time.time_ns() - self.last_time
        self.timestep += self.dt
        self.timestep += (
            self._lows_power
            * self._config["reactivity"]
            / self._config["speed"]
            * 1000000000.0
        )
        self.last_time = time.time_ns()

        t1 = self.time(self._config["speed"] * 5, timestep=self.timestep)
        t2 = self.time(self._config["speed"] * 6.5, timestep=self.timestep)

        self.c1[:] = np.linspace(0, 1, self.pixel_count)
        # np.subtract(self.c1, self.hl, out=self.c1)
        # np.abs(self.c1, out=self.c1)
        # np.divide(self.c1, self.hl, out=self.c1)
        np.subtract(1, self.c1, out=self.c1)

        self.s = np.ones(self.pixel_count)
        self.v = np.copy(self.c1)
        np.add(self.c1, t2, out=self.c1)

        self.array_sin(self.v)
        np.add(self.v, t1, out=self.v)
        self.array_sin(self.v)
        np.add(self.v, t1, out=self.v)
        self.array_sin(self.v)
        np.power(self.v, 2, out=self.v)
        np.multiply(self.v, self.bg_bright, out=self.v)

        self.hsv_array[:, 0] = self.c1

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
        self.hsv_array[:, 2] = self.v

        # blur and decay the strobe
        self.strobe_overlay *= self.strobe_decay_rate
        self.strobe_overlay = smooth(self.strobe_overlay, self.strobe_blur)

