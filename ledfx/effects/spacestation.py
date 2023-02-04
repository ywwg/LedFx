from random import randint
import time

import numpy as np
import voluptuous as vol

from ledfx.color import parse_color, validate_color
from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.hsv_effect import HSVEffect


class SpaceStationEffect(AudioReactiveEffect, HSVEffect):

    NAME = "Space Station"
    CATEGORY = "Atmospheric"

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "mirror",
                description="Mirror the effect",
                default=True,
            ): bool,
            vol.Optional(
                "speed",
                description="Effect Speed modifier",
                default=0.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.001, max=1)),
            vol.Optional(
                "strobe_width",
                description="Percussive strobe width, from one pixel to the full length",
                default=5,
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=100)),
            vol.Optional(
                "delay_width",
                description="pause time sort of",
                default=5,
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=2000)),
        }
    )

    def on_activate(self, pixel_count):
        self._pos_index = 0
        self._last_time = time.time_ns()
        self.h = np.zeros(pixel_count)
        self.s = np.ones(pixel_count)
        self.v = np.linspace(0, 1, pixel_count)
        # self.drop_frames = np.zeros(self.pixel_count, dtype=int)
        # self.drop_colors = np.zeros((3, self.pixel_count))

    def config_updated(self, config):
        try:
            self.h = np.ones(self.pixel_count) * self._config["gradient_roll"]
        except:
            pass

    def render_hsv(self):
        now_ns = time.time_ns()
        self.dt = now_ns - self._last_time
        self._last_time = now_ns
        self._pos_index += self.dt * self._config["speed"]
        LOOP_VAL = 1000000000
        PAUSE_WIDTH = self._config["delay_width"]
        width = self._config["strobe_width"]

        if self._pos_index > LOOP_VAL:
            self._pos_index -= LOOP_VAL
        self._pos_val = self._pos_index / LOOP_VAL
        self._pixel_index = int(self._pos_val * (self.pixel_count + PAUSE_WIDTH - width))
        if self._pixel_index < self.pixel_count - width:
            self.v = np.zeros(self.pixel_count)
            self.v[self._pixel_index:self._pixel_index+width] = np.ones(width)
            self.hsv_array[:, 2] = self.v
        else:
            self.hsv_array[:, 2] = np.zeros(self.pixel_count)

        self.hsv_array[:, 0] = self.h
        self.hsv_array[:, 1] = self.s
