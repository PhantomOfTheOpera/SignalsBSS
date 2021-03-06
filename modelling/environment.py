"""Environment class."""
import numpy as np

from modelling.source import Source
from modelling.microphone import Mic
from modelling.point import SoundPoint
from modelling.player import Player


class Environment:
    """Manages all transmissions."""
    __c = 340

    def __init__(self, sr: int = 22_050, mics: [] = None,
                 sources: [] = None):
        if SoundPoint.get_rate():
            raise RuntimeError("only one environment can exist")
        self.__sr = sr
        SoundPoint.set_rate(self)
        self._player = Player(self)

        self._mics = [mic if isinstance(mic, Mic)
                      else Mic(*mic[:-1], **mic[-1])
                      for mic in mics] if mics else []
        self._sources = [src if isinstance(src, Source)
                         else Source(*src)
                         for src in sources] if sources else []

    def get_rate(self):
        return self.__sr

    def transmit_all(self, real_shift: bool = True,
                     dec_loudness: bool = True):
        transmissions = [[src, mic, src - mic]
                         for src in self._sources
                         for mic in self._mics]
        transmissions.sort(key=lambda k: k[-1])
        max_dist = transmissions[-1][-1]
        for tr in transmissions:
            self._transmit(tr, max_dist, real_shift, dec_loudness)

        for mic in self._mics:
            mic.finalize()

    def _transmit(self, transmission: [Source, Mic, np.float],
                  max_dist: np.float, real_shift: bool, dec_loudness: bool):
        transmission[1] <<= (
            (transmission[0] <=
                self.__sr * (max_dist - transmission[-1]) / self.__c
             ) if real_shift else ~transmission[0]
        ) / ((transmission[-1] + 1) if dec_loudness else 1)

    def show_all_time_domain(self):
        for mic in self._mics:
            mic.show_time_domain()

    def play_at_mic(self, mic_id: int):
        self._player.play(self._mics[mic_id])

    def play_at_src(self, src_id: int):
        self._player.play(self._sources[src_id])

    def play(self, data: list):
        self._player.play(data)

    def get_data_at(self, mic_id: int):
        return ~self._mics[mic_id]

    def get_src_at(self, src_id: int):
        return ~self._sources[src_id]
