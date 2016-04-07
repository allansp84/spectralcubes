# -*- coding: utf-8 -*-

from antispoofing.spectralcubes.datasets.replayattack import ReplayAttack
from antispoofing.spectralcubes.datasets.casia import Casia
from antispoofing.spectralcubes.datasets.maskattack import MaskAttack
from antispoofing.spectralcubes.datasets.uvad import UVAD
from antispoofing.spectralcubes.datasets.reduceduvad import ReducedUVAD

registered_datasets = {0: ReplayAttack,
                       1: Casia,
                       2: MaskAttack,
                       3: UVAD,
                       4: ReducedUVAD,
                       }
