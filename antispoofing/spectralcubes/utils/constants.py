# -*- coding: utf-8 -*-

import os
from multiprocessing import cpu_count

measures_available = {0: "energy_phase",
                      1: "entropy_phase",
                      2: "energy_mag",
                      3: "entropy_mag",
                      4: "mutualinfo_phase",
                      5: "mutualinfo_mag",
                      6: "correlation_phase",
                      7: "correlation_mag",
                      }

rois_available = {0: "centerframe",
                  1: "wholeframe",
                  }

VIDEO_TYPE = 'mov'
N_CUBOID = 100
CUBOID_WIDTH = 32
CUBOID_DEPTH = 32
KERNEL_WIDTH = 3
SIGMA = 0.5
SEED = 7

# -- Constants used in the facelocation module.
BOX_SHAPE = (200, 200)
IMG_OP = 'crop'
IMG_CHANNEL = 3
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 4

_current_path = os.path.abspath(os.path.dirname(__file__))
CASCADE_PATH = os.path.join(_current_path, 'haarcascade_frontalface_default.xml')

N_JOBS = (cpu_count()-1) if ((cpu_count()) > 1) else 1
