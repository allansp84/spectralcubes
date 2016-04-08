# -*- coding: utf-8 -*-

from antispoofing.spectralcubes.utils.misc import modification_date
from antispoofing.spectralcubes.utils.misc import get_time
from antispoofing.spectralcubes.utils.misc import total_time_elapsed
from antispoofing.spectralcubes.utils.misc import RunInParallel
from antispoofing.spectralcubes.utils.misc import acc_threshold
from antispoofing.spectralcubes.utils.misc import eer_threshold
from antispoofing.spectralcubes.utils.misc import farfrr
from antispoofing.spectralcubes.utils.misc import calc_hter
from antispoofing.spectralcubes.utils.misc import ppndf_over_array
from antispoofing.spectralcubes.utils.misc import ppndf
from antispoofing.spectralcubes.utils.misc import det
from antispoofing.spectralcubes.utils.misc import det_axis
from antispoofing.spectralcubes.utils.misc import plot_score_distributions
from antispoofing.spectralcubes.utils.misc import retrieve_samples
from antispoofing.spectralcubes.utils.misc import mosaic
from antispoofing.spectralcubes.utils.misc import progressbar
from antispoofing.spectralcubes.utils.misc import creating_csv

from antispoofing.spectralcubes.utils.constants import measures_available
from antispoofing.spectralcubes.utils.constants import rois_available
from antispoofing.spectralcubes.utils.constants import VIDEO_TYPE
from antispoofing.spectralcubes.utils.constants import N_CUBOID
from antispoofing.spectralcubes.utils.constants import CUBOID_WIDTH
from antispoofing.spectralcubes.utils.constants import CUBOID_DEPTH
from antispoofing.spectralcubes.utils.constants import KERNEL_WIDTH
from antispoofing.spectralcubes.utils.constants import SIGMA
from antispoofing.spectralcubes.utils.constants import SEED
from antispoofing.spectralcubes.utils.constants import BOX_SHAPE
from antispoofing.spectralcubes.utils.constants import IMG_OP
from antispoofing.spectralcubes.utils.constants import IMG_CHANNEL
from antispoofing.spectralcubes.utils.constants import SCALE_FACTOR
from antispoofing.spectralcubes.utils.constants import MIN_NEIGHBORS
from antispoofing.spectralcubes.utils.constants import CASCADE_PATH
from antispoofing.spectralcubes.utils.constants import N_JOBS
from antispoofing.spectralcubes.utils.constants import CONST
