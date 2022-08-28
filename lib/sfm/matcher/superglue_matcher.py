from pathlib import Path
import numpy as np
import matplotlib.cm as cm
import torch

from lib.sg.matching import Matching
from lib.sg.utils import (make_matching_plot, AverageTimer, read_image,  
                          frame2tensor, vizTileRes)
from  lib.io import generateTiles
torch.set_grad_enabled(False)