import gc
import os
import numpy as np
from matplotlib import pyplot as plt
from visualize import vis_mhl


class Vis(vis_mhl.Vis):
    def __init__(self):
        super(Vis, self).__init__()
