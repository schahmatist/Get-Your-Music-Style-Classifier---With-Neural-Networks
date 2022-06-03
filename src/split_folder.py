import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')



image_dir='../data_music/temp/'
dest_dir='../data_music/sample_recomm_db'


import splitfolders

splitfolders.ratio(image_dir, output=dest_dir, seed=1337, ratio=(.5, .25, .25 ), group_prefix=None, move=False) # default values

