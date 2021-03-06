# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By  Justin Rose
'''

from __future__ import print_function

from utils import load_spectrograms
import os
from data_load import load_data
import numpy as np
import tqdm
import codecs
from hyperparams import Hyperparams as hp

from data_load import get_target_file_paths


files = get_target_file_paths()

for fpath in tqdm.tqdm(files):
    fname, mel, mag = load_spectrograms(fpath)
    if not os.path.exists("mels"): os.mkdir("mels")
    if not os.path.exists("mags"): os.mkdir("mags")

    np.save("target_mels/{}".format(fname.replace("wav", "npy")), mel)
    np.save("target_mags/{}".format(fname.replace("wav", "npy")), mag)
