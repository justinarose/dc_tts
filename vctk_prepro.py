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

filename = os.path.join(hp.target_data, 'fnames.txt')
fpaths = codecs.open(filename, 'r').readlines()

for fpath in tqdm.tqdm(fpaths):
	fpath = fpath.strip()
    fname, mel, mag = load_spectrograms(fpath)
    if not os.path.exists("target_mels"): os.mkdir("target_mels")
    if not os.path.exists("target_mags"): os.mkdir("target_mags")

    np.save("target_mels/{}".format(fname.replace("wav", "npy")), mel)
    np.save("target_mags/{}".format(fname.replace("wav", "npy")), mag)