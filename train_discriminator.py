# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from tqdm import tqdm

from data_load import get_batch, load_vocab, get_true_batch_discriminator
from hyperparams import Hyperparams as hp
from modules import *
from networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN, Discriminator
import tensorflow as tf
from utils import *
import sys

class DiscriminatorGraph:
    def __init__(self, mode="train"):
    	self.mels, self.mags, self.ys, self.fnames, self.num_batch = get_true_batch_discriminator()

    	training = True if mode=="train" else False

    	with tf.variable_scope("Discriminator"):
    		self.yLogits, self.yPred = Discriminator(self.mels, training=training)

    	with tf.variable_scope("gs"):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

    	self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.yLogits, labels=self.ys))
    	tf.summary.scalar('train/loss', self.loss)
    	

    	# Training Scheme
        self.lr = learning_rate_decay(hp.lr, self.global_step)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        tf.summary.scalar("lr", self.lr)

        ## gradient clipping
        self.gvs = self.optimizer.compute_gradients(self.loss)
        self.clipped = []
        for grad, var in self.gvs:
        	grad = tf.clip_by_value(grad, -1., 1.)
        	self.clipped.append((grad, var))
        	self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)


if __name__ == '__main__':

    g = DiscriminatorGraph(); print("Training Graph loaded")

    logdir = hp.logdir + "-" + 'discriminator'
    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=g.global_step)
    with sv.managed_session() as sess:
        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                gs, _ = sess.run([g.global_step, g.train_op])

                # Write checkpoint files at every 1k steps
                if gs % 1000 == 0:
                    sv.saver.save(sess, logdir + '/model_gs_{}'.format(str(gs // 1000).zfill(3) + "k"))

                # break
                if gs > hp.num_iterations: break

    print("Done")
