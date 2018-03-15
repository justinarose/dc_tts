# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from tqdm import tqdm

from data_load import get_batch, load_vocab, get_train_batch_discriminator, get_validation_batch_discriminator
from hyperparams import Hyperparams as hp
from modules import *
from networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN, Discriminator
import tensorflow as tf
from utils import *
import sys

class DiscriminatorGraph:
    def __init__(self, mode="train"):
        self.train_mels, _, self.train_ys, _, self.num_batch = get_train_batch_discriminator()
        self.val_mels, _, self.val_ys, _, _ = get_validation_batch_discriminator()

        self.am_validation = tf.placeholder_with_default(False,())

        self.mels = tf.cond(self.am_validation, lambda:self.val_mels, lambda:self.train_mels)
        self.ys = tf.cond(self.am_validation, lambda:self.val_ys, lambda:self.train_ys)

        training = True if mode=="train" else False

        with tf.variable_scope("Discriminator"):
            self.yLogits, self.yPred = Discriminator(self.mels, training=training)

        with tf.variable_scope("gs"):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.yLogits, labels=self.ys))
        self.roundedYPred = tf.greater(self.yPred, 0.5)
        self.correct = tf.equal(self.roundedYPred, tf.equal(self.ys,1.0))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, 'float'))
        
        self.train_loss_summary = tf.summary.scalar('train/train_loss', self.loss)
        self.train_acc_summary = tf.summary.scalar('train/train_acc', self.accuracy)
        self.validation_loss_summary = tf.summary.scalar('train/validation_loss', self.loss)
        self.validation_acc_summary = tf.summary.scalar('train/validation_acc', self.accuracy)

        # Training Scheme
        self.lr = learning_rate_decay(hp.lr, self.global_step)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.lr_summary = tf.summary.scalar("lr", self.lr)

        ## gradient clipping
        self.gvs = self.optimizer.compute_gradients(self.loss)
        self.clipped = []
        for grad, var in self.gvs:
            grad = tf.clip_by_value(grad, -1., 1.)
            self.clipped.append((grad, var))
            self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

        # Summary
        self.merged = tf.summary.merge([self.train_loss_summary, self.train_acc_summary, self.lr_summary])


if __name__ == '__main__':

    g = DiscriminatorGraph(); print("Training Graph loaded")

    logdir = hp.logdir + "-" + 'discriminator'
    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=g.global_step, summary_op=g.merged,
                             save_summaries_secs=60)
    with sv.managed_session() as sess:
        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                gs, _ = sess.run([g.global_step, g.train_op])

                # Write checkpoint files at every 1000 steps
                if gs % 1000 == 0:
                    sv.saver.save(sess, logdir + '/model_gs_{}'.format(str(gs // 1000).zfill(3) + "k"))

                # Write validation every 100 steps
                if gs % 100 == 0:
                    loss, acc = sess.run([g.validation_loss_summary, g.validation_acc_summary],
                                         feed_dict={g.am_validation: True})
                    sv.summary_writer.add_summary(loss, global_step = gs)
                    sv.summary_writer.add_summary(acc, global_step = gs)

                # break
                if gs > hp.num_iterations: break

    print("Done")
