#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

class GAN(object):

    def __init__(self, id='', gender='neutral'):
        self.id = id
        self.gender = gender

    def get_generator(self, z, slice_len=16384, nch=1, kernel_len=25, dim=64, use_batchnorm=False, upsample='zeros', train=False):
        pass

    def get_discriminator(self, x, kernel_len=25, dim=64, use_batchnorm=False, phaseshuffle_rad=0):
        pass

    def train(self, fps, sample_rate=16000, batch=64, save_secs=300, summary_secs=120, slice_len=16384, overlap_ratio=0., first_slice=False, pad_end=False, prefetch=0, latent_dim=100, kernel_len=25, gan_dim=64, use_batchnorm=False, disc_nupdates=5, loss='wgan-gp', genr_upsample='zeros', genr_pp=False, genr_pp_len=512, disc_phaseshuffle=2):
        pass

    def infer(self, latent_dim=100, slice_len=16384, kernel_len=25, gan_dim=64, use_batchnorm=False, genr_upsample='zeros', genr_pp=False, genr_pp_len=512):
        pass

    def preview(self, sample_rate=16000, genr_pp=False, preview_n=32):
        pass

    def incept(self, incept_metagraph_fp='./eval/inception/infer.meta', incept_ckpt_fp='./eval/inception/best_acc-103005', incept_n=5000, incept_k=10):
        pass
