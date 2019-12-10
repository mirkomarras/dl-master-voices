#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.io.wavfile import write as wavwrite
import matplotlib.pyplot as plt
from scipy.signal import freqz
from six.moves import xrange
from functools import reduce
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import pickle
import time
import os

matplotlib.use('Agg')

from helpers.datapipeline import data_pipeline_gan
from models.gan.model import GAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class WaveGAN(GAN):

  def __init__(self, id='', gender='neutral'):
    super().__init__(id, gender)
    self.name = 'wavegan'

    tf_dir = os.path.join('.', 'data', 'pt_models', self.name, self.gender)
    tf_v = str(len(os.listdir(tf_dir))) if not id else id
    out_dir = os.path.join(tf_dir, 'v' + tf_v)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    assert os.path.exists(os.path.join(out_dir))

    self.dir = out_dir
    self.id = tf_v
    print('> GAN model', self.dir)

  def conv1d_transpose(self, inputs, filters, kernel_width, stride=4, padding='same', upsample='zeros'):

    if upsample == 'zeros':
      return tf.layers.conv2d_transpose( tf.expand_dims(inputs, axis=1), filters, (1, kernel_width), strides=(1, stride), padding='same')[:, 0]
    elif upsample == 'nn':
      batch_size = tf.shape(inputs)[0]
      _, w, nch = inputs.get_shape().as_list()

      x = inputs

      x = tf.expand_dims(x, axis=1)
      x = tf.image.resize_nearest_neighbor(x, [1, w * stride])
      x = x[:, 0]

      return tf.layers.conv1d(x, ilters, kernel_width, 1, padding='same')
    else:
      raise NotImplementedError

  def lrelu(self, inputs, alpha=0.2):
    return tf.maximum(alpha * inputs, inputs)

  def apply_phaseshuffle(self, x, rad, pad_type='reflect'):
    b, x_len, nch = x.get_shape().as_list()

    phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

    x = x[:, phase_start:phase_start+x_len]
    x.set_shape([b, x_len, nch])

    return x

  def get_generator(self, z, slice_len=16384, kernel_len=25, gan_dim=64, use_batchnorm=False, genr_upsample='zeros', train=False):
    assert slice_len in [16384, 32768, 65536]
    batch_size = tf.shape(z)[0]

    if use_batchnorm:
      batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
    else:
      batchnorm = lambda x: x

    dim_mul = 16 if slice_len == 16384 else 32
    output = z
    with tf.variable_scope('z_project'):
      output = tf.layers.dense(output, 4 * 4 * gan_dim * dim_mul)
      output = tf.reshape(output, [batch_size, 16, gan_dim * dim_mul])
      output = batchnorm(output)
      output = tf.nn.relu(output)
    dim_mul //= 2

    with tf.variable_scope('upconv_0'):
      output = self.conv1d_transpose(output, gan_dim * dim_mul, kernel_len, 4, upsample=genr_upsample)
      output = batchnorm(output)
      output = tf.nn.relu(output)
    dim_mul //= 2

    with tf.variable_scope('upconv_1'):
      output = self.conv1d_transpose(output, gan_dim * dim_mul, kernel_len, 4, upsample=genr_upsample)
      output = batchnorm(output)
      output = tf.nn.relu(output)
    dim_mul //= 2

    with tf.variable_scope('upconv_2'):
      output = self.conv1d_transpose(output, gan_dim * dim_mul, kernel_len, 4, upsample=genr_upsample)
      output = batchnorm(output)
      output = tf.nn.relu(output)
    dim_mul //= 2

    with tf.variable_scope('upconv_3'):
      output = self.conv1d_transpose(output, gan_dim * dim_mul, kernel_len, 4, upsample=genr_upsample)
      output = batchnorm(output)
      output = tf.nn.relu(output)

    if slice_len == 16384:
      with tf.variable_scope('upconv_4'):
        output = self.conv1d_transpose(output, 1, kernel_len, 4, upsample=genr_upsample)
      output = tf.nn.tanh(output)
    elif slice_len == 32768:
      with tf.variable_scope('upconv_4'):
        output = self.conv1d_transpose(output, gan_dim, kernel_len, 4, upsample=genr_upsample)
        output = batchnorm(output)
        output = tf.nn.relu(output)
      with tf.variable_scope('upconv_5'):
        output = self.conv1d_transpose(output, 1, kernel_len, 2, upsample=genr_upsample)
      output = tf.nn.tanh(output)
    elif slice_len == 65536:
      with tf.variable_scope('upconv_4'):
        output = self.conv1d_transpose(output, gan_dim, kernel_len, 4, upsample=genr_upsample)
        output = batchnorm(output)
        output = tf.nn.relu(output)
      with tf.variable_scope('upconv_5'):
        output = self.conv1d_transpose(output, 1, kernel_len, 4, upsample=genr_upsample)
        output = tf.nn.tanh(output)

    if train and use_batchnorm:
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=tf.get_variable_scope().name)
      if slice_len == 16384:
        assert len(update_ops) == 10
      else:
        assert len(update_ops) == 12
      with tf.control_dependencies(update_ops):
        output = tf.identity(output)

    return output

  def get_discriminator(self, x, kernel_len=25, gan_dim=64, use_batchnorm=False, disc_phaseshuffle=2):
    batch_size = tf.shape(x)[0]
    slice_len = int(x.get_shape()[1])

    if use_batchnorm:
      batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
    else:
      batchnorm = lambda x: x

    if disc_phaseshuffle > 0:
      phaseshuffle = lambda x: self.apply_phaseshuffle(x, disc_phaseshuffle)
    else:
      phaseshuffle = lambda x: x

    output = x
    with tf.variable_scope('downconv_0'):
      output = tf.layers.conv1d(output, gan_dim, kernel_len, 4, padding='SAME')
      output = self.lrelu(output)
      output = phaseshuffle(output)

    with tf.variable_scope('downconv_1'):
      output = tf.layers.conv1d(output, gan_dim * 2, kernel_len, 4, padding='SAME')
      output = batchnorm(output)
      output = self.lrelu(output)
      output = phaseshuffle(output)

    with tf.variable_scope('downconv_2'):
      output = tf.layers.conv1d(output, gan_dim * 4, kernel_len, 4, padding='SAME')
      output = batchnorm(output)
      output = self.lrelu(output)
      output = phaseshuffle(output)

    with tf.variable_scope('downconv_3'):
      output = tf.layers.conv1d(output, gan_dim * 8, kernel_len, 4, padding='SAME')
      output = batchnorm(output)
      output = self.lrelu(output)
      output = phaseshuffle(output)

    with tf.variable_scope('downconv_4'):
      output = tf.layers.conv1d(output, gan_dim * 16, kernel_len, 4, padding='SAME')
      output = batchnorm(output)
      output = self.lrelu(output)

    if slice_len == 32768:
      with tf.variable_scope('downconv_5'):
        output = tf.layers.conv1d(output, gan_dim * 32, kernel_len, 2, padding='SAME')
        output = batchnorm(output)
        output = self.lrelu(output)
    elif slice_len == 65536:
      with tf.variable_scope('downconv_5'):
        output = tf.layers.conv1d(output, gan_dim * 32, kernel_len, 4, padding='SAME')
        output = batchnorm(output)
        output = self.lrelu(output)

    output = tf.reshape(output, [batch_size, -1])
    with tf.variable_scope('output'):
      output = tf.layers.dense(output, 1)[:, 0]

    return output

  def train(self, fps, sample_rate=16000, batch=64, save_secs=300, summary_secs=120, slice_len=16384, overlap_ratio=0., first_slice=False, pad_end=False, prefetch=0, latent_dim=100, kernel_len=25, gan_dim=64, use_batchnorm=False, disc_nupdates=5, loss='wgan-gp', genr_upsample='zeros', genr_pp=False, genr_pp_len=512, disc_phaseshuffle=2):
      with tf.name_scope('loader'):
          x = data_pipeline_gan(fps, batch_size=batch, slice_len=slice_len, decode_fs=sample_rate, decode_parallel_calls=4, slice_randomize_offset=False if first_slice else True,
                                slice_first_only=first_slice, slice_overlap_ratio=0. if first_slice else overlap_ratio, slice_pad_end=True if first_slice else pad_end, repeat=True,
                                shuffle=True, shuffle_buffer_size=4096, prefetch_size=batch * 4, prefetch_gpu_num=prefetch)[:, :, 0]

      # Make z vector
      z = tf.random_uniform([batch, latent_dim], -1., 1., dtype=tf.float32)

      # Make generator
      with tf.variable_scope('G'):
          G_z = self.get_generator(z=z, slice_len=slice_len, kernel_len=kernel_len, gan_dim=gan_dim, use_batchnorm=use_batchnorm, genr_upsample=genr_upsample, train=True)
          if genr_pp:
              with tf.variable_scope('pp_filt'):
                  G_z = tf.layers.conv1d(G_z, 1, genr_pp_len, use_bias=False, padding='same')
      G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

      # Print G summary
      print('> Generator vars')
      nparams = 0
      for v in G_vars:
          v_shape = v.get_shape().as_list()
          v_n = reduce(lambda x, y: x * y, v_shape)
          nparams += v_n
          print('>> {} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
      print('>> Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))

      # Summarize
      tf.summary.audio('x', x, sample_rate)
      tf.summary.audio('G_z', G_z, sample_rate)
      G_z_rms = tf.sqrt(tf.reduce_mean(tf.square(G_z[:, :, 0]), axis=1))
      x_rms = tf.sqrt(tf.reduce_mean(tf.square(x[:, :, 0]), axis=1))
      tf.summary.histogram('x_rms_batch', x_rms)
      tf.summary.histogram('G_z_rms_batch', G_z_rms)
      tf.summary.scalar('x_rms', tf.reduce_mean(x_rms))
      tf.summary.scalar('G_z_rms', tf.reduce_mean(G_z_rms))

      # Make real discriminator
      with tf.name_scope('D_x'), tf.variable_scope('D'):
          D_x = self.get_discriminator(x=x, kernel_len=kernel_len, gan_dim=gan_dim, use_batchnorm=use_batchnorm, disc_phaseshuffle=disc_phaseshuffle)
      D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')

      # Print D summary
      print('> Discriminator vars')
      nparams = 0
      for v in D_vars:
          v_shape = v.get_shape().as_list()
          v_n = reduce(lambda x, y: x * y, v_shape)
          nparams += v_n
          print('>> {} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
      print('>> Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))

      # Make fake discriminator
      with tf.name_scope('D_G_z'), tf.variable_scope('D', reuse=True):
          D_G_z = self.get_discriminator(x=G_z, kernel_len=kernel_len, gan_dim=gan_dim, use_batchnorm=use_batchnorm, disc_phaseshuffle=disc_phaseshuffle)

      # Create loss
      D_clip_weights = None
      if loss == 'dcgan':
          fake = tf.zeros([batch], dtype=tf.float32)
          real = tf.ones([batch], dtype=tf.float32)
          G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_z, labels=real))
          D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_z, labels=fake))
          D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_x, labels=real))
          D_loss /= 2.
      elif loss == 'lsgan':
          G_loss = tf.reduce_mean((D_G_z - 1.) ** 2)
          D_loss = tf.reduce_mean((D_x - 1.) ** 2)
          D_loss += tf.reduce_mean(D_G_z ** 2)
          D_loss /= 2.
      elif loss == 'wgan':
          G_loss = -tf.reduce_mean(D_G_z)
          D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)
          with tf.name_scope('D_clip_weights'):
              clip_ops = []
              for var in D_vars:
                  clip_bounds = [-.01, .01]
                  clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
              D_clip_weights = tf.group(*clip_ops)
      elif loss == 'wgan-gp':
          G_loss = -tf.reduce_mean(D_G_z)
          D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)
          alpha = tf.random_uniform(shape=[batch, 1, 1], minval=0., maxval=1.)
          differences = G_z - x
          interpolates = x + (alpha * differences)
          with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
              D_interp = self.get_discriminator(x=interpolates, kernel_len=kernel_len, gan_dim=gan_dim, use_batchnorm=use_batchnorm, disc_phaseshuffle=disc_phaseshuffle)
          LAMBDA = 10
          gradients = tf.gradients(D_interp, [interpolates])[0]
          slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
          gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
          D_loss += LAMBDA * gradient_penalty
      else:
          raise NotImplementedError()

      tf.summary.scalar('G_loss', G_loss)
      tf.summary.scalar('D_loss', D_loss)

      # Create optimizer
      if loss == 'dcgan':
          G_opt = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
          D_opt = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
      elif loss == 'lsgan':
          G_opt = tf.train.RMSPropOptimizer(learning_rate=1e-4)
          D_opt = tf.train.RMSPropOptimizer(learning_rate=1e-4)
      elif loss == 'wgan':
          G_opt = tf.train.RMSPropOptimizer(learning_rate=5e-5)
          D_opt = tf.train.RMSPropOptimizer(learning_rate=5e-5)
      elif loss == 'wgan-gp':
          G_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
          D_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
      else:
          raise NotImplementedError()

      # Create training ops
      self.g_optimizer = G_opt.minimize(G_loss, var_list=G_vars, global_step=tf.train.get_or_create_global_step())
      self.g_loss = G_loss

      self.d_optimizer = D_opt.minimize(D_loss, var_list=D_vars)
      self.d_loss = D_loss

      # Run training
      print('> Training steps')

      scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=1))
      with tf.train.MonitoredTrainingSession(checkpoint_dir=self.dir, save_checkpoint_secs=save_secs, save_summaries_secs=summary_secs, scaffold=scaffold) as sess:

          epochs = 0
          d_steps = 0
          g_steps = 0
          total_d_loss = 0
          total_g_loss = 0
          saved_ckpt = 0

          while True:

              for _ in xrange(disc_nupdates):

                  _, d_loss = sess.run([self.d_optimizer, self.d_loss])
                  total_d_loss += d_loss
                  d_steps += 1

                  if D_clip_weights is not None:
                      sess.run(D_clip_weights)

              _, g_loss = sess.run([self.g_optimizer, self.g_loss])
              total_g_loss += g_loss
              g_steps += 1

              latest_ckpt_fp = float(tf.train.latest_checkpoint(self.dir).split('-')[1])

              print('\r>> step %5.0f - loss_d %3.5f - loss_g %3.5f - saved_ckpt %5.0f' % (epochs, total_d_loss / (d_steps+1), total_g_loss / (g_steps+1), latest_ckpt_fp), end='')
              epochs += 1

  def infer(self, latent_dim=100, slice_len=16384, kernel_len=25, gan_dim=64, use_batchnorm=False, genr_upsample='zeros', genr_pp=False, genr_pp_len=512):
      # Subgraph that generates latent vectors
      samp_z_n = tf.placeholder(tf.int32, [], name='samp_z_n')
      samp_z = tf.random_uniform([samp_z_n, latent_dim], -1.0, 1.0, dtype=tf.float32, name='samp_z')

      # Input zo
      z = tf.placeholder(tf.float32, [None, latent_dim], name='z')
      flat_pad = tf.placeholder(tf.int32, [], name='flat_pad')

      # Execute generator
      with tf.variable_scope('G'):
          G_z = self.get_generator(z=z, slice_len=slice_len, kernel_len=kernel_len, gan_dim=gan_dim, use_batchnorm=use_batchnorm, genr_upsample=genr_upsample, train=False)
          if genr_pp:
              with tf.variable_scope('pp_filt'):
                  G_z = tf.layers.conv1d(G_z, 1, genr_pp_len, use_bias=False, padding='same')
      G_z = tf.identity(G_z, name='G_z')

      # Flatten batch
      nch = int(G_z.get_shape()[-1])
      G_z_padded = tf.pad(G_z, [[0, 0], [0, flat_pad], [0, 0]])
      G_z_flat = tf.reshape(G_z_padded, [-1, nch], name='G_z_flat')

      # Encode to int16
      def float_to_int16(x, name=None):
          x_int16 = x * 32767.
          x_int16 = tf.clip_by_value(x_int16, -32767., 32767.)
          x_int16 = tf.cast(x_int16, tf.int16, name=name)
          return x_int16

      G_z_int16 = float_to_int16(G_z, name='G_z_int16')
      G_z_flat_int16 = float_to_int16(G_z_flat, name='G_z_flat_int16')

      # Create saver
      G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')
      global_step = tf.train.get_or_create_global_step()
      saver = tf.train.Saver(G_vars + [global_step])

      # Export graph
      tf.train.write_graph(tf.get_default_graph(), self.dir, 'infer.pbtxt')

      # Export MetaGraph
      infer_metagraph_fp = os.path.join(self.dir, 'infer.meta')
      tf.train.export_meta_graph(filename=infer_metagraph_fp, clear_devices=True, saver_def=saver.as_saver_def())

      # Reset graph
      tf.reset_default_graph()

  def preview(self, sample_rate=16000, genr_pp=False, preview_n=32):
      preview_dir = os.path.join(self.dir, 'preview')
      if not os.path.isdir(preview_dir):
          os.makedirs(preview_dir)

      # Load graph
      infer_metagraph_fp = os.path.join(self.dir, 'infer.meta')
      graph = tf.get_default_graph()
      saver = tf.train.import_meta_graph(infer_metagraph_fp)

      # Generate z_i and z_o
      samp_feeds = {}
      samp_feeds[graph.get_tensor_by_name('samp_z_n:0')] = preview_n
      samp_fetches = {}
      samp_fetches['zs'] = graph.get_tensor_by_name('samp_z:0')
      with tf.Session() as sess:
          _samp_fetches = sess.run(samp_fetches, samp_feeds)
      _zs = _samp_fetches['zs']

      # Set up graph for generating preview images
      feeds = {}
      feeds[graph.get_tensor_by_name('z:0')] = _zs
      feeds[graph.get_tensor_by_name('flat_pad:0')] = int(sample_rate / 2)
      fetches = {}
      fetches['step'] = tf.train.get_or_create_global_step()
      fetches['G_z'] = graph.get_tensor_by_name('G_z:0')
      fetches['G_z_flat_int16'] = graph.get_tensor_by_name('G_z_flat_int16:0')
      if genr_pp:
          fetches['pp_filter'] = graph.get_tensor_by_name('G/pp_filt/conv1d/kernel:0')[:, 0, 0]

      # Summarize
      G_z = graph.get_tensor_by_name('G_z_flat:0')
      summaries = [tf.summary.audio('preview', tf.expand_dims(G_z, axis=0), sample_rate, max_outputs=1)]
      fetches['summaries'] = tf.summary.merge(summaries)
      summary_writer = tf.summary.FileWriter(preview_dir)

      # PP Summarize
      if genr_pp:
          pp_fp = tf.placeholder(tf.string, [])
          pp_bin = tf.read_file(pp_fp)
          pp_png = tf.image.decode_png(pp_bin)
          pp_summary = tf.summary.image('pp_filt', tf.expand_dims(pp_png, axis=0))

      # Waiting for checkpoints
      latest_ckpt_fp = tf.train.latest_checkpoint(self.dir)

      print('> checkpoint {}'.format(latest_ckpt_fp))

      with tf.Session() as sess:
          saver.restore(sess, latest_ckpt_fp)
          _fetches = sess.run(fetches, feeds)
          _step = _fetches['step']

      preview_fp = os.path.join(preview_dir, '{}.wav'.format(str(_step).zfill(5)))
      wavwrite(preview_fp, sample_rate, _fetches['G_z_flat_int16'])
      summary_writer.add_summary(_fetches['summaries'], _step)

      if genr_pp:
          w, h = freqz(_fetches['pp_filter'])
          fig = plt.figure()
          plt.title('Digital filter frequncy response')
          ax1 = fig.add_subplot(111)
          plt.plot(w, 20 * np.log10(abs(h)), 'b')
          plt.ylabel('Amplitude [dB]', color='b')
          plt.xlabel('Frequency [rad/sample]')
          ax2 = ax1.twinx()
          angles = np.unwrap(np.angle(h))
          plt.plot(w, angles, 'g')
          plt.ylabel('Angle (radians)', color='g')
          plt.grid()
          plt.axis('tight')
          _pp_fp = os.path.join(preview_dir, '{}_ppfilt.png'.format(str(_step).zfill(8)))
          plt.savefig(_pp_fp)
          with tf.Session() as sess:
              _summary = sess.run(pp_summary, {pp_fp: _pp_fp})
              summary_writer.add_summary(_summary, _step)

      print('> preview created in', preview_fp)