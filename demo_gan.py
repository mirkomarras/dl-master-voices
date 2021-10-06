import os
import sys
import time
import pickle

import numpy as np
import tensorflow as tf


from PyQt5 import QtCore, QtWidgets, QtGui

from matplotlib.backends.qt_compat import is_pyqt5


if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


from helpers import plotting, audio

from models import gan

import sounddevice as sd
import soundfile as sf


# ROOT_DIR = 'data/f/loss_ablation_selected'
# FLOWS = ['db0R 000', 'db0R 001', 'db0r 000', 'db0r 001', 'db0r 002', 'db0r 003', 'db0r 004', 'naive 001', 'naive 002', 'naive 003']

# ROOT_DIR = 'data/f_hpc/loss_ablation_selected'
# FLOWS = ['db0r 002']

# ROOT_DIR = 'data/f/extractor_cc'
# FLOWS = ['naive 000']
#
# ROOT_DIR = 'data/f/extractor_cc_simple/'
# FLOWS = '.*'


class SensorInteractWindow(QtWidgets.QMainWindow):

    def keyPressEvent(self, QKeyEvent):

        if QKeyEvent.key() == QtCore.Qt.Key_Space:
            self.index = (self.index + 1) % 128

        if QKeyEvent.key() == QtCore.Qt.Key_Backspace:
            self.index = (self.index - 1) % 128

        if QKeyEvent.key() == QtCore.Qt.Key_0:
            self.index = 0

        # if QKeyEvent.key() == QtCore.Qt.Key_Up:
        #     self._last_position = (self._last_position[0], self._last_position[1] - 1)

        # if QKeyEvent.key() == QtCore.Qt.Key_Down:
        #     self._last_position = (self._last_position[0], self._last_position[1] + 1)

        # if QKeyEvent.key() == QtCore.Qt.Key_Left:
        #     self._last_position = (self._last_position[0] - 1, self._last_position[1])

        # if QKeyEvent.key() == QtCore.Qt.Key_Right:
        #     self._last_position = (self._last_position[0] + 1, self._last_position[1])

        # if QKeyEvent.key() == QtCore.Qt.Key_K:
        #     self.k0 = self.flow.sample_spn(self.data.valid_patch_size_rgb // 2)

        # if QKeyEvent.key() == QtCore.Qt.Key_F:
        #     self.load_flow(1)

        if QKeyEvent.key() == QtCore.Qt.Key_P:
            s0 = self.get_current_spectrum()            
            # s0 = self.gan.generator(self.z)[-1]
            sp = s0.numpy().squeeze()
            sp = np.vstack((sp, np.zeros((1, sp.shape[1])), sp[:0:-1]))
            sp = np.clip(sp, 0, None)
            
            wave_rec = audio.spectrum_to_signal(sp.T, int(16000 * 2.57), 100)
            
            
            sd.play(wave_rec, 16000)


        if QKeyEvent.key() == QtCore.Qt.Key_S:
            self.z = self.gan.sample_z(1)

        # if QKeyEvent.key() == QtCore.Qt.Key_D:
        #     self._diff_rgb = not self._diff_rgb

        self.setWindowTitle(self.window_title)
        self.update_viz(*self._last_position)

    @property
    def label(self):
        pixel_label = 'RGGB'[self._pixel_selection] if self._pixel_selection < 4 else 'RGGB'
        return f'image: {self._image_id}, pixel_sel: {pixel_label}'

    @property
    def window_title(self):
        pos = 2 * (self._last_position[0] / 256 - 0.5)
        return f'Window {pos}'
    
    def get_current_spectrum(self):
        pixel_x, pixel_y = self._last_position
        
        if self.gan.latent_dist == 'normal':
            value = 6 * (self._last_position[0] / 256 - 0.5)
        else:
            value = (self._last_position[0] / 256)
                
        z0 = np.copy(self.z)
        z0[0, self.index] = value        
        
        return self.gan.generator(z0)[-1]

    def update_viz(self, pixel_x, pixel_y):
        self._last_position = (pixel_x, pixel_y)

        for ax in self.axes:
            ax.cla()

        # batch_x = self.data.next_validation_batch(self._image_id, 1)[0]
        # batch_x = tf.convert_to_tensor(batch_x)

        # k0 = tf.convert_to_tensor(self.k0)

        # with tf.GradientTape() as tape:
        #     tape.watch(batch_x)
        #     tape.watch(k0)
        #     batch_x0 = self.flow.sensor.process(batch_x, k0)
        #     if self._pixel_selection < 4:
        #         pixel = batch_x0[0, pixel_y, pixel_x, self._pixel_selection]
        #     else:
        #         pixel = batch_x0[0, pixel_y, pixel_x, :]

        # grad_x, grad_k = tape.gradient(pixel, (batch_x, k0))

        # batch_x = batch_x.numpy()
        # batch_x0 = batch_x0.numpy()
        # batch_y = self.flow.isp.process(batch_x).numpy()
        # batch_y0 = self.flow.isp.process(batch_x0).numpy()

        # if self._fft:
        #     if self._diff_rgb:
        #         diff_label = 'FFT($y$ - $y_0$)'
        #         batch_d = batch_y - batch_y0
        #         batch_d = image.fft_log_norm(batch_d)
        #         # batch_d = np.abs(image.fft_log_norm(batch_y)) - np.abs(image.fft_log_norm(batch_y0))
        #         batch_d = image.normalize(batch_d, 0.1)
        #     else:
        #         diff_label = 'FFT($x$ - $x_0$)'
        #         batch_d = batch_x - batch_x0
        #         batch_d = np.sum(batch_d, axis=-1, keepdims=True)
        #         batch_d = image.fft_log_norm(batch_d)
        #         # batch_d = np.abs(image.fft_log_norm(batch_y)) - np.abs(image.fft_log_norm(batch_y0))
        #         batch_d = image.normalize(batch_d, 0.1)
        # else:
        #     if self._diff_rgb:
        #         diff_label = '$y$ - $y_0$'
        #         batch_d = image.normalize_residual(batch_y - batch_y0)
        #     else:
        #         diff_label = '$x$ - $x_0$'
        #         batch_d = batch_x - batch_x0
        #         batch_d = np.sum(batch_d, axis=-1, keepdims=True)
        #         batch_d = image.normalize_residual(batch_d)

        # # grad_x = grad_x * batch_x
        # # grad_x = image.normalize_residual(grad_x.numpy())
        # grad_x = grad_x.numpy()
        # grad_x = np.moveaxis(grad_x.squeeze(), -1, 0)
        # grad_xt = plots.thumbnails(grad_x, 2)

        # # grad_k = grad_k * k0
        # # grad_k = image.normalize_residual(grad_k.numpy())
        # grad_k = grad_k.numpy()
        # grad_k = np.moveaxis(grad_k.squeeze(), -1, 0)
        # grad_kt = plots.thumbnails(grad_k, 2)
        
        # index = 0
        if self.gan.latent_dist == 'normal':
            value = 6 * (self._last_position[0] / 256 - 0.5)
        else:
            value = (self._last_position[0] / 256)
        
        # np.random.normal(size=(1,128))
        # z = tf.convert_to_tensor(z)
        
        sp = self.gan.generator(self.z)[-1].numpy()
        
        z0 = np.copy(self.z)
        z0[0, self.index] = value
        # self.z = tf.convert_to_tensor(z0)
        
        sp0 = self.gan.generator(z0)[-1].numpy()
        
        plotting.quickshow(sp, 'original sample', cmap='jet', axes=self.axes[0])
        plotting.quickshow(sp0, f'set idx={self.index} to {value:.2f}', cmap='jet', axes=self.axes[1])
        
        self.axes[2].plot(z0.ravel())
        self.axes[2].plot([self.index], z0[0, self.index], 'ro')
        if self.gan.latent_dist == 'normal':
            self.axes[2].set_ylim([-3.2, 3.2])
        else:
            self.axes[2].set_ylim([0.05, 1.05])

        # plots.image(batch_x[..., [0, 1, 3]], f'RAW {self.label}', axes=self.axes[0])
        # plots.image(batch_y0, 'RGB image ($y_0$)', axes=self.axes[1])
        # plots.image(batch_d, diff_label, axes=self.axes[2])
        # plots.image(grad_xt, '$\\nabla_x$', axes=self.axes[3], cmap='seismic', vrange=False)
        # plots.image(grad_kt, '$\\nabla_k$', axes=self.axes[4], cmap='RdGy', vrange=False)

        # self.fig.axes[0].annotate('o', (pixel_x, pixel_y), ha='center', va='center', color='red')

        self.setWindowTitle(self.window_title)
        self.fig.canvas.draw_idle()

    def __init__(self):
        super().__init__()

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QHBoxLayout(self._main)
        
        model = 'ms-gan'
        dataset = 'vctk'
        version = 0
        dist = 'normal'
        patch = 256
        
        self.gan = gan.MultiscaleGAN(dataset, version=version, patch=patch, width_ratio=1, min_output=8, latent_dist=dist)
        self.gan.load()

        self.index = 0        
        self.z = self.gan.sample_z(1)

        self.fig, self.axes = plotting.sub(3, ncols=3)
        self._onmove_disabled = False

        def onclick(event):
            self._onmove_disabled = not self._onmove_disabled
            pixel_x = int(event.xdata)
            pixel_y = int(event.ydata)
            self.update_viz(pixel_x, pixel_y)

        def onmove(event):
            if self._onmove_disabled:
                return

            try:
                pixel_x = int(event.xdata)
                pixel_y = int(event.ydata)
                self.update_viz(pixel_x, pixel_y)
            except:
                pass

        static_canvas = FigureCanvas(self.fig)
        layout.addWidget(static_canvas)

        self.fig.canvas.mpl_connect('button_press_event', onclick)
        self.fig.canvas.mpl_connect('motion_notify_event', onmove)
        self.update_viz(0, 0)
        self.setWindowTitle(self.window_title)


if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = SensorInteractWindow()
    app.show()
    qapp.exec_()
