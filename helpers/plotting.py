# -*- coding: utf-8 -*-

from skimage.transform import resize
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plot
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import glob
import os

from helpers import audio, utils

from loguru import logger

# Helper functions for plotting images in Python (wrapper over matplotlib)

__INTERACTIVE = False


def set_interactive(status=False):
    global __INTERACTIVE
    __INTERACTIVE = status


def get_figure(**kwargs):
    if __INTERACTIVE:
        import matplotlib.pyplot as plt
        return plt.figure(**kwargs)
    else:
        from matplotlib.figure import Figure
        return Figure(**kwargs)


def thumbnails(images, ncols=0, columnwise=False, cell_size=None, pad=0):
    """
    Return a numpy array with image thumbnails.
    """
    
    if type(images) is np.ndarray:
        
        n_images = images.shape[0]        
        n_channels = images.shape[-1]
        img_size = images.shape[1:]
        
        if len(img_size) == 2:
            img_size += (1,)
                        
    elif type(images) is list or type(images) is tuple:
        
        n_images = len(images)
        n_channels = images[0].shape[-1]
        img_size = list(images[0].shape)
        
        if len(img_size) == 2:
            img_size += (1,)
    
    ncols = ncols if ncols > 0 else n_images
    images_x = ncols or int(np.ceil(np.sqrt(n_images)))
    images_y = int(np.ceil(n_images / images_x))
    size = (images_y, images_x)

    if cell_size is not None:
        img_size = (cell_size, cell_size, img_size[-1])

    # Allocate space for the thumbnails
    output = np.zeros((size[0] * (img_size[0] + 2*pad), size[1] * (img_size[1] + 2*pad), img_size[2]))
        
    for r in range(n_images):
        bx = int(r % images_x)
        by = int(np.floor(r / images_x))

        if columnwise:
            by = int(r % images_y)
            bx = int(np.floor(r / images_y))

        current = images[r]
        
        if current.ndim > 3:
            current = current.squeeze()

        if current.shape[0] != img_size[0] or current.shape[1] != img_size[1]:
            current = resize(current, img_size[:-1], anti_aliasing=True)

        if len(current.shape) == 2:
            current = np.expand_dims(current, axis=2)

        if pad:
            if current.ndim == 2:
                np_pad = ((pad, pad), (pad, pad))
            elif current.ndim == 3:
                np_pad = ((pad, pad), (pad, pad), (0, 0))
            else:
                raise ValueError(f'Could not pad the current cell: {current.shape}')

            current = np.pad(current, np_pad)

        output[by*(img_size[0]+2*pad):(by+1)*(img_size[0]+2*pad), bx*(img_size[1]+2*pad):(bx+1)*(img_size[1]+2*pad), :] = current
        
    return output
    

def _imarray(img, n_images, fetch_hook, titles, figwidth=4, cmap='gray', ncols=0, fig=None, rowlabels=None, global_vrange=None):
    """
    Function for plotting arrays of images. Not intended to be used directly. See 'images' for typical use cases.
    """
    if n_images > 128:
        raise RuntimeError('The number of subplots exceeds reasonable limits ({})!'.format(n_images))                            
    
    if ncols == 0:
        ncols = int(np.ceil(np.sqrt(n_images)))
    elif ncols < 0:
        ncols = n_images // abs(ncols)

    subplot_x = ncols
    subplot_y = int(np.ceil(n_images / subplot_x))

    if rowlabels is not None and len(rowlabels) != subplot_y:
        raise ValueError('The number of rows does not match the provided labels!')
            
    if titles is not None and type(titles) is str:
        titles = [titles for x in range(n_images)]
        
    if titles is not None and len(titles) != n_images:
        raise ValueError('Provided titles ({}) do not match the number of images ({})!'.format(len(titles), n_images))

    fig = fig or get_figure(figsize=(figwidth * subplot_x, figwidth * subplot_y))

    for n in range(n_images):
        ax = fig.add_subplot(subplot_y, subplot_x, n + 1)

        if isinstance(global_vrange, bool) and global_vrange:
            vrange = np.max([np.max(np.abs(x)) for x in img])
            vrange = (-vrange, vrange)
        elif isinstance(global_vrange, bool) and not global_vrange:
            vrange = np.max(np.abs(fetch_hook(img, n)))
            vrange = (-vrange, vrange)
        elif isinstance(global_vrange, tuple):
            vrange = global_vrange
        else:
            vrange = None

        image(fetch_hook(img, n), titles[n] if titles is not None else None, axes=ax, cmap=cmap, vrange=vrange)
        if rowlabels is not None and n % subplot_x == 0:
            ax.set_ylabel(rowlabels[n // subplot_x])

    return fig
    

def images(imgs, titles=None, figwidth=4, cmap='gray', ncols=0, fig=None, rowlabels=None, global_vrange=None):
    """
    Plot a series of images (in various structures). Not thoroughly tested, but should work with:

    - np.ndarray of size (h,w,3) or (h,w)
    - lists or tuples of np.ndarray of size (h,w,3) or (h,w)    
    - np.ndarray of size (h,w,channels) -> channels shown separately
    - np.ndarray of size (1, h, w, channels)
    - np.ndarray of size (N, h, w, 3) and (N, h, w, 1)
    
    CAUTION: This function creates new figures without a canvas manager - this prevents memory leaks but makes it more
    difficult to plot figures in interactive notebooks. In case of trouble, you can supply a target figure created by
    a managed matplotlib interface - use fig=plt.figure().

    :param imgs: input image structure (see details above)
    :param titles: a single string or a list of strings matching the number of images in the structure
    :param figwidth: width of a single image in the figure
    :param cmap: color map
    :param ncols: number of columns or: 0 for sqrt(#images) cols; use negative to set the number of rows
    :param fig: specify the target figure for plotting
    :param global_vrange: display all images with the same intensity range
    """
        
    if type(imgs) is list or type(imgs) is tuple:
        
        n_images = len(imgs)
        
        def fetch_example(image, n):
            return image[n]        
                    
        return _imarray(imgs, n_images, fetch_example, titles, figwidth, cmap, ncols, fig, rowlabels, global_vrange)
            
    elif type(imgs) in [np.ndarray]:
        
        if imgs.ndim == 2 or (imgs.ndim == 3 and imgs.shape[-1] == 3):
            
            fig = fig or get_figure(tight_layout=True, figsize=(figwidth, figwidth))
            image(imgs, titles, axes=fig.gca(), cmap=cmap)
            
            return fig

        elif imgs.ndim == 3 and imgs.shape[-1] != 3:
                        
            def fetch_example(im, n):
                return im[..., n]
            
            n_images = imgs.shape[-1]

            if n_images > 100:
                imgs = np.moveaxis(imgs, 0, -1)
                n_images = imgs.shape[-1]
                                        
        elif imgs.ndim == 4 and (imgs.shape[-1] == 3 or imgs.shape[-1] == 1):
            
            n_images = imgs.shape[0]
            
            def fetch_example(im, n):
                return im[n]
            
        elif imgs.ndim == 4 and imgs.shape[0] == 1:

            n_images = imgs.shape[-1]
            
            def fetch_example(im, n):
                return im[..., n]

        else:
            raise ValueError('Unsupported array dimensions {}!'.format(imgs.shape))
            
        return _imarray(imgs, n_images, fetch_example, titles, figwidth, cmap, ncols, fig, rowlabels, global_vrange)
            
    else:
        raise ValueError('Unsupported array type {}!'.format(type(imgs)))
                
    return fig


def image(x, label=None, *, axes=None, cmap='gray', vrange=None):
    """
    Plot a single image, hide ticks & add a formatted title with patterns replaced as follows:
    - '()' -> '(height x width)'
    - '[]' -> '[min - max]'
    - '{}' -> '(height x width) / [min - max]'
    - '<>' -> 'avg ± std'
    """
    
    label = label if label is not None else '{}'
    
    x = x.squeeze()
    
    if any(ptn in label for ptn in ['{}', '()', '[]', '<>']):
        label = label.replace('{}', '() / []')
        label = label.replace('()', '({}x{})'.format(*x.shape[0:2]))
        label = label.replace('[]', '[{:.2f} - {:.2f}]'.format(np.min(x), np.max(x)))
        label = label.replace('<>', '{:.2f} ± {:.2f}'.format(np.mean(x), np.std(x)))
        
    if axes is None:
        fig = get_figure()
        axes = fig.gca()

    if vrange is None:
        axes.imshow(x, cmap=cmap)
    elif isinstance(vrange, tuple) and len(vrange) == 2:
        axes.imshow(x, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
    elif isinstance(vrange, bool) and vrange:
        axes.imshow(x, cmap=cmap, vmin=np.min(x), vmax=np.max(x))
    elif isinstance(vrange, bool) and not vrange:
        axes.imshow(x, cmap=cmap, vmin=-np.max(np.abs(x)), vmax=np.max(np.abs(x)))
    else:
        raise ValueError('Invalid vrange - supported types: None, 2-elem tuple & boolean')

    if len(label) > 0:
        axes.set_title(label)
    axes.set_xticks([])
    axes.set_yticks([])

    if 'fig' in locals():
        return fig


def sub(n_plots, figwidth=6, figheight=None, ncols=-1, fig=None, transpose=False, gridspec=None, sharex=False, sharey=False):
    """
    Create a figure and split it into subplots. Provides more consistent behavior than matplotlib. Key features:
    - the returned axes are always a list
    - automatically choose number of rows/columns based on the total number of plots
    - the extra subplots will be turned off
    - axes traversal order can be changed (column/row-wise)

    :param n_plots:
    :param figwidth:
    :param figheight:
    :param ncols:
    :param fig:
    :param transpose:
    :return:
    """

    if ncols == 0:
        ncols = int(np.ceil(np.sqrt(n_plots)))
    elif ncols < 0:
        ncols = n_plots // abs(ncols)

    figheight = figheight or figwidth

    subplot_x = ncols or int(np.ceil(np.sqrt(n_plots)))
    subplot_y = int(np.ceil(n_plots / subplot_x))

    if transpose:
        subplot_x, subplot_y = subplot_y, subplot_x

    fig = fig or get_figure(tight_layout=True, figsize=(figwidth * subplot_x, subplot_y * (figheight or figwidth * (subplot_y / subplot_x))))
    axes = fig.subplots(nrows=subplot_y, ncols=subplot_x, gridspec_kw=gridspec, sharex=sharex, sharey=sharey)
    axes_flat = []

    if not hasattr(axes, '__iter__'):
        axes = [axes]

    for ax in axes:
        
        if hasattr(ax, '__iter__'):
            for a in ax:
                if len(axes_flat) < n_plots:
                    axes_flat.append(a)
                else:
                    a.remove()
        else:
            if len(axes_flat) < n_plots:
                axes_flat.append(ax)
            else:
                ax.remove()

    if transpose:
        from itertools import product
        axes_flat = [axes_flat[j * subplot_x + i] for i, j in product(range(subplot_x), range(subplot_y))]
    
    return fig, axes_flat


def waveforms(wav, spectrums=True, sampling=16000):
    cols = 2 if spectrums else 1
    
    fig, axes = sub(len(wav) * cols, ncols=cols, figwidth=16, figheight=len(wav) * 3.5)
    
    for i, ax in enumerate(axes):
        if i % 2 == 0:
            ax.plot(wav[i // cols].squeeze(), alpha=1)
#             ax.set_xlim([0, 16384 // 8])
            ax.set_yticks([])
        else:
            spec = audio.get_np_spectrum(wav[i // cols].squeeze(), sampling)[0]
            quickshow(spec, axes=ax)
            ax.set_ylabel('Frequency')
    return fig


def sub(n_plots, figwidth=16, figheight=None, ncols=None):
    subplot_x = ncols or int(np.ceil(np.sqrt(n_plots)))
    subplot_y = int(np.ceil(n_plots / subplot_x))
    
    fig = plot.figure(tight_layout=True, figsize=(figwidth, figheight or figwidth * (subplot_y / subplot_x)))
    axes = fig.subplots(nrows=subplot_y, ncols=subplot_x)
    axes_flat = []

    for ax in axes:
        
        if hasattr(ax, '__iter__'):
            for a in ax:
                if len(axes_flat) < n_plots:
                    axes_flat.append(a)
                else:
                    a.remove()
        else:
            if len(axes_flat) < n_plots:
                axes_flat.append(ax)
            else:
                ax.remove()                
    
    return fig, axes_flat


def corrcoeff(a, b):
    """ Returns the normalized correlation coefficient between two arrays """
    a = (a - np.mean(a)) / (1e-9 + np.std(a))
    b = (b - np.mean(b)) / (1e-9 + np.std(b))
    return np.mean(a * b)


def rsquared(a, b):
    """ Returns the coefficient of determination (R^2) between two arrays (normalized) """
    from sklearn.metrics import r2_score
    a = (a - np.mean(a)) / (1e-9 + np.std(a))
    b = (b - np.mean(b)) / (1e-9 + np.std(b))
    return r2_score(a, b)


def correlation(x, y, xlabel=None, ylabel=None, title=None, axes=None, alpha=0.1, guide=False, color=None, kde=False, marginals=False):

    title = '{} : '.format(title) if title is not None else ''

    cc = corrcoeff(x.ravel(), y.ravel())
    r2 = rsquared(x.ravel(), y.ravel())

    if axes is None:
        fig = plt.figure()
        axes = fig.gca()
        
    x = x.ravel()
    y = y.ravel()

    axes.plot(x, y, '.', alpha=alpha, color=color, zorder=1)
    axes.set_title('{}corr {:.2f} / R2 {:.2f}'.format(title, cc, r2))

    if guide:
        p1 = min(np.min(x), np.min(y))
        p2 = max(np.max(x), np.max(y))
        axes.plot([p1, p2], [p1, p2], 'k--', alpha=0.3)
        span_x = np.max(x) - np.min(x)
        span_y = np.max(y) - np.min(y)
        axes.set_xlim([np.min(x) - span_x * 0.05, np.max(x) + span_x * 0.05])
        axes.set_ylim([np.min(y) - span_y * 0.05, np.max(y) + span_y * 0.05])
        
    if kde:
        span_x = np.max(x) - np.min(x)
        span_y = np.max(y) - np.min(y)
        xmin = np.min(x) - span_x * 0.05
        xmax = np.max(x) + span_x * 0.05
        ymin = np.min(y) - span_y * 0.05
        ymax = np.max(y) + span_y * 0.05
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = sps.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        cfset = axes.contourf(xx, yy, f, cmap='Blues', alpha=0.35, zorder=2)
        cset = axes.contour(xx, yy, f, colors='k', alpha=0.35, zorder=2)

    if marginals:
        yy = axes.get_ylim()
        xx = axes.get_xlim()

        # X marginal
        x_hist, x_bins = np.histogram(x.reshape((-1, )), bins=30)
        x_bins = np.convolve(x_bins, [0.5, 0.5], mode='valid')
        x_hist = x_hist / x_hist.max()
        axes.bar(x_bins, bottom=yy[1], height=0.1 * np.abs(yy[1] - yy[0]) * x_hist, zorder=-1, clip_on=False, alpha=0.5, width=x_bins[1] - x_bins[0])
        axes.set_ylim(yy)

        # Y marginal
        y_hist, y_bins = np.histogram(y.reshape((-1, )), bins=30)
        y_bins = np.convolve(y_bins, [0.5, 0.5], mode='valid')
        y_hist = y_hist / y_hist.max()
        axes.barh(y_bins, left=xx[1], width=0.1 * np.abs(xx[1] - xx[0]) * y_hist, zorder=-1, clip_on=False, alpha=0.5, height=y_bins[1] - y_bins[0])
        axes.set_xlim(xx)

    if xlabel is not None: axes.set_xlabel(xlabel)
    if ylabel is not None: axes.set_ylabel(ylabel)

    if 'fig' in locals():
        return fig


def imp_rate_change(mv_set, target_pop, net, policy):
    def get_subplot(imps_mv_eer, imps_sv_eer, imps_sv_far1, imps_mv_far1, gender_map, gender):
        # Distributions
        distr_mv_eer = (imps_mv_eer > 0).sum(axis=1) / imps_mv_eer.shape[1]
        distr_mv_far1 = (imps_mv_far1 > 0).sum(axis=1) / imps_mv_eer.shape[1]
        distr_sv_eer = (imps_sv_eer > 0).sum(axis=1) / imps_mv_eer.shape[1]
        distr_sv_far1 = (imps_sv_far1 > 0).sum(axis=1) / imps_mv_eer.shape[1]

        # Title
        plt.title('Attack to {} users ({}s)'.format(gender, gender_map.count(gender[0])))

        # Plotting
        xs = np.linspace(0, 1, 200)
        plt.plot(xs, gaussian_kde(distr_mv_eer)(xs), color='blue', linestyle='-', label=r'MVs $\tau_{EER}$')
        plt.plot(xs, gaussian_kde(distr_mv_far1)(xs), color='blue', linestyle='--', label=r'MVs $\tau_{FAR1\%}$')
        plt.plot(xs, gaussian_kde(distr_sv_eer)(xs), color='green', linestyle='-', label=r'SVs $\tau_{EER}$')
        plt.plot(xs, gaussian_kde(distr_sv_far1)(xs), color='green', linestyle='--', label=r'SVs $\tau_{FAR1\%}$')

        # Decorations
        plt.legend(ncol=2)
        plt.xlim(0, 1)
        plt.xlabel('Impersonation Rate')
        plt.ylim(0, None)
        plt.ylabel('Density')

        plt.grid()

    # Load impersonation rates
    mv_eer_path = os.path.join('..', 'data', 'vs_mv_data', mv_set, 'mv', target_pop + '-' + net + '-' + policy + '-eer.npz')
    imps_mv_eer = np.load(mv_eer_path, allow_pickle=True)['results'][()]['imps']
    imps_sv_eer = np.load(mv_eer_path.replace('mv' + os.sep, 'sv' + os.sep), allow_pickle=True)['results'][()]['imps']
    imps_sv_far1 = np.load(mv_eer_path.replace('mv' + os.sep, 'sv' + os.sep).replace('eer', 'far1'), allow_pickle=True)['results'][()]['imps']
    imps_mv_far1 = np.load(mv_eer_path.replace('eer', 'far1'), allow_pickle=True)['results'][()]['imps']

    # Load gender indexes
    pop_data = pd.read_csv(os.path.join('..', 'data', 'vs_mv_pairs', target_pop + '.csv'))
    gnds = pop_data.drop_duplicates('user_id')['gender'].to_list()
    m_idx = [i for i, x in enumerate(gnds) if x == 'm']
    f_idx = [i for i, x in enumerate(gnds) if x == 'f']

    # Grid
    plt.figure(figsize=(30, 10), dpi=300)
    plt.suptitle('mv_set={}, target_pop={}, net={}'.format(mv_set, target_pop, net), fontweight='bold')

    plt.subplot(1, 2, 1)
    get_subplot(imps_mv_eer[:, m_idx], imps_sv_eer[:, m_idx], imps_sv_far1[:, m_idx], imps_mv_far1[:, m_idx], gnds,
                'male')

    plt.subplot(1, 2, 2)
    get_subplot(imps_mv_eer[:, f_idx], imps_sv_eer[:, f_idx], imps_sv_far1[:, f_idx], imps_mv_far1[:, f_idx], gnds, 'female')

    plt.tight_layout()
    plt.show()


def imp_rate_scatter(mv_set, target_pop, net, policy):
    def get_subplot(imps_sv, imps_mv, level, gender):
        # Distributions
        distr_mv = (imps_mv > 0).sum(axis=1) / imps_mv.shape[1]
        distr_sv = (imps_sv > 0).sum(axis=1) / imps_sv.shape[1]

        # Plotting
        plt.title('level={}, gender={}'.format(level, gender))
        plt.scatter(distr_sv, distr_mv)
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--')

        # Decorations
        plt.xlim(0, 1)
        plt.xlabel('Impersonation Rate (before)')
        plt.ylim(0, 1)
        plt.ylabel('Impersonation Rate (after)')
        plt.grid()

    # Impersonation rates
    mv_eer_path = os.path.join('..', 'data', 'vs_mv_data', mv_set, 'mv', target_pop + '-' + net + '-' + policy + '-eer.npz')
    imps_mv_eer = np.load(mv_eer_path, allow_pickle=True)['results'][()]['imps']
    imps_sv_eer = np.load(mv_eer_path.replace('mv' + os.sep, 'sv' + os.sep), allow_pickle=True)['results'][()]['imps']
    imps_sv_far1 = np.load(mv_eer_path.replace('mv' + os.sep, 'sv' + os.sep).replace('eer', 'far1'), allow_pickle=True)['results'][()]['imps']
    imps_mv_far1 = np.load(mv_eer_path.replace('eer', 'far1'), allow_pickle=True)['results'][()]['imps']

    # Load gender indexes
    pop_data = pd.read_csv(os.path.join('..', 'data', 'vs_mv_pairs', target_pop + '.csv'))
    gnds = pop_data.drop_duplicates('user_id')['gender'].to_list()
    m_idx = [i for i, x in enumerate(gnds) if x == 'm']
    f_idx = [i for i, x in enumerate(gnds) if x == 'f']

    # Grid
    plt.figure(figsize=(30, 20))
    plt.suptitle('mv_set={}, target_pop={}, net={}'.format(mv_set, target_pop, net), fontweight='bold')

    # Female
    plt.subplot(2, 2, 1)
    get_subplot(imps_sv_eer[:, m_idx], imps_mv_eer[:, m_idx], 'eer', 'male')

    plt.subplot(2, 2, 2)
    get_subplot(imps_sv_eer[:, f_idx], imps_mv_eer[:, f_idx], 'eer', 'female')

    plt.subplot(2, 2, 3)
    get_subplot(imps_sv_far1[:, m_idx], imps_mv_far1[:, m_idx], 'far1', 'male')

    plt.subplot(2, 2, 4)
    get_subplot(imps_sv_far1[:, f_idx], imps_mv_far1[:, f_idx], 'far1', 'female')

    plt.tight_layout()
    plt.show()


def cross_asv_table(mv_sets, test_pop, policy, level, gender):
    print('test_pop={}, policy={}, level={}, gender={}'.format(test_pop, policy, level, gender))

    results = {}
    for mv_set in mv_sets:
        if '_' + gender[0] not in mv_set:
            continue

        base_path = os.path.join('..','data', 'vs_mv_data', mv_set, 'mv')
        source_asv = mv_set.split(os.sep)[0]

        # Find paths across asv
        paths = [os.path.join(base_path, p) for p in os.listdir(base_path) if test_pop in p and policy in p and level in p]
        paths.sort()

        # Loop across target asv
        results[source_asv] = {}
        for p in paths:
            target_asv = p.split('\\')[-1].split('-')[1]
            gnd_sv = np.load(p.replace('mv' + os.sep, 'sv' + os.sep), allow_pickle=True)['results'][()]['gnds']
            gnd_mv = np.load(p, allow_pickle=True)['results'][()]['gnds']
            gnd_sv_score = np.mean(gnd_sv[:, int(gender == 'female')])
            gnd_mv_score = np.mean(gnd_mv[:, int(gender == 'female')])
            results[source_asv][target_asv] = (np.round(gnd_sv_score, 2), np.round(gnd_mv_score, 2))

    return pd.DataFrame(results).transpose()


def v1(source_imps, max_pres):
    top_k_imp_rates = []

    for other in np.arange(len(source_imps)):
        top_k_sum_imps = np.sum(source_imps[np.array([other])], axis=0)
        top_k_imp_rates.append(np.sum(top_k_sum_imps > 0) / len(top_k_sum_imps))

    return np.argsort(-np.array(top_k_imp_rates))[:max_pres]


def v2(source_imps, max_pres):
    ordered_mv_list = []

    for _ in np.arange(max_pres):

        top_k_imp_rates = []
        for other in np.arange(len(source_imps)):
            top_k_sum_imps = np.sum(source_imps[np.array(ordered_mv_list + [other])], axis=0)
            top_k_imp_rates.append(np.sum(top_k_sum_imps > 0) / len(top_k_sum_imps))

        top_1 = np.argsort(-np.array(top_k_imp_rates))[0]
        ordered_mv_list.append(top_1)

    return ordered_mv_list


def multiple_presentation_table(source_pop, target_pop, mv_sets, net, policy, level, gender, max_pres=5):
    b = os.path.join('..', 'data', 'vs_mv_data')

    results = {}
    for mv_set in mv_sets:
        if '_' + gender[0] not in mv_set:
            continue

        results[mv_set] = []

        # Find best master voices in the train population
        path = os.path.join(b, mv_set, 'mv', source_pop + '-' + net + '-' + policy + '-' + level + '.npz')
        source_imps = np.load(path, allow_pickle=True)['results'][()]['imps']

        # Find user indexes for males and females
        pop_data = pd.read_csv(os.path.join('..', 'data', 'vs_mv_pairs', source_pop + '.csv'))
        gnds = pop_data.drop_duplicates('user_id')['gender'].to_list()
        g_idx = [i for i, x in enumerate(gnds) if not x == gender[0]]
        ng_idx = [i for i, x in enumerate(gnds) if x == gender[0]]

        source_imps[:, np.array(g_idx)] = 0

        # Find top k
        ordered_mv_list_v1 = v1(source_imps, max_pres)
        ordered_mv_list_v2 = v2(source_imps, max_pres)

        for pres in np.arange(1, max_pres + 1):
            # Retrieve impersonation rates
            d = os.path.join(b, mv_set, 'mv', target_pop + '-' + net + '-' + policy + '-' + level + '.npz')
            imps = np.load(d, allow_pickle=True)['results'][()]['imps']
            imps[:, np.array(g_idx)] = 0

            # Random k presentation attack
            top_k = ordered_mv_list_v1[:pres]
            top_k_sum_imps = np.sum(imps[top_k], axis=0)
            ran_k_imp_rate = np.sum(top_k_sum_imps[np.array(ng_idx)] > 0) / len(top_k_sum_imps[np.array(ng_idx)])

            # Top k presentation attack
            top_k = ordered_mv_list_v2[:pres]
            top_k_sum_imps = np.sum(imps[top_k], axis=0)
            top_k_imp_rate = np.sum(top_k_sum_imps[np.array(ng_idx)] > 0) / len(top_k_sum_imps[np.array(ng_idx)])

            # Pair reference and top
            results[mv_set].append((np.round(ran_k_imp_rate, 2), np.round(top_k_imp_rate, 2)))

    df = pd.DataFrame(results).transpose()
    df.columns = ['#Pres = {}'.format(i+1) for i in df.columns]

    return df

def hist(samples, bins=20, labels='', xlabel=None, guides=0, axes=None, alpha=0.4, scale=False, colors=None, guide=None, kde=False):
    if axes is None:
        fig = get_figure()
        axes = fig.gca()

    if isinstance(samples, np.ndarray) and utils.is_vector(samples):
        samples = [samples]
        labels = [labels]
        
    if utils.is_number(bins):
        s_min = np.min([np.min(s) for s in samples])
        s_max = np.max([np.max(s) for s in samples])
        
        if s_min == s_max:
            delta = 10 ** (np.log10(s_max) - 1)
            s_min -= delta
            s_max += delta
            
        cc = np.linspace(s_min, s_max, bins)
        
    else:
        cc = bins
        s_min = np.min(cc)
        s_max = np.max(cc)
        
    if len(np.unique(cc)) != len(cc):
        logger.warning(f'Duplicated histogram bins detected: {len(cc)} bins in [{s_min:.2f}, {s_min:.2f}]')        
        
    h_bins = utils.bin_edges(cc)

    h_max_global = 0

    for si, spls in enumerate(samples):

        p50 = np.percentile(spls, 50)
        p99 = np.percentile(spls, 99)
        p01 = np.percentile(spls, 1)

        if labels is not None:
            label = labels[si]
            label = label.replace('()', f'{p50:.2f}')
        else:
            label = None

        color = colors[si] if colors is not None else None

        h1 = axes.hist(spls.ravel(), h_bins, color=color, alpha=alpha, density=True, label=label)

        h_max = 1.05 * np.max(h1[0])
        if h_max > h_max_global:
            h_max_global = h_max
        color = h1[-1][0].get_facecolor()

        if kde:
            try:
                e_color = (color[0] * 0.75, color[1] * 0.75, color[2] * 0.75, 0.7)
                d_bins = np.linspace(h_bins[0], h_bins[-1], 200)
                kde_pos = sps.gaussian_kde(spls.ravel())
                axes.plot(d_bins, kde_pos.pdf(d_bins), color=e_color, linewidth=2)
            except np.linalg.LinAlgError as e:
                logger.warning(f'Plotting error (KDE): {e}')

        if guides & 1:
            axes.plot([p01, p01], [0, h_max], ':', color=color)

        if guides & 2:
            axes.plot([p50, p50], [0, h_max], ':', color=color)

        if guides & 4:
            axes.plot([p99, p99], [0, h_max], ':', color=color)

    axes.set_yticks([])

    if labels is not None:
        axes.legend()

    if scale:
        margin = (cc[-1] - cc[0]) / 100
        axes.set_xlim([cc[0] - margin, cc[-1] + margin])
        axes.set_ylim([0, h_max_global * 1.01])

    if guide is not None:
        axes.plot([guide * 1.01, guide * 1.01], [0, h_max_global], 'k:', alpha=0.5)

    if xlabel is not None:
        axes.set_xlabel(xlabel)

    if 'fig' in locals():
        return fig