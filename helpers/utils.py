import re
import os
import sys
import numpy as np

from loguru import logger

_float_types = {float, np.float, np.float16, np.float32, np.float64}

_numeric_types = {int, float, bool, np.bool, np.float, np.float16, np.float32, np.float64,
                           np.int, np.int8, np.int32, np.int16, np.int64,
                           np.uint, np.uint8, np.uint32, np.uint16, np.uint64}

def setup_logging(filename=None, long_date=False, level='INFO'):
    """
    Configure the logger to a compact format.
    :param filename: add an additional sink to the given file
    :param long_date: flag to choose a full or compact date format
    """

    if long_date:
        # If needed, can add "<cyan>{name}</cyan>:" to show file which triggered the message
        log_format = '<green>{time:YYYY-MM-DD HH:mm:ss}</> | <lvl>{level:8s}</lvl> | <lvl>{message}</>'
    else:
        log_format = '<green>{time:HH:mm:ss}</> | <lvl>{level:8s}</> | <lvl>{message}</lvl>'

    config = {
        "handlers": [
            {"sink": sys.stderr, "format": log_format, "level": level, "colorize": True}
        ],
    }

    if filename is not None:
        if '/' not in filename:
            filename = os.path.join('logs', filename)
        config['handlers'].append({"sink": filename, "serialize": False, "format": log_format, "level": level})

    logger.configure(**config)


def sanitize_path(name, sub='_'):
    return re.sub(r'[ _~*!+#@":"!<>\[\]]+', sub, name).strip(sub)

def is_number(value):
    return type(value) in _numeric_types


def is_numeric_type(t):
    return t in _numeric_types


def is_nan(value):
    if value is None:
        return True

    if is_number(value):
        return np.isnan(value)

    return False


def is_vector(data):

    if isinstance(data, list) and all(is_number(x) for x in data):
        return True
    elif isinstance(data, np.ndarray) and (data.ndim == 1 or (data.ndim == 2 and 1 in data.shape)):
        return True
    else:
        return False

def bin_edges(code_book):
    max_float = np.max(code_book)
    min_float = np.min(code_book)
    code_book_edges = np.convolve(code_book, [0.5, 0.5], mode='valid')
    code_book_edges = np.concatenate((np.array([min_float]), code_book_edges, np.array([max_float])), axis=0)
    return code_book_edges