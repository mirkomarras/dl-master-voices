import re
import os
import sys
from loguru import logger

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
