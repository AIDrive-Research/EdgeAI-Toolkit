import os
import platform
import sys

import numpy as np
from turbojpeg import TurboJPEG, TJFLAG_PROGRESSIVE

from logger import LOGGER

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)

arch = platform.machine().lower()
if 'arm' in arch or 'aarch64' in arch:
    LOGGER.info('Detect arm architecture')
    turbo_jpeg = TurboJPEG(lib_path=os.path.abspath(os.path.join(CURRENT_PATH, 'libturbojpeg_arm.so')))
elif 'x86_64' in arch or 'amd64' in arch or 'i386' in arch or 'i686' in arch:
    LOGGER.info('Detect x86 architecture')
    turbo_jpeg = TurboJPEG(lib_path=os.path.abspath(os.path.join(CURRENT_PATH, 'libturbojpeg_x86.so')))
else:
    LOGGER.error('Unknown architecture')
    turbo_jpeg = None


def bytes_to_mat(image: bytes, **kwargs):
    """
    bytes转mat
    Args:
        image: bytes
    Returns: mat or None
    """
    try:
        if turbo_jpeg is not None:
            return turbo_jpeg.decode(image)
        else:
            LOGGER.error('TurboJPEG is not initialized')
    except:
        LOGGER.exception('bytes_to_mat')
    return None


def mat_to_bytes(image: np.ndarray, **kwargs):
    """
    mat转bytes
    Args:
        image: mat
    Returns: bytes or None
    """
    try:
        if turbo_jpeg is not None:
            quality = kwargs.get('quality', 85)
            flags = kwargs.get('flags', TJFLAG_PROGRESSIVE)
            return turbo_jpeg.encode(image, quality=quality, flags=flags)
        else:
            LOGGER.error('TurboJPEG is not initialized')
    except:
        LOGGER.exception('mat_to_bytes')
    return None
