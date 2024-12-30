import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from logger import LOGGER


def bytes_to_base64(data: bytes):
    """
    bytes转base64 string
    Args:
        data: bytes
    Returns: base64 string
    """
    return base64.b64encode(data).decode('utf-8')


def base64_to_bytes(data: str):
    """
    base64 string转bytes
    Args:
        data: base64 string
    Returns: bytes
    """
    return base64.b64decode(data.encode('utf-8'))


def write_base64_image(image: str, image_path):
    """
    base64 string图像数据保存至文件
    Args:
        image: base64 string
        image_path: 图像保存路径
    Returns: True or False
    """
    try:
        with open(image_path, 'wb') as f:
            f.write(base64_to_bytes(image))
    except:
        LOGGER.exception('write_base64_image')
        return False
    return True


#########################################################opencv#########################################################
def bytes_to_opencv(image: bytes):
    """
    bytes转opencv
    Args:
        image: bytes
    Returns: opencv ndarray or None
    """
    try:
        return cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    except:
        LOGGER.exception('bytes_to_opencv')
        return None


def base64_to_opencv(image: str):
    """
    base64 string转opencv
    Args:
        image: base64 string
    Returns: opencv ndarray or None
    """
    try:
        return cv2.imdecode(np.frombuffer(base64_to_bytes(image), np.uint8), cv2.IMREAD_COLOR)
    except:
        LOGGER.exception('base64_to_opencv')
        return None


def opencv_to_bytes(image: np.ndarray):
    """
    opencv转bytes
    Args:
        image: opencv ndarray
    Returns: bytes or None
    """
    try:
        return cv2.imencode('.jpg', image)[1].tobytes()
    except:
        LOGGER.exception('opencv_to_bytes')
        return None


def opencv_to_base64(image: np.ndarray):
    """
    opencv转base64 string
    Args:
        image: opencv ndarray
    Returns: base64 string or None
    """
    try:
        return bytes_to_base64(cv2.imencode('.jpg', image)[1].tobytes())
    except:
        LOGGER.exception('opencv_to_base64')
        return None


#########################################################Pillow#########################################################
def bytes_to_pil(image: bytes):
    """
    bytes转PIL Image
    Args:
        image: bytes
    Returns: PIL Image or None
    """
    try:
        return Image.open(BytesIO(image))
    except:
        LOGGER.exception('bytes_to_pil')
        return None


def opencv_to_pil(image: np.ndarray):
    """
    opencv转PIL Image
    Args:
        image: opencv ndarray
    Returns: PIL Image or None
    """
    try:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except:
        LOGGER.exception('opencv_to_pil')
        return None


def base64_to_pil(image: str):
    """
    base64 string转PIL Image
    Args:
        image: base64 string
    Returns: PIL Image or None
    """
    try:
        return bytes_to_pil(base64_to_bytes(image))
    except:
        LOGGER.exception('base64_to_pil')
        return None


def pil_to_bytes(image: Image):
    """
    PIL Image转bytes
    Args:
        image: PIL Image
    Returns: bytes or None
    """
    try:
        image_bytes = BytesIO()
        image.save(image_bytes, format='JPEG')
        return image_bytes.getvalue()
    except:
        LOGGER.exception('pil_to_bytes')
        return None


def pil_to_opencv(image: Image):
    """
    PIL Image转opencv
    Args:
        image: PIL Image
    Returns: opencv ndarray or None
    """
    try:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except:
        LOGGER.exception('pil_to_opencv')
        return None


def pil_to_base64(image: Image):
    """
    PIL Image转base64 string
    Args:
        image: PIL Image
    Returns: base64 string or None
    """
    try:
        return bytes_to_base64(pil_to_bytes(image))
    except:
        LOGGER.exception('pil_to_base64')
        return None
