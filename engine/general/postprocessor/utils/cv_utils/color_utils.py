import random

import cv2
import numpy as np

from logger import LOGGER


def gen_random_rgb(seed=1):
    """
    生成随机RGB颜色
    Args:
        seed: 随机种子
    Returns: 随机RGB颜色
    """
    try:
        random.seed(seed)
        color = tuple([random.randint(0, 255) for _ in range(3)])
        return color
    except:
        LOGGER.exception('gen_random_rgb')
    return None


def rgb_reverse(image):
    """
    3通道图像反转，将RGB格式的图片转换为BGR格式，或者将BGR格式的图片转换为RGB格式
    Args:
        image: 图片，numpy格式，3通道
    Returns: 反转后的图片
    """
    try:
        return image[..., ::-1]
    except:
        LOGGER.exception('rgb_reverse')
    return None


def rgba_reverse(image):
    """
    4通道图像反转，将RGBA格式的图片转换为BGRA格式，或者将BGRA格式的图片转换为RGBA格式
    Args:
        image: 图片，numpy格式，4通道
    Returns: 反转后的图片
    """
    try:
        image_reversed = np.zeros_like(image)
        # 反转RGB
        image_reversed[..., :3] = image[..., :3][..., ::-1]
        # 保持Alpha不变
        image_reversed[..., 3] = image[..., 3]
        return image_reversed
    except:
        LOGGER.exception('rgba_reverse')
    return None


def rgb2gray(image, channel='bgr'):
    """
    将RGB格式的图片转换为灰度图
    Args:
        image: 图片，numpy格式，3通道
        channel: 图片通道，bgr或者rgb
    Returns: 灰度图
    """
    try:
        if channel == 'bgr':
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channel == 'rgb':
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            LOGGER.error('channel error')
    except:
        LOGGER.exception('rgb2gray')
    return None
