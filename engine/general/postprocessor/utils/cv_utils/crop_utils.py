import cv2
import numpy as np

from logger import LOGGER


def crop_rectangle(image, xyxy, bbox=True, reverse=False):
    """
    裁剪矩形区域
    Args:
        image: image，numpy格式
        xyxy: 矩形的左上角和右下角坐标
        bbox: 是否裁剪外接矩形
        reverse: 是否反转mask
    Returns: 裁剪之后的图片
    """
    try:
        if bbox:
            image = image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        else:
            points = [[xyxy[0], xyxy[1]], [xyxy[2], xyxy[1]], [xyxy[2], xyxy[3]], [xyxy[0], xyxy[3]]]
            np_points = np.array(points, dtype=np.int32)
            if not reverse:
                white_mask = np.zeros(image.shape, dtype=np.uint8)
                cv2.fillConvexPoly(white_mask, np_points, (255, 255, 255))
                image = cv2.bitwise_and(image, white_mask)
            else:
                black_mask = np.ones(image.shape, dtype=np.uint8) * 255
                cv2.fillConvexPoly(black_mask, np_points, (0, 0, 0))
                image = cv2.bitwise_and(image, black_mask)
        return image
    except:
        LOGGER.exception('crop_rectangle')
    return None


def crop_poly(image, points, bbox=True, reverse=False):
    """
    裁剪多边形区域
    Args:
        image: image，numpy格式
        points: 多边形的顶点坐标
        bbox: 是否裁剪外接矩形
        reverse: 是否反转mask
    Returns: 裁剪之后的图片
    """
    try:
        np_points = np.array(points, dtype=np.int32)
        if not reverse:
            white_mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.fillPoly(white_mask, [np_points], (255, 255, 255))
            image = cv2.bitwise_and(image, white_mask)
        else:
            black_mask = np.ones(image.shape, dtype=np.uint8) * 255
            cv2.fillPoly(black_mask, [np_points], (0, 0, 0))
            image = cv2.bitwise_and(image, black_mask)
        if bbox:
            x, y, w, h = cv2.boundingRect(np_points)
            image = image[y:y + h, x:x + w]
        return image
    except:
        LOGGER.exception('crop_poly')
    return None
