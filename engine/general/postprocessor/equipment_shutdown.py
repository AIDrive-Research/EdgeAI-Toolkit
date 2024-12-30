import cv2
import numpy as np

from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from window.ratio_window import RatioWindow
from .utils import json_utils
from .utils.image_utils import base64_to_opencv


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.threshold = None
        self.length = None
        self.area_th = None
        self.diff_th = None
        self.targets = {}
        self.check_interval = 1

    def _process(self, result, filter_result):
        hit = False
        if self.diff_th is None:
            self.diff_th = self.reserved_args['diff']
        if self.area_th is None:
            self.area_th = self.reserved_args['area']
        if self.length is None:
            self.length = self.reserved_args['length']
        if self.threshold is None:
            if self.length != 0:
                self.threshold = self.reserved_args['threshold'] / self.length
                self.threshold = 0 if self.threshold < 0 else self.threshold
                self.threshold = 1 if self.threshold > 1 else self.threshold
            else:
                self.threshold = 1
            LOGGER.info('source_id={}, alg_name={}, length={}, threshold={}'.format(
                self.source_id, self.alg_name, self.length, self.threshold))
        polygons = self._gen_polygons()
        for polygon in polygons.values():
            model_name, infer_image = next(iter(filter_result.items()))
            infer_image = base64_to_opencv(infer_image)
            gray_image = cv2.cvtColor(infer_image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
            polygon_str = json_utils.dumps(polygon['polygon'])
            target = self.targets.get(polygon_str)
            if not target:
                mask = np.zeros((infer_image.shape[0], infer_image.shape[1]), dtype=np.uint8)
                mask = cv2.fillPoly(
                    mask, [(np.array(polygon['polygon']) // self.scale).astype(np.int32).reshape((-1, 1, 2))], 255)
                self.targets[polygon_str] = {
                    'window': RatioWindow(self.length, self.threshold),
                    'hit': False,
                    'pre_target': gray_image,
                    'time': self.time,
                    'mask': mask,
                    'area': len(np.column_stack(np.where(mask > 0)))
                }
                continue
            if self.time - target['time'] <= self.check_interval:
                continue
            pre_gray_image = target['pre_target']
            gray_image = cv2.bitwise_and(gray_image, target['mask'])
            pre_gray_image = cv2.bitwise_and(pre_gray_image, target['mask'])
            delta = cv2.absdiff(pre_gray_image, gray_image)
            binary_image = cv2.threshold(delta, self.diff_th, 255, cv2.THRESH_BINARY)[1]
            area = len(np.column_stack(np.where(binary_image > 0)))
            if area / target['area'] < self.area_th:
                target['hit'] = True
            else:
                target['hit'] = False
            target['time'] = self.time
            target['pre_target'] = gray_image
            if target['window'].insert({'time': self.time, 'data': {'hit': target['hit']}}):
                hit = True
                polygon['color'] = self.alert_color
        result['hit'] = hit
        result['data']['bbox']['polygons'].update(polygons)
        return True

    def _filter(self, model_name, model_data):
        return model_data['engine_result']
