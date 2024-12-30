import cv2
import numpy as np

from postprocessor import Postprocessor as BasePostprocessor
from .utils import json_utils
from .utils.image_utils import base64_to_opencv


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.threshold = None
        self.length = None
        self.diff_th = None
        self.area_th = None
        self.targets = {}
        self.check_interval = 1

    def __check_motion(self, rectangles, pre_gray, cur_gray):
        hit = False
        delta = cv2.absdiff(pre_gray, cur_gray)
        binary_image = cv2.threshold(delta, self.diff_th, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= self.area_th:
                hit = True
                x, y, w, h = [int(x * self.scale) for x in cv2.boundingRect(contour)]
                rectangles.append(self._gen_rectangle([x, y, x + w, y + h], self.alert_color, '运动目标', 1))
        return hit, rectangles

    def _process(self, result, filter_result):
        hit = False
        if self.diff_th is None:
            self.diff_th = self.reserved_args['diff']
        if self.area_th is None:
            self.area_th = self.reserved_args['area']
        polygons = self._gen_polygons()
        rectangles = []
        model_name, infer_image = next(iter(filter_result.items()))
        infer_image = base64_to_opencv(infer_image)
        if not polygons:
            gray_image = cv2.cvtColor(infer_image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.GaussianBlur(gray_image, (21, 21), 0)
            if not self.targets:
                self.targets = {
                    'pre_target': gray_image,
                    'time': self.time
                }
            pre_gray_image = self.targets['pre_target']
            hit, rectangles = self.__check_motion(rectangles, pre_gray_image, gray_image)
            if self.time - self.targets['time'] > self.check_interval:
                self.targets['time'] = self.time
                self.targets['pre_target'] = gray_image
        for polygon in polygons.values():
            # gray_image如果放在外面，会影响告警
            gray_image = cv2.cvtColor(infer_image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.GaussianBlur(gray_image, (21, 21), 0)
            polygon_str = json_utils.dumps(polygon['polygon'])
            target = self.targets.get(polygon_str)
            if not target:
                mask = np.zeros((infer_image.shape[0], infer_image.shape[1]), dtype=np.uint8)
                mask = cv2.fillPoly(
                    mask, [(np.array(polygon['polygon']) // self.scale).astype(np.int32).reshape((-1, 1, 2))], 255)
                self.targets[polygon_str] = {
                    'pre_target': gray_image,
                    'time': self.time,
                    'mask': mask
                }
                continue
            gray_image = cv2.bitwise_and(gray_image, target['mask'])
            pre_gray_image = target['pre_target']
            pre_gray_image = cv2.bitwise_and(pre_gray_image, target['mask'])
            hit, rectangle = self.__check_motion(rectangles, pre_gray_image, gray_image)
            rectangles.extend(rectangle)
            if hit:
                polygon['color'] = self.alert_color
            if self.time - target['time'] > self.check_interval:
                target['time'] = self.time
                target['pre_target'] = gray_image
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(rectangles)
        result['data']['bbox']['polygons'].update(polygons)
        return True

    def _filter(self, model_name, model_data):
        return model_data['engine_result']
