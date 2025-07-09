import cv2
import numpy as np

from postprocessor import Postprocessor as BasePostprocessor
from .utils import json_utils
from .utils.cv_utils.crop_utils import crop_rectangle
from .utils.image_utils.turbojpegutils import bytes_to_mat


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.threshold = None
        self.roi = {}

    @staticmethod
    def __get_polygons_box(polygon):
        points = polygon['polygon']
        points = np.array(points)
        min_x = np.min(points[:, 0])
        min_y = np.min(points[:, 1])
        max_x = np.max(points[:, 0])
        max_y = np.max(points[:, 1])
        return [min_x, min_y, max_x, max_y]

    @staticmethod
    def __calculate_laplacian_variance(image):
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        variance = laplacian.var()
        return variance

    def _process(self, result, filter_result):
        hit = False
        if self.threshold is None:
            self.threshold = self.reserved_args['threshold']
        polygons = self._gen_polygons()
        _, infer_image = next(iter(filter_result.items()))
        infer_image = bytes_to_mat(infer_image)
        infer_image = cv2.cvtColor(infer_image, cv2.COLOR_BGR2GRAY)
        for polygon in polygons.values():
            polygon_key = json_utils.dumps(polygon['polygon'])
            roi = self.roi.get(polygon_key)
            if roi is None:
                roi = self.__get_polygons_box(polygon)
                self.roi[polygon_key] = roi
            cropped_image = crop_rectangle(infer_image, roi)
            variance = self.__calculate_laplacian_variance(cropped_image)
            if variance <= self.threshold:
                hit = True
                polygon['color'] = self.alert_color
        else:
            variance = self.__calculate_laplacian_variance(infer_image)
            if variance <= self.threshold:
                hit = True
        result['hit'] = hit
        result['data']['bbox']['polygons'].update(polygons)
        return True

    def _filter(self, model_name, model_data):
        return model_data['engine_result']
