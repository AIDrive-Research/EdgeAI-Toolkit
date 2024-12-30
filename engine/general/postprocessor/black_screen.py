import cv2
import numpy as np

from postprocessor import Postprocessor as BasePostprocessor
from .utils import json_utils
from .utils.image_utils import base64_to_opencv


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.threshold = None
        self.mask = {}

    def _process(self, result, filter_result):
        hit = False
        if self.threshold is None:
            self.threshold = self.reserved_args['threshold']
        polygons = self._gen_polygons()
        model_name, infer_image = next(iter(filter_result.items()))
        infer_image = base64_to_opencv(infer_image)
        infer_image = cv2.cvtColor(infer_image, cv2.COLOR_BGR2GRAY)
        for polygon in polygons.values():
            polygon_key = json_utils.dumps(polygon['polygon'])
            mask = self.mask.get(polygon_key)
            if mask is None:
                mask = np.zeros(infer_image.shape[:2], dtype=np.uint8)
                np_points = np.array(polygon['polygon'], dtype=np.int32)
                np_points = (np_points // self.scale).astype(np.int32)
                cv2.fillPoly(mask, [np_points], 255)
                self.mask[polygon_key] = mask
            polygon_pixels = infer_image[np.where(255 == mask)]
            if int(np.mean(polygon_pixels, axis=0)) <= self.threshold:
                hit = True
                polygon['color'] = self.alert_color
        result['hit'] = hit
        result['data']['bbox']['polygons'].update(polygons)
        return True

    def _filter(self, model_name, model_data):
        return model_data['engine_result']
