import numpy as np

from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from .utils import json_utils
from .utils.cv_utils.color_utils import rgb_reverse
from .utils.cv_utils.crop_utils import crop_rectangle
from .utils.image_utils import base64_to_opencv, opencv_to_base64


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.cls_model_name = 'zql_fog_classify'
        self.timeout = None
        self.reinfer_result = {}
        self.fog_label = [0,2]

    @staticmethod
    def __get_polygons_box(polygons):
        points = []
        for id_, info in polygons.items():
            points.extend(info['polygon'])
        points = np.array(points)
        min_x = np.min(points[:, 0])
        min_y = np.min(points[:, 1])
        max_x = np.max(points[:, 0])
        max_y = np.max(points[:, 1])
        return [min_x, min_y, max_x, max_y]

    def __reinfer(self, polygons):
        count = 0
        draw_image = base64_to_opencv(self.draw_image)
        if polygons:
            roi = self.__get_polygons_box(polygons)
            cropped_image = crop_rectangle(draw_image, roi)
        else:
            cropped_image = draw_image
        if cropped_image is not None:
            cropped_image = rgb_reverse(cropped_image)
            source_data = {
                'source_id': self.source_id,
                'time': self.time * 1000000,
                'infer_image': opencv_to_base64(cropped_image),
                'draw_image': None,
                'reserved_data': {
                    'specified_model': [self.cls_model_name],
                    'unsort': True
                }
            }
            self.rq_source.put(json_utils.dumps(source_data))
            count += 1
        if count > 0:
            self.reinfer_result[self.time] = {
                'count': count,
                'draw_image': self.draw_image,
                'result': []
            }
        return count

    def __check_expire(self):
        for time in list(self.reinfer_result.keys()):
            if time < self.time - self.timeout:
                LOGGER.warning('Reinfer result expired, source_id={}, alg_name={}, time={}, timeout={}'.format(
                    self.source_id, self.alg_name, time, self.timeout))
                del self.reinfer_result[time]
        return True

    def _process(self, result, filter_result):
        hit = False
        if self.timeout is None:
            self.timeout = (self.frame_interval / 1000) * 2
            LOGGER.info('source_id={}, alg_name={}, timeout={}'.format(self.source_id, self.alg_name, self.timeout))
        polygons = self._gen_polygons()
        if not self.reserved_data:
            count = self.__reinfer(polygons)
            if not count:
                self.__check_expire()
                result['hit'] = False
                result['data']['bbox']['polygons'].update(polygons)
                return True
            return False
        self.__check_expire()
        model_name, rectangles = next(iter(filter_result.items()))
        if model_name != self.cls_model_name:
            LOGGER.error('Get wrong model result, expect {}, but get {}'.format(self.cls_model_name, model_name))
            return False
        if self.reinfer_result.get(self.time) is None:
            LOGGER.warning('Not found reinfer result, time={}'.format(self.time))
            return False
        self.reinfer_result[self.time]['result'].append(rectangles)
        if len(self.reinfer_result[self.time]['result']) < self.reinfer_result[self.time]['count']:
            return False
        reinfer_result_ = self.reinfer_result.pop(self.time)
        self.draw_image = reinfer_result_['draw_image']
        for targets in reinfer_result_['result']:
            if not targets:
                continue
            hit = True
        result['hit'] = hit
        result['data']['bbox']['polygons'].update(polygons)
        return result

    def _filter(self, model_name, model_data):
        targets = []
        model_conf = model_data['model_conf']
        engine_result = model_data['engine_result']
        if engine_result:
            score = np.max(engine_result['output'])
            label = np.argmax(engine_result['output'])
            if score >= model_conf['args']['conf_thres'] and label in self.fog_label:
                targets.append(engine_result)
        return targets
