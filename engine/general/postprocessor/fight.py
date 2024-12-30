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
        self.fight_model_name = 'fight'
        self.person_model_name = 'person'
        self.timeout = None
        self.reinfer_result = {}
        self.image_list = []
        self.batch_size = 8
        self.fight_label = 1

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

    def __reinfer(self, polygons, filter_result):
        person_rectangles = filter_result.get(self.person_model_name)
        if person_rectangles is None:
            LOGGER.error('Person model result is None!')
            return False
        count = 0
        if len(person_rectangles) < 2:
            return count
        draw_image = base64_to_opencv(self.draw_image)
        if polygons:
            roi = self.__get_polygons_box(polygons)
            cropped_image = crop_rectangle(draw_image, roi)
        else:
            cropped_image = draw_image
        if cropped_image is not None:
            cropped_image = rgb_reverse(cropped_image)
            self.image_list.append(cropped_image)
            if len(self.image_list) > self.batch_size:
                self.image_list.pop(0)
            if len(self.image_list) == self.batch_size:
                source_data = {
                    'source_id': self.source_id,
                    'time': self.time * 1000000,
                    'infer_image': [opencv_to_base64(image) for image in self.image_list],
                    'draw_image': None,
                    'reserved_data': {
                        'specified_model': [self.fight_model_name],
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
            count = self.__reinfer(polygons, filter_result)
            if not count:
                self.__check_expire()
                result['hit'] = False
                result['data']['bbox']['polygons'].update(polygons)
                return True
            return False
        self.__check_expire()
        model_name, rectangles = next(iter(filter_result.items()))
        if model_name != self.fight_model_name:
            LOGGER.error('Get wrong model result, expect {}, but get {}'.format(self.fight_model_name, model_name))
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
        if model_name == self.fight_model_name and not self.reserved_data:
            return targets
        model_conf = model_data['model_conf']
        engine_result = model_data['engine_result']
        if model_name == self.person_model_name:
            for engine_result_ in engine_result:
                # 过滤掉置信度低于阈值的目标
                if not self._filter_by_conf(model_conf, engine_result_['conf']):
                    continue
                # 过滤掉不在label列表中的目标
                label = self._filter_by_label(model_conf, engine_result_['label'])
                if not label:
                    continue
                # 坐标缩放
                xyxy = self._scale(engine_result_['xyxy'])
                # 过滤掉不在多边形内的目标
                if not self._filter_by_roi(xyxy):
                    continue
                # 生成矩形框
                targets.append(self._gen_rectangle(xyxy, self.non_alert_color, label, engine_result_['conf']))
        elif model_name == self.fight_model_name:
            if engine_result:
                score = engine_result['output'][self.fight_label]
                if score >= model_conf['args']['conf_thres']:
                    targets.append(engine_result)
        return targets
