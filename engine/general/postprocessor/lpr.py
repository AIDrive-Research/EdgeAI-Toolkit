import numpy as np

from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from .utils import json_utils
from .utils.cv_utils.color_utils import rgb_reverse
from .utils.image_utils import base64_to_opencv, opencv_to_base64


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.model_name = 'lpr'
        self.timeout = None
        self.reinfer_result = {}

    def __reinfer(self):
        draw_image = base64_to_opencv(self.draw_image)
        draw_image = rgb_reverse(draw_image)
        source_data = {
            'source_id': self.source_id,
            'time': self.time * 1000000,
            'infer_image': opencv_to_base64(draw_image),
            'draw_image': None,
            'reserved_data': {
                'specified_model': [self.model_name],
                'unsort': True
            }
        }
        self.rq_source.put(json_utils.dumps(source_data))
        self.reinfer_result[self.time] = {
            'count': 1,
            'draw_image': self.draw_image,
            'result': []
        }
        return True

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
        if not self.reserved_data:
            self.__reinfer()
            return False
        self.__check_expire()
        polygons = self._gen_polygons()
        model_name, targets = next(iter(filter_result.items()))
        if self.reinfer_result.get(self.time) is None:
            LOGGER.warning('Not found reinfer result, time={}'.format(self.time))
            return False
        self.reinfer_result[self.time]['result'].append(targets)
        if len(self.reinfer_result[self.time]['result']) < self.reinfer_result[self.time]['count']:
            return False
        reinfer_result_ = self.reinfer_result.pop(self.time)
        self.draw_image = reinfer_result_['draw_image']
        for targets in reinfer_result_['result']:
            for lpr_rectangle in targets:
                if lpr_rectangle['ext']['plate_type'] in ['蓝牌', '黄牌'] and len(
                        lpr_rectangle['ext']['plate_code']) != 7:
                    continue
                if lpr_rectangle['ext']['plate_type'] == '绿牌' and len(lpr_rectangle['ext']['plate_code']) != 8:
                    continue
                hit = True
                label = '{} {}'.format(
                    self._get_ext(lpr_rectangle, 'plate_code'), self._get_ext(lpr_rectangle, 'plate_type'))
                lpr_rectangle['label'] = label
                lpr_rectangle['color'] = self.non_alert_color
                result['data']['bbox']['rectangles'].append(lpr_rectangle)
        result['hit'] = hit
        result['data']['bbox']['polygons'].update(polygons)
        return result

    def _filter(self, model_name, model_data):
        targets = []
        if not self.reserved_data:
            return targets
        model_conf = model_data['model_conf']
        engine_result = model_data['engine_result']
        for engine_result_ in engine_result:
            # 过滤掉置信度低于阈值的目标
            rec_conf = engine_result_['rec_conf']
            min_conf = np.min(np.array(rec_conf))
            if not self._filter_by_conf(model_conf, min_conf):
                continue
            # 坐标缩放
            xyxy = engine_result_['xyxy']
            # 过滤掉不在多边形内的目标
            if not self._filter_by_roi(xyxy):
                continue
            # 生成矩形框
            targets.append(self._gen_rectangle(
                xyxy, self.non_alert_color, None, engine_result_['det_conf'],
                plate_code=engine_result_['plate_code'], plate_type=engine_result_['plate_type']))
        return targets
