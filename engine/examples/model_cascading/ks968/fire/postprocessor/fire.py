from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from .utils import json_utils
from .utils.cv_utils.color_utils import rgb_reverse
from .utils.cv_utils.crop_utils import crop_rectangle
from .utils.cv_utils.geo_utils import calc_iou
from .utils.image_utils import base64_to_opencv, opencv_to_base64


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.det_model_name = 'zql_fire'
        self.cls_model_name = 'zql_fire_classify'
        self.iou = None
        self.timeout = None
        self.pre_n = 3
        self.pre_targets = []
        self.reinfer_result = {}
        self.fire_label = 0

    def __reinfer(self, filter_result):
        fire_rectangles = filter_result.get(self.det_model_name)
        if fire_rectangles is None:
            LOGGER.error('Fire model result is None!')
            return False
        fire_rectangles = sorted(fire_rectangles, key=lambda x: x['conf'], reverse=True)
        draw_image = base64_to_opencv(self.draw_image)
        count = 0
        for fire_rectangle in fire_rectangles:
            xyxy = fire_rectangle['xyxy']
            cropped_image = crop_rectangle(draw_image, xyxy)
            cropped_image = rgb_reverse(cropped_image)
            source_data = {
                'source_id': self.source_id,
                'time': self.time * 1000000,
                'infer_image': opencv_to_base64(cropped_image),
                'draw_image': None,
                'reserved_data': {
                    'specified_model': [self.cls_model_name],
                    'rectangle': fire_rectangle,
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
        if self.iou is None:
            self.iou = self.reserved_args['iou']
        if self.timeout is None:
            self.timeout = (self.frame_interval / 1000) * 2
            LOGGER.info('source_id={}, alg_name={}, timeout={}'.format(self.source_id, self.alg_name, self.timeout))
        polygons = self._gen_polygons()
        if not self.reserved_data:
            count = self.__reinfer(filter_result)
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
        self.reinfer_result[self.time]['result'].append((rectangles, self.reserved_data['rectangle']))
        if len(self.reinfer_result[self.time]['result']) < self.reinfer_result[self.time]['count']:
            return False
        reinfer_result_ = self.reinfer_result.pop(self.time)
        self.draw_image = reinfer_result_['draw_image']
        fire_rectangles = []
        for targets, rectangle in reinfer_result_['result']:
            if not targets:
                continue
            diff_num = 0
            for pre_targets_ in self.pre_targets:
                max_iou = 0
                for pre_target in pre_targets_:
                    iou = calc_iou(pre_target['xyxy'], rectangle['xyxy'])
                    max_iou = max(max_iou, iou)
                    if max_iou > 0 and max_iou <= self.iou:
                        diff_num += 1
                        break
            if diff_num and len(self.pre_targets) == self.pre_n:
                hit = True
                rectangle['color'] = self.alert_color
            fire_rectangles.append(rectangle)
        if len(fire_rectangles):
            self.pre_targets.append(fire_rectangles)
        if len(self.pre_targets) > self.pre_n:
            self.pre_targets.pop(0)
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(fire_rectangles)
        result['data']['bbox']['polygons'].update(polygons)
        return True

    def _filter(self, model_name, model_data):
        targets = []
        if model_name == self.cls_model_name and not self.reserved_data:
            return targets
        model_conf = model_data['model_conf']
        engine_result = model_data['engine_result']
        if model_name == self.det_model_name:
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
                targets.append(self._gen_rectangle(
                    xyxy, self._get_color(model_conf['label'], label), label, engine_result_['conf']))
        elif model_name == self.cls_model_name:
            score = engine_result['output'][self.fire_label]
            if score >= model_conf['args']['conf_thres']:
                targets.append(engine_result)
        return targets
