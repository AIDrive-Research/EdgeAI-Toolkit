from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from .utils import msgpack_utils
from .utils.cv_utils.color_utils import rgb_reverse
from .utils.cv_utils.crop_utils import crop_rectangle
from .utils.image_utils.turbojpegutils import bytes_to_mat, mat_to_bytes


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.det_model_name = 'zql_fight'
        self.cls_model_name = 'zql_fight_classify'
        self.distance = 10
        self.timeout = None
        self.reinfer_result = {}
        self.fight_label = 0

    def __reinfer(self, filter_result):
        rectangles = filter_result.get(self.det_model_name)
        if rectangles is None:
            LOGGER.error('Fall down model result is None!')
            return False
        draw_image = bytes_to_mat(self.draw_image)
        image_shape = draw_image.shape
        count = 0
        normal_rectangles = []
        for rectangle in rectangles:
            if rectangle['label'] not in self.alert_label:
                normal_rectangles.append(rectangle)
                continue
            xyxy = rectangle['xyxy']
            cropped_image = crop_rectangle(draw_image, xyxy)
            cropped_image = rgb_reverse(cropped_image)
            if (xyxy[0] < self.distance) \
                    or (xyxy[3] > image_shape[0] - self.distance) \
                    or (xyxy[2] > image_shape[1] - self.distance):
                continue
            source_data = {
                'source_id': self.source_id,
                'time': self.time * 1000000,
                'infer_image': mat_to_bytes(cropped_image),
                'draw_image': None,
                'reserved_data': {
                    'specified_model': [self.cls_model_name],
                    'rectangle': rectangle,
                    'unsort': True
                }
            }
            self.rq_source.put(msgpack_utils.dump(source_data))
            count += 1
        if count > 0:
            self.reinfer_result[self.time] = {
                'count': count,
                'draw_image': self.draw_image,
                'normal_rectangles': normal_rectangles,
                'result': []
            }
        return count, normal_rectangles

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
            count, normal_rectangles = self.__reinfer(filter_result)
            if not count:
                self.__check_expire()
                result['hit'] = False
                result['data']['bbox']['rectangles'].extend(normal_rectangles)
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
        rectangles = reinfer_result_['normal_rectangles']
        for targets, rectangle in reinfer_result_['result']:
            if not targets:
                rectangle['color'] = self.non_alert_color
                rectangle['label'] = '正常'
                rectangles.append(rectangle)
                continue
            hit = True
            rectangle['color'] = self.alert_color
            rectangles.append(rectangle)
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(rectangles)
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
            score = engine_result['output'][self.fight_label]
            if score >= model_conf['args']['conf_thres']:
                targets.append(engine_result)
        return targets
