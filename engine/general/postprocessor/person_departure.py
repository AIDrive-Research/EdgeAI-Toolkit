import time

from postprocessor import Postprocessor as BasePostprocessor
from .utils import json_utils
from .utils.time_utils import sec2hms


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.strategy = None
        self.threshold = None
        self.duration = {}

    def _process(self, result, filter_result):
        hit = False
        if self.strategy is None:
            self.strategy = self.reserved_args['strategy']
        if self.threshold is None:
            self.threshold = self.reserved_args['threshold']
        polygons = self._gen_polygons()
        model_name, rectangles = next(iter(filter_result.items()))
        for polygon in polygons.values():
            polygon_str = json_utils.dumps(polygon['polygon'])
            if polygon_str not in self.duration:
                self.duration[polygon_str] = {
                    'time': int(time.time()),
                    'duration': 0
                }
            count = 0
            for rectangle in rectangles:
                self._set_ext(rectangle, in_polygon=False)
                if self._is_rectangle_in_polygon(rectangle['xyxy'], polygon['polygon'], self.strategy):
                    self._set_ext(rectangle, in_polygon=True)
                    count += 1
                    rectangle['color'] = self.alert_color
            if count <= self.threshold:
                hit = True
                polygon['color'] = self.alert_color
                self._set_ext(polygon, result=count)
                self.duration[polygon_str]['duration'] = int(time.time()) - self.duration[polygon_str]['time']
                if polygon['name']:
                    polygon['name'] = '{}-{}'.format(polygon['name'], sec2hms(self.duration[polygon_str]['duration']))
                else:
                    polygon['name'] = '{}'.format(sec2hms(self.duration[polygon_str]['duration']))
            else:
                for rectangle in rectangles:
                    if self._get_ext(rectangle, 'in_polygon'):
                        rectangle['color'] = self.non_alert_color
                self.duration[polygon_str]['duration'] = 0
                self.duration[polygon_str]['time'] = int(time.time())
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(rectangles)
        result['data']['bbox']['polygons'].update(polygons)
        return True

    def _filter(self, model_name, model_data):
        targets = []
        model_conf = model_data['model_conf']
        engine_result = model_data['engine_result']
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
            # 生成矩形框
            targets.append(self._gen_rectangle(
                xyxy, self._get_color(model_conf['label'], label), label, engine_result_['conf']))
        return targets