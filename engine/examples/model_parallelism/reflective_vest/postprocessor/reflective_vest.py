from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from .utils.cv_utils.geo_utils import is_point_in_rectangle


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.reflective_vest_model_name = 'reflective_vest'
        self.person_model_name = 'person'
        self.alert_label = '未穿戴反光衣'

    def _process(self, result, filter_result):
        hit = False
        polygons = self._gen_polygons()
        reflective_vest_rectangles = filter_result.get(self.reflective_vest_model_name)
        if reflective_vest_rectangles is None:
            LOGGER.error('Reflective vest model result is None!')
            return False
        person_rectangles = filter_result.get(self.person_model_name)
        if person_rectangles is None:
            LOGGER.error('Person model result is None!')
            return False
        for person_rectangle in person_rectangles:
            for reflective_vest_rectangle in reflective_vest_rectangles:
                point = self._get_point(reflective_vest_rectangle['xyxy'], 'center')
                if is_point_in_rectangle(point, person_rectangle['xyxy']):
                    break
            else:
                if polygons:
                    for polygon in polygons.values():
                        if self._is_rectangle_in_polygon(person_rectangle['xyxy'], polygon['polygon'], 'center'):
                            break
                    else:
                        continue
                hit = True
                person_rectangle['color'] = self.alert_color
                person_rectangle['label'] = self.alert_label
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(person_rectangles)
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
            targets.append(self._gen_rectangle(xyxy, self.non_alert_color, label, engine_result_['conf']))
        return targets
