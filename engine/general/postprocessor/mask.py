from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from .utils.cv_utils.geo_utils import is_point_in_rectangle


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.mask_model_name = 'mask'
        self.person_model_name = 'person'

    def _process(self, result, filter_result):
        hit = False
        polygons = self._gen_polygons()
        mask_rectangles = filter_result.get(self.mask_model_name)
        if mask_rectangles is None:
            LOGGER.error('Mask model result is None!')
            return False
        no_mask_rectangles = [rectangle for rectangle in mask_rectangles if rectangle['label'] == self.alert_label[0]]
        person_rectangles = filter_result.get(self.person_model_name)
        if person_rectangles is None:
            LOGGER.error('Person model result is None!')
            return False
        for person_rectangle in person_rectangles:
            for no_mask_rectangle in no_mask_rectangles:
                point = self._get_point(no_mask_rectangle['xyxy'], 'center')
                if is_point_in_rectangle(point, person_rectangle['xyxy']):
                    hit = True
                    person_rectangle['color'] = self.alert_color
                    person_rectangle['label'] = self.alert_label[0]
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
            # 过滤掉不在多边形内的目标
            if model_name == self.person_model_name and (not self._filter_by_roi(xyxy)):
                continue
            # 生成矩形框
            targets.append(self._gen_rectangle(xyxy, self.non_alert_color, label, engine_result_['conf']))
        return targets
