import numpy as np

from postprocessor import Postprocessor as BasePostprocessor
from .utils.cv_utils.geo_utils import is_point_in_polygon


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)

    def _process(self, result, filter_result):
        hit = False
        polygons = self._gen_polygons()
        model_name, rectangles = next(iter(filter_result.items()))
        if rectangles:
            hit = True
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(rectangles)
        result['data']['bbox']['polygons'].update(polygons)
        return True

    @staticmethod
    def min_enclosing_rect(vertices):
        # 计算矩形的边界框
        min_x = np.min(vertices[:, 0])
        max_x = np.max(vertices[:, 0])
        min_y = np.min(vertices[:, 1])
        max_y = np.max(vertices[:, 1])
        # 获取最小外接矩形的四个顶点
        rect_vertices = [int(min_x), int(min_y), int(max_x), int(max_y)]
        return rect_vertices

    def _filter(self, model_name, model_data):
        targets = []
        model_conf = model_data['model_conf']
        engine_result = model_data['engine_result']
        for engine_result_ in engine_result:
            # 过滤掉检测置信度低于阈值的目标
            if not self._filter_by_conf(model_conf, engine_result_['det_conf']):
                continue
            # 过滤掉识别置信度低于阈值的目标
            if engine_result_['rec_conf'] < self.reserved_args['threshold']:
                continue
            # 过滤掉无label的目标
            label = engine_result_['rec']
            if not label:
                continue
            # 坐标缩放
            xyxy = [[int(x * self.scale), int(y * self.scale)] for x, y in engine_result_['xyxy']]
            center_x = sum(point[0] for point in xyxy) / 4
            center_y = sum(point[1] for point in xyxy) / 4
            xyxy = self.min_enclosing_rect(np.array(xyxy))
            # 过滤掉不在多边形内的点
            polygons = self.bbox.get('polygons', [])
            if polygons:
                for polygon in polygons:
                    if is_point_in_polygon((center_x, center_y), polygon['polygon']):
                        break
                else:
                    continue
            # 生成矩形框
            targets.append(self._gen_rectangle(xyxy, self.non_alert_color, label, engine_result_['rec_conf']))
        return targets
