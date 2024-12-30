import numpy as np

import gv
from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.index = None
        self.group_type = None
        self.similarity = None
        self.quality_thresh = None

    def __process_blacklist(self, feature, rectangle):
        id_, score = self.index.search(feature, self.similarity) if self.index is not None else (None, None)
        if id_ is not None:
            face_info = self.index.query(id_)
            if face_info:
                rectangle['color'] = self.alert_color
                rectangle['label'] = face_info['name']
                rectangle['conf'] = min(max(score + 0.3, 0.4), 0.95)
                self._set_ext(rectangle, face=face_info)
                return True
        rectangle['color'] = self.non_alert_color
        return False

    def __process_whitelist(self, feature, rectangle):
        id_, score = self.index.search(feature, self.similarity) if self.index is not None else (None, None)
        if id_ is not None:
            face_info = self.index.query(id_)
            if face_info:
                rectangle['color'] = self.non_alert_color
                rectangle['label'] = face_info['name']
                rectangle['conf'] = min(max(score + 0.3, 0.4), 0.95)
                return False
        rectangle['color'] = self.alert_color
        return True

    def _process(self, result, filter_result):
        hit = False
        if self.index is None:
            self.index = gv.index_dic.get(self.reserved_args['group_id'])
        if self.group_type is None:
            self.group_type = self.reserved_args['group_type']
        if self.similarity is None:
            self.similarity = max(self.reserved_args['similarity'] - 0.3, 0)
        polygons = self._gen_polygons()
        model_name, rectangles = next(iter(filter_result.items()))
        for rectangle in rectangles:
            feature = self._get_ext(rectangle, 'feature', pop=True)
            if feature is not None:
                np_feature = np.array(feature, dtype=np.float32)
                if 'blacklist' == self.group_type:
                    hit_ = self.__process_blacklist(np_feature, rectangle)
                elif 'whitelist' == self.group_type:
                    hit_ = self.__process_whitelist(np_feature, rectangle)
                else:
                    LOGGER.error('Unknown group_type: {}'.format(self.group_type))
                    continue
                if hit_:
                    hit = hit_
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(rectangles)
        result['data']['bbox']['polygons'].update(polygons)
        result['data']['group'] = {
            'id': self.index.group_id if self.index is not None else None,
            'name': self.index.group_name if self.index is not None else None
        }
        return True

    def _filter(self, model_name, model_data):
        targets = []
        if self.quality_thresh is None:
            self.quality_thresh = self.reserved_args['threshold']
        model_conf = model_data['model_conf']
        engine_result = model_data['engine_result']
        for engine_result_ in engine_result:
            # 过滤掉置信度低于阈值的目标
            if not self._filter_by_conf(model_conf, engine_result_['conf']):
                continue
            # 过滤掉人脸质量低于阈值的目标
            if engine_result_['quality'] < self.quality_thresh * 100:
                continue
            # 坐标缩放
            xyxy = self._scale(engine_result_['xyxy'])
            # 过滤掉不在多边形内的目标
            if not self._filter_by_roi(xyxy):
                continue
            # 生成矩形框
            targets.append(self._gen_rectangle(
                xyxy, self.non_alert_color, None, engine_result_['conf'],
                feature=engine_result_['feature']))
        return targets
