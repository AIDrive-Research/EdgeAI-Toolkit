import numpy as np

import gv
from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from .utils import msgpack_utils
from .utils.cv_utils.color_utils import rgb_reverse
from .utils.cv_utils.crop_utils import crop_rectangle
from .utils.cv_utils.geo_utils import is_point_in_rectangle
from .utils.image_utils.turbojpegutils import bytes_to_mat, mat_to_bytes


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.head_model_name = 'zql_helmet'
        self.person_model_name = 'zql_person'
        self.ppe_model_name = 'zql_ppe'
        self.index = None
        self.group_type = None
        self.similarity = None
        self.limit = None
        self.timeout = None
        self.reinfer_result = {}

    def __reinfer(self, filter_result):
        head_rectangles = filter_result.get(self.head_model_name)
        if head_rectangles is None:
            LOGGER.error('Head model result is None!')
            return False
        person_rectangles = filter_result.get(self.person_model_name)
        if person_rectangles is None:
            LOGGER.error('Person model result is None!')
            return False
        head_rectangles = sorted(head_rectangles, key=lambda x: x['conf'], reverse=True)
        draw_image = bytes_to_mat(self.draw_image)
        count = 0
        person_results = []
        for i in range(len(person_rectangles)):
            person_rectangle = person_rectangles[i]
            if count >= self.limit:
                person_results.extend(person_rectangles[i:])
                break
            hit_ = False
            for head_rectangle in head_rectangles:
                xyxy = head_rectangle['xyxy']
                if head_rectangle['ext'].get('hit'):
                    continue
                point = self._get_point(xyxy, strategy='center')
                if not is_point_in_rectangle(point, person_rectangle['xyxy']):
                    continue
                hit_ = True
                head_rectangle['ext']['hit'] = True
                cropped_image = crop_rectangle(draw_image, xyxy)
                cropped_image = rgb_reverse(cropped_image)
                source_data = {
                    'source_id': self.source_id,
                    'time': self.time * 1000000,
                    'infer_image': mat_to_bytes(cropped_image),
                    'draw_image': None,
                    'reserved_data': {
                        'specified_model': [self.ppe_model_name],
                        'person_rectangle': person_rectangle,
                        'unsort': True
                    }
                }
                self.rq_source.put(msgpack_utils.dump(source_data))
                count += 1
                break
            if not hit_:
                person_results.append(person_rectangle)
        if count > 0:
            self.reinfer_result[self.time] = {
                'count': count,
                'result': [],
                'draw_image': self.draw_image,
                'person_results': person_results
            }
        return count, person_rectangles

    def __check_expire(self):
        for time in list(self.reinfer_result.keys()):
            if time < self.time - self.timeout:
                LOGGER.warning('Reinfer result expired, source_id={}, alg_name={}, time={}, timeout={}'.format(
                    self.source_id, self.alg_name, time, self.timeout))
                del self.reinfer_result[time]
        return True

    def __process_blacklist(self, feature):
        id_, score = self.index.search(feature, self.similarity) if self.index is not None else (None, None)
        if id_ is not None:
            goggles_info = self.index.query(id_)
            if goggles_info:
                return True, goggles_info['name']
        return False, '人'

    def __process_whitelist(self, feature):
        id_, score = self.index.search(feature, self.similarity) if self.index is not None else (None, None)
        if id_ is not None:
            goggles_info = self.index.query(id_)
            if goggles_info:
                return False, goggles_info['name']
        return True, self.alert_label[0]

    def _process(self, result, filter_result):
        hit = False
        if self.index is None:
            self.index = gv.index_dic.get(self.reserved_args['group_id'])
        if self.group_type is None:
            self.group_type = self.reserved_args['group_type']
        if self.limit is None:
            self.limit = self.reserved_args['extra_model'][self.ppe_model_name]
        if self.similarity is None:
            self.similarity = max(self.reserved_args['similarity'] - 0.2, 0)
        if self.timeout is None:
            self.timeout = (self.frame_interval / 1000) * 2
            LOGGER.info('source_id={}, alg_name={}, timeout={}'.format(self.source_id, self.alg_name, self.timeout))
        polygons = self._gen_polygons()
        if not self.reserved_data:
            count, person_results = self.__reinfer(filter_result)
            if not count:
                self.__check_expire()
                result['data']['bbox']['rectangles'].extend(person_results)
                result['hit'] = False
                result['data']['bbox']['polygons'].update(polygons)
                return True
            return False
        self.__check_expire()
        model_name, targets = next(iter(filter_result.items()))
        if model_name != self.ppe_model_name:
            LOGGER.error(
                'Get wrong model result, expect {}, but get {}'.format(self.ppe_model_name, model_name))
            return False
        if self.reinfer_result.get(self.time) is None:
            LOGGER.warning('Not found reinfer result, time={}'.format(self.time))
            return False
        self.reinfer_result[self.time]['result'].append((targets, self.reserved_data['person_rectangle']))
        if len(self.reinfer_result[self.time]['result']) < self.reinfer_result[self.time]['count']:
            return False
        reinfer_result_ = self.reinfer_result.pop(self.time)
        self.draw_image = reinfer_result_['draw_image']
        for targets, person_rectangle in reinfer_result_['result']:
            for target in targets:
                feature = target.pop('feature', None)
                if feature is not None:
                    np_feature = np.array(feature, dtype=np.float32)
                    if 'blacklist' == self.group_type:
                        hit_, label = self.__process_blacklist(np_feature)
                    elif 'whitelist' == self.group_type:
                        hit_, label = self.__process_whitelist(np_feature)
                    else:
                        LOGGER.error('Unknown group_type: {}'.format(self.group_type))
                        continue
                    if hit_:
                        hit = hit_
                        person_rectangle['label'] = label
                        person_rectangle['color'] = self.alert_color
                        result['data']['bbox']['rectangles'].append(person_rectangle)
                    else:
                        result['data']['bbox']['rectangles'].append(person_rectangle)
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(reinfer_result_['person_results'])
        result['data']['bbox']['polygons'].update(polygons)
        result['data']['group'] = {
            'id': self.index.group_id if self.index is not None else None,
            'name': self.index.group_name if self.index is not None else None
        }
        return True

    def _filter(self, model_name, model_data):
        targets = []
        if model_name == self.ppe_model_name and not self.reserved_data:
            return targets
        model_conf = model_data['model_conf']
        engine_result = model_data['engine_result']
        if model_name in [self.head_model_name, self.person_model_name]:
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
        elif model_name == self.ppe_model_name:
            targets.append({
                'feature': engine_result
            })
        return targets
