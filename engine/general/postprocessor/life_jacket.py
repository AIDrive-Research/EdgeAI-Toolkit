import numpy as np

import gv
from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from .utils import json_utils
from .utils.cv_utils.color_utils import rgb_reverse
from .utils.cv_utils.crop_utils import crop_rectangle
from .utils.image_utils import base64_to_opencv, opencv_to_base64


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.person_model_name = 'pose'
        self.torso_model_name = 'ppe'
        self.alert_label = '未穿救生衣'
        self.non_alert_label = '人'
        self.index = None
        self.group_type = None
        self.similarity = None
        self.limit = None
        self.timeout = None
        self.reinfer_result = {}

    def __torso_data_prepare(self, xyxy, key_points):
        keypoints = []
        for point in key_points:
            keypoints.append([point[0] - xyxy[0], point[1] - xyxy[1], point[2]])
        keypoints = np.array(keypoints)
        # 鼻子，肩膀，胯部
        index = [0, 5, 6, 11, 12]
        attention_points = keypoints[index]
        bool_points = attention_points[:, 2] < self.reserved_args['pose_threshold']
        if any(bool_points):
            return None
        attention_points[bool_points, :] = 0
        nose_point = attention_points[0]
        shoulder_points = attention_points[1:3]
        hip_points = attention_points[3:5]
        data = {
            'nose_point': nose_point,
            'attention_points': attention_points,
            'shoulder_points': shoulder_points,
            'hip_points': hip_points
        }
        return data

    def __expand_width(self, image_shape, rect, expand_ratio=0.2):
        _, imgw, _ = image_shape
        xmin, ymin, xmax, ymax = rect[0], rect[1], rect[2], rect[3]
        width = xmax - xmin
        expand_width = int(width * expand_ratio)
        xmin = max(0, int(xmin - expand_width))
        xmax = min(imgw - 1, int(xmax + expand_width))
        return [xmin, ymin, xmax, ymax]

    def __gen_torso_region(self, image, data):
        nose_point = data['nose_point']
        left_shoulder_point = data['shoulder_points'][0]
        right_shoulder_point = data['shoulder_points'][1]
        left_hip_point = data['hip_points'][0]
        right_hip_point = data['hip_points'][1]
        if left_shoulder_point[0] < right_shoulder_point[0]:
            return None
        shoulder_y = min(left_shoulder_point[1], right_shoulder_point[1])
        min_y = int((nose_point[1] + shoulder_y) / 2)
        min_x = int(min(right_shoulder_point[0], right_hip_point[0]))
        max_x = int(max(left_shoulder_point[0], left_hip_point[0]))
        max_y = int(min(left_hip_point[1], right_hip_point[1]))
        bbox = self.__expand_width(image.shape, [min_x, min_y, max_x, max_y], 0.25)
        torso_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return torso_image

    def __reinfer(self, filter_result):
        person_rectangles = filter_result.get(self.person_model_name)
        if person_rectangles is None:
            LOGGER.error('Person model result is None!')
            return False
        person_results = []
        person_rectangles = sorted(person_rectangles, key=lambda x: x['conf'], reverse=True)
        draw_image = base64_to_opencv(self.draw_image)
        count = 0
        for i in range(len(person_rectangles)):
            if count >= self.limit:
                break
            xyxy = person_rectangles[i]['xyxy']
            cropped_image = crop_rectangle(draw_image, xyxy)
            cropped_image = rgb_reverse(cropped_image)
            data = self.__torso_data_prepare(xyxy, self._get_ext(person_rectangles[i], 'key_points'))
            if data is None:
                person_results.append(self._gen_rectangle(xyxy, (0, 255, 255), self.non_alert_label, None))
                continue
            torso_image = self.__gen_torso_region(cropped_image, data)
            if torso_image is None:
                person_results.append(self._gen_rectangle(xyxy, (0, 255, 255), self.non_alert_label, None))
                continue
            source_data = {
                'source_id': self.source_id,
                'time': self.time * 1000000,
                'infer_image': opencv_to_base64(torso_image),
                'draw_image': None,
                'reserved_data': {
                    'specified_model': [self.torso_model_name],
                    'xyxy': xyxy,
                    'unsort': True
                }
            }
            self.rq_source.put(json_utils.dumps(source_data))
            count += 1
        if count > 0:
            self.reinfer_result[self.time] = {
                'count': count,
                'result': [],
                'draw_image': self.draw_image,
                'person_results': person_results
            }
        return count, person_results

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
            torso_info = self.index.query(id_)
            if torso_info:
                return True, torso_info['name']
        return False, '人'

    def __process_whitelist(self, feature):
        id_, score = self.index.search(feature, self.similarity) if self.index is not None else (None, None)
        if id_ is not None:
            torso_info = self.index.query(id_)
            if torso_info:
                return False, torso_info['name']
        return True, '未穿救生衣'

    def _process(self, result, filter_result):
        hit = False
        if self.index is None:
            self.index = gv.index_dic.get(self.reserved_args['group_id'])
        if self.group_type is None:
            self.group_type = self.reserved_args['group_type']
        if self.limit is None:
            self.limit = self.reserved_args['extra_model'][self.torso_model_name]
        if self.similarity is None:
            self.similarity = max(self.reserved_args['similarity'] - 0.3, 0)
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
        if model_name != self.torso_model_name:
            LOGGER.error(
                'Get wrong model result, expect {}, but get {}'.format(self.torso_model_name, model_name))
            return False
        if self.reinfer_result.get(self.time) is None:
            LOGGER.warning('Not found reinfer result, time=gan{}'.format(self.time))
            return False
        self.reinfer_result[self.time]['result'].append((targets, self.reserved_data['xyxy']))
        if len(self.reinfer_result[self.time]['result']) < self.reinfer_result[self.time]['count']:
            return False
        reinfer_result_ = self.reinfer_result.pop(self.time)
        self.draw_image = reinfer_result_['draw_image']
        for targets, xyxy in reinfer_result_['result']:
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
                        result['data']['bbox']['rectangles'].append(self._gen_rectangle(
                            xyxy, self.alert_color, label, None))
                    else:
                        result['data']['bbox']['rectangles'].append(self._gen_rectangle(
                            xyxy, self.non_alert_color, label, None))
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
        if model_name == self.torso_model_name and not self.reserved_data:
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
                targets.append(self._gen_rectangle(
                    xyxy, self.non_alert_color, label, engine_result_['conf'],
                    key_points=[[int(x * self.scale), int(y * self.scale), s] for (
                        x, y, s) in engine_result_['key_points']]))
        elif model_name == self.torso_model_name:
            targets.append({
                'feature': engine_result
            })
        return targets
