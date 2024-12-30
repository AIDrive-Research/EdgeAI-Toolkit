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
        self.goggles_model_name = 'ppe'
        self.index = None
        self.group_type = None
        self.similarity = None
        self.limit = None
        self.timeout = None
        self.reinfer_result = {}
        self.angle_th = 45

    def __goggles_data_prepare(self, xyxy, key_points):
        keypoints = []
        for point in key_points:
            keypoints.append([point[0] - xyxy[0], point[1] - xyxy[1], point[2]])
        keypoints = np.array(keypoints)
        attention_points = keypoints[:5]
        bool_points = attention_points[:, 2] < self.reserved_args['pose_threshold']
        attention_points[bool_points, :] = 0
        # 鼻子、眼睛、耳朵
        nose_point = attention_points[0]
        eyes_points = attention_points[1:3]
        ear_points = attention_points[3:5]
        data = {
            'attention_points': attention_points,
            'nose_point': nose_point,
            'eyes_points': eyes_points,
            'ear_points': ear_points,
            'bool_points': bool_points
        }
        return data

    def __gen_goggles_region(self, image, data):
        img_h, img_w = image.shape[:2]
        # 鼻子、眼睛、耳朵必须在
        if not (sum(data['bool_points']) == 0):
            return None
        nose_point = data['nose_point']
        eyes_points = data['eyes_points']
        ear_points = data['ear_points']
        v1 = np.array((eyes_points[1][0] - eyes_points[0][0], eyes_points[1][1] - eyes_points[0][1]))
        v2 = np.array((eyes_points[1][0] - eyes_points[0][0], 0))
        angle = self.__vector_angle(v1, v2)
        # 头部倾斜角度
        if angle > self.angle_th:
            return None
        min_y, max_y = self.__compute_bbox_h(nose_point, eyes_points)
        min_x, max_x = self.__compute_bbox_x(eyes_points, ear_points)
        scale_w = 1.4
        scale_h = 1.2
        bbox = self.__expand_box([min_x, min_y, max_x, max_y], scale_w, scale_h, img_w, img_h)
        if (bbox is None) or (bbox[3] - bbox[1] < 10):
            return None
        goggles_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        return goggles_image

    @staticmethod
    def __vector_angle(vec1, vec2):
        x, y = vec1, vec2
        Lx = np.sqrt(x.dot(x))
        Ly = np.sqrt(y.dot(y))
        cos_angle = x.dot(y) / (Lx * Ly)
        angle = np.arccos(cos_angle)
        angle = angle * 360 / 2 / np.pi
        return angle

    @staticmethod
    def __compute_bbox_h(nose_point, eyes_points):
        # max_y
        max_y = int(nose_point[1])
        min_eyes_y = np.min(eyes_points[:, 1], axis=0)
        max_eyes_y = np.max(eyes_points[:, 1], axis=0)
        if min_eyes_y >= max_y or max_eyes_y >= max_y:
            return None, None
        h_half = max_y - min_eyes_y
        # min_y
        min_y = max(0, max_y - h_half * 2)
        return int(min_y), int(max_y)

    @staticmethod
    def __compute_bbox_x(eyes_points, ear_points):
        # max_x
        if ear_points[0][2] > 0:
            max_x = int(ear_points[0][0])
        else:
            max_x = int(np.max(eyes_points[:, 0], axis=0))
        # min_x
        if ear_points[1][2] > 0:
            min_x = int(ear_points[1][0])
        else:
            min_x = int(np.min(eyes_points[:, 0], axis=0))
        return int(min_x), int(max_x)

    @staticmethod
    def __expand_box(rectangle, scale_w, scale_h, img_w, img_h):
        min_x, min_y, max_x, max_y = rectangle
        # 计算矩形的中心点
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        # 计算矩形的宽度和高度
        width = max_x - min_x
        height = max_y - min_y
        # 按比例扩大矩形框
        new_width = width * scale_w
        new_height = height * scale_h
        # 计算扩大后的矩形框的坐标
        new_min_x = max(0, center_x - new_width / 2)
        new_max_x = min(img_w, center_x + new_width / 2)
        new_min_y = max(0, center_y - new_height / 2)
        new_max_y = min(img_h, center_y + new_height / 2)
        if new_min_x > new_max_x or new_min_y > new_max_y:
            return None
        return [int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)]

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
            data = self.__goggles_data_prepare(xyxy, self._get_ext(person_rectangles[i], 'key_points'))
            goggles_image = self.__gen_goggles_region(cropped_image, data)
            if goggles_image is None:
                person_results.append(self._gen_rectangle(xyxy, (0, 255, 255), '人', None))
                continue
            source_data = {
                'source_id': self.source_id,
                'time': self.time * 1000000,
                'infer_image': opencv_to_base64(goggles_image),
                'draw_image': None,
                'reserved_data': {
                    'specified_model': [self.goggles_model_name],
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
        return True, '未佩戴护目镜'

    def _process(self, result, filter_result):
        hit = False
        if self.index is None:
            self.index = gv.index_dic.get(self.reserved_args['group_id'])
        if self.group_type is None:
            self.group_type = self.reserved_args['group_type']
        if self.limit is None:
            self.limit = self.reserved_args['extra_model'][self.goggles_model_name]
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
        if model_name != self.goggles_model_name:
            LOGGER.error(
                'Get wrong model result, expect {}, but get {}'.format(self.goggles_model_name, model_name))
            return False
        if self.reinfer_result.get(self.time) is None:
            LOGGER.warning('Not found reinfer result, time={}'.format(self.time))
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
        if model_name == self.goggles_model_name and not self.reserved_data:
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
        elif model_name == self.goggles_model_name:
            targets.append({
                'feature': engine_result
            })
        return targets
