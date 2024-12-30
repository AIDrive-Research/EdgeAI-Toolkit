import numpy as np

from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from .utils import json_utils
from .utils.cv_utils.color_utils import rgb_reverse
from .utils.cv_utils.crop_utils import crop_rectangle
from .utils.cv_utils.geo_utils import is_point_in_polygon
from .utils.image_utils import base64_to_opencv, opencv_to_base64


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.person_model_name = 'person'
        self.handheld_item_model_name = 'hrnet'
        self.limit = None
        self.timeout = None
        self.reinfer_result = {}

    def __reinfer(self, filter_result):
        person_rectangles = filter_result.get(self.person_model_name)
        if person_rectangles is None:
            LOGGER.error('Person model result is None!')
            return False
        person_rectangles = sorted(person_rectangles, key=lambda x: x['conf'], reverse=True)
        draw_image = base64_to_opencv(self.draw_image)
        count = 0
        for i in range(self.limit):
            if i >= len(person_rectangles):
                break
            xyxy = person_rectangles[i]['xyxy']
            cropped_image, new_bbox = self.__expand_crop(draw_image, xyxy, expand_ratio=0.3)
            cropped_image = rgb_reverse(cropped_image)
            source_data = {
                'source_id': self.source_id,
                'time': self.time * 1000000,
                'infer_image': opencv_to_base64(cropped_image),
                'draw_image': None,
                'reserved_data': {
                    'xyxy': xyxy,
                    'specified_model': [self.handheld_item_model_name],
                    'offset': [new_bbox[0], new_bbox[1]],
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
        return True

    def __expand_crop(self, image, rect, expand_ratio=0.3):
        imgh, imgw, _ = image.shape
        xmin, ymin, xmax, ymax = rect[0], rect[1], rect[2], rect[3]
        h_half = (ymax - ymin) * (1 + expand_ratio) / 2.
        w_half = (xmax - xmin) * (1 + expand_ratio) / 2.
        if h_half > w_half * 4 / 3:
            w_half = h_half * 0.75
        center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
        ymin = max(0, int(center[0] - h_half))
        ymax = min(imgh - 1, int(center[0] + h_half))
        xmin = max(0, int(center[1] - w_half))
        xmax = min(imgw - 1, int(center[1] + w_half))
        return crop_rectangle(image, (xmin, ymin, xmax, ymax)), [xmin, ymin, xmax, ymax]

    def __check_expire(self):
        for time in list(self.reinfer_result.keys()):
            if time < self.time - self.timeout:
                LOGGER.warning('Reinfer result expired, source_id={}, alg_name={}, time={}, timeout={}'.format(
                    self.source_id, self.alg_name, time, self.timeout))
                del self.reinfer_result[time]
        return True

    @staticmethod
    def __vector_angle(vec1, vec2):
        x, y = vec1, vec2
        Lx = np.sqrt(x.dot(x))
        Ly = np.sqrt(y.dot(y))
        cos_angle = x.dot(y) / (Lx * Ly)
        angle = np.arccos(cos_angle)
        angle = angle * 360 / 2 / np.pi
        return angle

    def __check_handheld_item(self, targets, polygons):
        # 判断手是否抬起，如果是，则认为是货架拿取物品动作识别
        handheld_item_targets = []
        for target in targets:
            key_points = target['key_points']
            left = {
                'elbow': key_points[7],
                'hand': key_points[9]
            }
            right = {
                'elbow': key_points[8],
                'hand': key_points[10]
            }
            angle_left_elbow_hand, angle_right_elbow_hand = 0, 0
            # 左侧肘-手与地面的角度
            if left['elbow'][0] > 0 and left['elbow'][1] > 0 and left['hand'][0] > 0 and left['hand'][1] > 0:
                elbow_to_hand = np.array([left['hand'][0] - left['elbow'][0], left['hand'][1] - left['elbow'][1]])
                elbow_to_ground = np.array([0, -left['hand'][1] - left['elbow'][1]])
                angle_left_elbow_hand = self.__vector_angle(elbow_to_hand, elbow_to_ground)
            if angle_left_elbow_hand > self.reserved_args['threshold']:
                if polygons:
                    for polygon in polygons.values():
                        if is_point_in_polygon(left['hand'], polygon['polygon']):
                            handheld_item_targets.append(target)
                            break
                else:
                    handheld_item_targets.append(target)
            if handheld_item_targets:
                continue
            # 右侧肘-手与地面的角度
            if right['elbow'][0] > 0 and right['elbow'][1] > 0 and right['hand'][0] > 0 and right['hand'][1] > 0:
                elbow_to_hand = np.array([right['hand'][0] - right['elbow'][0], right['hand'][1] - right['elbow'][1]])
                elbow_to_ground = np.array([0, -right['hand'][1] - right['elbow'][1]])
                angle_right_elbow_hand = self.__vector_angle(elbow_to_hand, elbow_to_ground)
            if angle_right_elbow_hand > self.reserved_args['threshold']:
                if polygons:
                    for polygon in polygons.values():
                        if is_point_in_polygon(right['hand'], polygon['polygon']):
                            handheld_item_targets.append(target)
                            break
                else:
                    handheld_item_targets.append(target)
        return handheld_item_targets

    def _process(self, result, filter_result):
        hit = False
        if self.limit is None:
            self.limit = self.reserved_args['extra_model'][self.handheld_item_model_name]
        if self.timeout is None:
            self.timeout = (self.frame_interval / 1000) * 2
            LOGGER.info('source_id={}, alg_name={}, timeout={}'.format(self.source_id, self.alg_name, self.timeout))
        if not self.reserved_data:
            self.__reinfer(filter_result)
            return False
        self.__check_expire()
        polygons = self._gen_polygons()
        model_name, targets = next(iter(filter_result.items()))
        if model_name != self.handheld_item_model_name:
            LOGGER.error('Get wrong model result, expect {}, but get {}'.format(
                self.handheld_item_model_name, model_name))
            return False
        if self.reinfer_result.get(self.time) is None:
            LOGGER.warning('Not found reinfer result, time={}'.format(self.time))
            return False
        self.reinfer_result[self.time]['result'].append(
            (targets, self.reserved_data['offset'], self.reserved_data['xyxy']))
        if len(self.reinfer_result[self.time]['result']) < self.reinfer_result[self.time]['count']:
            return False
        reinfer_result_ = self.reinfer_result.pop(self.time)
        self.draw_image = reinfer_result_['draw_image']
        for targets, offset, xyxy in reinfer_result_['result']:
            if not targets:
                continue
            for i, target in enumerate(targets):
                for j in range(len(target['key_points'])):
                    target['key_points'][j][0] += offset[0]
                    target['key_points'][j][1] += offset[1]
            handheld_item_targets = self.__check_handheld_item(targets, polygons)
            if handheld_item_targets:
                for target in handheld_item_targets:
                    hit = True
                    result['data']['bbox']['rectangles'].append(self._gen_rectangle(
                        xyxy, self.alert_color, target['label'], target['conf']))
            else:
                for target in targets:
                    result['data']['bbox']['rectangles'].append(self._gen_rectangle(
                        xyxy, target['color'], target['label'], target['conf']))
        result['hit'] = hit
        result['data']['bbox']['polygons'].update(polygons)
        return True

    def _filter(self, model_name, model_data):
        targets = []
        if model_name == self.handheld_item_model_name and not self.reserved_data:
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
                # 生成矩形框
                targets.append(self._gen_rectangle(xyxy, self.non_alert_color, label, engine_result_['conf']))
        elif model_name == self.handheld_item_model_name:
            for engine_result_ in engine_result:
                # 过滤掉置信度低于阈值的目标
                if not self._filter_by_conf(model_conf, engine_result_['conf']):
                    continue
                label = self._get_label(model_conf['label'], engine_result_['label'])
                # 生成关键点目标
                targets.append({
                    'key_points': engine_result_['key_points'],
                    'color': self.non_alert_color,
                    'label': label,
                    'conf': engine_result_['conf']
                })
        return targets
