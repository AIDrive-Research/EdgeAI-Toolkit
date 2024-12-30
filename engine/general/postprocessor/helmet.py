from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from .utils import json_utils
from .utils.cv_utils.color_utils import rgb_reverse
from .utils.cv_utils.crop_utils import crop_rectangle
from .utils.cv_utils.geo_utils import is_point_in_rectangle
from .utils.image_utils import base64_to_opencv, opencv_to_base64


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.helmet_model_name = 'helmet'
        self.person_model_name = 'person'
        self.cls_model_name = 'helmet_classify'
        self.timeout = None
        self.reinfer_result = {}
        self.helmet_label = 0

    @staticmethod
    def __expand_box(rectangle, expand_w, expand_h, img_w, img_h):
        min_x, min_y, max_x, max_y = rectangle
        # 计算矩形的中心点
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        # 计算矩形的宽度和高度
        width = max_x - min_x
        height = max_y - min_y
        # 按比例扩大矩形框
        new_width = width + expand_w * 2
        new_height = height + expand_h * 2
        # 计算扩大后的矩形框的坐标
        new_min_x = max(0, center_x - new_width / 2)
        new_max_x = min(img_w, center_x + new_width / 2)
        new_min_y = max(0, center_y - new_height / 2)
        new_max_y = min(img_h, center_y + new_height / 2)
        if new_min_x > new_max_x or new_min_y > new_max_y:
            return None
        return [int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)]

    def __reinfer(self, filter_result):
        helmet_rectangles = filter_result.get(self.helmet_model_name)
        if helmet_rectangles is None:
            LOGGER.error('Helmet model result is None!')
            return False
        person_rectangles = filter_result.get(self.person_model_name)
        if person_rectangles is None:
            LOGGER.error('Person model result is None!')
            return False
        draw_image = base64_to_opencv(self.draw_image)
        img_h, img_w = draw_image.shape[:2]
        count = 0
        for helmet_rectangle in helmet_rectangles:
            if helmet_rectangle['label'] not in self.alert_label:
                continue
            xyxy = helmet_rectangle['xyxy']
            bbox = self.__expand_box(xyxy, 5, 5, img_w, img_h)
            cropped_image = crop_rectangle(draw_image, bbox)
            cropped_image = rgb_reverse(cropped_image)
            source_data = {
                'source_id': self.source_id,
                'time': self.time * 1000000,
                'infer_image': opencv_to_base64(cropped_image),
                'draw_image': None,
                'reserved_data': {
                    'specified_model': [self.cls_model_name],
                    'rectangle': helmet_rectangle,
                    'unsort': True
                }
            }
            self.rq_source.put(json_utils.dumps(source_data))
            count += 1
        if count > 0:
            self.reinfer_result[self.time] = {
                'count': count,
                'draw_image': self.draw_image,
                'person_rectangles': person_rectangles,
                'result': []
            }
        return count, person_rectangles

    @staticmethod
    def calc_iou(xyxy_head, xyxy_person):
        """
        计算两个矩形的IOU/head
        Args:
            xyxy1: 矩形1的左上角和右下角坐标
            xyxy2: 矩形2的左上角和右下角坐标
        Returns: IOU
        """
        left_max = max(xyxy_head[0], xyxy_person[0])
        right_min = min(xyxy_head[2], xyxy_person[2])
        up_max = max(xyxy_head[1], xyxy_person[1])
        down_min = min(xyxy_head[3], xyxy_person[3])
        if left_max >= right_min or down_min <= up_max:
            return 0
        s1 = (xyxy_head[2] - xyxy_head[0]) * (xyxy_head[3] - xyxy_head[1])
        s_cross = (down_min - up_max) * (right_min - left_max)
        return s_cross / s1

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
            count, person_results = self.__reinfer(filter_result)
            if not count:
                self.__check_expire()
                result['hit'] = hit
                result['data']['bbox']['rectangles'].extend(person_results)
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
        person_rectangles = reinfer_result_['person_rectangles']
        for targets, helmet_rectangle in reinfer_result_['result']:
            if not targets:
                continue
            max_iou = 0
            hit_person = None
            for i, person_rectangle in enumerate(person_rectangles):
                # 下边沿中心点
                point = ((helmet_rectangle['xyxy'][0] + helmet_rectangle['xyxy'][2]) / 2,
                         helmet_rectangle['xyxy'][3])
                if not is_point_in_rectangle(point, person_rectangle['xyxy']):
                    continue
                iou = self.calc_iou(helmet_rectangle['xyxy'], person_rectangle['xyxy'])
                if iou > max_iou:
                    max_iou = iou
                    hit_person = i
            if hit_person is not None and max_iou > 0.8:
                hit = True
                person_rectangles[hit_person]['color'] = self.alert_color
                person_rectangles[hit_person]['label'] = self.alert_label[0]
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(person_rectangles)
        result['data']['bbox']['polygons'].update(polygons)
        return result

    def _filter(self, model_name, model_data):
        targets = []
        if model_name == self.cls_model_name and not self.reserved_data:
            return targets
        model_conf = model_data['model_conf']
        engine_result = model_data['engine_result']
        if model_name in [self.helmet_model_name, self.person_model_name]:
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
        elif model_name == self.cls_model_name:
            score = engine_result['output'][self.helmet_label]
            if score >= model_conf['args']['conf_thres']:
                targets.append(engine_result)
        return targets
