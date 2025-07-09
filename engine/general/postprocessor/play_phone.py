from postprocessor import Postprocessor as BasePostprocessor
from .utils import json_utils
from .utils import msgpack_utils
from .utils.cv_utils.color_utils import rgb_reverse
from .utils.cv_utils.crop_utils import crop_rectangle
from .utils.cv_utils.geo_utils import is_rectangle_intersect
from .utils.image_utils.turbojpegutils import bytes_to_mat, mat_to_bytes


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.person_model_name = 'zql_person'
        self.play_phone_model_name = 'zql_play_phone'
        self.phone_label = 'phone'
        self.hand_label = 'hand'
        self.alert_label = '使用手机'
        self.limit = None
        self.timeout = None
        self.reinfer_result = {}

    @staticmethod
    def _expand_box(rectangle, scale_w, scale_h, img_w, img_h):
        min_x, min_y, max_x, max_y = rectangle
        min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
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
            return rectangle
        return [int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)]

    def __reinfer(self, filter_result):
        person_rectangles = filter_result.get(self.person_model_name)
        if person_rectangles is None:
            LOGGER.error('Person model result is None!')
            return False
        person_rectangles = sorted(person_rectangles, key=lambda x: x['conf'], reverse=True)
        draw_image = bytes_to_mat(self.draw_image)
        img_h, img_w = draw_image.shape[0], draw_image.shape[1]
        count = 0
        for i in range(self.limit):
            if i >= len(person_rectangles):
                break
            xyxy = person_rectangles[i]['xyxy']
            expand_box = self._expand_box(xyxy, 1.2, 1, img_w, img_h)
            cropped_image = crop_rectangle(draw_image, expand_box)
            cropped_image = rgb_reverse(cropped_image)
            source_data = {
                'source_id': self.source_id,
                'time': self.time * 1000000,
                'infer_image': mat_to_bytes(cropped_image),
                'draw_image': None,
                'reserved_data': {
                    'specified_model': [self.play_phone_model_name],
                    'xyxy': expand_box,
                    'unsort': True
                }
            }
            self.rq_source.put(msgpack_utils.dump(source_data))
            count += 1
        if count > 0:
            self.reinfer_result[self.time] = {
                'count': count,
                'draw_image': self.draw_image,
                'result': []
            }
        return count

    def __check_expire(self):
        for time in list(self.reinfer_result.keys()):
            if time < self.time - self.timeout:
                LOGGER.warning('Reinfer result expired, source_id={}, alg_name={}, time={}, timeout={}'.format(
                    self.source_id, self.alg_name, time, self.timeout))
                del self.reinfer_result[time]
        return True

    def __check_play_phone(self, rectangles):
        # 判断手机与手是否相交，如果是，则认为是使用手机行为
        phone_rectangles = [x for x in rectangles if x['label'] == self.phone_label]
        hand_rectangles = [x for x in rectangles if x['label'] == self.hand_label]
        for phone_rectangle in phone_rectangles:
            for hand_rectangle in hand_rectangles:
                if is_rectangle_intersect(phone_rectangle['xyxy'], hand_rectangle['xyxy']):
                    phone_rectangle['color'] = self.alert_color
                    phone_rectangle['label'] = self.alert_label
                    break
        return [x for x in phone_rectangles if x['label'] == self.alert_label]

    def _process(self, result, filter_result):
        hit = False
        if self.limit is None:
            self.limit = self.reserved_args['extra_model'][self.play_phone_model_name]
        if self.timeout is None:
            self.timeout = (self.frame_interval / 1000) * 2
            LOGGER.info('source_id={}, alg_name={}, timeout={}'.format(self.source_id, self.alg_name, self.timeout))
        polygons = self._gen_polygons()
        if not self.reserved_data:
            count = self.__reinfer(filter_result)
            if not count:
                self.__check_expire()
                result['hit'] = False
                result['data']['bbox']['polygons'].update(polygons)
                return True
            return False
        self.__check_expire()
        model_name, rectangles = next(iter(filter_result.items()))
        if model_name != self.play_phone_model_name:
            LOGGER.error('Get wrong model result, expect {}, but get {}'.format(self.play_phone_model_name, model_name))
            return False
        if self.reinfer_result.get(self.time) is None:
            LOGGER.warning('Not found reinfer result, time={}'.format(self.time))
            return False
        self.reinfer_result[self.time]['result'].append((rectangles, self.reserved_data['xyxy']))
        if len(self.reinfer_result[self.time]['result']) < self.reinfer_result[self.time]['count']:
            return False
        reinfer_result_ = self.reinfer_result.pop(self.time)
        self.draw_image = reinfer_result_['draw_image']
        person_rectangles = {}
        for rectangles, xyxy in reinfer_result_['result']:
            if json_utils.dumps(xyxy) not in person_rectangles:
                person_rectangles[json_utils.dumps(xyxy)] = self._gen_rectangle(xyxy, self.non_alert_color, '人', None)
            phone_rectangles = self.__check_play_phone(rectangles)
            if phone_rectangles:
                for phone_rectangle in phone_rectangles:
                    phone_rectangle['xyxy'][0] += xyxy[0]
                    phone_rectangle['xyxy'][1] += xyxy[1]
                    phone_rectangle['xyxy'][2] += xyxy[0]
                    phone_rectangle['xyxy'][3] += xyxy[1]
                hit = True
                result['data']['bbox']['rectangles'].extend(phone_rectangles)
                person_rectangles[json_utils.dumps(xyxy)]['color'] = self.alert_color
        result['hit'] = hit
        for _, rectangle in person_rectangles.items():
            result['data']['bbox']['rectangles'].append(rectangle)
        result['data']['bbox']['polygons'].update(polygons)
        return True

    def _filter(self, model_name, model_data):
        targets = []
        if model_name == self.play_phone_model_name and not self.reserved_data:
            return targets
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
            if model_name == self.person_model_name:
                # 坐标缩放
                xyxy = self._scale(engine_result_['xyxy'])
                # 过滤掉不在多边形内的目标
                if not self._filter_by_roi(xyxy):
                    continue
            else:
                xyxy = engine_result_['xyxy']
            # 生成矩形框
            targets.append(self._gen_rectangle(
                xyxy, self._get_color(model_conf['label'], label), label, engine_result_['conf']))
        return targets
