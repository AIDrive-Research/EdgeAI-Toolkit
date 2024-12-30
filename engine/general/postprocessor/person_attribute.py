import numpy as np

from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from .utils import json_utils
from .utils.cv_utils.color_utils import rgb_reverse
from .utils.cv_utils.crop_utils import crop_rectangle
from .utils.image_utils import base64_to_opencv, opencv_to_base64


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.det_model_name = 'person'
        self.rec_model_name = 'person_attribute'
        self.strategy = None
        self.timeout = None
        self.limit = None
        self.reinfer_result = {}
        self.args = {
            'threshold': 0.5,
            'hold_threshold': 0.6,
            'glasses_threshold': 0.3,
            'age_list': ['小于18', '18-60', '大于60'],
            'direct_list': ['正面', '侧面', '背面'],
            'bag_list': ['手提包', '单肩包', '双肩包'],
            'upper_list': ['上衣条纹', '上衣logo', '上衣格子', '上衣拼装'],
            'lower_list': ['下装带条纹', '下装图案', '长外套', '长裤', '短裤', '短裙&裙子']
        }

    def __reinfer(self, filter_result):
        person_rectangles = filter_result.get(self.det_model_name)
        if person_rectangles is None:
            LOGGER.error('Perons model result is None!')
            return False
        person_rectangles = sorted(person_rectangles, key=lambda x: x['conf'], reverse=True)
        draw_image = base64_to_opencv(self.draw_image)
        count = 0
        for i in range(self.limit):
            if i >= len(person_rectangles):
                break
            xyxy = person_rectangles[i]['xyxy']
            cropped_image = crop_rectangle(draw_image, xyxy)
            cropped_image = rgb_reverse(cropped_image)
            source_data = {
                'source_id': self.source_id,
                'time': self.time * 1000000,
                'infer_image': opencv_to_base64(cropped_image),
                'draw_image': None,
                'reserved_data': {
                    'specified_model': [self.rec_model_name],
                    'rectangle': xyxy,
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
        return count

    def __check_expire(self):
        for time in list(self.reinfer_result.keys()):
            if time < self.time - self.timeout:
                LOGGER.warning('Reinfer result expired, source_id={}, alg_name={}, time={}, timeout={}'.format(
                    self.source_id, self.alg_name, time, self.timeout))
                del self.reinfer_result[time]
        return True

    def __postprocess(self, result):
        label_res = []
        # gender
        gender = '女' if result[22] > self.args['threshold'] else '男'
        label_res.append(gender)
        # age
        age = self.args['age_list'][np.argmax(result[19:22])]
        label_res.append(age)
        # direction
        direction = self.args['direct_list'][np.argmax(result[23:])]
        label_res.append(direction)
        # glasses
        if result[1] > self.args['glasses_threshold']:
            label_res.append('戴眼镜')
        else:
            label_res.append('未戴眼镜')
        # hat
        if result[0] > self.args['threshold']:
            label_res.append('戴帽子')
        else:
            label_res.append('未戴帽子')
        # bag
        bag = self.args['bag_list'][np.argmax(result[15:18])]
        bag_score = result[15 + np.argmax(result[15:18])]
        bag_label = bag if bag_score > self.args['threshold'] else '无包'
        label_res.append(bag_label)
        # upper
        sleeve = '长袖' if result[3] > result[2] else '短袖'
        label_res.append(sleeve)
        # lower
        lower_res = result[8:14]
        has_lower = False
        for i, l in enumerate(lower_res):
            if l > self.args['threshold']:
                lower_label = '{}'.format(self.args['lower_list'][i])
                has_lower = True
        if not has_lower:
            lower_label = '{}'.format(self.args['lower_list'][np.argmax(lower_res)])
        label_res.append(lower_label)
        # shoe
        shoe = '穿靴' if result[14] > self.args['threshold'] else '无靴'
        label_res.append(shoe)
        return label_res

    def _process(self, result, filter_result):
        hit = False
        if self.limit is None:
            self.limit = self.reserved_args['extra_model'][self.rec_model_name]
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
        model_name, targets = next(iter(filter_result.items()))
        if model_name != self.rec_model_name:
            LOGGER.error('Get wrong model result, expect {}, but get {}'.format(self.rec_model_name, model_name))
            return False
        if self.reinfer_result.get(self.time) is None:
            LOGGER.warning('Not found reinfer result, time={}'.format(self.time))
            return False
        self.reinfer_result[self.time]['result'].append((targets, self.reserved_data['rectangle']))
        if len(self.reinfer_result[self.time]['result']) < self.reinfer_result[self.time]['count']:
            return False
        reinfer_result_ = self.reinfer_result.pop(self.time)
        self.draw_image = reinfer_result_['draw_image']
        for target, xyxy in reinfer_result_['result']:
            hit = True
            rec_result = target[0]['output']
            attribute = self.__postprocess(rec_result)
            rectangle = {
                'xyxy': xyxy,
                'color': self.alert_color,
                'label': None,
                'ext': {
                    'attribute': attribute
                }
            }
            result['data']['bbox']['rectangles'].append(rectangle)
        result['hit'] = hit
        result['data']['bbox']['polygons'].update(polygons)
        return True

    def _filter(self, model_name, model_data):
        targets = []
        if model_name == self.rec_model_name and not self.reserved_data:
            return targets
        if self.strategy is None:
            self.strategy = self.reserved_args['strategy']
        model_conf = model_data['model_conf']
        engine_result = model_data['engine_result']
        if model_name == self.det_model_name:
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
                if not self._filter_by_roi(xyxy, strategy=self.strategy):
                    continue
                # 生成矩形框
                targets.append(self._gen_rectangle(
                    xyxy, self._get_color(model_conf['label'], label), label, engine_result_['conf']))
        elif model_name == self.rec_model_name:
            targets.append(engine_result)
        return targets
