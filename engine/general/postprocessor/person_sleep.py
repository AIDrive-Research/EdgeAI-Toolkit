from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from tracker import Tracker
from window.ratio_window import RatioWindow
from .utils import json_utils
from .utils.cv_utils.color_utils import rgb_reverse
from .utils.cv_utils.crop_utils import crop_rectangle
from .utils.cv_utils.geo_utils import calc_iou
from .utils.image_utils import base64_to_opencv, opencv_to_base64


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.det_model_name = 'person'
        self.cls_model_name = 'sleep_classify'
        self.strategy = None
        self.iou = None
        self.length = None
        self.threshold = None
        self.tracker = None
        self.max_retain = 0
        self.targets = {}
        self.check_interval = 1
        self.pre_n = 5
        self.reinfer_result = {}
        self.timeout = None
        self.sleep_label = 0

    def __check_lost_target(self, tracker_result):
        for track_id in list(self.targets.keys()):
            if track_id not in tracker_result:
                self.targets[track_id]['lost'] += 1
            else:
                self.targets[track_id]['lost'] = 0
            if self.targets[track_id]['lost'] > self.max_retain:
                LOGGER.info('Target lost, source_id={}, alg_name={}, track_id={}, pre_target={}'.format(
                    self.source_id, self.alg_name, track_id, self.targets[track_id]['pre_targets']))
                del self.targets[track_id]
        return True

    def __reinfer(self, filter_result, polygons):
        person_rectangles = filter_result.get(self.det_model_name)
        if person_rectangles is None:
            LOGGER.error('Perons model result is None!')
            return False
        # 目标跟踪
        tracker_result = self.tracker.track(person_rectangles)
        # 检查丢失目标
        self.__check_lost_target(tracker_result)
        count = 0
        rectangles = []
        for track_id, rectangle in tracker_result.items():
            target = self.targets.get(track_id)
            if target is None:
                target = {
                    'window': RatioWindow(self.length, self.threshold),
                    'lost': 0,
                    'hit': False,
                    'pre_targets': [],
                    'time': self.time
                }
                self.targets[track_id] = target
            for polygon in polygons.values():
                if self._is_rectangle_in_polygon(rectangle['xyxy'], polygon['polygon'], self.strategy):
                    if len(target['pre_targets']) == self.pre_n and self.time - target['time'] > self.check_interval:
                        match_num = 0
                        for pre_target in target['pre_targets']:
                            if calc_iou(pre_target['xyxy'], rectangle['xyxy']) > self.iou:
                                match_num += 1
                        if match_num == self.pre_n:
                            target['hit'] = True
                        else:
                            target['hit'] = False
                    break
            if len(target['pre_targets']) < self.pre_n or self.time - target['time'] > self.check_interval:
                target['time'] = self.time
                target['pre_targets'].append(rectangle)
            if len(target['pre_targets']) > self.pre_n:
                target['pre_targets'].pop(0)
            if target['window'].insert({'time': self.time, 'data': {'hit': target['hit']}}):
                target['window'] = RatioWindow(self.length, self.threshold)
                draw_image = base64_to_opencv(self.draw_image)
                cropped_image = crop_rectangle(draw_image, rectangle['xyxy'])
                cropped_image = rgb_reverse(cropped_image)
                source_data = {
                    'source_id': self.source_id,
                    'time': self.time * 1000000,
                    'infer_image': opencv_to_base64(cropped_image),
                    'draw_image': None,
                    'reserved_data': {
                        'specified_model': [self.cls_model_name],
                        'rectangle': rectangle,
                        'unsort': True
                    }
                }
                self.rq_source.put(json_utils.dumps(source_data))
                count += 1
            else:
                rectangles.append(rectangle)
            if count > 0:
                self.reinfer_result[self.time] = {
                    'count': count,
                    'draw_image': self.draw_image,
                    'result': []
                }
        return count, rectangles

    def __check_expire(self):
        for time in list(self.reinfer_result.keys()):
            if time < self.time - self.timeout:
                LOGGER.warning('Reinfer result expired, source_id={}, alg_name={}, time={}, timeout={}'.format(
                    self.source_id, self.alg_name, time, self.timeout))
                del self.reinfer_result[time]
        return True

    def _process(self, result, filter_result):
        hit = False
        if self.iou is None:
            self.iou = self.reserved_args['iou']
        if self.length is None:
            self.length = self.reserved_args['length']
            self.pre_n = min(self.length, self.pre_n)
        if self.timeout is None:
            self.timeout = (self.frame_interval / 1000) * 2
            LOGGER.info('source_id={}, alg_name={}, timeout={}'.format(self.source_id, self.alg_name, self.timeout))
        if self.threshold is None:
            if self.length != 0:
                self.threshold = self.reserved_args['threshold'] / self.length
                self.threshold = 0 if self.threshold < 0 else self.threshold
                self.threshold = 1 if self.threshold > 1 else self.threshold
            else:
                self.threshold = 1
            LOGGER.info('source_id={}, alg_name={}, length={}, threshold={}'.format(
                self.source_id, self.alg_name, self.length, self.threshold))
        if self.tracker is None:
            self.tracker = Tracker(self.frame_interval)
            self.max_retain = self.tracker.track_buffer + 1
            LOGGER.info('Init tracker, source_id={}, alg_name={}, track_buffer={}'.format(
                self.source_id, self.alg_name, self.tracker.track_buffer))
        polygons = self._gen_polygons()
        person_rectangles = []
        if not self.reserved_data:
            count, person_rectangles = self.__reinfer(filter_result, polygons)
            if not count:
                self.__check_expire()
                result['hit'] = False
                result['data']['bbox']['rectangles'].extend(person_rectangles)
                result['data']['bbox']['polygons'].update(polygons)
                return True
            return False
        self.__check_expire()
        model_name, targets = next(iter(filter_result.items()))
        if model_name != self.cls_model_name:
            LOGGER.error('Get wrong model result, expect {}, but get {}'.format(self.cls_model_name, model_name))
            return False
        if self.reinfer_result.get(self.time) is None:
            LOGGER.warning('Not found reinfer result, time={}'.format(self.time))
            return False
        self.reinfer_result[self.time]['result'].append((targets, self.reserved_data['rectangle']))
        if len(self.reinfer_result[self.time]['result']) < self.reinfer_result[self.time]['count']:
            return False
        reinfer_result_ = self.reinfer_result.pop(self.time)
        self.draw_image = reinfer_result_['draw_image']
        for targets, rectangle in reinfer_result_['result']:
            if not targets:
                person_rectangles.append(rectangle)
                continue
            hit = True
            rectangle['color'] = self.alert_color
            rectangle['label'] = self.alert_label[0]
            person_rectangles.append(rectangle)
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(person_rectangles)
        result['data']['bbox']['polygons'].update(polygons)
        return True

    def _filter(self, model_name, model_data):
        targets = []
        if model_name == self.cls_model_name and not self.reserved_data:
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
        elif model_name == self.cls_model_name:
            score = engine_result['output'][self.sleep_label]
            if score >= model_conf['args']['conf_thres']:
                targets.append(engine_result)
        return targets
