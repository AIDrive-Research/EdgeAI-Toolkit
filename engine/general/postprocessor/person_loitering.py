from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from tracker import Tracker
from window.ratio_window import RatioWindow
from .utils.cv_utils.geo_utils import calc_iou


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.tracker = None
        self.iou = None
        self.length = None
        self.threshold = None
        self.max_retain = 0
        self.targets = {}
        self.check_interval = 1

    def __check_lost_target(self, tracker_result):
        for track_id in list(self.targets.keys()):
            if track_id not in tracker_result:
                self.targets[track_id]['lost'] += 1
            else:
                self.targets[track_id]['lost'] = 0
            if self.targets[track_id]['lost'] > self.max_retain:
                LOGGER.info('Target lost, source_id={}, alg_name={}, track_id={}, pre_target={}'.format(
                    self.source_id, self.alg_name, track_id, self.targets[track_id]['pre_target']))
                del self.targets[track_id]
        return True

    def _process(self, result, filter_result):
        hit = False
        if self.iou is None:
            self.iou = self.reserved_args['iou']
        if self.length is None:
            self.length = self.reserved_args['length']
        if self.threshold is None:
            if self.length != 0:
                self.threshold = self.reserved_args['threshold'] / self.length
                self.threshold = 0 if self.threshold < 0 else self.threshold
                self.threshold = 1 if self.threshold > 1 else self.threshold
            else:
                self.threshold = 1
            LOGGER.info('source_id={}, alg_name={}, length={}, threshold={}'.format(
                self.source_id, self.alg_name, self.length, self.threshold))
        polygons = self._gen_polygons()
        if self.tracker is None:
            self.tracker = Tracker(self.frame_interval)
            self.max_retain = self.tracker.track_buffer + 1
            LOGGER.info('Init tracker, source_id={}, alg_name={}, track_buffer={}'.format(
                self.source_id, self.alg_name, self.tracker.track_buffer))
        model_name, rectangles = next(iter(filter_result.items()))
        # 目标跟踪
        tracker_result = self.tracker.track(rectangles)
        # 检查丢失目标
        self.__check_lost_target(tracker_result)
        for track_id, rectangle in tracker_result.items():
            target = self.targets.get(track_id)
            if target is None:
                target = {
                    'window': RatioWindow(self.length, self.threshold),
                    'lost': 0,
                    'hit': False,
                    'pre_target': None,
                    'time': self.time
                }
                self.targets[track_id] = target
            if target['pre_target'] is not None and self.time - target['time'] > self.check_interval:
                if calc_iou(target['pre_target']['xyxy'], rectangle['xyxy']) < self.iou:
                    target['hit'] = True
                else:
                    target['hit'] = False
            if target['pre_target'] is None or self.time - target['time'] > self.check_interval:
                target['time'] = self.time
                target['pre_target'] = rectangle
            if target['window'].insert({'time': self.time, 'data': {'hit': target['hit']}}):
                hit = True
                rectangle['color'] = self.alert_color
                rectangle['label'] = self.alert_label[0]
            else:
                rectangle['color'] = self.non_alert_color
            result['data']['bbox']['rectangles'].append(rectangle)
        result['hit'] = hit
        result['data']['bbox']['polygons'].update(polygons)
        return True
