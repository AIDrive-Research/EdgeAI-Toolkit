from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from tracker import Tracker


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.strategy = None
        self.tracker = None
        self.max_retain = 0
        self.targets = {}
        self.lines = None

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
        if self.strategy is None:
            self.strategy = self.reserved_args['strategy']
        polygons = self._gen_polygons()
        if self.lines is None:
            self.lines = self._gen_lines()
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
                    'lost': 0,
                    'pre_target': None
                }
                self.targets[track_id] = target
            if target['pre_target'] is not None:
                for line in self.lines.values():
                    result_ = self._cross_line_counting(
                        target['pre_target']['xyxy'], rectangle['xyxy'],
                        line['line'], line['ext']['direction'], self.strategy)
                    if result_ is not None:
                        hit = True
                        rectangle['color'] = self.alert_color
                        self._merge_cross_line_counting_result(line['ext']['result'], result_)
                        break
                else:
                    rectangle['color'] = self.non_alert_color
            else:
                rectangle['color'] = self.non_alert_color
            target['pre_target'] = rectangle
            result['data']['bbox']['rectangles'].append(rectangle)
        result['hit'] = hit
        result['data']['bbox']['polygons'].update(polygons)
        result['data']['bbox']['lines'].update(self.lines)
        return True
