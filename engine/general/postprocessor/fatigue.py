import numpy as np

from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from tracker import Tracker
from window.ratio_window import RatioWindow


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.threshold = None
        self.length = None
        self.sensitivity = None
        self.tracker = None
        self.max_retain = 0
        self.targets = {}
        self.model_name = 'face_landmark'
        self.left_eye_index = [35, 36, 37, 39, 41, 42]
        self.right_eye_index = [89, 90, 91, 93, 95, 96]

    @staticmethod
    def __get_box(points):
        eye_points_array = np.array(points)
        min_x = np.min(eye_points_array[:, 0])
        max_x = np.max(eye_points_array[:, 0])
        min_y = np.min(eye_points_array[:, 1])
        max_y = np.max(eye_points_array[:, 1])
        return [int(min_x), int(min_y), int(max_x), int(max_y)]

    def __get_eye_rectangle(self, xyxy, hit):
        return self._gen_rectangle(
            xyxy=xyxy,
            color=self.alert_color if hit else self.non_alert_color,
            label=None,
            conf_=None
        )

    def __check_lost_target(self, tracker_result):
        for track_id in list(self.targets.keys()):
            if track_id not in tracker_result:
                self.targets[track_id]['lost'] += 1
            else:
                self.targets[track_id]['lost'] = 0
            if self.targets[track_id]['lost'] > self.max_retain:
                LOGGER.info('Target lost, source_id={}, alg_name={}, track_id={}'.format(
                    self.source_id, self.alg_name, track_id))
                del self.targets[track_id]
        return True

    def _process(self, result, filter_result):
        hit = False
        polygons = self._gen_polygons()
        if self.sensitivity is None:
            self.sensitivity = self.reserved_args['sensitivity']
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
        if self.tracker is None:
            self.tracker = Tracker(self.frame_interval)
            self.max_retain = self.tracker.track_buffer + 1
            LOGGER.info('Init tracker, source_id={}, alg_name={}, track_buffer={}'.format(
                self.source_id, self.alg_name, self.tracker.track_buffer))
        face_landmark_rectangles = filter_result.get(self.model_name)
        if face_landmark_rectangles is None:
            LOGGER.error('Face_landmark model result is None!')
            return result
        # 目标跟踪
        tracker_result = self.tracker.track(face_landmark_rectangles)
        # 检查丢失目标
        self.__check_lost_target(tracker_result)
        for track_id, rectangle in tracker_result.items():
            target = self.targets.get(track_id)
            if target is None:
                target = {
                    'window': RatioWindow(self.length, self.threshold),
                    'lost': 0,
                    'hit': False
                }
                self.targets[track_id] = target
                continue
            hit_flag = False
            lm = self._get_ext(rectangle, 'landmark', pop=True)
            left_eye_points = [lm[i] for i in self.left_eye_index]
            left_eye_box = self.__get_box(left_eye_points)
            right_eye_points = [lm[i] for i in self.right_eye_index]
            right_eye_box = self.__get_box(right_eye_points)
            if lm[39][0] - lm[35][0] == 0 or lm[93][0] - lm[89][0] == 0:
                result['data']['bbox']['rectangles'].append(self.__get_eye_rectangle(left_eye_box, hit_flag))
                result['data']['bbox']['rectangles'].append(self.__get_eye_rectangle(right_eye_box, hit_flag))
                continue
            era_left = (abs(lm[36][1] - lm[41][1]) + abs(lm[37][1] - lm[42][1])) / (abs(lm[39][0] - lm[35][0]) * 2)
            era_right = (abs(lm[90][1] - lm[95][1]) + abs(lm[91][1] - lm[96][1])) / (abs(lm[93][0] - lm[89][0]) * 2)
            if (era_left + era_right) / 2 < self.sensitivity:
                hit_flag = True
                target['hit'] = True
                rectangle['color'] = self.alert_color
            result['data']['bbox']['rectangles'].append(self.__get_eye_rectangle(left_eye_box, hit_flag))
            result['data']['bbox']['rectangles'].append(self.__get_eye_rectangle(right_eye_box, hit_flag))
            if target['window'].insert({'time': self.time, 'data': {'hit': target['hit']}}):
                hit = True
                rectangle['color'] = self.alert_color
                rectangle['label'] = self.alert_label[0]
                target['window'] = RatioWindow(self.length, self.threshold)
            else:
                rectangle['color'] = self.non_alert_color
        result['hit'] = hit
        result['data']['bbox']['rectangles'].extend(tracker_result.values())
        result['data']['bbox']['polygons'].update(polygons)
        return result

    def _filter(self, model_name, model_data):
        targets = []
        model_conf = model_data['model_conf']
        engine_result = model_data['engine_result']
        for engine_result_ in engine_result:
            # 过滤掉置信度低于阈值的目标
            if not self._filter_by_conf(model_conf, engine_result_['conf']):
                continue
            # 过滤掉不在label列表中的目标
            label = self._filter_by_label(model_conf, 0)
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
                landmark=[[int(x * self.scale), int(y * self.scale)] for x, y in engine_result_['landmark']]))
        return targets
