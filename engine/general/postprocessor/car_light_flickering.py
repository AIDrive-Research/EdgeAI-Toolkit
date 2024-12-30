import cv2 

from skimage.metrics import structural_similarity as compare_ssim

from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from tracker import Tracker
from .utils.cv_utils.geo_utils import is_point_in_rectangle
from .utils.cv_utils.geo_utils import calc_iou
from .utils.image_utils import base64_to_opencv


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.car_model_name = 'common'
        self.light_model_name = 'car_light'
        self.targets = {}
        self.iou = None
        self.threshold = None
        self.tracker = None
        self.length = None 
        self.max_retain = 0
        self.pre_image = None
        self.check_interval = 1

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
    
    def __compute_similarity(self, xyxy, pre_image, cur_image):
        pre_roi = pre_image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        cur_roi = cur_image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        score, _ = compare_ssim(pre_roi, cur_roi, full=True)
        return score

    def __check_flicker(self, target):
        if 2 not in target or 1 not in target:
            return False
        # 计算交替变化次数
        change_count = 0  
        for i in range(1, len(target)):
            if target[i] != target[i-1]:  # 相邻的元素不同
                change_count += 1  # 计入交替
        LOGGER.info('change_count: {}'.format(change_count))
        if change_count >= self.threshold:
            return True
        else:
            return False

    def _process(self, result, filter_result):
        hit = False
        if self.iou is None:
            self.iou = self.reserved_args['iou']
        if self.length is None:
            self.length = self.reserved_args['length']
        if self.threshold is None:
            self.threshold = self.reserved_args['threshold']
        polygons = self._gen_polygons()
        if self.tracker is None:
            self.tracker = Tracker(self.frame_interval)
            self.max_retain = self.tracker.track_buffer + 1
            LOGGER.info('Init tracker, source_id={}, alg_name={}, track_buffer={}'.format(
                self.source_id, self.alg_name, self.tracker.track_buffer))
        light_rectangles = filter_result.get(self.light_model_name)
        if light_rectangles is None:
            LOGGER.error('Car light model result is None!')
            return False
        car_rectangles = filter_result.get(self.car_model_name)
        if car_rectangles is None:
            LOGGER.error('Common model result is None!')
            return False
        LOGGER.info(car_rectangles)
        draw_image = base64_to_opencv(self.draw_image)
        gray_image = cv2.cvtColor(draw_image, cv2.COLOR_BGR2GRAY)
        tracker_result = self.tracker.track(car_rectangles)
        # 检查丢失目标
        self.__check_lost_target(tracker_result)
        for track_id, car_rectangle in tracker_result.items():
            target = self.targets.get(track_id)
            if target is None:
                target = {
                    'lost': 0,
                    'hit': [],
                    'pre_target': None,
                    'stationary': None,
                    'time': self.time
                }
                self.targets[track_id] = target
            if self.time - target['time'] < self.check_interval:
                result['data']['bbox']['rectangles'].append(car_rectangle)
                continue
            if target['pre_target'] is not None:
                if calc_iou(target['pre_target']['xyxy'], car_rectangle['xyxy']) < self.iou:
                    target['stationary'] = False
                else:
                    target['stationary'] = True
            # 车辆静止时判断车灯
            if target['stationary'] and self.pre_image is not None:
                for light_rectangle in light_rectangles:
                    point = self._get_point(light_rectangle['xyxy'], 'center')
                    if is_point_in_rectangle(point, car_rectangle['xyxy']):
                        # 检测到车灯，判断与上一帧是否相似
                        score = self.__compute_similarity(light_rectangle['xyxy'], self.pre_image, gray_image)
                        if score >= 0.8:
                            target['hit'].append(2)
                        else:
                            target['hit'].append(1)
                        break
                else:
                    target['hit'].append(0)
                if len(target['hit']) > self.length:
                    target['hit'].pop(0)        
                if len(target['hit']) == self.length and target['hit'][0]:
                    if target['hit'].count(2) == len(target['hit']):
                        hit = True
                        car_rectangle['color'] = self.alert_color
                        car_rectangle['label'] = '常亮异常'
                    elif self.__check_flicker(target['hit']):
                            hit = True
                            car_rectangle['color'] = self.alert_color
                            car_rectangle['label'] = '频闪异常'
            target['pre_target'] = car_rectangle
            result['data']['bbox']['rectangles'].append(car_rectangle)
        self.pre_image = gray_image.copy()
        result['hit'] = hit
        result['data']['bbox']['polygons'].update(polygons)
        return True

