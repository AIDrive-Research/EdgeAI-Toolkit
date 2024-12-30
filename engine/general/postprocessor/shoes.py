import math

import cv2
import numpy as np

import gv
from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from .utils import json_utils
from .utils.cv_utils.color_utils import rgb_reverse
from .utils.cv_utils.crop_utils import crop_rectangle
from .utils.image_utils import base64_to_opencv, opencv_to_base64, base64_to_bytes


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)
        self.person_model_name = 'pose'
        self.mask_model_name = 'segment'
        self.shoes_model_name = 'ppe'
        self.index = None
        self.group_type = None
        self.similarity = None
        self.limit = None
        self.timeout = None
        self.reinfer_result = {}
        self.min_scale = 0.052
        self.angle_th = 150
        self.image_width, self.image_height = None, None

    def __shoes_data_prepare(self, xyxy, key_points, mask):
        keypoints = []
        for point in key_points:
            keypoints.append([point[0] - xyxy[0], point[1] - xyxy[1], point[2]])
        keypoints = np.array(keypoints)
        attention_points = keypoints[11:]
        bool_points = attention_points[:, 2] < self.reserved_args['pose_threshold']
        attention_points[bool_points, :] = 0
        # 胯、膝盖、脚踝
        hip_points = attention_points[0:2]
        knee_points = attention_points[2:4]
        ankle_points = attention_points[4:]
        data = {
            'attention_points': attention_points,
            'hip_points': hip_points,
            'knee_points': knee_points,
            'ankle_points': ankle_points,
            'bool_points': bool_points,
            'mask': mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]].copy()
        }
        return data

    @staticmethod
    def __calculate_distance(points):
        x1, y1, _ = points[0]
        x2, y2, _ = points[1]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    @staticmethod
    def __estimate_foot_direction(p1, p2):
        # 计算两点的中点坐标
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        # 计算两点的斜率
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        # 计算中垂线的斜率
        slope = -1 / slope
        intercept = mid_y - slope * mid_x
        # 返回斜率、截距
        return (int(mid_x), int(mid_y)), slope, intercept

    @staticmethod
    def __perpendicular_line_equation(slope, c):
        """
        计算中垂线的方程：y = mx + c
        """

        def line_function(y):
            """
            根据斜截式方程，给定 y 计算对应的 x 值
            """
            return (y - c) / slope

        return line_function

    def __draw_perpendicular_line(self, ankle_points, mask):
        try:
            ankle_center = None
            if ankle_points[0][2] and ankle_points[1][2]:
                ankle_center, slope, c = self.__estimate_foot_direction(ankle_points[0], ankle_points[1])
                # 在mask图上画出中垂线
                line_func = self.__perpendicular_line_equation(slope, c)
                # 计算中垂线的起点和终点坐标
                y1 = 0
                x1 = int(line_func(y1))
                y2 = mask.shape[0] - 1
                x2 = int(line_func(y2))
                # 画出中垂线
                cv2.line(mask, (x1, y1), (x2, y2), 0, thickness=2)
        except:
            pass
        return ankle_center, mask

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
    def __remove_disconnected_components(image, point):
        result = image
        try:
            image_copy = image.copy()
            _, mask, _, _ = cv2.floodFill(image_copy, None, point, 0)
            result = image - mask
        except:
            pass
        return result

    @staticmethod
    def __calculate_foot_roi_rectangle(ankle_point, roi_length, img_w, img_h):
        # 计算矩形的中心点
        center_x = int(ankle_point[0])
        center_y = min(img_h, int(ankle_point[1] + roi_length / 4))
        # 计算扩大后的矩形框的坐标
        new_min_x = max(0, center_x - roi_length / 2)
        new_max_x = min(img_w - 1, new_min_x + roi_length - 1)
        new_min_y = max(0, center_y - roi_length / 2)
        new_max_y = min(img_h - 1, new_min_y + roi_length - 1)
        return [int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)]

    @staticmethod
    def __find_largest_connected_component(image):
        # 寻找连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

        # 寻找最大的连通区域（除去背景）
        max_label = 1
        max_size = stats[1, cv2.CC_STAT_AREA]
        for label in range(2, num_labels):
            if stats[label, cv2.CC_STAT_AREA] > max_size:
                max_label = label
                max_size = stats[label, cv2.CC_STAT_AREA]

        # 创建一个只包含最大连通区域的二值图像
        largest_connected_component = (labels == max_label).astype('uint8') * 255
        return largest_connected_component

    @staticmethod
    def __minimum_bounding_rectangle(image):
        # 找到不为零的点的坐标
        nonzero_points = np.column_stack(np.nonzero(image))
        # 找到最小外接矩形的四个顶点
        min_x = np.min(nonzero_points[:, 1])
        max_x = np.max(nonzero_points[:, 1])
        min_y = np.min(nonzero_points[:, 0])
        max_y = np.max(nonzero_points[:, 0])
        box = [min_x, min_y, max_x, max_y]
        return box

    def __adjust_feet_roi(self, mask, roi_box):
        try:
            # 查找最大连通区域
            largest_connected_component = self.__find_largest_connected_component(mask)
            # 计算最小外接矩形
            min_rect = self.__minimum_bounding_rectangle(largest_connected_component)
            result = [
                min_rect[0] + roi_box[0],
                min_rect[1] + roi_box[1],
                min(roi_box[0] + min_rect[2], roi_box[2]),
                min(roi_box[1] + min_rect[3], roi_box[3])
            ]
        except:
            return None
        return result

    @staticmethod
    def __point_inside_rectangle(point, rect):
        # 矩形的左上角和右下角坐标
        rect_x1, rect_y1, rect_x2, rect_y2 = rect
        # 点的坐标
        point_x, point_y, _ = point
        # 判断点是否在矩形框内部
        inside = (rect_x1 <= point_x <= rect_x2) and (rect_y1 <= point_y <= rect_y2)
        return inside

    def __gen_shoes_region(self, image, data):
        shoes_image_list = []
        img_h, img_w = image.shape[:2]
        # [1] 两个跨的点必须存在
        if not (data['hip_points'][0][2] and data['hip_points'][1][2]):
            return shoes_image_list
        # [2] 跨部两点顺序决定人体朝向
        if data['hip_points'][0][0] < data['hip_points'][1][0]:
            return shoes_image_list
        dis_hip = self.__calculate_distance(data['hip_points'])
        # [3] 如果两个脚踝点都存在，画出两个脚踝点的中垂线
        _, data['mask'] = self.__draw_perpendicular_line(data['ankle_points'], data['mask'])
        for i in range(2):
            hip_point = data['hip_points'][i]
            knee_point = data['knee_points'][i]
            ankle_point = data['ankle_points'][i]
            if not (hip_point[2] and knee_point[2] and ankle_point[2]):
                continue
            v1 = hip_point[:2] - knee_point[:2]
            v2 = ankle_point[:2] - knee_point[:2]
            angle_leg = self.__vector_angle(v1, v2)
            v3 = np.array([knee_point[0], ankle_point[1]]) - knee_point[:2]
            angle_ground = self.__vector_angle(v2, v3)
            thigh_length = np.linalg.norm(v1)
            calf_length = np.linalg.norm(v2)
            # 胯部两点距离过滤
            if dis_hip < max(thigh_length, calf_length) / 3:
                continue
            # 腿夹角过滤 & # 小腿与地面角度过滤
            if angle_leg < 160 or angle_ground > 30:
                continue
            # 将脚踝点之外的连通区域置为0
            foot_mask = self.__remove_disconnected_components(data['mask'], (int(ankle_point[0]), int(ankle_point[1])))
            # 根据脚踝点选定区域
            scale = 0.8
            roi_length = calf_length * scale
            roi_box = self.__calculate_foot_roi_rectangle(ankle_point, roi_length, img_w, img_h)
            roi_mask = foot_mask[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]]
            bbox = self.__adjust_feet_roi(roi_mask, roi_box)
            if bbox is None or (not self.__point_inside_rectangle(ankle_point, bbox)):
                continue
            max_size = max(img_h, img_w)
            if min((bbox[3] - bbox[1]) / max_size, (bbox[2] - bbox[0]) / max_size) < self.min_scale:
                continue
            shoes_image_list.append(image[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        return shoes_image_list

    @staticmethod
    def __vector_angle(vec1, vec2):
        x, y = vec1, vec2
        Lx = np.sqrt(x.dot(x))
        Ly = np.sqrt(y.dot(y))
        cos_angle = x.dot(y) / (Lx * Ly)
        angle = np.arccos(cos_angle)
        angle = angle * 360 / 2 / np.pi
        return angle

    def __reinfer(self, filter_result):
        person_rectangles = filter_result.get(self.person_model_name)
        if person_rectangles is None:
            LOGGER.error('Person model result is None!')
            return False
        mask_targets = filter_result.get(self.mask_model_name)
        if mask_targets is None:
            LOGGER.error('Mask model result is None!')
            return False
        mask = mask_targets[0].get('mask')
        if mask is None:
            LOGGER.error('Mask result is None!')
            return False
        person_results = []
        person_rectangles = sorted(person_rectangles, key=lambda x: x['conf'], reverse=True)
        draw_image = base64_to_opencv(self.draw_image)
        count = 0
        count_shoes = 0
        for i in range(len(person_rectangles)):
            if count >= self.limit:
                break
            xyxy = person_rectangles[i]['xyxy']
            cropped_image = crop_rectangle(draw_image, xyxy)
            cropped_image = rgb_reverse(cropped_image)
            data = self.__shoes_data_prepare(xyxy, self._get_ext(person_rectangles[i], 'key_points'), mask)
            shoes_image = self.__gen_shoes_region(cropped_image, data)
            if not shoes_image:
                person_results.append(self._gen_rectangle(xyxy, (0, 255, 255), '人', None))
                continue
            for shoes_image in shoes_image:
                source_data = {
                    'source_id': self.source_id,
                    'time': self.time * 1000000,
                    'infer_image': opencv_to_base64(shoes_image),
                    'draw_image': None,
                    'reserved_data': {
                        'specified_model': [self.shoes_model_name],
                        'xyxy': xyxy,
                        'unsort': True
                    }
                }
                self.rq_source.put(json_utils.dumps(source_data))
                count_shoes += 1
            count += 1
        if count_shoes > 0:
            self.reinfer_result[self.time] = {
                'count': count_shoes,
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

    def __gen_segment_mask(self, model_conf, engine_result):
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        for engine_result_ in engine_result:
            # 过滤掉置信度低于阈值的目标
            if engine_result_['conf'] < model_conf['args']['conf_thres']:
                continue
            result_mask = cv2.imdecode(np.frombuffer(base64_to_bytes(engine_result_['mask']), np.uint8),
                                       cv2.IMREAD_GRAYSCALE)
            result_mask = cv2.resize(result_mask, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            _, result_mask = cv2.threshold(result_mask, 127, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_or(mask, result_mask)
        return mask

    def __process_blacklist(self, feature):
        id_, score = self.index.search(feature, self.similarity) if self.index is not None else (None, None)
        if id_ is not None:
            shoes_info = self.index.query(id_)
            if shoes_info:
                return True, shoes_info['name']
        return False, '人'

    def __process_whitelist(self, feature):
        id_, score = self.index.search(feature, self.similarity) if self.index is not None else (None, None)
        if id_ is not None:
            shoes_info = self.index.query(id_)
            if shoes_info:
                return False, shoes_info['name']
        return True, '未穿防护鞋'

    def _process(self, result, filter_result):
        hit = False
        if self.index is None:
            self.index = gv.index_dic.get(self.reserved_args['group_id'])
        if self.group_type is None:
            self.group_type = self.reserved_args['group_type']
        if self.limit is None:
            self.limit = self.reserved_args['extra_model'][self.shoes_model_name]
        if self.similarity is None:
            self.similarity = max(self.reserved_args['similarity'] - 0.1, 0)
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
        if model_name != self.shoes_model_name:
            LOGGER.error('Get wrong model result, expect {}, but get {}'.format(self.shoes_model_name, model_name))
            return False
        if self.reinfer_result.get(self.time) is None:
            LOGGER.warning('Not found reinfer result, time=gan{}'.format(self.time))
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
        if model_name == self.shoes_model_name and not self.reserved_data:
            return targets
        if self.image_height is None:
            draw_image = base64_to_opencv(self.draw_image)
            self.image_height, self.image_width = draw_image.shape[:2]
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
        elif model_name == self.mask_model_name:
            mask = self.__gen_segment_mask(model_conf, engine_result)
            targets.append({
                'mask': mask
            })
        elif model_name == self.shoes_model_name:
            targets.append({
                'feature': engine_result
            })
        return targets
