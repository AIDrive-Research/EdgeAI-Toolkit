import cv2
import numpy as np

from logger import LOGGER
from postprocessor import Postprocessor as BasePostprocessor
from .utils import json_utils
from .utils.cv_utils.geo_utils import get_polygon_edges
from .utils.image_utils import base64_to_bytes
from .utils.unique_id_utils import get_object_id


class Postprocessor(BasePostprocessor):
    def __init__(self, source_id, alg_name):
        super().__init__(source_id, alg_name)

    @staticmethod
    def __get_line_polygon_intersect(line_points, polygon):
        line_vector = line_points[1] - line_points[0]
        polygon_edges = get_polygon_edges(polygon)
        intersections = []
        inter_points = []
        for edge in polygon_edges:
            p1, p2 = edge
            edge_vector = np.array(p2) - np.array(p1)
            denominator = np.cross(line_vector, edge_vector)
            if abs(denominator) > 1e-8:
                t = np.cross(p1 - line_points[0], edge_vector) / denominator
                u = np.cross(p1 - line_points[0], line_vector) / denominator
                if 0 <= t <= 1 and 0 <= u <= 1:
                    intersection = line_points[0] + t * line_vector
                    intersections.append({
                        'intersections': intersection.astype(np.int).tolist(),
                        'polygon_edge': edge,
                        'line_points': line_points.tolist(),
                        'polygon': json_utils.dumps(polygon)
                    })
        # 线在polygon外
        if not intersections:
            top = int(np.min(np.array(polygon)[:, 1]))
            bottom = int(np.max(np.array(polygon)[:, 1]))
            slope = line_vector[1] / line_vector[0]
            x_top = int((top - line_points[0][1]) / slope + line_points[0][0])
            x_bottom = int((bottom - line_points[1][1]) / slope + line_points[1][0])
            inter_points = [[x_top, top], [x_bottom, bottom]]
        return intersections, inter_points

    @staticmethod
    def __mask_to_line(mask):
        # mask拟合直线
        _, cols = mask.shape
        points = np.column_stack(np.where(mask > 0))
        points = points[:, [1, 0]]
        [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        line_points = np.array([(cols - 1, righty), (0, lefty)])
        return line_points

    @staticmethod
    def __get_polygon_top_bottom(polygon):
        min_top = None
        max_bottom = None
        polygon_points = np.array(polygon)
        center_y = np.mean(np.array(polygon_points)[:, 1])
        top_center = polygon_points[polygon_points[:, 1] < center_y]
        bottom_center = polygon_points[polygon_points[:, 1] > center_y]
        if top_center.size > 0:
            min_top = np.max(top_center[:, 1])
        if bottom_center.size > 0:
            max_bottom = np.min(bottom_center[:, 1])
        return min_top, max_bottom

    @staticmethod
    def __check_conveyor_belt(intersections, polygons_border):
        # 判断交点是否是侧边
        status = False
        for intersection in intersections:
            for info in intersection:
                polygon = info['polygon']
                top = polygons_border[polygon]['top']
                bottom = polygons_border[polygon]['bottom']
                polygon_edge_point_first = info['polygon_edge'][0]
                polygon_edge_point_second = info['polygon_edge'][1]
                if (polygon_edge_point_first[1] <= top and polygon_edge_point_second[1] <= top) or \
                        (polygon_edge_point_first[1] >= bottom and polygon_edge_point_second[1] >= bottom):
                    continue
                else:
                    status = True
        return status

    def __get_mask_polygon_intersect(self, mask, polygons):
        try:
            line_points = self.__mask_to_line(mask)
            intersections = []
            lines = []
            for polygon in polygons:
                intersection, inter_points = self.__get_line_polygon_intersect(line_points, polygon)
                if intersection:
                    line = []
                    intersections.append(intersection)
                    for info in intersection:
                        line.append(info['intersections'])
                    lines.append(line)
                else:
                    lines.append(inter_points)
        except:
            LOGGER.exception('__get_mask_polygon_intersect')
        return intersections, lines

    # def __show(self, mask, polygons, polygons_border, intersections):
    #     image = np.zeros((mask.shape[0], mask.shape[1], 3))
    #     cv2.polylines(image, [np.array(polygons)], isClosed=True, color=(0, 255, 0), thickness=2)
    #     for intersection in intersections:
    #         for info in intersection:
    #             polygon = info['polygon']
    #             top = polygons_border[polygon]['top']
    #             bottom = polygons_border[polygon]['bottom']
    #             cv2.line(image, (0, top), (image.shape[1], top), (255, 255, 255), 2)
    #             cv2.line(image, (0, bottom), (image.shape[1], bottom), (255, 255, 255), 2)
    #             inter = info['intersections']
    #             line_points = info['line_points']
    #             cv2.circle(image, (int(inter[0]), int(inter[1])), 3, (0, 0, 255), 3)
    #             cv2.circle(image, (int(inter[0]), int(inter[1])), 3, (0, 0, 255), 3)
    #             cv2.line(image, (line_points[0][0], line_points[0][1]), (line_points[1][0], line_points[1][1]),
    #                      (0, 255, 255), 2)
    #     cv2.imwrite('image.jpg', image)
    #     cv2.imwrite('mask.jpg', mask)
    #     return

    def _process(self, result, filter_result):
        hit = False
        polygons = self._gen_polygons()
        polygons_ = []
        polygons_border = {}
        for polygon in polygons.values():
            polygons_.append(polygon['polygon'])
            # polygon的上下内边界
            polygon_top, polygon_bottom = self.__get_polygon_top_bottom(polygon['polygon'])
            polygons_border[json_utils.dumps(polygon['polygon'])] = {
                'top': polygon_top,
                'bottom': polygon_bottom
            }
        model_name, rectangles = next(iter(filter_result.items()))
        lines = []
        for rectangle in rectangles:
            mask = rectangle['ext']['mask']
            # mask和polygons多边形的交点
            intersection, line = self.__get_mask_polygon_intersect(mask, polygons_)
            # 判断是否发生偏离
            status = self.__check_conveyor_belt(intersection, polygons_border)
            if status:
                hit = True
                for l_ in line:
                    data = {
                        'lines': [{
                            'id': get_object_id(),
                            'name': None,
                            'line': l_
                        }]
                    }
                    coveyor_line = self._gen_lines(data)
                    for _, v in coveyor_line.items():
                        v['color'] = self.alert_color
                    result['data']['bbox']['lines'].update(coveyor_line)
            else:
                for l_ in line:
                    data = {
                        'lines': [{
                            'id': get_object_id(),
                            'name': None,
                            'line': l_
                        }]
                    }
                    coveyor_line = self._gen_lines(data)
                    for _, v in coveyor_line.items():
                        v['color'] = self.non_alert_color
                    result['data']['bbox']['lines'].update(coveyor_line)
        result['hit'] = hit
        result['data']['bbox']['polygons'].update(polygons)
        return True

    def _filter(self, model_name, model_data):
        targets = []
        model_conf = model_data['model_conf']
        engine_result = model_data['engine_result']
        engine_result = sorted(engine_result, key=lambda x: x['conf'], reverse=True)
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
            mask = cv2.imdecode(np.frombuffer(base64_to_bytes(engine_result_['mask']), np.uint8), cv2.IMREAD_GRAYSCALE)
            height, width = mask.shape
            mask = cv2.resize(mask, (int(width * self.scale), int(height * self.scale)), interpolation=cv2.INTER_LINEAR)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            points = np.column_stack(np.where(mask > 0))[:, [1, 0]]
            # 生成矩形框
            targets.append(self._gen_rectangle(
                xyxy, self.non_alert_color, label, engine_result_['conf'],
                mask=mask, points=points))
        return targets
