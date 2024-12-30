import copy
import os
import random
import sys

import db.cross_line_counting
from config import conf
from db.operation import get_session
from logger import LOGGER
from redis_queue import RedisQueue
from .utils import json_utils
from .utils.cv_utils.geo_utils import is_point_in_polygon, is_seg_intersect
from .utils.time_utils import get_weekday, get_day_second

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURRENT_PATH)


class Postprocessor:
    rq_source = RedisQueue(conf.redis_queue_source, 300, conf.redis_host, conf.redis_port, conf.redis_db)

    def __init__(self, source_id, alg_name):
        self.source_id = source_id
        self.alg_name = alg_name
        # 告警颜色：红色
        self.alert_color = (0, 0, 255)
        # 非告警颜色：绿色
        self.non_alert_color = (0, 255, 0)
        # marker颜色：蓝色
        self.marker_color = (255, 0, 0)
        # 红色、绿色、蓝色
        self.exclude_color_pool = {
            self.alert_color, self.non_alert_color, self.marker_color
        }
        # 黄色、洋红、青色、橙色、紫色、海军蓝、深绿
        self.color_pool = {
            (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255), (128, 0, 128), (128, 0, 0), (0, 128, 0)
        }
        self.label2color = {}
        # 参数
        self.time = None
        self.alert_label = None
        self.bbox = None
        self.scale = None
        self.frame_interval = None
        self.reserved_args = None
        self.alg_type = None
        self.reserved_data = None
        # 结果
        self.draw_image = None
        self.result = {
            'hit': False,
            'data': {
                'bbox': {
                    'rectangles': [],
                    'polygons': {},
                    'lines': {}
                },
                'custom': {}
            }
        }

    def _process(self, result, filter_result):
        """
        处理过滤之后的结果，并生成最终结果，按需重写
        Args:
            result: 合并后的最终结果
            filter_result: 过滤后的结果
        Returns: True or False
        """
        hit = False
        polygons = self._gen_polygons()
        if filter_result:
            # 通常只会有一个模型
            model_name, rectangles = next(iter(filter_result.items()))
            for rectangle in rectangles:
                if rectangle['label'] in self.alert_label:
                    hit = True
                    rectangle['color'] = self.alert_color
            result['hit'] = hit
            result['data']['bbox']['rectangles'].extend(rectangles)
            result['data']['bbox']['polygons'].update(polygons)
        return True

    def _filter(self, model_name, model_data):
        """
        过滤掉不符合条件的目标，按需重写
        Args:
            model_name: 模型名称
            model_data: 模型数据
        Returns: 过滤后的结果
        """
        targets = []
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
            # 坐标缩放
            xyxy = self._scale(engine_result_['xyxy'])
            # 过滤掉不在多边形内的目标
            if not self._filter_by_roi(xyxy):
                continue
            # 生成矩形框
            targets.append(self._gen_rectangle(
                xyxy, self._get_color(model_conf['label'], label), label, engine_result_['conf']))
        return targets

    def postprocess(self, args, draw_image):
        """
        后处理，无需重写
        Args:
            args: 后处理参数
            draw_image: 用于绘制的图像数据，base64编码
        Returns: True or False, 合并后的最终结果
        """
        status = False
        result = copy.deepcopy(self.result)
        self.draw_image = draw_image
        try:
            self.time = args['time']
            if self.alert_label is None:
                self.alert_label = args['alert_label']
            if self.bbox is None:
                self.bbox = args['bbox']
            if self.scale is None:
                self.scale = args['scale']
            if self.frame_interval is None:
                self.frame_interval = args['frame_interval']
            if self.reserved_args is None:
                self.reserved_args = args['reserved_args']
            if self.alg_type is None:
                self.alg_type = args['alg_type']
            self.reserved_data = args.get('reserved_data', {})
            if self._is_in_plan(self.time, args['plan']):
                filter_result = {}
                for model_name, model_data in args['model'].items():
                    filter_result[model_name] = self._filter(model_name, model_data)
                status = self._process(result, filter_result)
                LOGGER.info('status:{}', status)
        except:
            LOGGER.exception('postprocess')
            LOGGER.error('Postprocess failed, source_id={}, alg_name={}'.format(self.source_id, self.alg_name))
        return status, result, self.draw_image

    ######################################################工具函数######################################################
    @staticmethod
    def _is_in_plan(timestamp, plan=None):
        """
        是否在计划时间内，无需重写
        Args:
            timestamp: 时间戳
            plan: 计划时间
        Returns: True or False
        """
        if plan is None:
            return True
        weekday = get_weekday(timestamp)
        if weekday is None:
            return False
        day_second = get_day_second(timestamp)
        if day_second is None:
            return False
        weekday_plan = plan.get(weekday)
        if weekday_plan is None:
            return False
        for time_slice in weekday_plan:
            if time_slice[0] <= day_second < time_slice[1]:
                return True
        return False

    @staticmethod
    def _get_label(label_conf, label: int):
        """
        获取label，无需重写
        Args:
            label_conf: label配置
            label: 标签，int类型
        Returns: label，str类型
        """
        label = label_conf['class2label'].get(str(label))
        if label is None:
            return None
        return label_conf.get('label_map', {}).get(label, label)

    def _gen_random_color(self, seed=1):
        """
        生成随机颜色，无需重写
        Args:
            seed: 随机种子
        Returns: 随机颜色
        """
        for color in self.color_pool:
            if color not in self.label2color.values():
                return color
        random.seed(seed)
        while True:
            color = tuple([random.randint(0, 255) for _ in range(3)])
            if color not in self.exclude_color_pool:
                return color

    def _get_color(self, label_conf, label: str):
        """
        获取颜色，无需重写
        Args:
            label_conf: label配置
            label: 标签，str类型
        Returns: 颜色
        """
        color = self.label2color.get(label)
        if color:
            return color
        label2color = label_conf.get('label2color', {})
        color = label2color.get(label)
        if color and isinstance(color, list) and 3 == len(color):
            self.label2color[label] = tuple(color)
            return color
        for exclude_color in self.label2color.values():
            self.exclude_color_pool.add(exclude_color)
        for exclude_color in label2color.values():
            self.exclude_color_pool.add(exclude_color)
        color = self._gen_random_color()
        self.label2color[label] = color
        return color

    @staticmethod
    def _get_point(rectangle, strategy):
        """
        获取矩形框的点，无需重写
        Args:
            rectangle: 矩形框
            strategy: 取点策略，center: 中心点，bottom: 下边沿中心点，top: 上边沿中心点，left: 左边沿中心点，right: 右边沿中心点
        Returns: 点坐标
        """
        if strategy not in ['center', 'bottom', 'top', 'left', 'right']:
            strategy = 'center'
            LOGGER.warning('Invalid strategy: {}, use default strategy: center'.format(strategy))
        if 'center' == strategy:
            return (rectangle[0] + rectangle[2]) / 2, (rectangle[1] + rectangle[3]) / 2
        elif 'bottom' == strategy:
            return (rectangle[0] + rectangle[2]) / 2, rectangle[3]
        elif 'top' == strategy:
            return (rectangle[0] + rectangle[2]) / 2, rectangle[1]
        elif 'left' == strategy:
            return rectangle[0], (rectangle[1] + rectangle[3]) / 2
        elif 'right' == strategy:
            return rectangle[2], (rectangle[1] + rectangle[3]) / 2

    def _is_rectangle_in_polygon(self, rectangle, polygon, strategy):
        """
        判断矩形框是否在多边形内，无需重写
        Args:
            rectangle: 矩形框
            polygon: 多边形顶点
            strategy: 取点策略，center: 中心点，bottom: 下边沿中心点，top: 上边沿中心点，left: 左边沿中心点，right: 右边沿中心点
        Returns: True or False
        """
        point = self._get_point(rectangle, strategy)
        return is_point_in_polygon(point, polygon)

    def _cross_line_counting(self, rectangle1, rectangle2, line, direction, strategy='center'):
        """
        跨线计数，无需重写
        Args:
            rectangle1: 上一次矩形框
            rectangle2: 当前矩形框
            line: 线段
            direction: 方向，l-r+: 左减右增，l+r-: 左增右减，u+d-: 上增下减，u-d+: 上减下增，r+: 右增，l+: 左增，u+: 上增，d+: 下增
            strategy: 取点策略，center: 中心点，bottom: 下边沿中心点，top: 上边沿中心点，left: 左边沿中心点，right: 右边沿中心点
        Returns: 计数结果
        """
        if direction not in ['l-r+', 'l+r-', 'u+d-', 'u-d+', 'r+', 'l+', 'u+', 'd+']:
            LOGGER.error('Invalid direction: {}'.format(direction))
            return None
        point1 = self._get_point(rectangle1, strategy)
        point2 = self._get_point(rectangle2, strategy)
        if is_seg_intersect((point1, point2), line):
            if 'l-r+' == direction:
                if point2[0] > point1[0]:
                    return {'increase': 1}
                elif point2[0] < point1[0]:
                    return {'decrease': 1}
            elif 'l+r-' == direction:
                if point2[0] < point1[0]:
                    return {'increase': 1}
                elif point2[0] > point1[0]:
                    return {'decrease': 1}
            elif 'u+d-' == direction:
                if point2[1] < point1[1]:
                    return {'increase': 1}
                elif point2[1] > point1[1]:
                    return {'decrease': 1}
            elif 'u-d+' == direction:
                if point2[1] > point1[1]:
                    return {'increase': 1}
                if point2[1] < point1[1]:
                    return {'decrease': 1}
            elif 'r+' == direction:
                if point2[0] > point1[0]:
                    return {'count': 1}
            elif 'l+' == direction:
                if point2[0] < point1[0]:
                    return {'count': 1}
            elif 'u+' == direction:
                if point2[1] < point1[1]:
                    return {'count': 1}
            elif 'd+' == direction:
                if point2[1] > point1[1]:
                    return {'count': 1}
        return None

    @staticmethod
    def _merge_cross_line_counting_result(result1, result2):
        """
        合并跨线计数结果，无需重写
        Args:
            result1: 上一次计数结果
            result2: 当前计数结果
        Returns: True or False
        """
        if result2.get('count') is not None:
            result1['count'] += result2['count']
        else:
            if result2.get('increase') is not None:
                result1['increase'] += result2['increase']
                result1['delta'] += result2['increase']
            elif result2.get('decrease') is not None:
                result1['decrease'] += result2['decrease']
                result1['delta'] -= result2['decrease']
        return True

    @staticmethod
    def _set_ext(obj: dict, *args, **kwargs):
        """
        设置扩展字段
        Args:
            obj: dict对象
            *args: 待删除的key
            **kwargs: 待添加的key-value
        Returns: True or False
        """
        if 'ext' not in obj:
            obj['ext'] = {}
        for key in args:
            if key in obj.get('ext', {}):
                del obj['ext'][key]
        for key, value in kwargs.items():
            obj['ext'][key] = value
        return True

    @staticmethod
    def _get_ext(obj: dict, key, pop=False):
        """
        获取扩展字段的值
        Args:
            obj: dict对象
            key: key
            pop: 是否删除key
        Returns: value
        """
        if not pop:
            return obj.get('ext', {}).get(key)
        return obj.get('ext', {}).pop(key, None)

    def _gen_rectangle(self, xyxy, color, label, conf_, **kwargs):
        """
        生成矩形框，无需重写
        Args:
            xyxy: 矩形左上右下顶点坐标
            color: 颜色
            label: 标签
            conf_: 置信度
            **kwargs: 扩展参数
        Returns: 矩形框
        """
        rectangle = {
            'xyxy': xyxy,
            'color': color,
            'label': label,
            'conf': conf_
        }
        self._set_ext(rectangle, **kwargs)
        return rectangle

    def _gen_polygons(self, data=None):
        """
        生成多边形，无需重写
        Returns: 多边形
        """
        polygons = self.bbox.get('polygons', []) if data is None else data.get('polygons', [])
        polygons_ = {}
        for polygon in polygons:
            polygons_[polygon['id']] = {
                'name': polygon['name'],
                'polygon': polygon['polygon'],
                'color': self.marker_color
            }
            self._set_ext(polygons_[polygon['id']])
        if 'counting' == self.alg_type:
            for polygon in polygons:
                self._set_ext(polygons_[polygon['id']], result=None)
        return polygons_

    def _gen_lines(self, data=None):
        """
        生成线段，无需重写
        Returns: 线段
        """
        lines = self.bbox.get('lines', []) if data is None else data.get('lines', [])
        lines_ = {}
        for line in lines:
            lines_[line['id']] = {
                'name': line['name'],
                'line': line['line'],
                'color': self.marker_color
            }
            self._set_ext(lines_[line['id']])
        if 'cross_line_counting' == self.alg_type:
            with get_session() as db_session:
                for line in lines:
                    status, row = db.cross_line_counting.query(db_session, line['id'])
                    if status and row:
                        self._set_ext(
                            lines_[line['id']],
                            direction=line['direction'],
                            action=line['action'],
                            result=json_utils.loads(row['result'])
                        )
                    else:
                        del lines_[line['id']]
        return lines_

    def _scale(self, xyxy):
        """
        坐标缩放，无需重写
        Args:
            xyxy: 坐标
        Returns: 缩放后的坐标
        """
        return [int(x * self.scale) for x in xyxy]

    @staticmethod
    def _filter_by_conf(model_conf, conf_):
        """
        过滤掉置信度低于阈值的目标，无需重写
        Args:
            model_conf: 模型配置
            conf_: 阈值
        Returns: True or False
        """
        return conf_ >= model_conf['args']['conf_thres']

    def _filter_by_label(self, model_conf, label):
        """
        过滤掉不在label列表中的目标，无需重写
        Args:
            model_conf: 模型配置
            label: 标签
        Returns: label
        """
        return self._get_label(model_conf['label'], label)

    def _filter_by_roi(self, xyxy, strategy='center'):
        """
        过滤掉中心点不在多边形内的目标，无需重写
        Args:
            xyxy: 坐标
            strategy: 取点策略，center: 中心点，bottom: 下边沿中心点，top: 上边沿中心点，left: 左边沿中心点，right: 右边沿中心点
        Returns: True or False
        """
        polygons = self.bbox.get('polygons', [])
        if polygons:
            for polygon in polygons:
                if self._is_rectangle_in_polygon(xyxy, polygon['polygon'], strategy):
                    return True
            else:
                return False
        return True
