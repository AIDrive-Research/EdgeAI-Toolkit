def is_point_in_segment(point, segment):
    """
    判断点是否在线段上
    Args:
        point: 点的坐标
        segment: 线段的两个端点
    Returns: True or False
    """
    start, end = segment
    if (min(start[0], end[0]) <= point[0] <= max(start[0], end[0]) and
            min(start[1], end[1]) <= point[1] <= max(start[1], end[1])):
        if start[0] == end[0] or start[1] == end[1]:
            return True
        if point[1] == start[1]:
            return point[0] == start[0]
        return (end[0] - start[0]) / (end[1] - start[1]) == (point[0] - start[0]) / (point[1] - start[1])
    else:
        return False


def is_ray_intersect_segment(point, segment):
    """
    判断射线是否与边相交，射线方向为水平向右
    Args:
        point: 射线起点
        segment: 线段的两个端点
    Returns: True or False
    """
    start, end = segment
    if start[1] == end[1]:
        return False
    if start[0] < point[0] and end[0] < point[0]:
        return False
    if start[1] > point[1] and end[1] > point[1]:
        return False
    if start[1] < point[1] and end[1] < point[1]:
        return False
    if (start[1] == point[1] and end[1] > point[1]) or (end[1] == point[1] and start[1] > point[1]):
        return False
    node_x = end[0] - (end[1] - point[1]) * (end[0] - start[0]) / (end[1] - start[1])
    if node_x < point[0]:
        return False
    return True


def get_polygon_edges(polygon):
    """
    获取多边形的边
    Args:
        polygon: 多边形的顶点
    Returns: 边的列表
    """
    edges = []
    for i in range(len(polygon) - 1):
        edges.append((polygon[i], polygon[i + 1]))
    edges.append((polygon[len(polygon) - 1], polygon[0]))
    return edges


def is_point_in_polygon(point, polygon):
    """
    判断点是否在多边形内
    Args:
        point: 点的坐标
        polygon: 多边形的顶点
    Returns: True or False
    """
    n = 0
    for edge in get_polygon_edges(polygon):
        if is_point_in_segment(point, edge):
            return True
        else:
            if is_ray_intersect_segment(point, edge):
                n += 1
    return 1 == n % 2


def is_point_in_rectangle(point, xyxy):
    """
    判断点是否在矩形内
    Args:
        point: 点的坐标
        xyxy: 矩形的左上角和右下角坐标
    Returns: True or False
    """
    x, y = point
    x_min, y_min, x_max, y_max = xyxy
    return x_min <= x <= x_max and y_min <= y <= y_max


def is_rectangle_intersect(xyxy1, xyxy2):
    """
    判断两个矩形是否相交
    Args:
        xyxy1: 矩形1的左上角和右下角坐标
        xyxy2: 矩形2的左上角和右下角坐标
    Returns: True or False
    """
    x_min_1, y_min_1, x_max_1, y_max_1 = xyxy1
    x_min_2, y_min_2, x_max_2, y_max_2 = xyxy2
    x_min = max(x_min_1, x_min_2)
    y_min = max(y_min_1, y_min_2)
    x_max = min(x_max_1, x_max_2)
    y_max = min(y_max_1, y_max_2)
    if x_min > x_max or y_min > y_max:
        return False
    return True


def cross(point1, point2, point3):
    """
    计算向量point1-point3和point1-point2的叉积
    Args:
        point1: 向量起点
        point2: 向量终点1
        point3: 向量终点2
    Returns: 叉积
    """
    x1 = point2[0] - point1[0]
    y1 = point2[1] - point1[1]
    x2 = point3[0] - point1[0]
    y2 = point3[1] - point1[1]
    return x1 * y2 - x2 * y1


def is_seg_intersect(segment1, segment2):
    """
    判断两个线段是否相交
    Args:
        segment1: 线段1的两个端点
        segment2: 线段2的两个端点
    Returns: True or False
    """
    point1, point2 = segment1
    point3, point4 = segment2
    xyxy1 = min(point1[0], point2[0]), min(point1[1], point2[1]), max(point1[0], point2[0]), max(point1[1], point2[1])
    xyxy2 = min(point3[0], point4[0]), min(point3[1], point4[1]), max(point3[0], point4[0]), max(point3[1], point4[1])
    # 跨立判断之前可以先做矩形相交判断（性能优化）
    if (is_rectangle_intersect(xyxy1, xyxy2) and (
            cross(point1, point2, point3) * cross(point1, point2, point4) <= 0) and (
            cross(point3, point4, point1) * cross(point3, point4, point2) <= 0)):
        return True
    return False


def calc_iou(xyxy1, xyxy2):
    """
    计算两个矩形的IOU
    Args:
        xyxy1: 矩形1的左上角和右下角坐标
        xyxy2: 矩形2的左上角和右下角坐标
    Returns: IOU
    """
    left_max = max(xyxy1[0], xyxy2[0])
    right_min = min(xyxy1[2], xyxy2[2])
    up_max = max(xyxy1[1], xyxy2[1])
    down_min = min(xyxy1[3], xyxy2[3])
    if left_max >= right_min or down_min <= up_max:
        return 0
    s1 = (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1])
    s2 = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])
    s_cross = (down_min - up_max) * (right_min - left_max)
    return s_cross / (s1 + s2 - s_cross)
