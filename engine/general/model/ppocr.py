import math
import os

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from logger import LOGGER
from model import RknnModel


class Model(RknnModel):
    default_args = {
        # detection
        'max_num': 5,
        'img_size': 640,
        'conf_thres': 0.5
    }

    def __init__(self, acc_id, name, conf):
        super().__init__(acc_id, name, conf, ['ppdet', 'pprec'])
        self.infer_before_process_op, self.det_re_process_op = self.__get_process()

    def _load_model(self, path):
        try:
            if super()._load_model(path):
                self.postprocess_op = process_pred(os.path.join(path, 'ppocr_keys_v1.txt'), 'ch', True)
        except:
            LOGGER.exception('_load_model')
            return False
        return True

    def _load_args(self, args):
        try:
            self.max_num = args.get('max_num', self.default_args['max_num'])
            self.img_size = args.get('img_size', self.default_args['img_size'])
            self.conf_thres = args.get('conf_thres', self.default_args['conf_thres'])
        except:
            LOGGER.exception('_load_args')
            return False
        return True

    def __get_process(self):
        det_db_thresh = 0.3
        max_candidates = 2000
        unclip_ratio = 1.6
        use_dilation = True

        pre_process_list = [{
            'DetResizeForTest': {
                'image_shape': [self.img_size, self.img_size]
            }
        }, {
            'NormalizeImage': {
                'std': [1., 1., 1.],
                'mean': [0., 0., 0.],
                'scale': '1.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]

        infer_before_process_op = self.__create_operators(pre_process_list)
        det_re_process_op = DBPostProcess(det_db_thresh, self.conf_thres, max_candidates, unclip_ratio, use_dilation)
        return infer_before_process_op, det_re_process_op

    def __create_operators(self, op_param_list, global_config=None):
        """
        create operators based on the config

        Args:
            params(list): a dict list, used to create some operators
        """
        assert isinstance(op_param_list, list), ('operator config should be a list')
        ops = []
        for operator in op_param_list:
            assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
            op_name = list(operator)[0]
            param = {} if operator[op_name] is None else operator[op_name]
            if global_config is not None:
                param.update(global_config)
            op = eval(op_name)(**param)
            ops.append(op)
        return ops

    @staticmethod
    def __transform(data, ops=None):
        """ transform """
        if ops is None:
            ops = []
        for op in ops:
            data = op(data)
            if data is None:
                return None
        return data

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, scores, image_shape):
        img_height, img_width = image_shape[1:3]
        dt_boxes_new = []
        scores_new = []
        for box, score in zip(dt_boxes, scores):
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
            scores_new.append(score)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes, scores_new

    def sorted_boxes(self, dt_boxes, scores):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        comb = zip(dt_boxes, scores)
        # sorted_comb = sorted(comb, key=lambda x: (x[0][0][1], x[0][0][0]))
        sorted_comb = sorted(comb, key=lambda x: x[1], reverse=True)
        sorted_comb = sorted_comb[:self.max_num]
        sorted_boxes, scores = zip(*sorted_comb)
        _boxes = list(sorted_boxes)
        return _boxes, scores

    def resize_norm_img(self, img, max_wh_ratio):
        x = 320
        imgC, imgH, imgW = [int(v) for v in "3,48,320".split(",")]
        assert imgC == img.shape[2]
        imgW = int((32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = x
        if math.ceil(imgH * ratio) > imgW:
            resized_w = x
        resized_image = cv2.resize(img, (x, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1))
        padding_im = np.zeros((imgC, imgH, x), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __infer_det(self, image):
        dt_boxes, scores = [], []
        try:
            # 预处理
            image = image.copy()
            data = {'image': image}
            data = self.__transform(data, self.infer_before_process_op)
            image, shape_list = data
            image = np.expand_dims(image, axis=0)
            shape_list = np.expand_dims(shape_list, axis=0)
            image = np.transpose(image, (0, 2, 3, 1))
            outputs = self._rknn_infer('ppdet', [image])
            # 后处理
            post_res = self.det_re_process_op(outputs[0], shape_list)
            dt_boxes = post_res[0]['points']
            scores = post_res[0]['scores']
            if scores:
                dt_boxes, scores = self.filter_tag_det_res(dt_boxes, scores, image.shape)
                dt_boxes, scores = self.sorted_boxes(dt_boxes, scores)
        except:
            LOGGER.exception('__infer_det')
        return dt_boxes, scores

    def get_rotate_crop_image(self, img, points):
        img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img, M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def get_img_res(self, image, process_op):
        h, w = image.shape[:2]
        image = self.resize_norm_img(image, w * 1.0 / h)
        image = image[np.newaxis, :]
        image = np.transpose(image, (0, 2, 3, 1))
        outputs = self._rknn_infer('pprec', [image])
        result = process_op(outputs[0])
        return result

    def __infer_rec(self, image, bbox):
        img_crop = self.get_rotate_crop_image(image, bbox)
        res = self.get_img_res(img_crop, self.postprocess_op)
        return res[0]

    @staticmethod
    def min_enclosing_rect(vertices):
        # 计算矩形的边界框
        min_x = np.min(vertices[:, 0])
        max_x = np.max(vertices[:, 0])
        min_y = np.min(vertices[:, 1])
        max_y = np.max(vertices[:, 1])

        # 获取最小外接矩形的四个顶点
        rect_vertices = [int(min_x), int(min_y), int(max_x), int(max_y)]
        return rect_vertices

    def infer(self, image, **kwargs):
        """
        字符检测+字符识别
        Args:
            image: 图像数据，ndarray类型，RGB格式（BGR格式需转换）
        Returns: infer_result
        """
        infer_result = []
        if self.status:
            try:
                image = image[..., ::-1]
                dt_boxes, dt_confs = self.__infer_det(image)
                if dt_boxes:
                    for bbox, det_conf in zip(dt_boxes, dt_confs):
                        rec_info = self.__infer_rec(image, bbox)
                        obj = {
                            'xyxy': bbox.tolist(),
                            'det_conf': round(float(det_conf), 2),
                            'rec': rec_info[0],
                            'rec_conf': round(float(rec_info[1]), 2)
                        }
                        infer_result.append(obj)
            except:
                LOGGER.exception('infer')
        return infer_result


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=2.0, use_dilation=False, **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(np.array(segmentation[batch_index]).astype(np.uint8), self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
            boxes_batch.append({'points': boxes, 'scores': scores})
        return boxes_batch


class process_pred(object):
    def __init__(self, character_dict_path=None, character_type='ch', use_space_char=False):
        self.character_str = ''
        with open(character_dict_path, 'rb') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip('\n').strip('\r\n')
                self.character_str += line
        if use_space_char:
            self.character_str += ' '
        dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        result_list = []
        ignored_tokens = [0]
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def __call__(self, preds, label=None):
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label


class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        self.image_shape = kwargs['image_shape']

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape
        img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_type1(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # return img, np.array([ori_h, ori_w])
        return img, [ratio_h, ratio_w]


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
        data['image'] = (img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list
