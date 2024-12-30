import cv2
import numpy as np

from logger import LOGGER
from model import RknnModel


class Model(RknnModel):
    default_args = {
        # detection
        'img_size': 640,
        'nms_thres': 0.5,
        'conf_thres': 0.3,
        'rec_size': (48, 168),
        'anchor': [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    }

    def __init__(self, acc_id, name, conf):
        super().__init__(acc_id, name, conf, ['det', 'rec'])
        self.color = ['黑牌', '蓝牌', '绿牌', '白牌', '黄牌']
        self.plate_name = (r'#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危'
                           r'0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品')

    def __nms_boxes(self, boxes):
        index = np.argsort(boxes[:, 4])[::-1]
        keep = []
        while index.size > 0:
            i = index[0]
            keep.append(i)
            x1 = np.maximum(boxes[i, 0], boxes[index[1:], 0])
            y1 = np.maximum(boxes[i, 1], boxes[index[1:], 1])
            x2 = np.minimum(boxes[i, 2], boxes[index[1:], 2])
            y2 = np.minimum(boxes[i, 3], boxes[index[1:], 3])
            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)
            inter_area = w * h
            union_area = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + (
                    boxes[index[1:], 2] - boxes[index[1:], 0]) * (boxes[index[1:], 3] - boxes[index[1:], 1])
            iou = inter_area / (union_area - inter_area)
            idx = np.where(iou <= self.nms_thres)[0]
            index = index[idx + 1]
        return keep

    @staticmethod
    def __restore_box(boxes, r, left, top):
        boxes[:, [0, 2, 5, 7, 9, 11]] -= left
        boxes[:, [1, 3, 6, 8, 10, 12]] -= top
        boxes[:, [0, 2, 5, 7, 9, 11]] /= r
        boxes[:, [1, 3, 6, 8, 10, 12]] /= r
        return boxes

    @staticmethod
    def __get_rotate_crop_image(img, points):
        assert len(points) == 4, 'shape of points must be 4*2'
        img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
        points = points.astype(np.float32)
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    @staticmethod
    def __softmax(x, axis=1):
        # 计算指数
        exp_x = np.exp(x)
        # 对每行求和
        sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
        # 计算softmax
        softmax_output = exp_x / sum_exp_x
        return softmax_output

    @staticmethod
    def __get_split_merge(img):
        h, w, c = img.shape
        img_upper = img[0:int(5 / 12 * h), :]
        img_lower = img[int(1 / 3 * h):, :]
        img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
        new_img = np.hstack((img_upper, img_lower))
        return new_img

    def __post_process(self, outputs, scale, dw, dh, image_shape):
        choice = outputs[:, :, 4] > self.conf_thres
        outputs = outputs[choice]
        outputs[:, 13:15] *= outputs[:, 4:5]
        box = outputs[:, :4]
        boxes = self._xywh2xyxy(box)
        score = np.max(outputs[:, 13:15], axis=-1, keepdims=True)
        index = np.argmax(outputs[:, 13:15], axis=-1).reshape(-1, 1)
        output = np.concatenate((boxes, score, outputs[:, 5:13], index), axis=1)
        reserve_ = self.__nms_boxes(output)
        output = output[reserve_]
        output = self.__restore_box(output, scale, dw, dh)
        return output

    def __decode(self, preds):
        pre = 0
        newPreds = []
        index = []
        for i in range(len(preds)):
            if preds[i] != 0 and preds[i] != pre:
                newPreds.append(preds[i])
                index.append(i)
            pre = preds[i]
        return newPreds, index

    def __infer_det(self, image):
        # 预处理
        scale = 1
        raw_width, raw_height = image.shape[1], image.shape[0]
        if max(image.shape[:2]) != self.img_size:
            scale = self.img_size / max(image.shape[:2])
            if raw_height > raw_width:
                image = cv2.resize(image, (int(raw_width * scale), self.img_size))
            else:
                image = cv2.resize(image, (self.img_size, int(raw_height * scale)))
        image, dw, dh = self._letterbox(image, (self.img_size, self.img_size))
        image = np.expand_dims(image, axis=0)
        outputs = self._rknn_infer('det', [image])
        # 后处理
        bbox_list = self.__post_process(outputs[0], scale, dw, dh, image.shape)
        return bbox_list

    def __infer_rec(self, layer_num, pad):
        if layer_num:
            # double
            pad = self.__get_split_merge(pad)
        data = pad[:, :, ::-1]
        data, _, _ = self._letterbox(data, self.rec_size, stretch=True)
        data = np.expand_dims(data, axis=0)
        outputs = self._rknn_infer('rec', [data])
        if outputs:
            preds = self.__softmax(outputs[0], axis=2)
            prob = np.max(preds, axis=2).reshape(-1)
            index = np.argmax(preds, axis=2).reshape(-1)
            new_preds, new_index = self.__decode(index)
            rec_conf = prob[new_index]
            plate_code = ""
            for i in new_preds:
                plate_code += self.plate_name[i]
            if len(plate_code) < 7:
                return None, None, None
            color_preds = self.__softmax(outputs[1])
            color_index = np.argmax(color_preds)
            plate_type = self.color[color_index]
        else:
            plate_code, rec_conf = '', 0.0
        if plate_code is not None and len(plate_code) < 7:
            plate_code = None
        return plate_code, plate_type, rec_conf

    def _load_args(self, args):
        try:
            self.img_size = args.get('img_size', self.default_args['img_size'])
            self.nms_thres = args.get('nms_thres', self.default_args['nms_thres'])
            self.conf_thres = args.get('conf_thres', self.default_args['conf_thres'])
            self.rec_size = args.get('rec_size', self.default_args['rec_size'])
        except:
            LOGGER.exception('_load_args')
            return False
        return True

    def infer(self, data, **kwargs):
        """
        车牌识别
        Args:
            data: 图像数据，ndarray类型，RGB格式（BGR格式需转换）
        Returns: infer_result
        """
        infer_result = []
        if self.status:
            try:
                image = data
                out_list = self.__infer_det(image)
                for out in out_list:
                    xyxy = out[:4].astype(int)
                    score = out[4]
                    land_marks = out[5:13].reshape(4, 2).astype(int)
                    layer_num = int(out[13])
                    pad = self.__get_rotate_crop_image(image, land_marks)
                    plate_code, plate_type, rec_conf = self.__infer_rec(layer_num, pad)
                    if plate_code:
                        obj = {
                            'plate_code': plate_code,
                            'plate_type': plate_type,
                            'det_conf': round(score, 2),
                            'rec_conf': rec_conf.tolist()
                        }
                        xyxy = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        obj['xyxy'] = [xyxy[0] if xyxy[0] >= 0 else 0,
                                       xyxy[1] if xyxy[1] >= 0 else 0,
                                       xyxy[2] if xyxy[2] <= image.shape[1] else image.shape[1],
                                       xyxy[3] if xyxy[3] <= image.shape[0] else image.shape[1]]
                        infer_result.append(obj)
            except:
                LOGGER.exception('infer')
        return infer_result
