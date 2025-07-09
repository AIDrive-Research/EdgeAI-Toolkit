import cv2
import numpy as np

from logger import LOGGER
from model import RknnModel


class Model(RknnModel):
    default_args = {
        'max_num': 5,
        # detection
        'img_size': 640,
        'nms_thres': 0.4,
        'conf_thres': 0.5,
        'fmc': 3,
        'feat_stride_fpn': [8, 16, 32],
        'num_anchors': 2,
        'center_cache': {},
        'use_kps': False,
        'face_size': 112
    }

    def __init__(self, acc_id, name, conf):
        super().__init__(acc_id, name, conf, ['det', 'landmark'])

    @staticmethod
    def __expand_crop(image_shape, rect, expand_ratio=0.3):
        imgh, imgw, _ = image_shape
        xmin, ymin, xmax, ymax = rect[0], rect[1], rect[2], rect[3]
        h_half = (ymax - ymin) * (1 + expand_ratio) / 2.
        w_half = (xmax - xmin) * (1 + expand_ratio) / 2.
        if h_half > w_half * 4 / 3:
            w_half = h_half * 0.75
        center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
        ymin = max(0, int(center[0] - h_half))
        ymax = min(imgh - 1, int(center[0] + h_half))
        xmin = max(0, int(center[1] - w_half))
        xmax = min(imgw - 1, int(center[1] + w_half))
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def __distance2bbox(points, distance, dw, dh, max_shape=None):
        x1 = points[:, 0] - distance[:, 0] - dw
        y1 = points[:, 1] - distance[:, 1] - dh
        x2 = points[:, 0] + distance[:, 2] - dw
        y2 = points[:, 1] + distance[:, 3] - dh
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def __distance2kps(points, distance, dw, dh, max_shape=None):
        distance = distance.reshape(distance.shape[0], -1, 2)
        px = points[:, None, :2] + distance
        px[:, :, 0] -= dw
        px[:, :, 1] -= dh
        if max_shape is not None:
            px[:, :, 0] = np.clip(px[:, :, 0], 0, max_shape[1])
            px[:, :, 1] = np.clip(px[:, :, 1], 0, max_shape[0])
        return px.reshape(points.shape[0], -1)

    def __post_process(self, outputs, dw, dh, image_shape):
        score_list = []
        bbox_list = []
        kps_list = []
        image_shape = image_shape[1:]
        for idx, stride in enumerate(self.feat_stride_fpn):
            scores = outputs[idx][0, :, :]
            bbox_preds = outputs[idx + self.fmc][0, :, :] * stride
            if self.use_kps:
                kps_preds = outputs[idx + self.fmc * 2][0, :, :] * stride
            height = self.img_size // stride
            width = self.img_size // stride
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self.num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self.num_anchors, axis=1).reshape((-1, 2))
            pos_inds = np.where(scores >= self.conf_thres)[0]
            bboxes = self.__distance2bbox(anchor_centers, bbox_preds, dw, dh, image_shape)
            for i in range(len(bboxes)):
                bboxes[i] = self.__expand_crop(image_shape, bboxes[i], expand_ratio=0.1)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            if len(pos_scores):
                score_list.append(pos_scores)
                bbox_list.append(pos_bboxes)
                if self.use_kps:
                    kpss = self.__distance2kps(anchor_centers, kps_preds, dw, dh, image_shape)
                    kpss = kpss.reshape((kpss.shape[0], -1, 2))
                    pos_kpss = kpss[pos_inds]
                    kps_list.append(pos_kpss)
        return bbox_list, score_list, kps_list

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
        bbox_list, score_list, kps_list = self.__post_process(outputs, dw, dh, image.shape)
        if not bbox_list:
            return None
        # 合并 score_list、bbox_list、kps_list
        scores = np.vstack(score_list)
        bboxes = np.vstack(bbox_list) / scale
        kpss = np.vstack(kps_list) / scale if self.use_kps else None
        # 计算检测结果
        order = scores.ravel().argsort()[::-1]
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order]
        keep = self._nms_boxes(pre_det[:, :4], pre_det[:, -1])
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order][keep, :, :]
        # 限制人脸个数
        if self.max_num > 0 and det.shape[0] > self.max_num:
            det = det[:self.max_num]
            if self.use_kps:
                kpss = kpss[:self.max_num]
        return det[:, :4], det[:, -1], kpss

    def __infer_landmark(self, image):
        scale = self.face_size / max(image.shape[:2])
        image, dw, dh = self._letterbox(image, (self.face_size, self.face_size))
        image = np.expand_dims(image, axis=0)
        outputs = self._rknn_infer('landmark', [image])
        landmarks = outputs[1].reshape(-1, 2)
        landmarks[:, 0] = (landmarks[:, 0] * image.shape[2] - dw) / scale
        landmarks[:, 1] = (landmarks[:, 1] * image.shape[1] - dh) / scale
        return landmarks

    def _load_args(self, args):
        try:
            self.max_num = args.get('max_num', self.default_args['max_num'])
            self.img_size = args.get('img_size', self.default_args['img_size'])
            self.nms_thres = args.get('nms_thres', self.default_args['nms_thres'])
            self.conf_thres = args.get('conf_thres', self.default_args['conf_thres'])
            self.fmc = args.get('fmc', self.default_args['fmc'])
            self.feat_stride_fpn = args.get('feat_stride_fpn', self.default_args['feat_stride_fpn'])
            self.num_anchors = args.get('num_anchors', self.default_args['num_anchors'])
            self.center_cache = args.get('center_cache', self.default_args['center_cache'])
            self.use_kps = args.get('use_kps', self.default_args['use_kps'])
            self.face_size = args.get('face_size', self.default_args['face_size'])
        except:
            LOGGER.exception('_load_args')
            return False
        return True

    def infer(self, data, **kwargs):
        """
        人脸检测+人脸关键点提取
        Args:
            data: 图像数据，ndarray类型，RGB格式（BGR格式需转换）
        Returns: infer_result
        """
        infer_result = []
        if self.status:
            try:
                image = data
                ret = self.__infer_det(image)
                if ret is not None:
                    boxes, scores, _ = ret
                    for i, xyxy in enumerate(boxes):
                        face_crop = image[int(xyxy[1]): int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        face_landmark = self.__infer_landmark(face_crop)
                        face_landmark[:, 0] = face_landmark[:, 0] + xyxy[0]
                        face_landmark[:, 1] = face_landmark[:, 1] + xyxy[1]
                        face_landmark = [[int(x), int(y)] for x, y in face_landmark]
                        obj = {
                            'conf': round(float(scores[i]), 2),
                            'landmark': face_landmark
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
